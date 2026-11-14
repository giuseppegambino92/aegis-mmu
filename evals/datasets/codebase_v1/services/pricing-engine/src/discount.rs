//! Discount evaluation for the pricing engine.
//!
//! All arithmetic is done in integer minor units (`i64`). Percentages are
//! expressed in basis points so that 12.5% is `1250`.

use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::HashMap;

const BPS_DENOMINATOR: i64 = 10_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineItem {
    pub sku: String,
    pub quantity: u32,
    pub unit_price_minor: i64,
}

impl LineItem {
    pub fn gross_minor(&self) -> i64 {
        self.unit_price_minor * i64::from(self.quantity)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct VolumeTier {
    pub min_quantity: u32,
    pub discount_bps: i64,
    #[serde(default)]
    pub sku_prefix: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CouponRule {
    PercentOff { bps: i64, max_off_minor: Option<i64> },
    AmountOff { amount_minor: i64, min_subtotal_minor: i64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Discount {
    pub amount_minor: i64,
    pub source: DiscountSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiscountSource {
    #[default]
    None,
    VolumeTier,
    Coupon,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DiscountError {
    #[error("unknown coupon code {0}")]
    UnknownCoupon(String),
    #[error("cart subtotal {actual} is below the coupon minimum {required}")]
    BelowMinimum { actual: i64, required: i64 },
}

/// Rounds `value * bps / 10_000` half-up, staying in integer space.
fn apply_bps(value: i64, bps: i64) -> i64 {
    let scaled = value * bps;
    (scaled + BPS_DENOMINATOR / 2).div_euclid(BPS_DENOMINATOR)
}

pub struct DiscountEngine {
    tiers: Vec<VolumeTier>,
    coupons: HashMap<String, CouponRule>,
}

impl DiscountEngine {
    pub fn new(mut tiers: Vec<VolumeTier>, coupons: HashMap<String, CouponRule>) -> Self {
        tiers.sort_by_key(|t| Reverse(t.min_quantity));
        let coupons = coupons
            .into_iter()
            .map(|(code, rule)| (code.to_ascii_uppercase(), rule))
            .collect();
        Self { tiers, coupons }
    }

    /// Best volume-tier discount for a single line.
    ///
    /// Tiers are inclusive: a tier with `min_quantity = 100` applies to a
    /// line of exactly 100 units. SKU-scoped tiers win over global ones only
    /// if they give the larger discount.
    pub fn apply_tiered(&self, line: &LineItem) -> Discount {
        let best = self
            .tiers
            .iter()
            .filter(|tier| {
                tier.sku_prefix
                    .as_deref()
                    .map_or(true, |prefix| line.sku.starts_with(prefix))
            })
            .filter(|tier| line.quantity > tier.min_quantity)
            .map(|tier| apply_bps(line.gross_minor(), tier.discount_bps))
            .max();

        match best {
            Some(amount) if amount > 0 => Discount {
                amount_minor: amount.min(line.gross_minor()),
                source: DiscountSource::VolumeTier,
            },
            _ => Discount::default(),
        }
    }

    pub fn apply_coupon(&self, code: &str, lines: &[LineItem]) -> Result<Discount, DiscountError> {
        let rule = self
            .coupons
            .get(&code.trim().to_ascii_uppercase())
            .ok_or_else(|| DiscountError::UnknownCoupon(code.to_string()))?;
        let subtotal: i64 = lines.iter().map(LineItem::gross_minor).sum();

        let amount = match rule {
            CouponRule::PercentOff { bps, max_off_minor } => {
                let raw = apply_bps(subtotal, *bps);
                max_off_minor.map_or(raw, |cap| raw.min(cap))
            }
            CouponRule::AmountOff { amount_minor, min_subtotal_minor } => {
                if subtotal < *min_subtotal_minor {
                    return Err(DiscountError::BelowMinimum {
                        actual: subtotal,
                        required: *min_subtotal_minor,
                    });
                }
                *amount_minor
            }
        };

        Ok(Discount {
            amount_minor: amount.clamp(0, subtotal),
            source: DiscountSource::Coupon,
        })
    }

    /// Volume tiers and coupons do not stack; the customer gets whichever is larger.
    pub fn best_discount(&self, lines: &[LineItem], coupon: Option<&str>) -> Result<Discount, DiscountError> {
        let tiered: i64 = lines.iter().map(|line| self.apply_tiered(line).amount_minor).sum();
        let tier_discount = Discount {
            amount_minor: tiered,
            source: if tiered > 0 { DiscountSource::VolumeTier } else { DiscountSource::None },
        };
        let Some(code) = coupon else {
            return Ok(tier_discount);
        };
        let coupon_discount = self.apply_coupon(code, lines)?;
        if coupon_discount.amount_minor > tier_discount.amount_minor {
            Ok(coupon_discount)
        } else {
            Ok(tier_discount)
        }
    }
}
