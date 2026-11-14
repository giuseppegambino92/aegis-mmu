//! Discount evaluation for the pricing engine.
//!
//! All arithmetic is done in integer minor units (`i64`). Percentages are
//! expressed in basis points so that 12.5% is `1250`.
//!
//! The engine answers two questions for the checkout service:
//!
//! 1. What is the best discount for this cart (`best_discount`)?
//! 2. How is an order-level discount split across lines (`allocate_discount`),
//!    which the tax service and refunds need per line.

use serde::Deserialize;
use std::cmp::Reverse;
use std::collections::HashMap;

const BPS_DENOMINATOR: i64 = 10_000;
/// Coupon codes longer than this are rejected before lookup.
const MAX_COUPON_CODE_LEN: usize = 32;

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

    /// Gift cards and donations are never discounted.
    pub fn is_discountable(&self) -> bool {
        !(self.sku.starts_with("GC-") || self.sku.starts_with("DON-"))
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

/// One line's share of an order-level discount.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineAllocation {
    pub sku: String,
    pub discount_minor: i64,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DiscountError {
    #[error("unknown coupon code {0}")]
    UnknownCoupon(String),
    #[error("cart subtotal {actual} is below the coupon minimum {required}")]
    BelowMinimum { actual: i64, required: i64 },
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("discount {discount} exceeds the discountable subtotal {subtotal}")]
    DiscountTooLarge { discount: i64, subtotal: i64 },
}

/// Rounds `value * bps / 10_000` half-up, staying in integer space.
fn apply_bps(value: i64, bps: i64) -> i64 {
    let scaled = value * bps;
    (scaled + BPS_DENOMINATOR / 2).div_euclid(BPS_DENOMINATOR)
}

fn validate_tiers(tiers: &[VolumeTier]) -> Result<(), DiscountError> {
    for tier in tiers {
        if tier.min_quantity == 0 {
            return Err(DiscountError::InvalidConfig(
                "volume tier min_quantity must be at least 1".to_string(),
            ));
        }
        if !(0..=BPS_DENOMINATOR).contains(&tier.discount_bps) {
            return Err(DiscountError::InvalidConfig(format!(
                "volume tier discount {} bps is out of range",
                tier.discount_bps
            )));
        }
    }
    Ok(())
}

fn validate_coupon(code: &str, rule: &CouponRule) -> Result<(), DiscountError> {
    match rule {
        CouponRule::PercentOff { bps, max_off_minor } => {
            if !(1..=BPS_DENOMINATOR).contains(bps) {
                return Err(DiscountError::InvalidConfig(format!("coupon {code}: {bps} bps is out of range")));
            }
            if matches!(max_off_minor, Some(cap) if *cap <= 0) {
                return Err(DiscountError::InvalidConfig(format!("coupon {code}: cap must be positive")));
            }
        }
        CouponRule::AmountOff { amount_minor, min_subtotal_minor } => {
            if *amount_minor <= 0 || *min_subtotal_minor < 0 {
                return Err(DiscountError::InvalidConfig(format!("coupon {code}: amounts must be positive")));
            }
        }
    }
    Ok(())
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

    /// Like `new`, but rejects configuration that would produce nonsense
    /// discounts. Used when loading rules from the merchant dashboard.
    pub fn try_new(tiers: Vec<VolumeTier>, coupons: HashMap<String, CouponRule>) -> Result<Self, DiscountError> {
        validate_tiers(&tiers)?;
        for (code, rule) in &coupons {
            if code.trim().is_empty() || code.len() > MAX_COUPON_CODE_LEN {
                return Err(DiscountError::InvalidConfig(format!("invalid coupon code {code:?}")));
            }
            validate_coupon(code, rule)?;
        }
        Ok(Self::new(tiers, coupons))
    }

    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }

    /// Best volume-tier discount for a single line.
    ///
    /// Tiers are inclusive: a tier with `min_quantity = 100` applies to a
    /// line of exactly 100 units. SKU-scoped tiers win over global ones only
    /// if they give the larger discount.
    pub fn apply_tiered(&self, line: &LineItem) -> Discount {
        if !line.is_discountable() || line.quantity == 0 {
            return Discount::default();
        }
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
        let normalized = code.trim().to_ascii_uppercase();
        if normalized.is_empty() || normalized.len() > MAX_COUPON_CODE_LEN {
            return Err(DiscountError::UnknownCoupon(code.to_string()));
        }
        let rule = self
            .coupons
            .get(&normalized)
            .ok_or_else(|| DiscountError::UnknownCoupon(code.to_string()))?;
        // Only discountable lines count towards both the discount base and
        // coupon minimums; a gift card must not unlock "$10 off $50".
        let subtotal: i64 = lines
            .iter()
            .filter(|line| line.is_discountable())
            .map(LineItem::gross_minor)
            .sum();

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

    /// Split an order-level discount across the discountable lines in
    /// proportion to their gross amounts.
    ///
    /// The shares must sum to exactly `discount_minor`: the tax service
    /// computes tax per line on `gross - share` and reconciles the sum against
    /// the order total, and partial refunds prorate per line from these
    /// numbers. Leftover minor units from rounding go to the largest lines.
    pub fn allocate_discount(
        &self,
        lines: &[LineItem],
        discount_minor: i64,
    ) -> Result<Vec<LineAllocation>, DiscountError> {
        if discount_minor < 0 {
            return Err(DiscountError::InvalidConfig("discount must be non-negative".to_string()));
        }
        let eligible: Vec<(usize, i64)> = lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.is_discountable() && line.gross_minor() > 0)
            .map(|(index, line)| (index, line.gross_minor()))
            .collect();
        // Denominator for the proportional split.
        let subtotal: i64 = lines.iter().map(LineItem::gross_minor).sum();
        if discount_minor > subtotal {
            return Err(DiscountError::DiscountTooLarge { discount: discount_minor, subtotal });
        }

        let mut shares: Vec<i64> = vec![0; lines.len()];
        if discount_minor == 0 || subtotal == 0 {
            return Ok(Self::to_allocations(lines, &shares));
        }

        // Floor each proportional share. i128 avoids overflow on large B2B
        // orders where gross * discount exceeds i64.
        for (index, gross) in &eligible {
            let exact = i128::from(*gross) * i128::from(discount_minor) / i128::from(subtotal);
            shares[*index] = exact as i64;
        }

        // Largest lines first, ties broken by original position for determinism.
        let mut by_size = eligible.clone();
        by_size.sort_by_key(|(index, gross)| (Reverse(*gross), *index));
        let allocated: i64 = shares.iter().sum();
        let leftover = discount_minor - allocated;
        for (index, gross) in by_size.iter().take(leftover.max(0) as usize) {
            // Never push a line's discount above its own gross.
            if shares[*index] < *gross {
                shares[*index] += 1;
            }
        }

        Ok(Self::to_allocations(lines, &shares))
    }

    fn to_allocations(lines: &[LineItem], shares: &[i64]) -> Vec<LineAllocation> {
        lines
            .iter()
            .zip(shares)
            .map(|(line, share)| LineAllocation {
                sku: line.sku.clone(),
                discount_minor: *share,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(sku: &str, quantity: u32, unit: i64) -> LineItem {
        LineItem {
            sku: sku.to_string(),
            quantity,
            unit_price_minor: unit,
        }
    }

    #[test]
    fn bps_rounds_half_up() {
        assert_eq!(apply_bps(1_000, 1_250), 125);
        assert_eq!(apply_bps(5, 1_000), 1);
        assert_eq!(apply_bps(4, 1_000), 0);
    }

    #[test]
    fn percent_coupon_respects_cap() {
        let mut coupons = HashMap::new();
        coupons.insert(
            "save20".to_string(),
            CouponRule::PercentOff { bps: 2_000, max_off_minor: Some(1_500) },
        );
        let engine = DiscountEngine::new(Vec::new(), coupons);
        let discount = engine.apply_coupon("SAVE20", &[line("MUG-1", 10, 1_000)]).unwrap();
        assert_eq!(discount.amount_minor, 1_500);
        assert_eq!(discount.source, DiscountSource::Coupon);
    }

    #[test]
    fn gift_cards_do_not_count_towards_minimum() {
        let mut coupons = HashMap::new();
        coupons.insert(
            "TENOFF".to_string(),
            CouponRule::AmountOff { amount_minor: 1_000, min_subtotal_minor: 5_000 },
        );
        let engine = DiscountEngine::new(Vec::new(), coupons);
        let result = engine.apply_coupon("tenoff", &[line("GC-50", 1, 5_000), line("MUG-1", 1, 1_200)]);
        assert_eq!(result, Err(DiscountError::BelowMinimum { actual: 1_200, required: 5_000 }));
    }

    #[test]
    fn unknown_coupon_is_reported_verbatim() {
        let engine = DiscountEngine::new(Vec::new(), HashMap::new());
        assert_eq!(
            engine.apply_coupon(" nope ", &[]),
            Err(DiscountError::UnknownCoupon(" nope ".to_string()))
        );
    }
}
