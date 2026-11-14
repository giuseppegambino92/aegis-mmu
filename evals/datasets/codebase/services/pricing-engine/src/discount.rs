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
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

const BPS_DENOMINATOR: i64 = 10_000;
/// Coupon codes longer than this are rejected before lookup.
const MAX_COUPON_CODE_LEN: usize = 32;
/// SKU prefixes for lines that are never discounted.
const NON_DISCOUNTABLE_PREFIXES: [&str; 2] = ["GC-", "DON-"];

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
    ///
    /// SKUs come from the catalogue service in upper case, but marketplace
    /// sellers have been seen sending lower-case prefixes, so the comparison
    /// ignores ASCII case. Empty SKUs are treated as discountable because the
    /// catalogue rejects them long before they reach pricing.
    pub fn is_discountable(&self) -> bool {
        let sku = self.sku.trim();
        if sku.is_empty() {
            return true;
        }
        let excluded = NON_DISCOUNTABLE_PREFIXES.iter().any(|prefix| {
            sku.get(..prefix.len())
                .map_or(false, |head| head == *prefix)
        });
        if excluded {
            debug!(sku = %self.sku, "line excluded from discounts");
        }
        !excluded
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
///
/// The product is computed in `i128`: a B2B line of a few million euros at a
/// high basis-point rate would overflow `i64` before the division. Negative
/// inputs only occur in refund reversals, where `div_euclid` keeps rounding
/// consistent with the forward calculation.
fn apply_bps(value: i64, bps: i64) -> i64 {
    if value == 0 || bps == 0 {
        return 0;
    }
    let scaled = i128::from(value) * i128::from(bps);
    let denominator = i128::from(BPS_DENOMINATOR);
    let rounded = (scaled + denominator / 2).div_euclid(denominator);
    match i64::try_from(rounded) {
        Ok(amount) => amount,
        Err(_) => {
            // Unreachable for bps <= 10_000, but never wrap silently.
            warn!(value, bps, "basis-point product out of range; saturating");
            if rounded.is_negative() { i64::MIN } else { i64::MAX }
        }
    }
}

fn validate_tiers(tiers: &[VolumeTier]) -> Result<(), DiscountError> {
    // Two tiers with the same threshold and scope are ambiguous: whichever
    // sorts first would win, and that depends on dashboard save order.
    let mut seen: HashSet<(u32, Option<&str>)> = HashSet::with_capacity(tiers.len());
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
        if matches!(tier.sku_prefix.as_deref(), Some(prefix) if prefix.trim().is_empty()) {
            return Err(DiscountError::InvalidConfig(
                "volume tier sku_prefix must not be blank; omit it for a global tier".to_string(),
            ));
        }
        if !seen.insert((tier.min_quantity, tier.sku_prefix.as_deref())) {
            return Err(DiscountError::InvalidConfig(format!(
                "duplicate volume tier for min_quantity {}",
                tier.min_quantity
            )));
        }
    }
    debug!(count = tiers.len(), "volume tiers validated");
    Ok(())
}

fn validate_coupon(code: &str, rule: &CouponRule) -> Result<(), DiscountError> {
    match rule {
        CouponRule::PercentOff { bps, max_off_minor } => {
            // Zero-percent coupons are almost always a dashboard typo and
            // would show up as "applied" with nothing taken off.
            if !(1..BPS_DENOMINATOR).contains(bps) {
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
            if *min_subtotal_minor > 0 && *amount_minor > *min_subtotal_minor {
                // Allowed (a "free item over $X" promotion), but the clamp in
                // apply_coupon will cap it at the cart subtotal.
                warn!(code, amount_minor, min_subtotal_minor, "amount-off coupon exceeds its own minimum");
            }
        }
    }
    if !code.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
        return Err(DiscountError::InvalidConfig(format!(
            "coupon {code:?}: codes may only contain letters, digits, '-' and '_'"
        )));
    }
    Ok(())
}

pub struct DiscountEngine {
    tiers: Vec<VolumeTier>,
    coupons: HashMap<String, CouponRule>,
}

impl DiscountEngine {
    pub fn new(mut tiers: Vec<VolumeTier>, coupons: HashMap<String, CouponRule>) -> Self {
        // Highest threshold first, so the scan in apply_tiered sees the most
        // specific tiers early; ties keep their configured order.
        tiers.sort_by_key(|t| Reverse(t.min_quantity));
        let mut normalized: HashMap<String, CouponRule> = HashMap::with_capacity(coupons.len());
        for (code, rule) in coupons {
            let key = code.to_ascii_uppercase();
            if normalized.insert(key.clone(), rule).is_some() {
                // "save10" and "SAVE10" collapse to one code; last one wins.
                warn!(code = %key, "coupon code defined twice with different casing");
            }
        }
        debug!(tiers = tiers.len(), coupons = normalized.len(), "discount engine built");
        Self { tiers, coupons: normalized }
    }

    /// Like `new`, but rejects configuration that would produce nonsense
    /// discounts. Used when loading rules from the merchant dashboard.
    pub fn try_new(tiers: Vec<VolumeTier>, coupons: HashMap<String, CouponRule>) -> Result<Self, DiscountError> {
        validate_tiers(&tiers)?;
        let mut normalized_codes: HashSet<String> = HashSet::with_capacity(coupons.len());
        for (code, rule) in &coupons {
            let trimmed = code.trim();
            if trimmed.is_empty() || trimmed.len() > MAX_COUPON_CODE_LEN {
                return Err(DiscountError::InvalidConfig(format!("invalid coupon code {code:?}")));
            }
            validate_coupon(trimmed, rule)?;
            // `new` folds case; reject configurations where that would
            // silently drop one of two distinct rules.
            if !normalized_codes.insert(trimmed.to_ascii_uppercase()) {
                return Err(DiscountError::InvalidConfig(format!(
                    "coupon code {code:?} differs from another only by case"
                )));
            }
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
        if line.unit_price_minor <= 0 {
            // Free samples and price-adjustment lines have nothing to discount.
            return Discount::default();
        }
        let gross = line.gross_minor();
        let best = self
            .tiers
            .iter()
            .filter(|tier| {
                tier.sku_prefix
                    .as_deref()
                    .map_or(true, |prefix| line.sku.starts_with(prefix))
            })
            .filter(|tier| line.quantity > tier.min_quantity)
            .map(|tier| apply_bps(gross, tier.discount_bps))
            .max();

        match best {
            Some(amount) if amount > 0 => {
                // A 100% tier must not push the line negative.
                let capped = amount.min(gross);
                debug!(sku = %line.sku, quantity = line.quantity, discount = capped, "volume tier applied");
                Discount {
                    amount_minor: capped,
                    source: DiscountSource::VolumeTier,
                }
            }
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

        let clamped = amount.clamp(0, subtotal);
        if clamped != amount {
            debug!(code = %normalized, amount, subtotal, "coupon amount clamped to subtotal");
        }
        Ok(Discount {
            amount_minor: clamped,
            source: DiscountSource::Coupon,
        })
    }

    /// Volume tiers and coupons do not stack; the customer gets whichever is larger.
    pub fn best_discount(&self, lines: &[LineItem], coupon: Option<&str>) -> Result<Discount, DiscountError> {
        if lines.is_empty() {
            return Ok(Discount::default());
        }
        let tiered: i64 = lines.iter().map(|line| self.apply_tiered(line).amount_minor).sum();
        let tier_discount = Discount {
            amount_minor: tiered,
            source: if tiered > 0 { DiscountSource::VolumeTier } else { DiscountSource::None },
        };
        let Some(code) = coupon.map(str::trim).filter(|code| !code.is_empty()) else {
            return Ok(tier_discount);
        };
        let coupon_discount = self.apply_coupon(code, lines)?;
        // Ties go to the volume tier: it needs no redemption bookkeeping and
        // leaves the coupon unused for a later order.
        let chosen = if coupon_discount.amount_minor >= tier_discount.amount_minor {
            coupon_discount
        } else {
            tier_discount
        };
        debug!(
            tiered = tier_discount.amount_minor,
            coupon = coupon_discount.amount_minor,
            source = ?chosen.source,
            "best discount selected"
        );
        Ok(chosen)
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
        debug_assert_eq!(lines.len(), shares.len(), "one share per line");
        let total: i64 = shares.iter().sum();
        let allocations: Vec<LineAllocation> = lines
            .iter()
            .zip(shares)
            .map(|(line, share)| LineAllocation {
                sku: line.sku.clone(),
                discount_minor: *share,
            })
            .collect();
        // Non-discountable lines are kept with a zero share so callers can
        // zip the result back onto the order lines by position.
        let zero_lines = allocations.iter().filter(|a| a.discount_minor == 0).count();
        debug!(lines = allocations.len(), zero_lines, total, "discount allocated");
        allocations
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
        // Zero short-circuits regardless of the other operand.
        assert_eq!(apply_bps(0, 2_500), 0);
        assert_eq!(apply_bps(9_999, 0), 0);
        // Large B2B amounts must not overflow the intermediate product.
        assert_eq!(apply_bps(900_000_000_000_000, 5_000), 450_000_000_000_000);
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
        // Below the cap the plain percentage applies.
        let small = engine.apply_coupon("save20", &[line("MUG-1", 2, 1_000)]).unwrap();
        assert_eq!(small.amount_minor, 400);
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
        // Donations are excluded the same way.
        let donation = engine.apply_coupon("TENOFF", &[line("DON-RED-CROSS", 1, 9_000), line("MUG-1", 1, 1_200)]);
        assert!(matches!(donation, Err(DiscountError::BelowMinimum { .. })));
    }

    #[test]
    fn unknown_coupon_is_reported_verbatim() {
        let engine = DiscountEngine::new(Vec::new(), HashMap::new());
        assert_eq!(
            engine.apply_coupon(" nope ", &[]),
            Err(DiscountError::UnknownCoupon(" nope ".to_string()))
        );
        let too_long = "X".repeat(MAX_COUPON_CODE_LEN + 1);
        assert_eq!(
            engine.apply_coupon(&too_long, &[line("MUG-1", 1, 100)]),
            Err(DiscountError::UnknownCoupon(too_long.clone()))
        );
    }
}
