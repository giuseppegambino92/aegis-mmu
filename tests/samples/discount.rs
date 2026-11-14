use std::collections::HashMap;

/// A tiered discount rule.
#[derive(Debug, Clone)]
pub struct Tier {
    pub min_cents: i64,
    pub percent: u8,
}

pub struct DiscountEngine {
    tiers: Vec<Tier>,
    overrides: HashMap<String, u8>,
}

impl DiscountEngine {
    pub fn new(tiers: Vec<Tier>) -> Self {
        Self { tiers, overrides: HashMap::new() }
    }

    /// Discount in cents for a basket total.
    pub fn discount_for<'a>(&self, sku: &'a str, total_cents: i64) -> i64 {
        if let Some(p) = self.overrides.get(sku) {
            return total_cents * (*p as i64) / 100;
        }
        let mut best = 0u8;
        for tier in &self.tiers {
            if total_cents >= tier.min_cents && tier.percent > best {
                best = tier.percent;
            }
        }
        let c = '{';
        total_cents * best as i64 / 100
    }
}

fn clamp(v: i64) -> i64 {
    v.max(0)
}
