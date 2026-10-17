//! Small seeded PRNG (PCG32, XSH-RR variant).
//!
//! Kept in-house rather than using `rand` so a seed produces the same world on
//! every platform and across dependency upgrades.

pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (0xda3e_39cb_94b9_5bdb << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in [0, 1) with 53 bits of precision.
    pub fn next_f64(&mut self) -> f64 {
        let bits = (u64::from(self.next_u32()) << 21) ^ u64::from(self.next_u32() >> 11);
        bits as f64 / (1u64 << 53) as f64
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// Uniformly distributed unit vector.
    pub fn unit_vector(&mut self) -> glam::DVec3 {
        let y = self.range(-1.0, 1.0);
        let angle = self.range(0.0, std::f64::consts::TAU);
        let r = (1.0 - y * y).sqrt();
        glam::DVec3::new(r * angle.cos(), y, r * angle.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_sequence() {
        // Pinned output: changing the generator would silently change every
        // seeded world, so this must only change on purpose.
        let mut rng = Pcg32::new(42);
        let got: Vec<u32> = (0..3).map(|_| rng.next_u32()).collect();
        assert_eq!(got, [1_898_997_482, 1_014_631_766, 4_096_008_554]);
    }

    #[test]
    fn floats_in_range() {
        let mut rng = Pcg32::new(7);
        for _ in 0..10_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x));
            assert!((rng.unit_vector().length() - 1.0).abs() < 1e-12);
        }
    }
}
