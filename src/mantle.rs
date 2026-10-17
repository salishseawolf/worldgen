//! Mantle flow: the slow convection under the plates, as a tangent velocity
//! field on the unit sphere.
//!
//! The field has the two parts geophysicists split plate-scale flow into:
//! poloidal flow (spreading out of upwellings, converging into downwellings)
//! and toroidal flow (swirling, the part that makes transform faults). Each
//! part is a sum of smooth bumps. Today's plates carry about a third of
//! their motion as toroidal (Lithgow-Bertelloni & Richards 1998), which is
//! the default split here.
//!
//! Velocities are in radians per million years on the unit sphere; see
//! [`CM_PER_YEAR`].

use glam::DVec3;

use crate::geo::EARTH_RADIUS_M;
use crate::grid::Grid;
use crate::rng::Pcg32;

/// One cm/yr of surface speed, in radians per million years.
pub const CM_PER_YEAR: f64 = 1e4 / EARTH_RADIUS_M;

const UPWELLINGS: usize = 5;
const DOWNWELLINGS: usize = 8;
const VORTICES: usize = 6;
/// Angular widths (radians): broad upwellings, narrower downwellings.
const UPWELLING_WIDTH: f64 = 0.6;
const DOWNWELLING_WIDTH: f64 = 0.35;
const VORTEX_WIDTH: f64 = 0.5;
/// Sinking slabs make narrow downwellings under trenches.
const SLAB_WIDTH: f64 = 0.12;
/// Area-weighted RMS speed of the field (cm/yr) and its toroidal share of
/// kinetic energy.
pub const RMS_CM_YEAR: f64 = 3.0;
pub const TOROIDAL_FRACTION: f64 = 1.0 / 3.0;

/// A Gaussian bump `strength * exp(-(angle / width)^2)` around `center`.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub center: DVec3,
    pub strength: f64,
    pub width: f64,
    /// Cosine of three widths: beyond it the bump is under 0.02% of its
    /// peak and is skipped.
    reach_cos: f64,
}

impl Feature {
    pub fn new(center: DVec3, strength: f64, width: f64) -> Self {
        Self {
            center,
            strength,
            width,
            reach_cos: (3.0 * width).min(std::f64::consts::PI).cos(),
        }
    }

    /// Height of the bump at unit point `p`.
    fn value(&self, p: DVec3) -> f64 {
        if p.dot(self.center) < self.reach_cos {
            return 0.0;
        }
        let angle = p.angle_between(self.center);
        self.strength * (-(angle / self.width).powi(2)).exp()
    }

    /// Surface gradient of the bump at unit point `p`.
    fn gradient(&self, p: DVec3) -> DVec3 {
        // A dot product rules out distant points before any trigonometry.
        if p.dot(self.center) < self.reach_cos {
            return DVec3::ZERO;
        }
        let toward = self.center - p * self.center.dot(p);
        let len = toward.length();
        if len < 1e-12 {
            return DVec3::ZERO;
        }
        let angle = p.angle_between(self.center);
        let w2 = self.width * self.width;
        let g = (-angle * angle / w2).exp();
        toward / len * (2.0 * angle / w2 * g * self.strength)
    }
}

pub struct MantleFlow {
    /// Positive strength wells up and spreads out; negative sinks.
    pub sources: Vec<Feature>,
    /// Positive strength turns counterclockwise seen from outside.
    pub vortices: Vec<Feature>,
    poloidal_scale: f64,
    toroidal_scale: f64,
}

impl MantleFlow {
    pub fn random(grid: &Grid, rng: &mut Pcg32) -> Self {
        let mut feature =
            |strength: f64, width: f64| Feature::new(rng.unit_vector(), strength, width);
        let mut sources: Vec<Feature> = (0..UPWELLINGS)
            .map(|_| feature(1.0, UPWELLING_WIDTH))
            .collect();
        sources.extend((0..DOWNWELLINGS).map(|_| feature(-1.0, DOWNWELLING_WIDTH)));
        let vortices = (0..VORTICES)
            .map(|i| feature(if i % 2 == 0 { 1.0 } else { -1.0 }, VORTEX_WIDTH))
            .collect();
        let mut flow = Self {
            sources,
            vortices,
            poloidal_scale: 1.0,
            toroidal_scale: 1.0,
        };
        flow.normalize(grid);
        flow
    }

    /// Adds a narrow downwelling where a slab sinks, with `strength`
    /// relative to an ordinary downwelling.
    pub fn add_slab(&mut self, at: DVec3, strength: f64) {
        self.sources.push(Feature::new(at, -strength, SLAB_WIDTH));
    }

    fn poloidal(&self, p: DVec3) -> DVec3 {
        -self.sources.iter().map(|f| f.gradient(p)).sum::<DVec3>()
    }

    fn toroidal(&self, p: DVec3) -> DVec3 {
        // The gradient points at each center; crossing it with the outward
        // normal turns it a quarter turn counterclockwise around the center.
        self.vortices
            .iter()
            .map(|f| f.gradient(p))
            .sum::<DVec3>()
            .cross(p)
    }

    /// How strongly the mantle rises (positive) or sinks (negative) under
    /// unit point `p`, in arbitrary units: the potential whose downhill
    /// gradient is the spreading flow. Swirling flow has no vertical part.
    pub fn upwelling(&self, p: DVec3) -> f64 {
        self.sources.iter().map(|f| f.value(p)).sum()
    }

    /// Flow velocity (rad/Myr, tangent) at unit point `p`.
    pub fn velocity(&self, p: DVec3) -> DVec3 {
        self.poloidal(p) * self.poloidal_scale + self.toroidal(p) * self.toroidal_scale
    }

    /// Velocity at every grid cell.
    pub fn sample(&self, grid: &Grid) -> Vec<DVec3> {
        grid.centers.iter().map(|&p| self.velocity(p)).collect()
    }

    /// Rescales both parts to [`RMS_CM_YEAR`] overall, split by
    /// [`TOROIDAL_FRACTION`] of the energy. Grid cells are close enough to
    /// equal-area for the RMS to be a plain mean over cells.
    pub fn normalize(&mut self, grid: &Grid) {
        let rms = |f: &dyn Fn(DVec3) -> DVec3| {
            let sum: f64 = grid.centers.iter().map(|&p| f(p).length_squared()).sum();
            (sum / grid.len() as f64).sqrt().max(1e-12)
        };
        let target = RMS_CM_YEAR * CM_PER_YEAR;
        self.poloidal_scale =
            target * (1.0 - TOROIDAL_FRACTION).sqrt() / rms(&|p| self.poloidal(p));
        self.toroidal_scale = target * TOROIDAL_FRACTION.sqrt() / rms(&|p| self.toroidal(p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    #[test]
    fn flow_is_tangent_with_target_speed_and_split() {
        let grid = Grid::new(Resolution::Two);
        let flow = MantleFlow::random(&grid, &mut Pcg32::new(3));
        let mut total = 0.0;
        let mut toroidal = 0.0;
        for &p in &grid.centers {
            let v = flow.velocity(p);
            assert!(v.dot(p).abs() < 1e-12, "tangent");
            total += v.length_squared();
            toroidal += (flow.toroidal(p) * flow.toroidal_scale).length_squared();
        }
        let rms = (total / grid.len() as f64).sqrt() / CM_PER_YEAR;
        // The parts aren't exactly orthogonal on a sampled grid.
        assert!((rms - RMS_CM_YEAR).abs() < 0.3 * RMS_CM_YEAR, "{rms}");
        let share = toroidal / total;
        assert!((share - TOROIDAL_FRACTION).abs() < 0.1, "{share}");
    }

    #[test]
    fn upwelling_rises_and_downwelling_sinks() {
        let flow = MantleFlow {
            sources: vec![
                Feature::new(DVec3::Y, 1.0, 0.5),
                Feature::new(-DVec3::Y, -1.0, 0.5),
            ],
            vortices: vec![Feature::new(DVec3::X, 1.0, 0.5)],
            poloidal_scale: 1.0,
            toroidal_scale: 1.0,
        };
        assert!((flow.upwelling(DVec3::Y) - 1.0).abs() < 1e-9);
        assert!((flow.upwelling(-DVec3::Y) + 1.0).abs() < 1e-9);
        // The vortex at +X adds nothing vertical.
        assert!(flow.upwelling(DVec3::X).abs() < 1e-3);
    }

    #[test]
    fn upwelling_spreads_and_vortex_turns() {
        let center = DVec3::Y;
        let near = DVec3::new(0.2, 1.0, 0.0).normalize();
        let up = MantleFlow {
            sources: vec![Feature::new(center, 1.0, 0.5)],
            vortices: vec![],
            poloidal_scale: 1.0,
            toroidal_scale: 1.0,
        };
        let away = (near - center).normalize();
        assert!(
            up.velocity(near).dot(away) > 0.0,
            "flows away from upwelling"
        );

        let swirl = MantleFlow {
            sources: vec![],
            vortices: up.sources.clone(),
            ..up
        };
        let v = swirl.velocity(near);
        assert!(v.dot(away).abs() < 1e-12, "no spreading");
        // Counterclockwise seen from outside, above the center.
        assert!(away.cross(v).dot(near) > 0.0, "turns counterclockwise");
    }
}
