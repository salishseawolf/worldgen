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
//! The layout follows Earth's deep mantle, whose structure is dominated by
//! two huge, hot regions at its base, under Africa and the Pacific, roughly
//! opposite each other (the large low-shear-velocity provinces), with a
//! ring of sinking slabs between them around the Pacific. Each province
//! feeds a broad upwelling; downwellings sit along the ring between.
//!
//! Velocities are in radians per million years on the unit sphere; see
//! [`CM_PER_YEAR`].

use glam::DVec3;

use std::f64::consts::TAU;

use crate::geo::{EARTH_RADIUS_M, LonLat};
use crate::grid::Grid;
use crate::rng::Pcg32;

/// One cm/yr of surface speed, in radians per million years.
pub const CM_PER_YEAR: f64 = 1e4 / EARTH_RADIUS_M;

const DOWNWELLINGS: usize = 8;
const VORTICES: usize = 6;
/// Deep provinces: angular radius range (radians, about 2,900 to 3,800 km)
/// and how far the second strays from exactly opposite the first.
const PROVINCE_RADIUS: (f64, f64) = (0.45, 0.6);
const PROVINCE_OFFSET: f64 = 0.3;
/// Outline irregularity: relative amplitude of the 2- and 3-lobed wobbles.
const PROVINCE_WOBBLE: f64 = 0.12;
/// Downwellings sit within this angle (radians) of the ring halfway between
/// the provinces, and are spread around it with this much angular jitter.
const RING_SPREAD: f64 = 0.25;
const RING_JITTER: f64 = 0.3;
/// Angular widths (radians): narrower downwellings, broad swirls.
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

/// A hot, dense region at the base of the mantle, seen from above: a rough
/// cap around `center`. Its edges are where deep plumes tend to rise
/// ("plume generation zones", Burke, Torsvik and colleagues).
#[derive(Clone, Copy, Debug)]
pub struct Province {
    pub center: DVec3,
    /// Mean angular radius (radians).
    pub radius: f64,
    /// (relative amplitude, phase) of 2- and 3-lobed outline wobbles.
    wobble: [(f64, f64); 2],
}

impl Province {
    fn random(center: DVec3, rng: &mut Pcg32) -> Self {
        let radius = rng.range(PROVINCE_RADIUS.0, PROVINCE_RADIUS.1);
        let mut wobble = || (rng.range(0.0, PROVINCE_WOBBLE), rng.range(0.0, TAU));
        let wobble = [wobble(), wobble()];
        Self {
            center,
            radius,
            wobble,
        }
    }

    /// Unit vectors pointing north and east at the center.
    fn basis(&self) -> (DVec3, DVec3) {
        let (east, north) = LonLat::from_vec(self.center).east_north();
        (north, east)
    }

    /// Angular radius toward `azimuth` (radians clockwise from north).
    pub fn radius_toward(&self, azimuth: f64) -> f64 {
        let [(a2, p2), (a3, p3)] = self.wobble;
        self.radius * (1.0 + a2 * (2.0 * azimuth + p2).sin() + a3 * (3.0 * azimuth + p3).sin())
    }

    /// The point on the outline toward `azimuth`.
    pub fn edge_point(&self, azimuth: f64) -> DVec3 {
        let (north, east) = self.basis();
        let dir = north * azimuth.cos() + east * azimuth.sin();
        let r = self.radius_toward(azimuth);
        self.center * r.cos() + dir * r.sin()
    }

    /// Angular distance (radians) from `p` to the outline, negative inside.
    pub fn edge_distance(&self, p: DVec3) -> f64 {
        let (north, east) = self.basis();
        let toward = p - self.center * self.center.dot(p);
        let azimuth = toward.dot(east).atan2(toward.dot(north));
        p.angle_between(self.center) - self.radius_toward(azimuth)
    }

    /// `points` points around the outline.
    pub fn outline(&self, points: usize) -> Vec<DVec3> {
        (0..points)
            .map(|i| self.edge_point(TAU * i as f64 / points as f64))
            .collect()
    }
}

pub struct MantleFlow {
    /// The two deep provinces.
    pub provinces: Vec<Province>,
    /// Positive strength wells up and spreads out; negative sinks.
    pub sources: Vec<Feature>,
    /// Positive strength turns counterclockwise seen from outside.
    pub vortices: Vec<Feature>,
    poloidal_scale: f64,
    toroidal_scale: f64,
}

impl MantleFlow {
    /// Earth-like layout: two roughly opposite deep provinces, each with a
    /// broad upwelling, downwellings around the ring between them, and
    /// swirls anywhere.
    pub fn earthlike(grid: &Grid, rng: &mut Pcg32) -> Self {
        let first = rng.unit_vector();
        let second = (-first + rng.unit_vector() * PROVINCE_OFFSET).normalize();
        let provinces = vec![Province::random(first, rng), Province::random(second, rng)];
        let mut sources: Vec<Feature> = provinces
            .iter()
            .map(|p| Feature::new(p.center, 1.0, p.radius))
            .collect();

        // The ring halfway between: perpendicular to the line joining the
        // provinces.
        let axis = (provinces[0].center - provinces[1].center).normalize();
        let u = axis.any_orthonormal_vector();
        let v = axis.cross(u);
        for k in 0..DOWNWELLINGS {
            let around =
                TAU * k as f64 / DOWNWELLINGS as f64 + rng.range(-RING_JITTER, RING_JITTER);
            let off = rng.range(-RING_SPREAD, RING_SPREAD);
            let dir = u * around.cos() + v * around.sin();
            let at = dir * off.cos() + axis * off.sin();
            sources.push(Feature::new(at, -1.0, DOWNWELLING_WIDTH));
        }

        let vortices = (0..VORTICES)
            .map(|i| {
                let spin = if i % 2 == 0 { 1.0 } else { -1.0 };
                Feature::new(rng.unit_vector(), spin, VORTEX_WIDTH)
            })
            .collect();
        let mut flow = Self {
            provinces,
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
        self.normalize_and_sample(grid);
    }

    /// [`normalize`](Self::normalize), returning the velocity at every
    /// cell as [`sample`](Self::sample) would. Evaluating the features is
    /// the costly part, so this does it once for both.
    pub fn normalize_and_sample(&mut self, grid: &Grid) -> Vec<DVec3> {
        let parts: Vec<(DVec3, DVec3)> = grid
            .centers
            .iter()
            .map(|&p| (self.poloidal(p), self.toroidal(p)))
            .collect();
        let rms = |pick: fn(&(DVec3, DVec3)) -> DVec3| {
            let sum: f64 = parts.iter().map(|x| pick(x).length_squared()).sum();
            (sum / grid.len() as f64).sqrt().max(1e-12)
        };
        let target = RMS_CM_YEAR * CM_PER_YEAR;
        self.poloidal_scale = target * (1.0 - TOROIDAL_FRACTION).sqrt() / rms(|x| x.0);
        self.toroidal_scale = target * TOROIDAL_FRACTION.sqrt() / rms(|x| x.1);
        parts
            .iter()
            .map(|(pol, tor)| *pol * self.poloidal_scale + *tor * self.toroidal_scale)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    #[test]
    fn flow_is_tangent_with_target_speed_and_split() {
        let grid = Grid::new(Resolution::Two);
        let flow = MantleFlow::earthlike(&grid, &mut Pcg32::new(3));
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
    fn earthlike_layout() {
        let grid = Grid::new(Resolution::Two);
        for seed in 0..10 {
            let flow = MantleFlow::earthlike(&grid, &mut Pcg32::new(seed));
            let [a, b] = [flow.provinces[0], flow.provinces[1]];
            // Roughly opposite each other.
            let apart = a.center.angle_between(b.center).to_degrees();
            assert!(apart > 150.0, "seed {seed}: {apart:.0}°");
            // Rising over each province, sinking on the ring between.
            assert!(flow.upwelling(a.center) > 0.5 && flow.upwelling(b.center) > 0.5);
            let axis = (a.center - b.center).normalize();
            for f in flow.sources.iter().filter(|f| f.strength < 0.0) {
                assert!(f.center.dot(axis).abs() < RING_SPREAD.sin() + 1e-9);
            }
        }
    }

    #[test]
    fn province_outline_and_edge_distance() {
        let p = Province::random(DVec3::Y, &mut Pcg32::new(1));
        for &q in &p.outline(36) {
            assert!(p.edge_distance(q).abs() < 1e-9, "outline lies on the edge");
        }
        assert!(p.edge_distance(DVec3::Y) < 0.0, "center is inside");
        assert!(p.edge_distance(-DVec3::Y) > 0.0, "antipode is outside");
    }

    #[test]
    fn upwelling_rises_and_downwelling_sinks() {
        let flow = MantleFlow {
            provinces: vec![],
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
            provinces: vec![],
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
