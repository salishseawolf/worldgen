//! Crust: what each grid cell's surface is made of, and how high it sits.
//!
//! Crust type belongs to cells, not plates: like Earth's North American
//! plate, one plate can carry a continent and half an ocean. Elevation comes
//! from physics rather than noise:
//!
//! * Oceanic crust sinks as it cools and ages: half-space cooling while
//!   young, then the Parsons & Sclater (1977) plate model.
//! * Continental crust floats on the mantle by Airy isostasy: thicker crust
//!   stands higher, like Tibet's 70 to 80 km crust under about 5 km of
//!   plateau.

use crate::grid::Grid;
use crate::plates::{flood_fill, smooth_noise};
use crate::rng::Pcg32;

/// Share of the surface that is continental crust (Earth: about 40%,
/// counting continental shelves).
pub const CONTINENTAL_FRACTION: f64 = 0.35;
const CONTINENTS: usize = 6;
/// Continent seeds closer than this (radians) are rejected.
const CONTINENT_SPACING: f64 = 0.5;
/// Coastline roughness: crossing cost is exp(COAST_ROUGHNESS * noise).
const COAST_ROUGHNESS: f64 = 1.5;

pub const OCEANIC_THICKNESS_KM: f64 = 7.0;
pub const CONTINENTAL_THICKNESS_KM: f64 = 35.0;
/// Continental crust this thick floats with its top at sea level.
const SEA_LEVEL_THICKNESS_KM: f64 = 33.0;
/// Crust and mantle densities (kg/m³) for Airy isostasy.
const RHO_CRUST: f64 = 2800.0;
const RHO_MANTLE: f64 = 3300.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrustKind {
    Continental,
    Oceanic,
}

impl CrustKind {
    pub fn is_continental(self) -> bool {
        self == CrustKind::Continental
    }
}

/// Continental crust as a few blobs with rough coasts, covering
/// [`CONTINENTAL_FRACTION`] of the cells. Each continent grows from a seed
/// at its own speed, and all stop at the same moment, so sizes vary.
pub fn continents(grid: &Grid, rng: &mut Pcg32) -> Vec<CrustKind> {
    let mut seeds: Vec<u32> = Vec::new();
    while seeds.len() < CONTINENTS {
        let cell = (rng.next_f64() * grid.len() as f64) as u32;
        let p = grid.centers[cell as usize];
        if seeds
            .iter()
            .all(|&s| grid.centers[s as usize].angle_between(p) >= CONTINENT_SPACING)
        {
            seeds.push(cell);
        }
    }
    let speeds: Vec<f64> = seeds.iter().map(|_| rng.range(0.5, 1.5)).collect();
    let coarse = smooth_noise(grid, rng, 30);
    let fine = smooth_noise(grid, rng, 4);
    let cost: Vec<f64> = coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| (COAST_ROUGHNESS * (0.7 * c + 0.3 * f)).exp())
        .collect();
    let (_, arrival) = flood_fill(grid, &seeds, &speeds, &cost);

    let mut order: Vec<usize> = (0..grid.len()).collect();
    order.sort_by(|&a, &b| arrival[a].total_cmp(&arrival[b]));
    let mut kind = vec![CrustKind::Oceanic; grid.len()];
    let land = (CONTINENTAL_FRACTION * grid.len() as f64) as usize;
    for &cell in &order[..land] {
        kind[cell] = CrustKind::Continental;
    }
    kind
}

/// Seafloor elevation (m, negative) for oceanic crust `age` million years
/// old. Young crust follows half-space cooling, depth growing with the
/// square root of age; older crust levels off (Parsons & Sclater 1977).
/// The two meet within about 10 m at 20 Myr.
pub fn ocean_depth_m(age: f64) -> f64 {
    let age = age.max(0.0);
    if age < 20.0 {
        -(2500.0 + 350.0 * age.sqrt())
    } else {
        -(6400.0 - 3200.0 * (-age / 62.8).exp())
    }
}

/// Elevation (m) of continental crust `thickness` km thick by Airy
/// isostasy: each extra km of crust raises the surface by
/// `1 - ρ_crust/ρ_mantle` km, the rest becoming root.
pub fn continental_elevation_m(thickness_km: f64) -> f64 {
    (thickness_km - SEA_LEVEL_THICKNESS_KM) * 1000.0 * (1.0 - RHO_CRUST / RHO_MANTLE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    #[test]
    fn depth_follows_age() {
        assert_eq!(ocean_depth_m(0.0), -2500.0);
        // Half-space and plate model agree where they hand over.
        let (young, old) = (-(2500.0 + 350.0 * 20f64.sqrt()), ocean_depth_m(20.0));
        assert!((young - old).abs() < 15.0, "{young} vs {old}");
        // Monotonic deepening, levelling off toward 6400 m.
        let mut last = 0.0;
        for age in (0..=200).step_by(5) {
            let d = ocean_depth_m(age as f64);
            assert!(d <= last);
            last = d;
        }
        assert!(ocean_depth_m(200.0) > -6400.0 && ocean_depth_m(200.0) < -6000.0);
    }

    #[test]
    fn isostasy_matches_earth() {
        // Typical continent a little above sea level; Tibet about 5 km up.
        let typical = continental_elevation_m(CONTINENTAL_THICKNESS_KM);
        assert!((0.0..600.0).contains(&typical), "{typical}");
        let tibet = continental_elevation_m(70.0);
        assert!((4500.0..6500.0).contains(&tibet), "{tibet}");
    }

    #[test]
    fn continents_cover_target_share() {
        let grid = Grid::new(Resolution::Three);
        let kind = continents(&grid, &mut Pcg32::new(5));
        let land = kind.iter().filter(|k| k.is_continental()).count() as f64;
        let share = land / grid.len() as f64;
        assert!((share - CONTINENTAL_FRACTION).abs() < 0.01, "{share}");
    }
}
