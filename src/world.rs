//! A whole generated world: plates, crust, mantle flow and elevation, all
//! from one seed.
//!
//! Generation, in order:
//! 1. Continents as per-cell crust, and a mantle flow field.
//! 2. Major plates; their motion from mantle drag; their boundaries.
//! 3. Minor plates seeded behind trenches (back-arc plates, after Mallard et
//!    al. 2016).
//! 4. A few rounds of: classify boundaries, date the seafloor from the
//!    ridges, add sinking slabs to the mantle, re-solve motion with slab
//!    pull, ridge push and continental drag.
//! 5. Elevation from crust state plus boundary features, then hotspot
//!    island chains.
//!
//! This is a snapshot consistent with present-day motion; nothing is
//! stepped through time yet, so mountain belts and trenches are sized from
//! convergence rates rather than grown.

use glam::{DQuat, DVec3};

use crate::crust::{
    self, CONTINENTAL_THICKNESS_KM, CrustKind, OCEANIC_THICKNESS_KM, continental_elevation_m,
    ocean_depth_m,
};
use crate::geo::LonLat;
use crate::grid::Grid;
use crate::mantle::MantleFlow;
use crate::plates::{Plate, Plates};
use crate::rng::{Pcg32, Stream};
use crate::tectonics::{
    Boundary, BoundaryKind, DEFAULT_OCEAN_AGE_MYR, Edge, RADIUS_KM, classify, flood, ocean_ages,
    solve_motion,
};

/// Rounds of classify, date, re-solve.
const MOTION_ROUNDS: usize = 3;
/// Sinking-slab downwellings added per this many subduction edges.
const SLAB_SAMPLE: usize = 4;
/// Earth has about 40 to 50 catalogued hotspots, of which only 7 to 9 look
/// like strong plumes from deep in the mantle (Courtillot et al. 2003); the
/// rest are weaker. Strength 1 is Hawaii-like.
const STRONG_HOTSPOTS: usize = 8;
const WEAK_HOTSPOTS: usize = 30;
const STRONG_STRENGTH: (f64, f64) = (0.7, 1.0);
const WEAK_STRENGTH: (f64, f64) = (0.1, 0.35);
/// How far back hotspot tracks are traced (Myr).
const HOTSPOT_TRACK_MYR: u32 = 90;

/// Boundary feature sizes. Profiles are in km from the boundary; heights
/// and thickening scale with the convergence rate up to `FULL_RATE`.
const FULL_RATE_CM_YEAR: f64 = 5.0;
const TRENCH_DEPTH_M: f64 = 3500.0;
const TRENCH_WIDTH_KM: f64 = 70.0;
/// Volcanic arcs stand this far behind the trench.
const ARC_DISTANCE_KM: f64 = 180.0;
const ARC_WIDTH_KM: f64 = 90.0;
const ISLAND_ARC_M: f64 = 3500.0;
/// Andes-style arcs on continents thicken the crust.
const CONTINENTAL_ARC_KM: f64 = 28.0;
/// Collisions build a plateau this wide, then taper off.
const COLLISION_KM: f64 = 35.0;
const PLATEAU_KM: f64 = 300.0;
const PLATEAU_TAPER_KM: f64 = 450.0;
const RIFT_THINNING_KM: f64 = 12.0;
const RIFT_WIDTH_KM: f64 = 60.0;
/// Continental crust thins toward the coast into a shelf.
const SHELF_THINNING_KM: f64 = 4.0;
const SHELF_WIDTH_KM: f64 = 80.0;
/// Dynamic topography: rising mantle lifts the surface and sinking mantle
/// (including slabs) drags it down, by up to about a kilometer over
/// thousands of kilometers (Africa's superswell, Indonesia's low).
const DYNAMIC_TOPOGRAPHY_M: f64 = 800.0;
/// Hotspot swell: a broad, low dome over the plume (the Hawaiian swell is
/// about 1 km high and 1,000 km across). It trails a little downstream
/// while the lithosphere that passed over the plume cools.
const SWELL_M: f64 = 1000.0;
const SWELL_RADIUS_KM: f64 = 350.0;
const SWELL_TRAIL_MYR: u32 = 15;
const SWELL_DECAY_MYR: f64 = 6.0;
/// Hotspot volcanoes stand apart, about this far along the track.
const VOLCANO_SPACING_KM: f64 = 150.0;
/// Fresh volcano height above the surrounding surface, and how fast it
/// subsides and erodes (Myr). Ocean volcanoes are huge (Hawaii rises about
/// 9 km from the seafloor) and sink into seamounts; continental volcanic
/// fields are modest and wear down sooner.
const VOLCANO_OCEAN_M: f64 = 7500.0;
const VOLCANO_OCEAN_DECAY_MYR: f64 = 30.0;
const VOLCANO_LAND_M: f64 = 600.0;
const VOLCANO_LAND_DECAY_MYR: f64 = 10.0;
/// Share of a volcano's height on its flanks (the neighboring cells).
const VOLCANO_FLANK: f64 = 0.35;

pub struct Crust {
    pub kind: Vec<CrustKind>,
    pub thickness_km: Vec<f64>,
    /// Oceanic crust age (Myr); 0 for continental cells.
    pub age_myr: Vec<f64>,
    pub elevation_m: Vec<f64>,
}

pub struct World {
    pub plates: Plates,
    pub crust: Crust,
    pub mantle: MantleFlow,
    pub boundaries: Vec<Boundary>,
    pub hotspots: Vec<Hotspot>,
}

/// A mantle plume, fixed in the mantle.
#[derive(Clone, Copy, Debug)]
pub struct Hotspot {
    pub position: DVec3,
    /// 0 to 1; 1 is Hawaii-like. Scales the swell and volcanoes, and weaker
    /// plumes' volcanoes are also shorter-lived.
    pub strength: f64,
}

impl Hotspot {
    /// The world's plumes: a few strong ones and many weak ones.
    fn scatter(seed: u64) -> Vec<Self> {
        let mut rng = Pcg32::for_part(seed, Stream::Hotspots);
        let mut spots: Vec<Self> = (0..STRONG_HOTSPOTS)
            .map(|_| Self::random(&mut rng, STRONG_STRENGTH))
            .collect();
        spots.extend((0..WEAK_HOTSPOTS).map(|_| Self::random(&mut rng, WEAK_STRENGTH)));
        spots
    }

    fn random(rng: &mut Pcg32, (lo, hi): (f64, f64)) -> Self {
        Self {
            position: rng.unit_vector(),
            strength: rng.range(lo, hi),
        }
    }
}

impl World {
    pub fn generate(grid: &Grid, edges: &[Edge], seed: u64) -> Self {
        let kind = crust::continents(grid, &mut Pcg32::for_part(seed, Stream::Continents));
        let mut mantle = MantleFlow::random(grid, &mut Pcg32::for_part(seed, Stream::Mantle));
        let mut flow = mantle.sample(grid);
        let mut age = vec![DEFAULT_OCEAN_AGE_MYR; grid.len()];

        let mut plates = Plates::generate_with(grid, seed, |majors| {
            let count = majors.iter().max().map_or(0, |&m| m as usize + 1);
            let omega = solve_motion(grid, edges, majors, count, &kind, &flow, &[], &age);
            let boundaries = classify(edges, majors, &omega, &kind, &age);
            // Back-arc sites: the overriding side of each trench.
            boundaries
                .iter()
                .filter(|b| b.kind == BoundaryKind::Subduction)
                .map(|b| b.overriding(edges))
                .collect()
        });
        let count = plates.plates.len();

        let mut omega = solve_motion(
            grid,
            edges,
            &plates.cell_plate,
            count,
            &kind,
            &flow,
            &[],
            &age,
        );
        let mut boundaries = classify(edges, &plates.cell_plate, &omega, &kind, &age);
        for round in 0..MOTION_ROUNDS {
            age = ocean_ages(grid, edges, &plates.cell_plate, &kind, &boundaries);
            if round == 0 {
                // Slabs sink under trenches and pull the mantle down with
                // them (slab suction).
                for b in boundaries
                    .iter()
                    .filter(|b| b.kind == BoundaryKind::Subduction)
                    .step_by(SLAB_SAMPLE)
                {
                    let strength = (b.rate / FULL_RATE_CM_YEAR).min(1.0) * 0.3;
                    mantle.add_slab(grid.centers[b.subducting as usize], strength);
                }
                mantle.normalize(grid);
                flow = mantle.sample(grid);
            }
            omega = solve_motion(
                grid,
                edges,
                &plates.cell_plate,
                count,
                &kind,
                &flow,
                &boundaries,
                &age,
            );
            boundaries = classify(edges, &plates.cell_plate, &omega, &kind, &age);
        }
        age = ocean_ages(grid, edges, &plates.cell_plate, &kind, &boundaries);
        for (plate, w) in plates.plates.iter_mut().zip(&omega) {
            plate.omega = *w;
        }

        let crust = build_crust(grid, edges, &plates.cell_plate, kind, age, &boundaries);
        let mut world = World {
            plates,
            crust,
            mantle,
            boundaries,
            hotspots: Hotspot::scatter(seed),
        };
        world.add_hotspot_tracks(grid);
        let dynamic = dynamic_topography(grid, &world.mantle);
        for (e, d) in world.crust.elevation_m.iter_mut().zip(dynamic) {
            *e += d;
        }
        world
    }

    fn add_hotspot_tracks(&mut self, grid: &Grid) {
        let relief = hotspot_relief(
            grid,
            &self.plates.cell_plate,
            &self.plates.plates,
            &self.crust.kind,
            &self.hotspots,
        );
        for (e, r) in self.crust.elevation_m.iter_mut().zip(relief) {
            *e += r;
        }
    }
}

/// Elevation (m) from vertical mantle flow at every cell: the mantle's
/// upwelling potential, with rises and sinks each scaled so their strongest
/// reaches [`DYNAMIC_TOPOGRAPHY_M`]. Scaled separately because slab
/// downwellings crowd along trenches and add up, which would otherwise
/// leave upwellings only a fraction of the lows.
fn dynamic_topography(grid: &Grid, mantle: &MantleFlow) -> Vec<f64> {
    let up: Vec<f64> = grid.centers.iter().map(|&p| mantle.upwelling(p)).collect();
    let rise = up.iter().fold(0.0f64, |m, &u| m.max(u));
    let sink = up.iter().fold(0.0f64, |m, &u| m.max(-u));
    up.iter()
        .map(|&u| match u {
            u if u > 0.0 && rise > 1e-12 => DYNAMIC_TOPOGRAPHY_M * u / rise,
            u if u < 0.0 && sink > 1e-12 => DYNAMIC_TOPOGRAPHY_M * u / sink,
            _ => 0.0,
        })
        .collect()
}

/// Elevation added by hotspots. A plume fixed in the mantle raises a broad
/// swell around itself and builds a volcano on whatever passes over it;
/// plate motion carries older volcanoes away downstream while they subside
/// and erode, leaving a chain of separate volcanoes whose ages increase
/// along the track (like Hawaii and the Emperor seamounts on the ocean, or
/// Yellowstone and the Snake River Plain on a continent).
fn hotspot_relief(
    grid: &Grid,
    cell_plate: &[u32],
    plates: &[Plate],
    kind: &[CrustKind],
    hotspots: &[Hotspot],
) -> Vec<f64> {
    let mut swell = vec![0.0f64; grid.len()];
    let mut volcano = vec![0.0f64; grid.len()];
    for hotspot in hotspots {
        let (plume, strength) = (hotspot.position, hotspot.strength);
        // Weaker plumes: lower, narrower swells and shorter-lived volcanoes.
        let swell_radius = SWELL_RADIUS_KM * strength.sqrt();
        let swell_reach = (3.0 * swell_radius / RADIUS_KM).cos();
        let lifetime = 0.4 + 0.6 * strength;
        let plate = cell_plate[grid.cell_at(LonLat::from_vec(plume)) as usize];
        let omega = plates[plate as usize].omega;
        // Where the plate material now at `t` Myr downstream sat over the
        // plume: carried along by the plate's rotation since.
        let carried = |t: f64| {
            if omega.length() < 1e-12 {
                plume
            } else {
                DQuat::from_axis_angle(omega.normalize(), omega.length() * t) * plume
            }
        };

        // Cells the swell can reach anywhere along its trail, found once
        // rather than scanning the whole grid for every trail point.
        let trail_angle = omega.length() * SWELL_TRAIL_MYR as f64;
        let near_cos = (3.0 * swell_radius / RADIUS_KM + trail_angle)
            .min(std::f64::consts::PI)
            .cos();
        let near: Vec<usize> = (0..grid.len())
            .filter(|&c| grid.centers[c].dot(plume) > near_cos)
            .collect();
        for t in 0..=SWELL_TRAIL_MYR {
            let t = t as f64;
            let center = carried(t);
            let height = strength * SWELL_M * (-t / SWELL_DECAY_MYR).exp();
            for &c in &near {
                let p = grid.centers[c];
                if p.dot(center) > swell_reach {
                    let km = p.angle_between(center) * RADIUS_KM;
                    let h = height * (-(km / swell_radius).powi(2)).exp();
                    swell[c] = swell[c].max(h);
                }
            }
        }

        // One volcano per VOLCANO_SPACING_KM of plate travel.
        let km_per_myr = omega.cross(plume).length() * RADIUS_KM;
        let spacing = (VOLCANO_SPACING_KM / km_per_myr.max(1e-3)).clamp(1.0, 15.0);
        let mut t = 0.0;
        while t <= HOTSPOT_TRACK_MYR as f64 {
            let cell = grid.cell_at(LonLat::from_vec(carried(t)));
            if cell_plate[cell as usize] != plate {
                break;
            }
            let height = strength
                * if kind[cell as usize].is_continental() {
                    VOLCANO_LAND_M * (-t / (VOLCANO_LAND_DECAY_MYR * lifetime)).exp()
                } else {
                    VOLCANO_OCEAN_M * (-t / (VOLCANO_OCEAN_DECAY_MYR * lifetime)).exp()
                };
            volcano[cell as usize] = volcano[cell as usize].max(height);
            for n in grid.neighbors_of(cell) {
                volcano[n as usize] = volcano[n as usize].max(VOLCANO_FLANK * height);
            }
            t += spacing;
        }
    }
    swell.iter().zip(&volcano).map(|(s, v)| s + v).collect()
}

/// Thickness and elevation for every cell from crust kind, seafloor age and
/// the boundaries nearby.
fn build_crust(
    grid: &Grid,
    edges: &[Edge],
    cell_plate: &[u32],
    kind: Vec<CrustKind>,
    age: Vec<f64>,
    boundaries: &[Boundary],
) -> Crust {
    let n = grid.len();
    // Uniform crust to start: every bit of relief below comes from a
    // mechanism (boundaries, seafloor cooling, shelves, hotspots), none
    // from noise.
    let mut thickness: Vec<f64> = (0..n)
        .map(|c| {
            if kind[c].is_continental() {
                CONTINENTAL_THICKNESS_KM
            } else {
                OCEANIC_THICKNESS_KM
            }
        })
        .collect();
    // Extra seafloor relief (m) from trenches and island arcs.
    let mut relief = vec![0.0f64; n];

    // Continental shelves: crust thins toward the open ocean.
    let coast = flood(
        grid,
        None,
        (0..n as u32)
            .filter(|&c| !kind[c as usize].is_continental())
            .map(|c| (c, 0.0)),
        |_, _, _, km| Some(km),
    );
    for c in 0..n {
        if let (true, Some((km, _))) = (kind[c].is_continental(), coast[c]) {
            thickness[c] -= SHELF_THINNING_KM * (-km / SHELF_WIDTH_KM).exp();
        }
    }

    // Distance (km) within each plate from boundary cells of one kind,
    // carrying that boundary's rate scaled to 0..1.
    let reach = |pick: &dyn Fn(&Boundary) -> Vec<u32>| {
        let sources = boundaries.iter().flat_map(|b| {
            let strength = (b.rate / FULL_RATE_CM_YEAR).min(1.0);
            pick(b).into_iter().map(move |c| (c, strength))
        });
        flood(grid, Some(cell_plate), sources, |_, _, _, km| Some(km))
    };
    let gaussian = |km: f64, center: f64, width: f64| (-((km - center) / width).powi(2)).exp();

    let trench = reach(&|b| match b.kind {
        BoundaryKind::Subduction => vec![b.subducting],
        _ => vec![],
    });
    let arc = reach(&|b| match b.kind {
        BoundaryKind::Subduction => vec![b.overriding(edges)],
        _ => vec![],
    });
    let both_sides = |want: BoundaryKind| {
        move |b: &Boundary| {
            if b.kind == want {
                let e = &edges[b.edge as usize];
                vec![e.a, e.b]
            } else {
                vec![]
            }
        }
    };
    let collision = reach(&both_sides(BoundaryKind::Collision));
    let rift = reach(&both_sides(BoundaryKind::Rift));

    for c in 0..n {
        let continental = kind[c].is_continental();
        // Where an arc and a collision overlap, the stronger wins rather
        // than the two adding up: thickened crust spreads sideways instead
        // of piling ever higher (Earth's thickest, under Tibet, is 70-80 km).
        let mut thickening = 0.0f64;
        if let Some((km, s)) = trench[c]
            && !continental
        {
            relief[c] -= TRENCH_DEPTH_M * s * gaussian(km, 0.0, TRENCH_WIDTH_KM);
        }
        if let Some((km, s)) = arc[c] {
            let profile = s * gaussian(km, ARC_DISTANCE_KM, ARC_WIDTH_KM);
            if continental {
                thickening = thickening.max(CONTINENTAL_ARC_KM * profile);
            } else {
                relief[c] += ISLAND_ARC_M * profile;
            }
        }
        if let (Some((km, s)), true) = (collision[c], continental) {
            let taper = 1.0 - smoothstep(PLATEAU_KM, PLATEAU_KM + PLATEAU_TAPER_KM, km);
            thickening = thickening.max(COLLISION_KM * s * taper);
        }
        thickness[c] += thickening;
        if let (Some((km, s)), true) = (rift[c], continental) {
            thickness[c] -= RIFT_THINNING_KM * s.max(0.3) * gaussian(km, 0.0, RIFT_WIDTH_KM);
        }
    }

    let elevation = (0..n)
        .map(|c| {
            if kind[c].is_continental() {
                continental_elevation_m(thickness[c])
            } else {
                ocean_depth_m(age[c]) + relief[c]
            }
        })
        .collect();
    Crust {
        kind,
        thickness_km: thickness,
        age_myr: age,
        elevation_m: elevation,
    }
}

fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mantle::CM_PER_YEAR;
    use crate::tectonics::edges;
    use h3o::Resolution;

    struct Setup {
        grid: Grid,
        edges: Vec<Edge>,
    }

    fn setup() -> Setup {
        let grid = Grid::new(Resolution::Three);
        let edges = edges(&grid);
        Setup { grid, edges }
    }

    #[test]
    fn same_seed_same_world() {
        let s = setup();
        let a = World::generate(&s.grid, &s.edges, 7);
        let b = World::generate(&s.grid, &s.edges, 7);
        assert_eq!(a.plates.cell_plate, b.plates.cell_plate);
        assert_eq!(a.crust.elevation_m, b.crust.elevation_m);
    }

    #[test]
    fn earthlike_statistics() {
        let s = setup();
        for seed in [1, 2, 3] {
            let w = World::generate(&s.grid, &s.edges, seed);
            let e = &w.crust.elevation_m;
            let n = e.len() as f64;
            let below = e.iter().filter(|&&h| h < 0.0).count() as f64 / n;
            // Earth: about 71% below sea level.
            assert!(
                (0.55..0.8).contains(&below),
                "seed {seed}: {below} under water"
            );
            let max = e.iter().cloned().fold(f64::MIN, f64::max);
            let min = e.iter().cloned().fold(f64::MAX, f64::min);
            assert!(max > 2000.0, "seed {seed}: highest {max}");
            assert!(min < -6000.0, "seed {seed}: deepest {min}");
            let thickest = (0..e.len())
                .filter(|&c| w.crust.kind[c].is_continental())
                .map(|c| w.crust.thickness_km[c])
                .fold(0.0, f64::max);
            assert!(thickest < 80.0, "seed {seed}: crust {thickest:.0} km thick");
            // Every boundary type shows up somewhere on a world this size.
            for kind in [
                BoundaryKind::Ridge,
                BoundaryKind::Subduction,
                BoundaryKind::Transform,
            ] {
                assert!(
                    w.boundaries.iter().any(|b| b.kind == kind),
                    "seed {seed}: no {kind:?}"
                );
            }
        }
    }

    #[test]
    fn hotspot_makes_swell_and_separate_volcanoes() {
        let grid = Grid::new(Resolution::Three);
        let cell_plate = vec![0; grid.len()];
        // One plate turning about the pole at 5 cm/yr at the equator, so
        // the track from a plume at (0, 0) runs due east.
        let plates = [Plate {
            color: [0.0; 3],
            omega: DVec3::Y * 5.0 * crate::mantle::CM_PER_YEAR,
        }];
        let plume = LonLat::new(0.0, 0.0);
        let at = |lon: f64, lat: f64| grid.cell_at(LonLat::new(lon, lat)) as usize;
        let ocean = vec![CrustKind::Oceanic; grid.len()];
        let hawaii = [Hotspot {
            position: plume.to_unit(),
            strength: 1.0,
        }];
        let r = hotspot_relief(&grid, &cell_plate, &plates, &ocean, &hawaii);

        // Fresh ocean volcano plus swell at the plume.
        assert!(r[at(0.0, 0.0)] > VOLCANO_OCEAN_M, "{}", r[at(0.0, 0.0)]);
        // The swell is broad: 300 km to the side, off the track, still
        // raised by a few hundred meters.
        let side = r[at(0.0, 2.7)];
        assert!((200.0..SWELL_M).contains(&side), "{side}");
        // Volcanoes stand apart: along the track, some cells between
        // volcanoes are far lower than the volcanoes either side.
        let track: Vec<f64> = (0..60).map(|i| r[at(i as f64 * 0.25, 0.0)]).collect();
        let dips = track
            .windows(3)
            .filter(|w| w[1] < 0.6 * w[0].min(w[2]))
            .count();
        assert!(dips > 0, "{track:?}");

        // On a continent the same plume makes modest relief.
        let land = vec![CrustKind::Continental; grid.len()];
        let r = hotspot_relief(&grid, &cell_plate, &plates, &land, &hawaii);
        let peak = r.iter().cloned().fold(0.0, f64::max);
        assert!(peak <= VOLCANO_LAND_M + SWELL_M + 1.0, "{peak}");
        // 40 Myr downstream (2,000 km east), the track has all but gone.
        let old = r[at(2000.0 / RADIUS_KM.to_radians(), 0.0)];
        assert!(old < 50.0, "{old}");

        // A weak plume makes far less: a seamount, not an island.
        let weak = [Hotspot {
            position: plume.to_unit(),
            strength: 0.2,
        }];
        let r = hotspot_relief(&grid, &cell_plate, &plates, &ocean, &weak);
        let peak = r.iter().cloned().fold(0.0, f64::max);
        assert!(peak < 0.25 * (VOLCANO_OCEAN_M + SWELL_M), "{peak}");
    }

    #[test]
    fn mantle_flow_lifts_and_lowers_the_surface() {
        let s = setup();
        let w = World::generate(&s.grid, &s.edges, 3);
        let d = dynamic_topography(&s.grid, &w.mantle);
        let (min, max) = d
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(min >= -DYNAMIC_TOPOGRAPHY_M - 1e-9 && max <= DYNAMIC_TOPOGRAPHY_M + 1e-9);
        // Both directions occur: there are upwellings and sinking slabs.
        assert!(max > 0.3 * DYNAMIC_TOPOGRAPHY_M, "{max}");
        assert!(min < -0.3 * DYNAMIC_TOPOGRAPHY_M, "{min}");
        // Highest over the strongest upwelling.
        let top = (0..s.grid.len())
            .max_by(|&a, &b| d[a].total_cmp(&d[b]))
            .unwrap();
        let up = w.mantle.upwelling(s.grid.centers[top]);
        assert!(
            s.grid
                .centers
                .iter()
                .all(|&p| w.mantle.upwelling(p) <= up + 1e-9)
        );
    }

    #[test]
    fn continental_interiors_have_no_noise() {
        // Far from every plate boundary and coast, no mechanism reaches, so
        // continental crust keeps its starting thickness exactly. Any
        // variation there would be noise shaping terrain.
        let s = setup();
        let w = World::generate(&s.grid, &s.edges, 11);
        let kind = &w.crust.kind;
        let sources = (0..s.grid.len() as u32).filter(|&c| {
            let on_boundary = s
                .grid
                .neighbors_of(c)
                .any(|n| w.plates.cell_plate[n as usize] != w.plates.cell_plate[c as usize]);
            on_boundary || !kind[c as usize].is_continental()
        });
        let reach = flood(&s.grid, None, sources.map(|c| (c, 0.0)), |_, _, _, km| {
            Some(km)
        });
        let mut checked = 0;
        for c in 0..s.grid.len() {
            if let (true, Some((km, _))) = (kind[c].is_continental(), reach[c])
                && km > 1000.0
            {
                let t = w.crust.thickness_km[c];
                assert!(
                    (t - CONTINENTAL_THICKNESS_KM).abs() < 0.01,
                    "cell {c}: {t} km"
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no deep interior to check on this seed");
    }

    #[test]
    fn slab_attached_plates_move_faster() {
        // On Earth, major plates with slabs move several times faster than
        // those without (Forsyth & Uyeda 1975). Microplates are left out.
        let s = setup();
        let (mut with, mut without) = (Vec::new(), Vec::new());
        for seed in [1, 2, 3, 4] {
            let w = World::generate(&s.grid, &s.edges, seed);
            let count = w.plates.plates.len();
            let mut slab = vec![0.0; count];
            for b in w
                .boundaries
                .iter()
                .filter(|b| b.kind == BoundaryKind::Subduction)
            {
                slab[w.plates.cell_plate[b.subducting as usize] as usize] += 1.0;
            }
            // Mean surface speed (cm/yr) of each plate.
            let mut speed = vec![(0.0, 0.0); count];
            for (c, &p) in s.grid.centers.iter().enumerate() {
                let plate = w.plates.cell_plate[c] as usize;
                speed[plate].0 += w.plates.plates[plate].omega.cross(p).length() / CM_PER_YEAR;
                speed[plate].1 += 1.0;
            }
            for (i, (sum, cells)) in speed.into_iter().enumerate() {
                if cells < 1500.0 {
                    continue;
                }
                let v = sum / cells;
                if slab[i] >= 30.0 {
                    with.push(v)
                } else if slab[i] < 5.0 {
                    without.push(v)
                }
            }
        }
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let (fast, slow) = (mean(&with), mean(&without));
        assert!(
            fast > 1.5 * slow,
            "with slabs {fast:.1} cm/yr, without {slow:.1}"
        );
    }
}
