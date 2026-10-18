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

use crate::crust::{self, Column, CrustKind, CrustProvince, SurfaceRock};
use crate::geo::LonLat;
use crate::grid::Grid;
use crate::mantle::MantleFlow;
use crate::plates::{Plate, Plates};
use crate::rng::{Pcg32, Stream};
use crate::tectonics::{
    Boundary, BoundaryForce, BoundaryKind, DEFAULT_OCEAN_AGE_MYR, Edge, RADIUS_KM, boundary_forces,
    classify, flood, ocean_ages, solve_motion,
};

/// Rounds of classify, date, re-solve.
const MOTION_ROUNDS: usize = 3;
/// Sinking-slab downwellings added per this many subduction edges.
const SLAB_SAMPLE: usize = 4;
/// Hotspots by origin, after Courtillot et al. (2003): of Earth's 40 to 50,
/// about 7 to 9 are primary (deep), about 20 secondary and the rest
/// tertiary. See [`PlumeKind`].
const PRIMARY_HOTSPOTS: usize = 8;
const SECONDARY_HOTSPOTS: usize = 16;
const TERTIARY_HOTSPOTS: usize = 14;
/// Primary plumes along a province edge are evenly spaced, each shifted by
/// up to this share of the spacing, then scattered this far (radians, about
/// 130 km) off the outline.
const PRIMARY_SPACING_JITTER: f64 = 0.25;
const PLUME_EDGE_JITTER: f64 = 0.02;
/// Tertiary plumes scatter this far (radians, about 190 km) from a ridge or
/// rift.
const TERTIARY_JITTER: f64 = 0.03;
/// Attempts to find a secondary plume site before giving up on one.
const SECONDARY_ATTEMPTS: usize = 200;
/// Secondary plumes come off the rising domes, so their strength scales
/// with the local mantle rise, by between this and 1.
const MIN_UPWELLING_FACTOR: f64 = 0.6;
/// Hotspot volcanoes younger than this (Myr) show bare lava.
const FRESH_VOLCANO_MYR: f64 = 10.0;
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
/// Island arcs and Andes-style continental arcs thicken the crust with new
/// crust of intermediate to felsic composition.
const ISLAND_ARC_CRUST_KM: f64 = 12.0;
const ISLAND_ARC_FELSIC: f64 = 0.3;
const CONTINENTAL_ARC_KM: f64 = 28.0;
const CONTINENTAL_ARC_FELSIC: f64 = 0.6;
/// Collisions build a plateau this wide, then taper off.
const COLLISION_KM: f64 = 35.0;
const PLATEAU_KM: f64 = 300.0;
const PLATEAU_TAPER_KM: f64 = 450.0;
const RIFT_THINNING_KM: f64 = 12.0;
const RIFT_WIDTH_KM: f64 = 60.0;
/// Ocean sediment: slow open-ocean settling (km per Myr of seafloor age)
/// plus thick wedges off continents (GlobSed: thickness grows with age and
/// near land).
const PELAGIC_KM_PER_MYR: f64 = 0.003;
const MARGIN_SEDIMENT_KM: f64 = 2.5;
const MARGIN_WIDTH_KM: f64 = 250.0;
/// Seafloor with less sediment than this shows its basalt.
const BARE_SEAFLOOR_KM: f64 = 0.05;
/// Continental ground above this has had its sedimentary cover eroded off,
/// exposing granite and gneiss basement.
const EXPOSED_BASEMENT_M: f64 = 1200.0;
/// Continental crust thins toward the coast into a shelf.
/// Passive margins: continents end not in a cliff but in a zone of
/// stretched, thinned crust (shelf, slope and rise) a few hundred km wide.
/// Continental crust thins over this width toward the coast, to this
/// thickness at the shoreline...
const MARGIN_CONTINENT_KM: f64 = 300.0;
const COASTAL_CRUST_KM: f64 = 28.0;
/// ...and the seafloor just offshore is thickened transitional crust,
/// thinning back to normal oceanic crust over this width.
const MARGIN_OCEAN_KM: f64 = 300.0;
const TRANSITIONAL_CRUST_KM: f64 = 18.0;
const TRANSITIONAL_FELSIC: f64 = 0.3;
/// Neighbor-averaging passes that blend columns across continental province
/// edges (each pass reaches about one cell, roughly 120 km between cell
/// centers).
const PROVINCE_BLEND_PASSES: usize = 2;
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
    pub province: Vec<CrustProvince>,
    /// Each cell's lithosphere: sediment, crust and mantle lithosphere.
    pub column: Vec<Column>,
    /// Oceanic crust age (Myr); 0 for continental cells.
    pub age_myr: Vec<f64>,
    pub elevation_m: Vec<f64>,
    pub rock: Vec<SurfaceRock>,
}

pub struct World {
    pub plates: Plates,
    pub crust: Crust,
    pub mantle: MantleFlow,
    pub boundaries: Vec<Boundary>,
    /// The boundary forces behind the plates' motion, for display.
    pub forces: Vec<BoundaryForce>,
    pub hotspots: Vec<Hotspot>,
    /// Each hotspot's track: where the plate now carries ground that sat
    /// over the plume 0, 1, 2, ... Myr ago, while it stays on that plate.
    pub hotspot_tracks: Vec<Vec<DVec3>>,
    /// For each cell, the hotspot volcano standing there, if any.
    pub hotspot_marks: Vec<Option<HotspotMark>>,
    /// Elevation (m) each cell gains or loses from vertical mantle flow.
    pub dynamic_m: Vec<f64>,
}

/// A hotspot volcano on a cell.
#[derive(Clone, Copy, Debug)]
pub struct HotspotMark {
    /// Index into `World::hotspots`.
    pub hotspot: u32,
    /// Time since it stood over the plume.
    pub age_myr: f64,
}

/// Where a plume comes from, after Courtillot et al. (2003).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlumeKind {
    /// Deep plumes from the edges of the deep mantle provinces (Hawaii,
    /// Iceland, Afar, Réunion).
    Primary,
    /// Smaller plumes off the tops of the broad domes above the provinces
    /// (the island chains over the Pacific superswell).
    Secondary,
    /// Shallow melting where the plate is being stretched: near ridges and
    /// rifts.
    Tertiary,
}

impl PlumeKind {
    pub fn name(self) -> &'static str {
        match self {
            PlumeKind::Primary => "primary",
            PlumeKind::Secondary => "secondary",
            PlumeKind::Tertiary => "tertiary",
        }
    }

    /// Strength range; draws are skewed toward the low end, since most
    /// hotspots are weak and a few very strong (Sleep 1990).
    fn strength_range(self) -> (f64, f64) {
        match self {
            PlumeKind::Primary => (0.6, 1.0),
            PlumeKind::Secondary => (0.25, 0.6),
            PlumeKind::Tertiary => (0.1, 0.35),
        }
    }
}

/// A mantle plume, fixed in the mantle.
#[derive(Clone, Copy, Debug)]
pub struct Hotspot {
    pub kind: PlumeKind,
    pub position: DVec3,
    /// 0 to 1; 1 is Hawaii-like. Scales the swell and volcanoes, and weaker
    /// plumes' volcanoes are also shorter-lived.
    pub strength: f64,
}

impl Hotspot {
    /// The world's plumes, each placed by where its kind comes from:
    /// primaries spaced along the province edges (more on the larger one),
    /// secondaries inside the provinces where the mantle rises fastest,
    /// tertiaries near ridges and rifts.
    fn scatter(
        seed: u64,
        mantle: &MantleFlow,
        edges: &[Edge],
        boundaries: &[Boundary],
    ) -> Vec<Self> {
        let mut rng = Pcg32::for_part(seed, Stream::Hotspots);
        let provinces = &mantle.provinces;
        // Upwelling at a province center is about the strongest anywhere.
        let peak_rise = provinces
            .iter()
            .map(|p| mantle.upwelling(p.center))
            .fold(1e-12, f64::max);
        // Only secondaries depend on the dome they rise from: primaries are
        // fed from the base of the mantle, tertiaries by shallow stretching.
        let strength = |rng: &mut Pcg32, kind: PlumeKind, at: DVec3| {
            let (lo, hi) = kind.strength_range();
            let skewed = lo + (hi - lo) * rng.next_f64().powi(2);
            if kind != PlumeKind::Secondary {
                return skewed;
            }
            let rise = (mantle.upwelling(at) / peak_rise).clamp(0.0, 1.0);
            skewed * (MIN_UPWELLING_FACTOR + (1.0 - MIN_UPWELLING_FACTOR) * rise)
        };
        let mut spots = Vec::new();
        let mut add = |rng: &mut Pcg32, kind: PlumeKind, position: DVec3| {
            let strength = strength(rng, kind, position);
            spots.push(Self {
                kind,
                position,
                strength,
            });
        };

        // Primary: shared between the provinces by perimeter, then evenly
        // around each edge from a random starting direction.
        let total_radius: f64 = provinces.iter().map(|p| p.radius).sum();
        let mut placed = 0;
        for (i, p) in provinces.iter().enumerate() {
            let n = if i + 1 == provinces.len() {
                PRIMARY_HOTSPOTS - placed
            } else {
                ((PRIMARY_HOTSPOTS as f64 * p.radius / total_radius).round() as usize)
                    .min(PRIMARY_HOTSPOTS - placed)
            };
            placed += n;
            let step = std::f64::consts::TAU / n.max(1) as f64;
            let start = rng.range(0.0, std::f64::consts::TAU);
            for k in 0..n {
                let azimuth = start
                    + step
                        * (k as f64 + rng.range(-PRIMARY_SPACING_JITTER, PRIMARY_SPACING_JITTER));
                let edge = p.edge_point(azimuth);
                let at = (edge + rng.unit_vector() * PLUME_EDGE_JITTER).normalize();
                add(&mut rng, PlumeKind::Primary, at);
            }
        }

        // Secondary: inside a province, kept more often where the mantle
        // rises faster.
        for k in 0..SECONDARY_HOTSPOTS {
            let p = &provinces[k % provinces.len()];
            let site = (0..SECONDARY_ATTEMPTS).find_map(|_| {
                let at = (p.center + rng.unit_vector() * p.radius).normalize();
                let rise = (mantle.upwelling(at) / peak_rise).clamp(0.0, 1.0);
                (p.edge_distance(at) < 0.0 && rng.next_f64() < rise).then_some(at)
            });
            add(&mut rng, PlumeKind::Secondary, site.unwrap_or(p.center));
        }

        // Tertiary: near a spreading ridge or rift, anywhere if there are
        // none.
        let divergent: Vec<DVec3> = boundaries
            .iter()
            .filter(|b| matches!(b.kind, BoundaryKind::Ridge | BoundaryKind::Rift))
            .map(|b| edges[b.edge as usize].mid)
            .collect();
        for _ in 0..TERTIARY_HOTSPOTS {
            let at = if divergent.is_empty() {
                rng.unit_vector()
            } else {
                let mid = divergent[(rng.next_f64() * divergent.len() as f64) as usize];
                (mid + rng.unit_vector() * TERTIARY_JITTER).normalize()
            };
            add(&mut rng, PlumeKind::Tertiary, at);
        }
        spots
    }

    /// Weaker plumes' volcanoes are shorter-lived and their tracks shorter.
    fn lifetime(&self) -> f64 {
        0.4 + 0.6 * self.strength
    }

    /// Where a plate turning at `omega` has carried, after `t` Myr, the
    /// ground that sat over the plume.
    fn carried(&self, omega: DVec3, t: f64) -> DVec3 {
        if omega.length() < 1e-12 {
            self.position
        } else {
            DQuat::from_axis_angle(omega.normalize(), omega.length() * t) * self.position
        }
    }
}

/// The track of `hotspot`, one point per Myr, until it leaves the plate the
/// plume is under or runs out of the hotspot's track length.
fn hotspot_track(
    grid: &Grid,
    cell_plate: &[u32],
    plates: &[Plate],
    hotspot: &Hotspot,
) -> Vec<DVec3> {
    let plate = cell_plate[grid.cell_at(LonLat::from_vec(hotspot.position)) as usize];
    let omega = plates[plate as usize].omega;
    let myr = (HOTSPOT_TRACK_MYR as f64 * hotspot.lifetime()) as u32;
    (0..=myr)
        .map(|t| hotspot.carried(omega, t as f64))
        .take_while(|&p| cell_plate[grid.cell_at(LonLat::from_vec(p)) as usize] == plate)
        .collect()
}

impl World {
    pub fn generate(grid: &Grid, edges: &[Edge], seed: u64) -> Self {
        let (kind, province) =
            crust::continents(grid, &mut Pcg32::for_part(seed, Stream::Continents));
        let mut mantle = MantleFlow::earthlike(grid, &mut Pcg32::for_part(seed, Stream::Mantle));
        let mut flow = mantle.sample(grid);
        let mut age = vec![DEFAULT_OCEAN_AGE_MYR; grid.len()];
        // Each cell's lithosphere before boundary features build on it: what
        // decides which side sinks, how hard slabs pull and how keels drag.
        let base_columns = |age: &[f64]| -> Vec<Column> {
            (0..grid.len())
                .map(|c| {
                    province[c]
                        .column()
                        .unwrap_or_else(|| Column::oceanic(age[c], 0.0))
                })
                .collect()
        };
        let mut columns = base_columns(&age);

        let mut plates = Plates::generate_with(grid, seed, |majors| {
            let count = majors.iter().max().map_or(0, |&m| m as usize + 1);
            let omega = solve_motion(grid, edges, majors, count, &columns, &flow, &[]);
            let boundaries = classify(edges, majors, &omega, &kind, &columns);
            // Back-arc sites: the overriding side of each trench.
            boundaries
                .iter()
                .filter(|b| b.kind == BoundaryKind::Subduction)
                .map(|b| b.overriding(edges))
                .collect()
        });
        let count = plates.plates.len();

        let mut omega = solve_motion(grid, edges, &plates.cell_plate, count, &columns, &flow, &[]);
        let mut boundaries = classify(edges, &plates.cell_plate, &omega, &kind, &columns);
        for round in 0..MOTION_ROUNDS {
            age = ocean_ages(grid, edges, &plates.cell_plate, &kind, &boundaries);
            columns = base_columns(&age);
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
                flow = mantle.normalize_and_sample(grid);
            }
            omega = solve_motion(
                grid,
                edges,
                &plates.cell_plate,
                count,
                &columns,
                &flow,
                &boundaries,
            );
            boundaries = classify(edges, &plates.cell_plate, &omega, &kind, &columns);
        }
        age = ocean_ages(grid, edges, &plates.cell_plate, &kind, &boundaries);
        let forces = boundary_forces(grid, edges, &boundaries, &base_columns(&age));
        for (plate, w) in plates.plates.iter_mut().zip(&omega) {
            plate.omega = *w;
        }

        let mut crust = build_crust(
            grid,
            edges,
            &plates.cell_plate,
            kind,
            province,
            age,
            &boundaries,
        );
        let hotspots = Hotspot::scatter(seed, &mantle, edges, &boundaries);
        let relief = hotspot_relief(
            grid,
            &plates.cell_plate,
            &plates.plates,
            &crust.kind,
            &hotspots,
        );
        let dynamic_m = dynamic_topography(grid, &mantle);
        // Fresh hotspot volcanoes show their lava.
        for (c, mark) in relief.marks.iter().enumerate() {
            if let Some(m) = mark
                && m.age_myr < FRESH_VOLCANO_MYR
            {
                crust.rock[c] = if crust.kind[c].is_continental() {
                    SurfaceRock::FloodBasalt
                } else {
                    SurfaceRock::Basalt
                };
            }
        }
        for (c, e) in crust.elevation_m.iter_mut().enumerate() {
            *e += relief.elevation[c] + dynamic_m[c];
        }
        World {
            hotspot_tracks: hotspots
                .iter()
                .map(|h| hotspot_track(grid, &plates.cell_plate, &plates.plates, h))
                .collect(),
            hotspot_marks: relief.marks,
            hotspots,
            plates,
            crust,
            mantle,
            boundaries,
            forces,
            dynamic_m,
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
) -> HotspotRelief {
    let mut swell = vec![0.0f64; grid.len()];
    let mut volcano = vec![0.0f64; grid.len()];
    let mut marks: Vec<Option<HotspotMark>> = vec![None; grid.len()];
    for (index, hotspot) in hotspots.iter().enumerate() {
        let (plume, strength) = (hotspot.position, hotspot.strength);
        // Weaker plumes: lower, narrower swells and shorter-lived volcanoes.
        let swell_radius = SWELL_RADIUS_KM * strength.sqrt();
        let swell_reach = (3.0 * swell_radius / RADIUS_KM).cos();
        let lifetime = hotspot.lifetime();
        let plate = cell_plate[grid.cell_at(LonLat::from_vec(plume)) as usize];
        let omega = plates[plate as usize].omega;
        let carried = |t: f64| hotspot.carried(omega, t);

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
        while t <= HOTSPOT_TRACK_MYR as f64 * lifetime {
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
            let mark = HotspotMark {
                hotspot: index as u32,
                age_myr: t,
            };
            let mut raise = |c: u32, h: f64| {
                if h > volcano[c as usize] {
                    volcano[c as usize] = h;
                    marks[c as usize] = Some(mark);
                }
            };
            raise(cell, height);
            for n in grid.neighbors_of(cell) {
                raise(n, VOLCANO_FLANK * height);
            }
            t += spacing;
        }
    }
    HotspotRelief {
        elevation: swell.iter().zip(&volcano).map(|(s, v)| s + v).collect(),
        marks,
    }
}

struct HotspotRelief {
    elevation: Vec<f64>,
    marks: Vec<Option<HotspotMark>>,
}

/// Thickness and elevation for every cell from crust kind, seafloor age and
/// Every cell's column (sediment, crust, mantle lithosphere), its surface
/// rock, and the elevation it floats at, from crust kind, seafloor age and
/// the boundaries nearby.
fn build_crust(
    grid: &Grid,
    edges: &[Edge],
    cell_plate: &[u32],
    kind: Vec<CrustKind>,
    mut province: Vec<CrustProvince>,
    age: Vec<f64>,
    boundaries: &[Boundary],
) -> Crust {
    let n = grid.len();
    // Distance (km) from the nearest cell that isn't of `kind`, ignoring
    // plates: how far inland, or how far offshore.
    let distance_from_other = |continental: bool| {
        flood(
            grid,
            None,
            (0..n as u32)
                .filter(|&c| kind[c as usize].is_continental() != continental)
                .map(|c| (c, 0.0)),
            |_, _, _, km| Some(km),
        )
    };
    let inland = distance_from_other(true);
    let offshore = distance_from_other(false);

    // Uniform columns to start: every bit of relief below comes from a
    // mechanism (boundaries, seafloor cooling, sediment, shelves, hotspots),
    // none from noise.
    // Distances above run between cell centers; the coastline itself lies
    // about half a cell from the nearest center on either side.
    let half_cell =
        grid.centers[0].angle_between(grid.centers[grid.neighbors[0] as usize]) * RADIUS_KM / 2.0;
    // Strength of the margin taper: 1 at the coastline, falling to 0 at
    // `width` from it.
    let taper = |km: f64, width: f64| (1.0 - (km - half_cell) / width).clamp(0.0, 1.0);
    let mut column: Vec<Column> = (0..n)
        .map(|c| {
            if let Some(base) = province[c].column() {
                base
            } else {
                // Sediment builds up slowly on the open ocean floor and
                // thickly off continents, from what rivers carry off the land.
                let km = offshore[c].map_or(f64::MAX, |(km, _)| km);
                let sediment = PELAGIC_KM_PER_MYR * age[c]
                    + MARGIN_SEDIMENT_KM * (-km / MARGIN_WIDTH_KM).exp();
                Column::oceanic(age[c], sediment)
            }
        })
        .collect();
    blend_province_edges(grid, &kind, &mut column);
    for c in 0..n {
        let col = &mut column[c];
        if kind[c].is_continental() {
            // Stretched continental crust thinning toward the coast.
            let t = taper(
                inland[c].map_or(f64::MAX, |(km, _)| km),
                MARGIN_CONTINENT_KM,
            );
            col.crust_km -= (col.crust_km - COASTAL_CRUST_KM).max(0.0) * t;
        } else {
            // Transitional crust just offshore, between continental and
            // oceanic in thickness and composition.
            let t = taper(offshore[c].map_or(f64::MAX, |(km, _)| km), MARGIN_OCEAN_KM);
            let extra = (TRANSITIONAL_CRUST_KM - col.crust_km).max(0.0) * t;
            thicken(col, extra, TRANSITIONAL_FELSIC);
        }
    }
    // Seafloor relief (m) that isostasy doesn't explain: trenches, where the
    // sinking plate bends down.
    let mut relief = vec![0.0f64; n];

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

    // How strongly each cell is shaped by each feature, 0 to 1, for the
    // surface rock.
    let mut arc_at = vec![0.0f64; n];
    let mut collision_at = vec![0.0f64; n];
    let mut rift_at = vec![0.0f64; n];
    for c in 0..n {
        let continental = kind[c].is_continental();
        if let Some((km, s)) = trench[c]
            && !continental
        {
            relief[c] -= TRENCH_DEPTH_M * s * gaussian(km, 0.0, TRENCH_WIDTH_KM);
        }
        if let Some((km, s)) = arc[c] {
            arc_at[c] = s * gaussian(km, ARC_DISTANCE_KM, ARC_WIDTH_KM);
        }
        if let (Some((km, s)), true) = (collision[c], continental) {
            let taper = 1.0 - smoothstep(PLATEAU_KM, PLATEAU_KM + PLATEAU_TAPER_KM, km);
            collision_at[c] = s * taper;
        }
        if let (Some((km, s)), true) = (rift[c], continental) {
            rift_at[c] = s.max(0.3) * gaussian(km, 0.0, RIFT_WIDTH_KM);
        }

        let col = &mut column[c];
        // Active boundaries rework the crust they touch: whatever its age,
        // it now has young, Phanerozoic lithosphere.
        if continental
            && [arc_at[c], collision_at[c], rift_at[c]]
                .iter()
                .any(|&x| x > 0.3)
        {
            province[c] = CrustProvince::Phanerozoic;
            let young = CrustProvince::Phanerozoic.column().expect("continental");
            col.mantle_km = young.mantle_km;
            col.depletion = young.depletion;
        }
        if continental {
            // Where an arc and a collision overlap, the stronger wins rather
            // than the two adding up: thickened crust spreads sideways
            // instead of piling ever higher (Earth's thickest, under Tibet,
            // is 70-80 km). Arcs add new, more felsic crust; collisions
            // stack the crust that is already there.
            let arc_km = CONTINENTAL_ARC_KM * arc_at[c];
            let collision_km = COLLISION_KM * collision_at[c];
            if arc_km >= collision_km {
                thicken(col, arc_km, CONTINENTAL_ARC_FELSIC);
            } else {
                thicken(col, collision_km, col.felsic);
            }
            col.crust_km -= RIFT_THINNING_KM * rift_at[c];
        } else {
            // Island arcs build new crust of intermediate composition.
            thicken(col, ISLAND_ARC_CRUST_KM * arc_at[c], ISLAND_ARC_FELSIC);
        }
    }

    let elevation: Vec<f64> = (0..n)
        .map(|c| column[c].elevation_m() + relief[c])
        .collect();
    let rock = (0..n)
        .map(|c| {
            let strong = |x: f64| x > 0.3;
            if kind[c].is_continental() {
                if strong(collision_at[c]) {
                    SurfaceRock::Metamorphic
                } else if strong(arc_at[c]) {
                    SurfaceRock::Andesite
                } else if strong(rift_at[c]) || elevation[c] < EXPOSED_BASEMENT_M {
                    SurfaceRock::Sediment
                } else {
                    SurfaceRock::Granite
                }
            } else if strong(arc_at[c]) {
                SurfaceRock::Andesite
            } else if column[c].sediment_km < BARE_SEAFLOOR_KM {
                SurfaceRock::Basalt
            } else {
                SurfaceRock::OceanSediment
            }
        })
        .collect();
    Crust {
        kind,
        province,
        column,
        age_myr: age,
        elevation_m: elevation,
        rock,
    }
}

/// Smooths continental columns across province edges, so roots and crust
/// step gradually rather than in walls. Each layer's thickness and its
/// buoyancy are averaged with continental neighbors, then composition and
/// depletion are recomputed to match: buoyancy, and so elevation, blends
/// smoothly too. Oceanic cells take no part.
fn blend_province_edges(grid: &Grid, kind: &[CrustKind], column: &mut [Column]) {
    let rho_a = crust::RHO_ASTHENOSPHERE;
    // (crust km, crust buoyancy, mantle km, mantle buoyancy)
    let parts = |c: &Column| {
        [
            c.crust_km,
            (rho_a - crust::crust_density(c.felsic)) * c.crust_km,
            c.mantle_km,
            (rho_a - crust::mantle_lithosphere_density(c.depletion)) * c.mantle_km,
        ]
    };
    for _ in 0..PROVINCE_BLEND_PASSES {
        let blended: Vec<Option<[f64; 4]>> = (0..grid.len())
            .map(|c| {
                if !kind[c].is_continental() {
                    return None;
                }
                let cells: Vec<u32> = std::iter::once(c as u32)
                    .chain(grid.neighbors_of(c as u32))
                    .filter(|&x| kind[x as usize].is_continental())
                    .collect();
                let mut sum = [0.0; 4];
                for &x in &cells {
                    for (s, p) in sum.iter_mut().zip(parts(&column[x as usize])) {
                        *s += p;
                    }
                }
                Some(sum.map(|s| s / cells.len() as f64))
            })
            .collect();
        for (col, blend) in column.iter_mut().zip(blended) {
            if let Some([crust_km, crust_b, mantle_km, mantle_b]) = blend {
                col.crust_km = crust_km;
                col.felsic = crust::felsic_for(crust_km, crust_b);
                col.mantle_km = mantle_km;
                col.depletion = crust::depletion_for(mantle_km, mantle_b);
            }
        }
    }
}

/// Adds `km` of crust of composition `felsic` to a column, mixing the
/// composition by thickness.
fn thicken(col: &mut Column, km: f64, felsic: f64) {
    if km > 0.0 {
        col.felsic = (col.felsic * col.crust_km + felsic * km) / (col.crust_km + km);
        col.crust_km += km;
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
                .map(|c| w.crust.column[c].crust_km)
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
            kind: PlumeKind::Primary,
            position: plume.to_unit(),
            strength: 1.0,
        }];
        let r = hotspot_relief(&grid, &cell_plate, &plates, &ocean, &hawaii).elevation;

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
        let r = hotspot_relief(&grid, &cell_plate, &plates, &land, &hawaii).elevation;
        let peak = r.iter().cloned().fold(0.0, f64::max);
        assert!(peak <= VOLCANO_LAND_M + SWELL_M + 1.0, "{peak}");
        // 40 Myr downstream (2,000 km east), the track has all but gone.
        let old = r[at(2000.0 / RADIUS_KM.to_radians(), 0.0)];
        assert!(old < 50.0, "{old}");

        // A weak plume makes far less: a seamount, not an island.
        let weak = [Hotspot {
            kind: PlumeKind::Tertiary,
            position: plume.to_unit(),
            strength: 0.2,
        }];
        let r = hotspot_relief(&grid, &cell_plate, &plates, &ocean, &weak).elevation;
        let peak = r.iter().cloned().fold(0.0, f64::max);
        assert!(peak < 0.25 * (VOLCANO_OCEAN_M + SWELL_M), "{peak}");
    }

    #[test]
    fn plumes_rise_where_their_kind_comes_from() {
        let s = setup();
        for seed in [5, 6, 7] {
            let w = World::generate(&s.grid, &s.edges, seed);
            let of = |kind| w.hotspots.iter().filter(move |h| h.kind == kind);
            assert_eq!(of(PlumeKind::Primary).count(), PRIMARY_HOTSPOTS);
            assert_eq!(of(PlumeKind::Secondary).count(), SECONDARY_HOTSPOTS);
            assert_eq!(of(PlumeKind::Tertiary).count(), TERTIARY_HOTSPOTS);
            let provinces = &w.mantle.provinces;
            let to_edge = |p: DVec3| {
                provinces
                    .iter()
                    .map(|v| v.edge_distance(p))
                    .fold(f64::MAX, |m, d| if d.abs() < m.abs() { d } else { m })
            };
            for h in of(PlumeKind::Primary) {
                let d = to_edge(h.position).abs();
                assert!(
                    d < 2.0 * PLUME_EDGE_JITTER,
                    "seed {seed}: primary {d:.3} rad off an edge"
                );
            }
            for h in of(PlumeKind::Secondary) {
                assert!(
                    provinces
                        .iter()
                        .any(|p| p.edge_distance(h.position) <= 1e-9),
                    "seed {seed}: secondary outside every province"
                );
            }
            let divergent: Vec<DVec3> = w
                .boundaries
                .iter()
                .filter(|b| matches!(b.kind, BoundaryKind::Ridge | BoundaryKind::Rift))
                .map(|b| s.edges[b.edge as usize].mid)
                .collect();
            for h in of(PlumeKind::Tertiary) {
                let near = divergent
                    .iter()
                    .map(|m| m.angle_between(h.position))
                    .fold(f64::MAX, f64::min);
                assert!(
                    near < 2.0 * TERTIARY_JITTER,
                    "seed {seed}: tertiary {near:.3} rad from a ridge"
                );
            }
            // Strength ranks by kind: every primary beats every tertiary.
            let weakest_primary = of(PlumeKind::Primary)
                .map(|h| h.strength)
                .fold(1.0, f64::min);
            let strongest_tertiary = of(PlumeKind::Tertiary)
                .map(|h| h.strength)
                .fold(0.0, f64::max);
            assert!(weakest_primary > strongest_tertiary, "seed {seed}");
        }
        let w = World::generate(&s.grid, &s.edges, 5);
        for (h, track) in w.hotspots.iter().zip(&w.hotspot_tracks) {
            assert!(
                track[0].abs_diff_eq(h.position, 1e-12),
                "tracks start at the plume"
            );
        }
        // Volcano records point at real hotspots and plausible ages.
        let marked: Vec<HotspotMark> = w.hotspot_marks.iter().flatten().copied().collect();
        assert!(!marked.is_empty());
        assert!(
            marked
                .iter()
                .all(|m| (m.hotspot as usize) < w.hotspots.len()
                    && (0.0..=HOTSPOT_TRACK_MYR as f64).contains(&m.age_myr))
        );
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
            // Province edges blend the columns either side.
            let province_edge = s
                .grid
                .neighbors_of(c)
                .any(|n| w.crust.province[n as usize] != w.crust.province[c as usize]);
            on_boundary || province_edge || !kind[c as usize].is_continental()
        });
        let reach = flood(&s.grid, None, sources.map(|c| (c, 0.0)), |_, _, _, km| {
            Some(km)
        });
        let mut checked = 0;
        for c in 0..s.grid.len() {
            if let (true, Some((km, _))) = (kind[c].is_continental(), reach[c])
                && km > 1000.0
            {
                // Its province's starting crust, untouched.
                let t = w.crust.column[c].crust_km;
                let base = w.crust.province[c].column().unwrap().crust_km;
                assert!((t - base).abs() < 0.01, "cell {c}: {t} km, not {base}");
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
