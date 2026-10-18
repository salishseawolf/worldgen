//! Moving the world through time.
//!
//! Each plate carries its own crust: a full grid of crust samples in the
//! plate's own frame, plus a rotation that places that frame on the globe.
//! Moving a plate only updates its rotation, so crust moves exactly, with no
//! blurring from resampling a moving pattern onto fixed cells (the approach
//! of Cortial et al. 2019).
//!
//! Each step, every world cell looks up which plates' crust now covers it:
//!
//! * one plate: the cell shows that crust;
//! * several: continental crust wins over oceanic, and otherwise the most
//!   buoyant wins. Losing seafloor is subducted (removed); losing
//!   continental crust, too light to sink, is merged into the winner as
//!   thickened crust (a collision), so continental crust is conserved.
//!   Crust thickened past `MAX_CRUST_KM` spreads into the neighboring
//!   crust, widening the mountain belt into a plateau;
//! * none: the plates are pulling apart, and new oceanic crust forms there
//!   on the most common neighboring plate.
//!
//! The GPU runs this in `shaders/timestep.wgsl`; this is the reference
//! version, used by the tests and kept in step with the shader.

use bytemuck::{Pod, Zeroable};
use glam::{DQuat, DVec3};

use crate::crust::{
    Column, CrustKind, CrustProvince, OCEANIC_THICKNESS_KM, SurfaceRock, oceanic_mantle_km,
};
use crate::geo::LonLat;
use crate::grid::{Grid, NO_NEIGHBOR};
use crate::tectonics::{COLLISION_BUOYANCY, Edge, boundary_forces, classify, solve_motion_scaled};
use crate::world::{BARE_SEAFLOOR_KM, PELAGIC_KM_PER_MYR, World};

/// Plates the GPU buffers have room for.
pub const MAX_PLATES: usize = 32;
/// Time per step (Myr): plates move under one cell (about 120 km) per step
/// at up to 10 cm/yr (100 km/Myr).
pub const STEP_MYR: f64 = 1.0;
/// Plate motion is re-solved for the world as it stands every this many
/// Myr (`resolve_motion`).
pub const RESOLVE_MYR: f64 = 5.0;
/// Colliding crust thickens to about this (km); Tibet's is 70 to 80. Crust
/// past it flows out into the neighboring crust each step.
pub const MAX_CRUST_KM: f64 = 80.0;
/// Seafloor that spilled crust has thickened to this (km) counts as
/// continental from then on: about where crust is light enough not to sink.
pub const ACCRETED_KM: f64 = 20.0;
/// A cell weighs at most this many plates' crust at once: a winner and two
/// losers.
pub const MAX_CANDIDATES: usize = 3;

/// `Sample::flags` bits: continental crust, then province (2 bits) and
/// surface rock (3 bits). Must match timestep.wgsl.
pub const FLAG_CONTINENTAL: u32 = 1;
pub const PROVINCE_SHIFT: u32 = 1;
pub const ROCK_SHIFT: u32 = 3;

/// One crust sample in a plate's frame. The layout is shared with the GPU.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct Sample {
    pub flags: u32,
    pub crust_km: f32,
    pub felsic: f32,
    /// Continental mantle lithosphere; oceanic lithosphere follows age.
    pub mantle_km: f32,
    pub depletion: f32,
    /// Seafloor age (Myr); unused for continental crust.
    pub age_myr: f32,
    pub sediment_km: f32,
    /// Relief isostasy doesn't explain, carried with the crust: trenches
    /// and hotspot volcanoes (m).
    pub relief_m: f32,
}

// The CPU reference only runs in native tests.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
impl Sample {
    /// Brand-new seafloor at a spreading ridge.
    pub fn new_ocean() -> Self {
        Self {
            flags: (CrustProvince::Oceanic as u32) << PROVINCE_SHIFT
                | (SurfaceRock::Basalt as u32) << ROCK_SHIFT,
            crust_km: OCEANIC_THICKNESS_KM as f32,
            felsic: 0.0,
            mantle_km: 0.0,
            depletion: 0.0,
            age_myr: 0.0,
            sediment_km: 0.0,
            relief_m: 0.0,
        }
    }

    pub fn is_continental(&self) -> bool {
        self.flags & FLAG_CONTINENTAL != 0
    }

    pub fn province(&self) -> CrustProvince {
        CrustProvince::ALL[((self.flags >> PROVINCE_SHIFT) & 3) as usize]
    }

    /// Surface rock as it shows: seafloor turns from bare basalt to
    /// sediment-covered as sediment settles on it.
    pub fn rock(&self) -> SurfaceRock {
        let stored = SurfaceRock::ALL[((self.flags >> ROCK_SHIFT) & 7) as usize];
        match stored {
            SurfaceRock::Basalt | SurfaceRock::OceanSediment if !self.is_continental() => {
                if (self.sediment_km as f64) < BARE_SEAFLOOR_KM {
                    SurfaceRock::Basalt
                } else {
                    SurfaceRock::OceanSediment
                }
            }
            other => other,
        }
    }

    pub fn column(&self) -> Column {
        let mantle_km = if self.is_continental() {
            self.mantle_km as f64
        } else {
            oceanic_mantle_km(self.age_myr as f64)
        };
        Column {
            sediment_km: self.sediment_km as f64,
            crust_km: self.crust_km as f64,
            felsic: self.felsic as f64,
            mantle_km,
            depletion: self.depletion as f64,
        }
    }
}

/// One world cell's crust as the GPU reports it back. The layout is shared
/// with timestep.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ColumnOut {
    /// `u32::MAX` while nothing covers the cell.
    pub plate: u32,
    /// As `Sample::flags`, with the rock as it shows.
    pub flags: u32,
    pub sediment_km: f32,
    pub crust_km: f32,
    pub felsic: f32,
    pub mantle_km: f32,
    pub depletion: f32,
    pub age_myr: f32,
    pub elevation_m: f32,
    pub _pad: f32,
}

impl ColumnOut {
    /// The world cell's lithosphere, with the mantle as it stands (oceanic
    /// mantle already follows age).
    pub fn column(&self) -> Column {
        Column {
            sediment_km: self.sediment_km as f64,
            crust_km: self.crust_km as f64,
            felsic: self.felsic as f64,
            mantle_km: self.mantle_km as f64,
            depletion: self.depletion as f64,
        }
    }
}

/// Brings `world` up to date with world cells read back from the GPU:
/// crust, plate ownership, boundaries and their forces. Cells nothing
/// covers yet keep what they had. Plate motions are unchanged.
pub fn apply_columns(world: &mut World, grid: &Grid, edges: &[Edge], out: &[ColumnOut]) {
    let crust = &mut world.crust;
    for (c, o) in out.iter().enumerate() {
        if o.plate == u32::MAX {
            continue;
        }
        let continental = o.flags & FLAG_CONTINENTAL != 0;
        world.plates.cell_plate[c] = o.plate;
        crust.kind[c] = if continental {
            CrustKind::Continental
        } else {
            CrustKind::Oceanic
        };
        crust.province[c] = CrustProvince::ALL[((o.flags >> PROVINCE_SHIFT) & 3) as usize];
        crust.rock[c] = SurfaceRock::ALL[((o.flags >> ROCK_SHIFT) & 7) as usize];
        crust.column[c] = o.column();
        crust.age_myr[c] = if continental { 0.0 } else { o.age_myr as f64 };
        crust.elevation_m[c] = o.elevation_m as f64;
    }
    // Volcano marks sat on the old ground; the volcanoes themselves ride
    // with the crust as relief.
    world.hotspot_marks.iter_mut().for_each(|m| *m = None);
    let omega: Vec<DVec3> = world.plates.plates.iter().map(|p| p.omega).collect();
    world.boundaries = classify(
        edges,
        &world.plates.cell_plate,
        &omega,
        &crust.kind,
        &crust.column,
    );
    world.forces = boundary_forces(grid, edges, &world.boundaries, &crust.column);
}

/// Adds `km` of spilled crust, carrying `felsic_km` of felsic rock, to a
/// sample. Seafloor buried under enough of it becomes continental: young
/// crust over the oceanic mantle it had, its surface the shed sediment.
fn spill(s: &mut Sample, km: f64, felsic_km: f64) {
    let crust = s.crust_km as f64;
    s.felsic = ((s.felsic as f64 * crust + felsic_km) / (crust + km)) as f32;
    s.crust_km = (crust + km) as f32;
    if !s.is_continental() && s.crust_km as f64 >= ACCRETED_KM {
        s.mantle_km = oceanic_mantle_km(s.age_myr as f64) as f32;
        s.depletion = 0.0;
        s.flags = FLAG_CONTINENTAL
            | (CrustProvince::Phanerozoic as u32) << PROVINCE_SHIFT
            | (SurfaceRock::Sediment as u32) << ROCK_SHIFT;
    }
}

/// Re-solves the plates' motion for the world as it now stands (crust and
/// boundaries, as `apply_columns` left them), keeping the speed units fixed
/// at generation, then reclassifies the boundaries for the new motion.
pub fn resolve_motion(world: &mut World, grid: &Grid, edges: &[Edge]) {
    let (omega, _) = solve_motion_scaled(
        grid,
        edges,
        &world.plates.cell_plate,
        world.plates.plates.len(),
        &world.crust.column,
        &world.flow,
        &world.boundaries,
        Some(world.motion_scale),
    );
    for (plate, w) in world.plates.plates.iter_mut().zip(&omega) {
        plate.omega = *w;
    }
    let crust = &world.crust;
    world.boundaries = classify(
        edges,
        &world.plates.cell_plate,
        &omega,
        &crust.kind,
        &crust.column,
    );
    world.forces = boundary_forces(grid, edges, &world.boundaries, &crust.column);
}

/// Constants timestep.wgsl takes from the Rust side, as WGSL.
fn wgsl_prelude() -> String {
    let mut out = crate::crust::wgsl_constants();
    for (name, v) in [
        ("COLLISION_BUOYANCY", COLLISION_BUOYANCY),
        ("MAX_CRUST_KM", MAX_CRUST_KM),
        ("ACCRETED_KM", ACCRETED_KM),
        ("PELAGIC_KM_PER_MYR", PELAGIC_KM_PER_MYR),
        ("BARE_SEAFLOOR_KM", BARE_SEAFLOOR_KM),
    ] {
        out += &format!("const {name}: f32 = {v:?};\n");
    }
    for (name, v) in [
        ("MAX_PLATES", MAX_PLATES as u32),
        ("MAX_CANDIDATES", MAX_CANDIDATES as u32),
        ("FLAG_CONTINENTAL", FLAG_CONTINENTAL),
        ("PROVINCE_SHIFT", PROVINCE_SHIFT),
        ("ROCK_SHIFT", ROCK_SHIFT),
        ("PROVINCE_OCEANIC", CrustProvince::Oceanic as u32),
        ("PROVINCE_PHANEROZOIC", CrustProvince::Phanerozoic as u32),
        ("ROCK_SEDIMENT", SurfaceRock::Sediment as u32),
        ("ROCK_BASALT", SurfaceRock::Basalt as u32),
        ("ROCK_OCEAN_SEDIMENT", SurfaceRock::OceanSediment as u32),
    ] {
        out += &format!("const {name}: u32 = {v}u;\n");
    }
    out
}

/// The full time-step compute shader source.
pub fn wgsl_source() -> String {
    wgsl_prelude()
        + include_str!("shaders/cube.wgsl")
        + include_str!("shaders/lookup.wgsl")
        + include_str!("shaders/timestep.wgsl")
}

/// Every plate's crust and where its frame sits.
pub struct Frames {
    pub cells: usize,
    pub plates: usize,
    /// Rotation from each plate's frame to the globe.
    pub rotation: Vec<DQuat>,
    /// `plates × cells` samples, plate-major: sample `p * cells + r` is the
    /// crust at plate p's reference cell r.
    pub samples: Vec<Sample>,
    pub alive: Vec<bool>,
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub time_myr: f64,
}

/// Which crust each world cell shows after a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub struct Cover {
    pub plate: u32,
    /// Reference cell in the plate's frame.
    pub sample: u32,
}

impl Frames {
    /// The world at time 0: every cell's crust on its plate, in place.
    pub fn from_world(world: &World, grid: &Grid) -> Self {
        let cells = grid.len();
        let plates = world.plates.plates.len();
        assert!(plates <= MAX_PLATES, "{plates} plates");
        let mut frames = Self {
            cells,
            plates,
            rotation: vec![DQuat::IDENTITY; plates],
            samples: vec![Sample::zeroed(); plates * cells],
            alive: vec![false; plates * cells],
            time_myr: 0.0,
        };
        let crust = &world.crust;
        for c in 0..cells {
            let col = crust.column[c];
            let continental = crust.kind[c].is_continental();
            // Whatever elevation the column alone doesn't explain, apart
            // from the mantle-fixed parts (dynamic topography and swells),
            // rides with the crust: trenches and volcanoes.
            let relief =
                crust.elevation_m[c] - col.elevation_m() - world.dynamic_m[c] - world.swell_m[c];
            let i = world.plates.cell_plate[c] as usize * cells + c;
            frames.samples[i] = Sample {
                flags: u32::from(continental)
                    | (crust.province[c] as u32) << PROVINCE_SHIFT
                    | (crust.rock[c] as u32) << ROCK_SHIFT,
                crust_km: col.crust_km as f32,
                felsic: col.felsic as f32,
                mantle_km: col.mantle_km as f32,
                depletion: col.depletion as f32,
                age_myr: crust.age_myr[c] as f32,
                sediment_km: col.sediment_km as f32,
                relief_m: relief as f32,
            };
            frames.alive[i] = true;
        }
        frames
    }

    /// The reference cell of plate `p` under world point `point`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fn reference(&self, grid: &Grid, p: usize, point: DVec3) -> u32 {
        grid.cell_at(LonLat::from_vec(self.rotation[p].inverse() * point))
    }

    /// Advances `dt` Myr with plates turning at `omega` (rad/Myr). Returns
    /// what each world cell shows afterwards (None where nothing covers it
    /// yet, at the heart of a wide new gap).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn step(&mut self, grid: &Grid, omega: &[DVec3], dt: f64) -> Vec<Option<Cover>> {
        for (rot, w) in self.rotation.iter_mut().zip(omega) {
            if w.length() > 0.0 {
                *rot = (DQuat::from_axis_angle(w.normalize(), w.length() * dt) * *rot).normalize();
            }
        }
        let n = self.cells;

        // Resolve: the most buoyant covering sample wins each cell.
        let mut owner: Vec<Option<Cover>> = vec![None; n];
        let mut losers: Vec<Vec<(usize, bool)>> = vec![Vec::new(); n];
        for (d, &center) in grid.centers.iter().enumerate() {
            let mut candidates: Vec<(usize, u32, f64, bool)> = Vec::new();
            for p in 0..self.plates {
                let r = self.reference(grid, p, center);
                let i = p * n + r as usize;
                if self.alive[i] && candidates.len() < MAX_CANDIDATES {
                    let s = &self.samples[i];
                    candidates.push((p, r, s.column().buoyancy(), s.is_continental()));
                }
            }
            let Some(&(wp, wr, wb, _)) = candidates
                .iter()
                .max_by(|a, b| a.3.cmp(&b.3).then(a.2.total_cmp(&b.2)).then(b.0.cmp(&a.0)))
            else {
                continue;
            };
            owner[d] = Some(Cover {
                plate: wp as u32,
                sample: wr,
            });
            for &(p, r, b, continental) in &candidates {
                if p != wp {
                    let merge = continental || wb.min(b) >= COLLISION_BUOYANCY;
                    losers[d].push((p * n + r as usize, merge));
                }
            }
        }

        // Diverge: uncovered cells grow new crust on the most common
        // neighboring plate (ties to the lowest id).
        let mut new: Vec<Option<Cover>> = vec![None; n];
        for d in 0..n {
            if owner[d].is_some() {
                continue;
            }
            let mut counts = [0u32; MAX_PLATES];
            for &nb in &grid.neighbors[d * 6..d * 6 + 6] {
                if nb != NO_NEIGHBOR
                    && let Some(o) = owner[nb as usize]
                {
                    counts[o.plate as usize] += 1;
                }
            }
            let best = (0..self.plates).max_by(|&a, &b| counts[a].cmp(&counts[b]).then(b.cmp(&a)));
            if let Some(q) = best.filter(|&q| counts[q] > 0) {
                new[d] = Some(Cover {
                    plate: q as u32,
                    sample: self.reference(grid, q, grid.centers[d]),
                });
            }
        }

        // Apply: remove losers (merging continental and other buoyant ones
        // into the winner, once each), then plant new crust.
        let mut added = vec![0.0f64; self.samples.len()];
        for d in 0..n {
            for &(i, merge) in &losers[d] {
                if std::mem::replace(&mut self.alive[i], false)
                    && merge
                    && let Some(o) = owner[d]
                {
                    added[o.plate as usize * n + o.sample as usize] +=
                        self.samples[i].crust_km as f64;
                }
            }
        }
        for c in new.iter().flatten() {
            let i = c.plate as usize * n + c.sample as usize;
            if !self.alive[i] {
                self.samples[i] = Sample::new_ocean();
                self.alive[i] = true;
            }
        }

        // Age: seafloor gets older and gathers sediment; merged crust
        // thickens its winner, and crust past the cap is shared out among
        // the sample's living neighbors on its plate (with none, it stays).
        let neighbors = |i: usize| {
            let (p, r) = (i / n, i % n);
            grid.neighbors[r * 6..r * 6 + 6]
                .iter()
                .filter(|&&nb| nb != NO_NEIGHBOR)
                .map(move |&nb| p * n + nb as usize)
        };
        // Per sample: crust it sends each living neighbor (km), and its
        // felsic share.
        let mut share = vec![(0.0f64, 0.0f64); self.samples.len()];
        for i in 0..self.samples.len() {
            if !self.alive[i] {
                continue;
            }
            let crust = self.samples[i].crust_km as f64 + added[i];
            let living = neighbors(i).filter(|&j| self.alive[j]).count();
            let s = &mut self.samples[i];
            if crust > MAX_CRUST_KM && living > 0 {
                share[i] = ((crust - MAX_CRUST_KM) / living as f64, s.felsic as f64);
                s.crust_km = MAX_CRUST_KM as f32;
            } else {
                s.crust_km = crust as f32;
            }
            if !s.is_continental() {
                s.age_myr += dt as f32;
                s.sediment_km += (PELAGIC_KM_PER_MYR * dt) as f32;
            }
        }
        // Spill.
        for i in 0..self.samples.len() {
            if !self.alive[i] {
                continue;
            }
            let (km, felsic_km) = neighbors(i)
                .map(|j| share[j])
                .fold((0.0, 0.0), |(k, f), (sk, sf)| (k + sk, f + sk * sf));
            if km > 0.0 {
                spill(&mut self.samples[i], km, felsic_km);
            }
        }
        self.time_myr += dt;

        owner.iter().zip(&new).map(|(o, n)| o.or(*n)).collect()
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn sample(&self, c: Cover) -> &Sample {
        &self.samples[c.plate as usize * self.cells + c.sample as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crust::CrustProvince;
    use h3o::Resolution;

    fn grid() -> Grid {
        Grid::new(Resolution::Two)
    }

    /// Frames from a plate and sample for each cell.
    fn frames(grid: &Grid, plates: usize, make: impl Fn(DVec3) -> (usize, Sample)) -> Frames {
        let cells = grid.len();
        let mut f = Frames {
            cells,
            plates,
            rotation: vec![DQuat::IDENTITY; plates],
            samples: vec![Sample::zeroed(); plates * cells],
            alive: vec![false; plates * cells],
            time_myr: 0.0,
        };
        for (c, &p) in grid.centers.iter().enumerate() {
            let (plate, s) = make(p);
            f.samples[plate * cells + c] = s;
            f.alive[plate * cells + c] = true;
        }
        f
    }

    fn continent() -> Sample {
        let col = CrustProvince::Proterozoic.column().unwrap();
        Sample {
            flags: FLAG_CONTINENTAL | (CrustProvince::Proterozoic as u32) << PROVINCE_SHIFT,
            crust_km: col.crust_km as f32,
            felsic: col.felsic as f32,
            mantle_km: col.mantle_km as f32,
            depletion: col.depletion as f32,
            ..Sample::new_ocean()
        }
    }

    fn old_ocean() -> Sample {
        Sample {
            age_myr: 80.0,
            ..Sample::new_ocean()
        }
    }

    #[test]
    fn rigid_motion_carries_crust_exactly() {
        let g = grid();
        // One plate; a continental cap around +X on otherwise old seafloor.
        let mut f = frames(&g, 1, |p| {
            (0, if p.x > 0.9 { continent() } else { old_ocean() })
        });
        let land = f
            .alive
            .iter()
            .zip(&f.samples)
            .filter(|(a, s)| **a && s.is_continental())
            .count();
        // Turning about +Y at 0.01 rad/Myr for 50 Myr: 0.5 rad.
        let omega = [DVec3::Y * 0.01];
        let mut cover = Vec::new();
        for _ in 0..50 {
            cover = f.step(&g, &omega, STEP_MYR);
        }
        assert!((f.time_myr - 50.0).abs() < 1e-9);
        // Nothing lost or made: one plate always covers everything.
        assert_eq!(f.alive.iter().filter(|&&a| a).count(), g.len());
        let still = f
            .alive
            .iter()
            .zip(&f.samples)
            .filter(|(a, s)| **a && s.is_continental())
            .count();
        assert_eq!(still, land);
        // The cap has turned with the plate: what was at +X is now at
        // R(0.5 rad about Y) * X.
        let moved = DQuat::from_axis_angle(DVec3::Y, 0.5) * DVec3::X;
        let at = g.cell_at(LonLat::from_vec(moved)) as usize;
        assert!(f.sample(cover[at].unwrap()).is_continental());
        let old = g.cell_at(LonLat::from_vec(DVec3::X)) as usize;
        assert!(!f.sample(cover[old].unwrap()).is_continental());
    }

    /// North plate 0 and south plate 1, turning in opposite senses about
    /// X: they pull apart at 90°E (-Z) and press together at 90°W (+Z).
    fn split(g: &Grid, north: Sample, south: Sample) -> Frames {
        frames(g, 2, |p| if p.y >= 0.0 { (0, north) } else { (1, south) })
    }
    const SPLIT: [DVec3; 2] = [DVec3::new(0.01, 0.0, 0.0), DVec3::new(-0.01, 0.0, 0.0)];

    #[test]
    fn spreading_makes_new_seafloor_that_ages() {
        let g = grid();
        let mut f = split(&g, old_ocean(), old_ocean());
        let mut cover = Vec::new();
        for _ in 0..20 {
            cover = f.step(&g, &SPLIT, STEP_MYR);
        }
        // Along 90°E, seafloor is youngest at the ridge (the equator) and
        // older away from it.
        let age = |lat: f64| {
            let c = g.cell_at(LonLat::new(90.0, lat)) as usize;
            f.sample(cover[c].expect("covered")).age_myr
        };
        assert!(age(0.0) < 10.0, "ridge {}", age(0.0));
        assert!(age(0.0) < age(15.0), "{} then {}", age(0.0), age(15.0));
        // Brand-new crust is bare basalt.
        let ridge = g.cell_at(LonLat::new(90.0, 0.0)) as usize;
        assert_eq!(f.sample(cover[ridge].unwrap()).rock(), SurfaceRock::Basalt);
    }

    #[test]
    fn seafloor_sinks_under_continent_and_continent_is_kept() {
        let g = grid();
        let mut f = split(&g, continent(), old_ocean());
        let count = |f: &Frames, p: usize| {
            f.alive[p * f.cells..(p + 1) * f.cells]
                .iter()
                .filter(|&&a| a)
                .count()
        };
        let (land0, sea0) = (count(&f, 0), count(&f, 1));
        for _ in 0..20 {
            f.step(&g, &SPLIT, STEP_MYR);
        }
        // Seafloor is consumed where the plates converge; the continent
        // keeps every sample (it gains new seafloor only where they part).
        assert!(count(&f, 1) < sea0, "{} of {sea0}", count(&f, 1));
        assert!(count(&f, 0) >= land0);
        let continental = |f: &Frames| {
            f.alive
                .iter()
                .zip(&f.samples)
                .filter(|(a, s)| **a && s.is_continental())
                .count()
        };
        assert_eq!(continental(&f), land0);
    }

    #[test]
    fn colliding_continents_thicken_and_keep_their_crust() {
        let g = grid();
        let mut f = split(&g, continent(), continent());
        let volume = |f: &Frames| -> f64 {
            f.alive
                .iter()
                .zip(&f.samples)
                .filter(|(a, s)| **a && s.is_continental())
                .map(|(_, s)| s.crust_km as f64)
                .sum()
        };
        let before = volume(&f);
        let thickest = |f: &Frames| f.samples.iter().map(|s| s.crust_km).fold(0.0, f32::max);
        for _ in 0..10 {
            f.step(&g, &SPLIT, STEP_MYR);
        }
        // Where they meet, crust thickens (merged, not lost); continental
        // crust volume holds, short of the thickness cap.
        assert!(
            thickest(&f) > continent().crust_km + 10.0,
            "{}",
            thickest(&f)
        );
        let after = volume(&f);
        assert!(
            (after - before).abs() / before < 0.02,
            "{before} -> {after}"
        );
    }

    #[test]
    fn shader_compiles() {
        // Browsers only check WGSL at runtime; validate it here, with the
        // generated constants.
        let source = wgsl_source();
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        assert_eq!(std::mem::size_of::<ColumnOut>(), 40);
    }

    /// What the view pass reports for each world cell.
    fn view(f: &Frames, world: &World, covers: &[Option<Cover>]) -> Vec<ColumnOut> {
        covers
            .iter()
            .enumerate()
            .map(|(d, c)| match c {
                None => ColumnOut {
                    plate: u32::MAX,
                    ..ColumnOut::zeroed()
                },
                Some(c) => {
                    let s = f.sample(*c);
                    let col = s.column();
                    ColumnOut {
                        plate: c.plate,
                        flags: s.flags & !(7 << ROCK_SHIFT) | (s.rock() as u32) << ROCK_SHIFT,
                        sediment_km: s.sediment_km,
                        crust_km: s.crust_km,
                        felsic: s.felsic,
                        mantle_km: col.mantle_km as f32,
                        depletion: s.depletion,
                        age_myr: s.age_myr,
                        elevation_m: (col.elevation_m()
                            + s.relief_m as f64
                            + world.dynamic_m[d]
                            + world.swell_m[d]) as f32,
                        _pad: 0.0,
                    }
                }
            })
            .collect()
    }

    #[test]
    fn readback_updates_world() {
        let g = Grid::new(h3o::Resolution::Three);
        let edges = crate::tectonics::edges(&g);
        let mut world = World::generate(&g, &edges, 7);
        let mut f = Frames::from_world(&world, &g);
        let omega: Vec<DVec3> = world.plates.plates.iter().map(|p| p.omega).collect();

        // A step of nothing moving reads back the world as it was.
        let still = vec![DVec3::ZERO; omega.len()];
        let covers = f.step(&g, &still, 0.0);
        let before = world.crust.elevation_m.clone();
        let out = view(&f, &world, &covers);
        apply_columns(&mut world, &g, &edges, &out);
        for (a, b) in before.iter().zip(&world.crust.elevation_m) {
            assert!((a - b).abs() < 1.0, "{a} vs {b}");
        }

        for _ in 0..10 {
            let covers = f.step(&g, &omega, STEP_MYR);
            let out = view(&f, &world, &covers);
            apply_columns(&mut world, &g, &edges, &out);
        }
        assert!(world.hotspot_marks.iter().all(Option::is_none));
        assert!(!world.boundaries.is_empty() && !world.forces.is_empty());
        // Plates moved: ownership changed somewhere, and every boundary
        // still separates two plates.
        for b in &world.boundaries {
            let e = &edges[b.edge as usize];
            let cp = &world.plates.cell_plate;
            assert_ne!(cp[e.a as usize], cp[e.b as usize]);
        }
    }

    /// The run the GPU does: steps, with the world read back and motion
    /// re-solved every `RESOLVE_MYR`. Calls `each` with the time, world,
    /// frames and covers after every step.
    fn run(
        g: &Grid,
        edges: &[Edge],
        world: &mut World,
        myr: usize,
        mut each: impl FnMut(usize, &World, &Frames, &[Option<Cover>]),
    ) {
        let mut f = Frames::from_world(world, g);
        let mut omega: Vec<DVec3> = world.plates.plates.iter().map(|p| p.omega).collect();
        for t in 1..=myr {
            let covers = f.step(g, &omega, STEP_MYR);
            if f.time_myr % RESOLVE_MYR == 0.0 {
                let out = view(&f, world, &covers);
                apply_columns(world, g, edges, &out);
                resolve_motion(world, g, edges);
                omega = world.plates.plates.iter().map(|p| p.omega).collect();
            }
            each(t, world, &f, &covers);
        }
    }

    #[test]
    fn continents_survive_collisions() {
        // A world whose continents collide early on (seed from a reported
        // tear: a continent bulldozed through another at full speed,
        // leaving new seafloor in its wake and losing crust as it went).
        let g = grid();
        let edges = crate::tectonics::edges(&g);
        let mut world = World::generate(&g, &edges, 2743546521);
        let volume = |f: &Frames| -> f64 {
            (0..f.samples.len())
                .filter(|&i| f.alive[i] && f.samples[i].is_continental())
                .map(|i| f.samples[i].crust_km as f64)
                .sum()
        };
        let start = volume(&Frames::from_world(&world, &g));
        run(&g, &edges, &mut world, 60, |t, w, f, _| {
            // Continental crust is neither subducted nor lost to the cap
            // (accreting seafloor may add a little).
            let ratio = volume(f) / start;
            assert!((0.995..1.02).contains(&ratio), "t={t}: {ratio}");
            if t % RESOLVE_MYR as usize == 0 {
                // Colliding plates have slowed to a crawl (unchecked, they
                // close at 3 cm/yr or more).
                let rates: Vec<f64> = w
                    .boundaries
                    .iter()
                    .filter(|b| b.kind == crate::tectonics::BoundaryKind::Collision)
                    .map(|b| b.rate)
                    .collect();
                let mean = rates.iter().sum::<f64>() / rates.len().max(1) as f64;
                assert!(mean < 1.5, "t={t}: {mean} cm/yr");
            }
        });
    }

    #[test]
    fn crust_past_the_cap_spills_and_buries_seafloor() {
        let mut s = Sample::new_ocean();
        spill(&mut s, 6.0, 6.0 * 0.5);
        // 7 km of basalt under 6 km of half-felsic crust.
        assert!(!s.is_continental());
        assert!((s.crust_km - 13.0).abs() < 1e-5);
        assert!((s.felsic - 3.0 / 13.0).abs() < 1e-5);
        spill(&mut s, 10.0, 5.0);
        assert!(s.is_continental());
        assert_eq!(s.province(), CrustProvince::Phanerozoic);
        assert_eq!(s.rock(), SurfaceRock::Sediment);
    }

    #[test]
    fn sample_fields_round_trip() {
        let s = continent();
        assert!(s.is_continental());
        assert_eq!(s.province(), CrustProvince::Proterozoic);
        assert_eq!(std::mem::size_of::<Sample>(), 32);
        let ocean = Sample::new_ocean();
        assert_eq!(ocean.column().mantle_km, 0.0);
        let older = Sample {
            sediment_km: 0.3,
            age_myr: 100.0,
            ..ocean
        };
        assert_eq!(older.rock(), SurfaceRock::OceanSediment);
        assert!(older.column().mantle_km > 70.0);
    }
}
