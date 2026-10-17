//! Tectonic plates: a partition of the grid into connected regions.
//!
//! Plates grow from seed cells by a multi-source Dijkstra flood fill. Each
//! plate has its own growth speed, which spreads the plate sizes the way
//! Earth's are (a few huge plates, many small ones), and each cell has a
//! spatially smooth crossing cost, which makes boundaries wander instead of
//! following straight great-circle arcs.

use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap};

use glam::DVec3;

use crate::grid::Grid;
use crate::palette::srgb_to_linear;
use crate::rng::{Pcg32, Stream};

pub const MAJOR_PLATES: usize = 8;
pub const MINOR_PLATES: usize = 10;

const MAJOR_SPEED: (f64, f64) = (0.6, 1.4);
const MINOR_SPEED: (f64, f64) = (0.15, 0.35);
/// Seeds closer than this fraction of the even spacing are rejected.
const MAJOR_SPACING: f64 = 0.6;
const MINOR_SPACING: f64 = 0.25;
/// Attempts to move enclosed plates before merging them away.
const RESEED_ROUNDS: usize = 8;
/// Crossing cost per cell is exp(ROUGHNESS * noise), noise in [-1, 1].
const ROUGHNESS: f64 = 1.0;
/// Neighbor-averaging passes for the coarse and fine noise layers.
const NOISE_BLUR: (usize, usize) = (40, 8);

#[derive(Clone, Debug)]
pub struct Plate {
    /// Linear RGB.
    pub color: [f32; 3],
    /// Angular velocity (Euler pole times rate, rad/Myr): a surface point
    /// `p` moves at `omega.cross(p)`. Zero until `tectonics` solves it.
    pub omega: DVec3,
}

pub struct Plates {
    /// Plate id for each grid cell.
    pub cell_plate: Vec<u32>,
    pub plates: Vec<Plate>,
}

impl Plate {
    /// Surface speed at unit point `p` on a planet of `radius` meters.
    pub fn speed_cm_per_year(&self, p: DVec3, radius: f64) -> f64 {
        // rad/Myr * m = m/Myr, and 1 m/Myr = 1e-4 cm/yr.
        self.omega.cross(p).length() * radius * 1e-4
    }
}

impl Plates {
    /// Plates with minor plates seeded on any boundary between majors.
    #[cfg(test)]
    pub fn generate(grid: &Grid, seed: u64) -> Self {
        Self::generate_with(grid, seed, |majors| boundary_cells(grid, majors, None))
    }

    /// Like `generate`, but `minor_sites` picks where minor plates may start,
    /// given the plate id of each cell when only the majors exist. It should
    /// return cells on boundaries between majors (so minors border at least
    /// two plates); if it returns none, any such boundary is used.
    pub fn generate_with(
        grid: &Grid,
        seed: u64,
        minor_sites: impl FnOnce(&[u32]) -> Vec<u32>,
    ) -> Self {
        let mut rng = Pcg32::for_part(seed, Stream::Plates);
        let cost = crossing_cost(grid, &mut rng);
        let speeds: Vec<f64> = (0..MAJOR_PLATES + MINOR_PLATES)
            .map(|i| {
                let (lo, hi) = if i < MAJOR_PLATES {
                    MAJOR_SPEED
                } else {
                    MINOR_SPEED
                };
                rng.range(lo, hi)
            })
            .collect();

        // Major plates anywhere.
        let even_spacing = |count: usize| (4.0 * std::f64::consts::PI / count as f64).sqrt();
        let all_cells: Vec<u32> = (0..grid.len() as u32).collect();
        let mut seeds: Vec<u32> = Vec::new();
        place_seeds(
            grid,
            &mut rng,
            &all_cells,
            &mut seeds,
            MAJOR_PLATES,
            MAJOR_SPACING * even_spacing(MAJOR_PLATES),
        );

        // Minor plates on the boundaries between major plates, the way
        // Earth's microplates sit between larger ones. A seed there grows
        // into both sides, so it borders at least two plates.
        let (majors, _) = flood_fill(grid, &seeds, &speeds, &cost);
        let mut sites = minor_sites(&majors);
        if sites.is_empty() {
            sites = boundary_cells(grid, &majors, None);
        }
        place_seeds(
            grid,
            &mut rng,
            &sites,
            &mut seeds,
            MINOR_PLATES,
            MINOR_SPACING * even_spacing(MAJOR_PLATES),
        );
        let (mut cell_plate, _) = flood_fill(grid, &seeds, &speeds, &cost);

        // A plate surrounded by a single other plate is not realistic. Move
        // its seed onto a boundary between other plates and grow again; if
        // that keeps failing, fold it into the plate around it.
        for _ in 0..RESEED_ROUNDS {
            let enclosed = enclosed_plates(grid, &cell_plate, seeds.len());
            if enclosed.is_empty() {
                break;
            }
            for plate in enclosed {
                let candidates = boundary_cells(grid, &cell_plate, Some(plate as u32));
                if !candidates.is_empty() {
                    seeds[plate] = candidates[(rng.next_f64() * candidates.len() as f64) as usize];
                }
            }
            cell_plate = flood_fill(grid, &seeds, &speeds, &cost).0;
        }
        let count = merge_enclosed(grid, &mut cell_plate, seeds.len());

        let hue_offset = rng.next_f64();
        let plates: Vec<Plate> = (0..count)
            .map(|i| Plate {
                color: plate_color(i, hue_offset),
                omega: DVec3::ZERO,
            })
            .collect();
        Self { cell_plate, plates }
    }

    pub fn plate_at(&self, cell: u32) -> &Plate {
        &self.plates[self.cell_plate[cell as usize] as usize]
    }
}

/// Picks `count` seed cells at random from `candidates`, at least `min_angle`
/// from every existing seed. The spacing relaxes if that is too crowded.
fn place_seeds(
    grid: &Grid,
    rng: &mut Pcg32,
    candidates: &[u32],
    seeds: &mut Vec<u32>,
    count: usize,
    min_angle: f64,
) {
    let mut min_angle = min_angle;
    let target = seeds.len() + count;
    let mut attempts = 0;
    while seeds.len() < target {
        let cell = candidates[(rng.next_f64() * candidates.len() as f64) as usize];
        let p = grid.centers[cell as usize];
        if seeds
            .iter()
            .all(|&s| grid.centers[s as usize].angle_between(p) >= min_angle)
        {
            seeds.push(cell);
        }
        attempts += 1;
        if attempts % 1000 == 0 {
            min_angle *= 0.9;
        }
    }
}

/// Cells with a neighbor on another plate. With `exclude`, only boundaries
/// between other plates: neither the cell nor any neighbor is on `exclude`.
pub fn boundary_cells(grid: &Grid, cell_plate: &[u32], exclude: Option<u32>) -> Vec<u32> {
    (0..grid.len() as u32)
        .filter(|&c| {
            let plate = cell_plate[c as usize];
            let around = || grid.neighbors_of(c).map(|n| cell_plate[n as usize]);
            let touches_other = around().any(|p| p != plate);
            match exclude {
                Some(x) => touches_other && plate != x && around().all(|p| p != x),
                None => touches_other,
            }
        })
        .collect()
}

/// For each plate, the set of other plates it touches.
pub fn plate_neighbors(grid: &Grid, cell_plate: &[u32], count: usize) -> Vec<BTreeSet<u32>> {
    let mut sets = vec![BTreeSet::new(); count];
    for (a, b) in grid.adjacent_pairs() {
        let (pa, pb) = (cell_plate[a as usize], cell_plate[b as usize]);
        if pa != pb {
            sets[pa as usize].insert(pb);
            sets[pb as usize].insert(pa);
        }
    }
    sets
}

/// Plates that touch fewer than two other plates, i.e. lie inside one.
/// With only two plates on the globe, each touching the other is enough.
fn enclosed_plates(grid: &Grid, cell_plate: &[u32], count: usize) -> Vec<usize> {
    let required = 2.min(count.saturating_sub(1));
    plate_neighbors(grid, cell_plate, count)
        .iter()
        .enumerate()
        .filter(|(_, n)| n.len() < required)
        .map(|(i, _)| i)
        .collect()
}

/// Folds each enclosed plate into the plate around it and renumbers the
/// rest densely. Returns the new plate count.
fn merge_enclosed(grid: &Grid, cell_plate: &mut [u32], mut count: usize) -> usize {
    while let Some(&plate) = enclosed_plates(grid, cell_plate, count).last() {
        let host = *plate_neighbors(grid, cell_plate, count)[plate]
            .first()
            .expect("an enclosed plate has a host");
        for p in cell_plate.iter_mut() {
            if *p == plate as u32 {
                *p = host;
            }
            if *p > plate as u32 {
                *p -= 1;
            }
        }
        count -= 1;
    }
    count
}

/// Per-cell crossing cost from two layers of smoothed white noise.
fn crossing_cost(grid: &Grid, rng: &mut Pcg32) -> Vec<f64> {
    let coarse = smooth_noise(grid, rng, NOISE_BLUR.0);
    let fine = smooth_noise(grid, rng, NOISE_BLUR.1);
    coarse
        .iter()
        .zip(&fine)
        .map(|(c, f)| (ROUGHNESS * (0.65 * c + 0.35 * f)).exp())
        .collect()
}

/// White noise averaged over neighbors `passes` times, rescaled to [-1, 1].
pub fn smooth_noise(grid: &Grid, rng: &mut Pcg32, passes: usize) -> Vec<f64> {
    let mut values: Vec<f64> = (0..grid.len()).map(|_| rng.range(-1.0, 1.0)).collect();
    for _ in 0..passes {
        values = (0..grid.len() as u32)
            .map(|i| {
                let (sum, n) = grid
                    .neighbors_of(i)
                    .fold((values[i as usize], 1.0), |(s, n), j| {
                        (s + values[j as usize], n + 1.0)
                    });
                sum / n
            })
            .collect();
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let max = values
        .iter()
        .map(|v| (v - mean).abs())
        .fold(f64::EPSILON, f64::max);
    values.iter().map(|v| (v - mean) / max).collect()
}

/// Queue entry ordered so the cheapest pops first from a max-heap.
struct Front {
    cost: f64,
    cell: u32,
    plate: u32,
}

impl PartialEq for Front {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Front {}
impl PartialOrd for Front {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Front {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

/// Every cell goes to the plate that reaches it first. A cell is only ever
/// claimed from a neighbor of the same plate, so every plate is connected.
/// Also returns the cost at which each cell was claimed, its arrival time.
pub fn flood_fill(
    grid: &Grid,
    seeds: &[u32],
    speeds: &[f64],
    cost: &[f64],
) -> (Vec<u32>, Vec<f64>) {
    const UNCLAIMED: u32 = u32::MAX;
    let mut owner = vec![UNCLAIMED; grid.len()];
    let mut arrival = vec![0.0; grid.len()];
    let mut heap: BinaryHeap<Front> = seeds
        .iter()
        .enumerate()
        .map(|(plate, &cell)| Front {
            cost: 0.0,
            cell,
            plate: plate as u32,
        })
        .collect();
    while let Some(Front {
        cost: c,
        cell,
        plate,
    }) = heap.pop()
    {
        if owner[cell as usize] != UNCLAIMED {
            continue;
        }
        owner[cell as usize] = plate;
        arrival[cell as usize] = c;
        let here = grid.centers[cell as usize];
        for next in grid.neighbors_of(cell) {
            if owner[next as usize] == UNCLAIMED {
                let step = here.angle_between(grid.centers[next as usize])
                    * 0.5
                    * (cost[cell as usize] + cost[next as usize]);
                heap.push(Front {
                    cost: c + step / speeds[plate as usize],
                    cell: next,
                    plate,
                });
            }
        }
    }
    (owner, arrival)
}

/// Distinct hues by golden-angle stepping.
fn plate_color(index: usize, hue_offset: f64) -> [f32; 3] {
    let hue = (hue_offset + index as f64 * 0.618_033_988_75).fract();
    hsv_to_srgb(hue, 0.5, 0.75).map(|c| srgb_to_linear(c) as f32)
}

fn hsv_to_srgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    let f = |n: f64| {
        let k = (n + h * 6.0) % 6.0;
        v - v * s * k.min(4.0 - k).clamp(0.0, 1.0)
    };
    [f(5.0), f(3.0), f(1.0)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    fn grid() -> Grid {
        Grid::new(Resolution::Three)
    }

    #[test]
    fn every_cell_assigned_and_plates_connected() {
        let g = grid();
        let p = Plates::generate(&g, 1);
        assert!(p.plates.len() > MAJOR_PLATES && p.plates.len() <= MAJOR_PLATES + MINOR_PLATES);
        assert!(
            p.cell_plate
                .iter()
                .all(|&id| (id as usize) < p.plates.len())
        );
        for id in 0..p.plates.len() {
            // Search within the plate from any one of its cells.
            let start = p.cell_plate.iter().position(|&x| x == id as u32).unwrap();
            let mut seen = vec![false; g.len()];
            let mut stack = vec![start as u32];
            seen[start] = true;
            let mut reached = 0;
            while let Some(c) = stack.pop() {
                reached += 1;
                for n in g.neighbors_of(c) {
                    if !seen[n as usize] && p.cell_plate[n as usize] == id as u32 {
                        seen[n as usize] = true;
                        stack.push(n);
                    }
                }
            }
            let size = p.cell_plate.iter().filter(|&&x| x == id as u32).count();
            assert_eq!(reached, size, "plate {id} is split");
        }
    }

    #[test]
    fn no_plate_lies_inside_another() {
        let g = grid();
        for seed in 0..8 {
            let p = Plates::generate(&g, seed);
            let neighbors = plate_neighbors(&g, &p.cell_plate, p.plates.len());
            for (id, n) in neighbors.iter().enumerate() {
                assert!(n.len() >= 2, "seed {seed}: plate {id} touches only {n:?}");
            }
        }
    }

    #[test]
    fn merge_folds_enclosed_plate_into_host() {
        let g = grid();
        // Plate 1 is a single cell inside plate 0 (the north). Plates 2 and
        // 3 split the south, so the other three each touch two plates.
        let mut cell_plate: Vec<u32> = g
            .centers
            .iter()
            .map(|c| match (c.y > 0.0, c.x > 0.0) {
                (true, _) => 0,
                (false, true) => 2,
                (false, false) => 3,
            })
            .collect();
        let inside = (0..g.len()).find(|&i| g.centers[i].y > 0.9).unwrap();
        cell_plate[inside] = 1;
        assert_eq!(enclosed_plates(&g, &cell_plate, 4), vec![1]);
        let count = merge_enclosed(&g, &mut cell_plate, 4);
        assert_eq!(count, 3);
        assert_eq!(cell_plate[inside], 0);
        // Plates 2 and 3 are renumbered down to 1 and 2.
        assert!(cell_plate.iter().all(|&p| p < 3));
        assert!(enclosed_plates(&g, &cell_plate, 3).is_empty());
    }

    #[test]
    fn same_seed_same_world() {
        let g = grid();
        let a = Plates::generate(&g, 42);
        let b = Plates::generate(&g, 42);
        assert_eq!(a.cell_plate, b.cell_plate);
        assert_ne!(a.cell_plate, Plates::generate(&g, 43).cell_plate);
    }

    #[test]
    fn plate_sizes_spread() {
        let g = grid();
        for seed in 0..5 {
            let p = Plates::generate(&g, seed);
            let mut sizes = vec![0usize; p.plates.len()];
            for &id in &p.cell_plate {
                sizes[id as usize] += 1;
            }
            let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
            assert!(*min > 0, "seed {seed}: empty plate");
            assert!(max / min >= 5, "seed {seed}: sizes {sizes:?}");
        }
    }
}
