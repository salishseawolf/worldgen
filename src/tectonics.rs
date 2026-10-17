//! Plate motion and plate boundaries.
//!
//! Each plate's angular velocity comes from a balance of torques: mantle flow
//! drags on the plate's base (more under continents, whose deep keels
//! resist), and boundary forces act on its edges. Slab pull dominates: a
//! slab sinking at a trench pulls its plate with a force that grows with the
//! square root of the slab's age (McNutt 1984), which is why plates attached
//! to slabs move several times faster than those riding over them (Forsyth &
//! Uyeda 1975; Conrad & Lithgow-Bertelloni 2002). Ridge push is about a
//! tenth as strong.
//!
//! Every boundary edge is then classified from the plates' relative motion,
//! and subduction polarity follows buoyancy: continents don't sink, and of
//! two oceanic plates the older, denser one goes down.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use glam::{DMat3, DVec3};

use crate::crust::CrustKind;
use crate::geo::EARTH_RADIUS_M;
use crate::grid::Grid;
use crate::mantle::CM_PER_YEAR;

/// Basal drag per cell relative to oceanic lithosphere: continental keels
/// reach deeper into the mantle and resist more.
const CONTINENTAL_DRAG: f64 = 3.0;
/// Boundary forces per unit edge length, in the same units as basal drag
/// (drag coefficient times rad/Myr per cell). Slab pull is for a 100 Myr
/// slab and scales with sqrt(age).
const SLAB_PULL: f64 = 0.4;
const RIDGE_PUSH: f64 = 0.04;
/// Overriding plates are drawn gently toward the trench.
const TRENCH_SUCTION: f64 = 0.06;
/// Boundaries resist motion (fault friction, coupling to the neighbor):
/// extra drag per unit edge length, stronger where continents collide.
const EDGE_DRAG: f64 = 3.0;
const COLLISION_DRAG: f64 = 6.0;
/// Boundary direction is averaged over edges this close (radians, about
/// 250 km), smoothing out the zigzag of hexagon sides.
const NORMAL_SMOOTHING: f64 = 0.04;
/// Area-weighted RMS plate speed (cm/yr) after solving; Earth's is about 4
/// to 5 in a no-net-rotation frame.
pub const PLATE_RMS_CM_YEAR: f64 = 4.5;
/// Below this share of the relative speed across a boundary, the motion
/// counts as sliding past (transform) rather than opening or closing.
const TRANSFORM_SHARE: f64 = 0.4;
/// Relative speeds below this (cm/yr) are treated as transform/inactive.
const MIN_RATE_CM_YEAR: f64 = 0.3;
/// Seafloor older than this has been subducted on Earth.
pub const MAX_OCEAN_AGE_MYR: f64 = 200.0;
/// Age for oceanic cells no ridge on their plate reaches.
pub const DEFAULT_OCEAN_AGE_MYR: f64 = 120.0;
/// Planet radius in km, for distances.
pub const RADIUS_KM: f64 = EARTH_RADIUS_M / 1000.0;

/// An edge shared by two neighboring cells.
pub struct Edge {
    pub a: u32,
    pub b: u32,
    /// The edge's vertices, unit vectors.
    pub path: Vec<DVec3>,
    pub mid: DVec3,
    /// Unit tangent at `mid` pointing from cell `a` toward cell `b`.
    pub normal: DVec3,
    /// Arc length (radians).
    pub length: f64,
}

pub fn edges(grid: &Grid) -> Vec<Edge> {
    grid.shared_edges()
        .map(|(a, b, path)| {
            let mid = path.iter().sum::<DVec3>().normalize();
            let across = grid.centers[b as usize] - grid.centers[a as usize];
            let normal = (across - mid * across.dot(mid)).normalize();
            let length = path.windows(2).map(|w| w[0].angle_between(w[1])).sum();
            Edge {
                a,
                b,
                path,
                mid,
                normal,
                length,
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundaryKind {
    /// Oceanic spreading ridge: new seafloor forms here.
    Ridge,
    /// Divergence through continental crust: a rift valley.
    Rift,
    /// Convergence where `subducting` sinks under the other side.
    Subduction,
    /// Continent meets continent: crust thickens into mountains.
    Collision,
    Transform,
}

/// A plate-boundary edge and what happens there.
#[derive(Clone, Copy, Debug)]
pub struct Boundary {
    /// Index into the edge list.
    pub edge: u32,
    pub kind: BoundaryKind,
    /// Opening, closing or sliding rate (cm/yr).
    pub rate: f64,
    /// For subduction, the cell on the sinking side (edge `a` or `b`).
    pub subducting: u32,
    /// Unit tangent across the boundary from the edge's `a` side to its `b`
    /// side, averaged along the boundary nearby.
    pub normal: DVec3,
}

impl Boundary {
    /// The cell on the side that stays up.
    pub fn overriding(&self, edges: &[Edge]) -> u32 {
        let e = &edges[self.edge as usize];
        if self.subducting == e.a { e.b } else { e.a }
    }
}

/// Angular velocity (rad/Myr) of each plate from the torque balance on it,
/// rescaled so the RMS surface speed is [`PLATE_RMS_CM_YEAR`].
///
/// For plate cells `p` with drag `k` over mantle flow `v`, basal drag
/// torque `Σ k p × (v − ω × p)` balances boundary torques `T`:
/// `Σ k (I − p pᵀ) ω = Σ k p × v + T`, a 3×3 solve per plate.
#[allow(clippy::too_many_arguments)]
pub fn solve_motion(
    grid: &Grid,
    edges: &[Edge],
    cell_plate: &[u32],
    plate_count: usize,
    crust: &[CrustKind],
    mantle: &[DVec3],
    boundaries: &[Boundary],
    age: &[f64],
) -> Vec<DVec3> {
    let mut lhs = vec![DMat3::ZERO; plate_count];
    let mut rhs = vec![DVec3::ZERO; plate_count];
    for (cell, &p) in grid.centers.iter().enumerate() {
        let plate = cell_plate[cell] as usize;
        let k = if crust[cell].is_continental() {
            CONTINENTAL_DRAG
        } else {
            1.0
        };
        lhs[plate] += (DMat3::IDENTITY - DMat3::from_cols(p * p.x, p * p.y, p * p.z)) * k;
        rhs[plate] += p.cross(mantle[cell]) * k;
    }

    // Edge lengths in units of a typical cell edge, so forces don't depend
    // on grid resolution relative to drag (which sums over cells).
    let cell_edge = (4.0 * std::f64::consts::PI / grid.len() as f64).sqrt();
    let mut push = |cell: u32, e: &Edge, normal: DVec3, toward_other: bool, strength: f64| {
        let dir = if (cell == e.a) == toward_other {
            normal
        } else {
            -normal
        };
        let force = dir * strength * e.length / cell_edge;
        rhs[cell_plate[cell as usize] as usize] += e.mid.cross(force);
    };
    for b in boundaries {
        let e = &edges[b.edge as usize];
        // Resistance along the edge, on both plates.
        let drag = if b.kind == BoundaryKind::Collision {
            COLLISION_DRAG
        } else {
            EDGE_DRAG
        } * e.length
            / cell_edge;
        let m = e.mid;
        for cell in [e.a, e.b] {
            lhs[cell_plate[cell as usize] as usize] +=
                (DMat3::IDENTITY - DMat3::from_cols(m * m.x, m * m.y, m * m.z)) * drag;
        }
        let n = b.normal;
        match b.kind {
            BoundaryKind::Subduction => {
                let slab_age = age[b.subducting as usize].clamp(10.0, 150.0);
                push(
                    b.subducting,
                    e,
                    n,
                    true,
                    SLAB_PULL * (slab_age / 100.0).sqrt(),
                );
                push(b.overriding(edges), e, n, true, TRENCH_SUCTION);
            }
            BoundaryKind::Ridge => {
                push(e.a, e, n, false, RIDGE_PUSH);
                push(e.b, e, n, false, RIDGE_PUSH);
            }
            _ => {}
        }
    }

    let mut omega: Vec<DVec3> = lhs
        .iter()
        .zip(&rhs)
        .map(|(m, r)| {
            if m.determinant().abs() < 1e-9 {
                DVec3::ZERO
            } else {
                m.inverse() * *r
            }
        })
        .collect();
    let mean_sq: f64 = grid
        .centers
        .iter()
        .enumerate()
        .map(|(cell, &p)| omega[cell_plate[cell] as usize].cross(p).length_squared())
        .sum::<f64>()
        / grid.len() as f64;
    if mean_sq > 0.0 {
        let scale = PLATE_RMS_CM_YEAR * CM_PER_YEAR / mean_sq.sqrt();
        omega.iter_mut().for_each(|w| *w *= scale);
    }
    omega
}

/// Classifies every edge between two plates.
pub fn classify(
    edges: &[Edge],
    cell_plate: &[u32],
    omega: &[DVec3],
    crust: &[CrustKind],
    age: &[f64],
) -> Vec<Boundary> {
    boundary_normals(edges, cell_plate)
        .into_iter()
        .map(|(i, normal)| {
            let e = &edges[i as usize];
            let (pa, pb) = (cell_plate[e.a as usize], cell_plate[e.b as usize]);
            // Velocity of a's side relative to b's; positive `closing`
            // means a moves toward b.
            let rel = omega[pa as usize].cross(e.mid) - omega[pb as usize].cross(e.mid);
            let closing = rel.dot(normal);
            let speed = rel.length();
            let to_cm = 1.0 / CM_PER_YEAR;
            let (ka, kb) = (crust[e.a as usize], crust[e.b as usize]);
            let (kind, rate, subducting) =
                if speed * to_cm < MIN_RATE_CM_YEAR || closing.abs() < TRANSFORM_SHARE * speed {
                    (BoundaryKind::Transform, speed * to_cm, e.a)
                } else if closing > 0.0 {
                    let rate = closing * to_cm;
                    match (ka.is_continental(), kb.is_continental()) {
                        (true, true) => (BoundaryKind::Collision, rate, e.a),
                        (false, true) => (BoundaryKind::Subduction, rate, e.a),
                        (true, false) => (BoundaryKind::Subduction, rate, e.b),
                        (false, false) => {
                            let older = if age[e.a as usize] >= age[e.b as usize] {
                                e.a
                            } else {
                                e.b
                            };
                            (BoundaryKind::Subduction, rate, older)
                        }
                    }
                } else {
                    let kind = if ka.is_continental() || kb.is_continental() {
                        BoundaryKind::Rift
                    } else {
                        BoundaryKind::Ridge
                    };
                    (kind, -closing * to_cm, e.a)
                };
            Boundary {
                edge: i,
                kind,
                rate,
                subducting,
                normal,
            }
        })
        .collect()
}

/// Smoothed boundary normals for every edge between two plates, as
/// (edge index, normal from the edge's `a` side to its `b` side). Each is
/// the mean of the normals of nearby edges between the same two plates.
fn boundary_normals(edges: &[Edge], cell_plate: &[u32]) -> Vec<(u32, DVec3)> {
    // Group boundary edges by plate pair, orienting normals from the lower
    // plate id to the higher.
    let mut groups: HashMap<(u32, u32), Vec<(u32, DVec3)>> = HashMap::new();
    for (i, e) in edges.iter().enumerate() {
        let (pa, pb) = (cell_plate[e.a as usize], cell_plate[e.b as usize]);
        if pa != pb {
            let n = if pa < pb { e.normal } else { -e.normal };
            groups
                .entry((pa.min(pb), pa.max(pb)))
                .or_default()
                .push((i as u32, n));
        }
    }
    let cos_limit = NORMAL_SMOOTHING.cos();
    let mut out: Vec<(u32, DVec3)> = groups
        .values()
        .flat_map(|group| {
            group.iter().map(|&(i, _)| {
                let e = &edges[i as usize];
                let sum: DVec3 = group
                    .iter()
                    .filter(|(j, _)| edges[*j as usize].mid.dot(e.mid) >= cos_limit)
                    .map(|(_, n)| *n)
                    .sum();
                let tangent = sum - e.mid * sum.dot(e.mid);
                let n = if tangent.length() > 1e-12 {
                    tangent.normalize()
                } else {
                    e.normal
                };
                let flip = cell_plate[e.a as usize] > cell_plate[e.b as usize];
                (i, if flip { -n } else { n })
            })
        })
        .collect();
    // HashMap order varies run to run; keep results deterministic.
    out.sort_by_key(|(i, _)| *i);
    out
}

/// Seafloor age (Myr) of every cell: distance from the nearest ridge on the
/// same plate divided by that ridge's half spreading rate. Continental cells
/// get 0; oceanic cells no ridge reaches get [`DEFAULT_OCEAN_AGE_MYR`].
pub fn ocean_ages(
    grid: &Grid,
    edges: &[Edge],
    cell_plate: &[u32],
    crust: &[CrustKind],
    boundaries: &[Boundary],
) -> Vec<f64> {
    // Half rate in km/Myr: 1 cm/yr is 10 km/Myr, half of it per side.
    let sources = boundaries
        .iter()
        .filter(|b| b.kind == BoundaryKind::Ridge)
        .flat_map(|b| {
            let e = &edges[b.edge as usize];
            let half_rate = (b.rate * 5.0).max(5.0);
            [(e.a, half_rate), (e.b, half_rate)]
        });
    let reach = flood(grid, Some(cell_plate), sources, |_, to, half_rate, km| {
        (!crust[to as usize].is_continental()).then_some(km / half_rate)
    });
    reach
        .iter()
        .zip(crust)
        .map(|(r, kind)| match (kind, r) {
            (CrustKind::Continental, _) => 0.0,
            (_, Some((age, _))) => age.min(MAX_OCEAN_AGE_MYR),
            (_, None) => DEFAULT_OCEAN_AGE_MYR,
        })
        .collect()
}

/// Multi-source Dijkstra over grid cells. Each source carries a value;
/// every cell reached gets the total cost and the value of its cheapest
/// source. `step(from, to, value, km)` gives the cost of moving between
/// neighbors `km` apart, or None to block. With `cell_plate`, the search
/// never crosses into another plate.
pub fn flood(
    grid: &Grid,
    cell_plate: Option<&[u32]>,
    sources: impl IntoIterator<Item = (u32, f64)>,
    step: impl Fn(u32, u32, f64, f64) -> Option<f64>,
) -> Vec<Option<(f64, f64)>> {
    let mut best: Vec<Option<(f64, f64)>> = vec![None; grid.len()];
    let mut heap: BinaryHeap<Entry> = sources
        .into_iter()
        .map(|(cell, value)| Entry {
            cost: 0.0,
            cell,
            value,
        })
        .collect();
    while let Some(Entry { cost, cell, value }) = heap.pop() {
        if best[cell as usize].is_some() {
            continue;
        }
        best[cell as usize] = Some((cost, value));
        let here = grid.centers[cell as usize];
        for next in grid.neighbors_of(cell) {
            if best[next as usize].is_some()
                || cell_plate.is_some_and(|cp| cp[next as usize] != cp[cell as usize])
            {
                continue;
            }
            let km = here.angle_between(grid.centers[next as usize]) * RADIUS_KM;
            if let Some(c) = step(cell, next, value, km) {
                heap.push(Entry {
                    cost: cost + c,
                    cell: next,
                    value,
                });
            }
        }
    }
    best
}

/// Heap entry ordered so the cheapest pops first.
struct Entry {
    cost: f64,
    cell: u32,
    value: f64,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Entry {}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.cell.cmp(&self.cell))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::Resolution;

    /// Two plates split at the equator: north (0) and south (1).
    fn hemispheres(grid: &Grid) -> Vec<u32> {
        grid.centers
            .iter()
            .map(|c| if c.y >= 0.0 { 0 } else { 1 })
            .collect()
    }

    #[test]
    fn edge_normals_point_from_a_to_b() {
        let grid = Grid::new(Resolution::Two);
        for e in edges(&grid).iter().step_by(17) {
            assert!(
                e.normal
                    .dot(grid.centers[e.b as usize] - grid.centers[e.a as usize])
                    > 0.0
            );
            assert!(e.normal.dot(e.mid).abs() < 1e-9);
        }
    }

    #[test]
    fn drag_alone_recovers_rigid_rotation() {
        // Mantle flow that is itself a rigid rotation carries the plate
        // along with exactly that rotation.
        let grid = Grid::new(Resolution::Two);
        let cell_plate = vec![0; grid.len()];
        let spin = DVec3::new(0.3, 1.0, -0.2).normalize() * 0.01;
        let mantle: Vec<DVec3> = grid.centers.iter().map(|&p| spin.cross(p)).collect();
        let crust = vec![CrustKind::Oceanic; grid.len()];
        let omega = solve_motion(
            &grid,
            &edges(&grid),
            &cell_plate,
            1,
            &crust,
            &mantle,
            &[],
            &[],
        );
        // Same axis; the speed is renormalized.
        assert!(omega[0].normalize().abs_diff_eq(spin.normalize(), 1e-9));
    }

    #[test]
    fn slab_pull_drives_plate_toward_trench() {
        let grid = Grid::new(Resolution::Two);
        let edges = edges(&grid);
        let cell_plate = hemispheres(&grid);
        let crust = vec![CrustKind::Oceanic; grid.len()];
        let still = vec![DVec3::ZERO; grid.len()];
        // The south plate subducts northward under the north plate all
        // along the equator.
        let boundaries: Vec<Boundary> = edges
            .iter()
            .enumerate()
            .filter(|(_, e)| cell_plate[e.a as usize] != cell_plate[e.b as usize])
            .map(|(i, e)| Boundary {
                edge: i as u32,
                kind: BoundaryKind::Subduction,
                rate: 5.0,
                subducting: if cell_plate[e.a as usize] == 1 {
                    e.a
                } else {
                    e.b
                },
                // North, whichever side `a` is on.
                normal: if cell_plate[e.a as usize] == 1 {
                    DVec3::Y
                } else {
                    -DVec3::Y
                },
            })
            .collect();
        let age = vec![100.0; grid.len()];
        // A ring of equal pull has no net torque, so only the half of the
        // trench with x > 0 pulls.
        let half: Vec<Boundary> = boundaries
            .iter()
            .copied()
            .filter(|b| edges[b.edge as usize].mid.x > 0.0)
            .collect();
        let omega_half = solve_motion(&grid, &edges, &cell_plate, 2, &crust, &still, &half, &age);
        // The south plate's surface at the trench (lon 0) moves north.
        let at_trench = omega_half[1].cross(DVec3::X);
        assert!(at_trench.y > 0.0, "{at_trench}");
        // It outpaces the overriding plate.
        assert!(omega_half[1].length() > 2.0 * omega_half[0].length());
    }

    #[test]
    fn classification_and_polarity() {
        let grid = Grid::new(Resolution::Two);
        let edges = edges(&grid);
        let cell_plate = hemispheres(&grid);
        // North plate turns about +X, south about -X. At 90°W (+Z) the
        // north side moves south and the south side north; at 90°E (-Z)
        // they move apart.
        let omega = [DVec3::X * 0.01, -DVec3::X * 0.01];
        let oceanic = vec![CrustKind::Oceanic; grid.len()];
        let mut age = vec![50.0; grid.len()];
        for (c, a) in age.iter_mut().enumerate() {
            if cell_plate[c] == 1 {
                *a = 120.0;
            }
        }
        let kinds = |crust: &[CrustKind]| {
            classify(&edges, &cell_plate, &omega, crust, &age)
                .into_iter()
                .filter(|b| edges[b.edge as usize].mid.dot(DVec3::Z) > 0.99)
                .collect::<Vec<_>>()
        };
        for b in kinds(&oceanic) {
            assert_eq!(b.kind, BoundaryKind::Subduction);
            assert_eq!(cell_plate[b.subducting as usize], 1, "older plate sinks");
        }
        // Make the south continental: the younger oceanic north sinks.
        let mut mixed = oceanic.clone();
        for (c, k) in mixed.iter_mut().enumerate() {
            if cell_plate[c] == 1 {
                *k = CrustKind::Continental;
            }
        }
        for b in kinds(&mixed) {
            assert_eq!(cell_plate[b.subducting as usize], 0, "continents float");
        }
        let continental = vec![CrustKind::Continental; grid.len()];
        assert!(
            kinds(&continental)
                .iter()
                .all(|b| b.kind == BoundaryKind::Collision)
        );
        assert!(!kinds(&oceanic).is_empty());
        let apart = classify(&edges, &cell_plate, &omega, &oceanic, &age)
            .into_iter()
            .filter(|b| edges[b.edge as usize].mid.dot(-DVec3::Z) > 0.99)
            .collect::<Vec<_>>();
        assert!(!apart.is_empty() && apart.iter().all(|b| b.kind == BoundaryKind::Ridge));
    }

    #[test]
    fn seafloor_ages_from_ridge() {
        let grid = Grid::new(Resolution::Three);
        let edges = edges(&grid);
        let cell_plate = hemispheres(&grid);
        let crust = vec![CrustKind::Oceanic; grid.len()];
        let ridge: Vec<Boundary> = edges
            .iter()
            .enumerate()
            .filter(|(_, e)| cell_plate[e.a as usize] != cell_plate[e.b as usize])
            .map(|(i, e)| Boundary {
                edge: i as u32,
                kind: BoundaryKind::Ridge,
                rate: 4.0,
                subducting: e.a,
                normal: e.normal,
            })
            .collect();
        let age = ocean_ages(&grid, &edges, &cell_plate, &crust, &ridge);
        // 2 cm/yr per side is 20 km/Myr: 1000 km from the ridge is 50 Myr.
        let at = |lat: f64| {
            let c = grid.cell_at(crate::geo::LonLat::new(0.0, lat));
            age[c as usize]
        };
        let km_per_deg = RADIUS_KM.to_radians();
        let expected = 10.0 * km_per_deg / 20.0;
        assert!(
            (at(10.0) - expected).abs() < 0.15 * expected,
            "{} vs {expected}",
            at(10.0)
        );
        assert!(at(1.0) < at(10.0) && at(10.0) < at(20.0));
        assert!(age.iter().all(|&a| a <= MAX_OCEAN_AGE_MYR));
    }
}
