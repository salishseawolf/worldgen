//! The simulation grid: every H3 cell at one resolution, flattened into dense
//! arrays so per-cell data can live in plain GPU buffers.
//!
//! The H3 index is the stable, GIS-facing key for a cell; everything inside
//! the program uses the dense id (its position in `cells`).

use glam::DVec3;
use h3o::{CellIndex, LatLng, Resolution};

use crate::geo::LonLat;

/// Neighbor slot for the missing sixth neighbor of a pentagon.
pub const NO_NEIGHBOR: u32 = u32::MAX;

pub struct Grid {
    pub resolution: Resolution,
    /// Dense id -> H3 index, sorted so lookups can binary-search.
    pub cells: Vec<CellIndex>,
    /// Cell centers as unit vectors in render space.
    pub centers: Vec<DVec3>,
    /// Six neighbor ids per cell, padded with [`NO_NEIGHBOR`].
    pub neighbors: Vec<u32>,
}

impl Grid {
    pub fn new(resolution: Resolution) -> Self {
        let mut cells: Vec<CellIndex> = CellIndex::base_cells()
            .flat_map(|base| base.children(resolution))
            .collect();
        cells.sort_unstable();
        let centers = cells.iter().map(|&c| to_unit(LatLng::from(c))).collect();
        let mut grid = Self {
            resolution,
            cells,
            centers,
            neighbors: Vec::new(),
        };
        grid.neighbors = grid
            .cells
            .iter()
            .flat_map(|&cell| {
                let mut slots = [NO_NEIGHBOR; 6];
                for (slot, edge) in slots.iter_mut().zip(cell.edges()) {
                    *slot = grid.id(edge.destination()).expect("neighbor in grid");
                }
                slots
            })
            .collect();
        grid
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn id(&self, cell: CellIndex) -> Option<u32> {
        self.cells.binary_search(&cell).ok().map(|i| i as u32)
    }

    pub fn cell_at(&self, p: LonLat) -> u32 {
        self.id(h3_cell(p, self.resolution)).expect("cell in grid")
    }

    /// Neighbor ids of `id`, skipping the empty pentagon slot.
    pub fn neighbors_of(&self, id: u32) -> impl Iterator<Item = u32> + '_ {
        let start = id as usize * 6;
        self.neighbors[start..start + 6]
            .iter()
            .copied()
            .filter(|&n| n != NO_NEIGHBOR)
    }

    /// Boundary of `id` as unit vectors, counterclockwise seen from outside.
    pub fn boundary(&self, id: u32) -> Vec<DVec3> {
        self.cells[id as usize]
            .boundary()
            .iter()
            .map(|&ll| to_unit(ll))
            .collect()
    }

    /// Each pair of neighboring cells, once, as `(a, b)` with `a < b`.
    pub fn adjacent_pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..self.len() as u32).flat_map(move |a| {
            self.neighbors_of(a)
                .filter(move |&b| a < b)
                .map(move |b| (a, b))
        })
    }

    /// Each edge shared by two cells, once, as its endpoint vertices. Edges
    /// on icosahedron faces can carry an extra distortion vertex, so a path
    /// may have more than two points.
    pub fn shared_edges(&self) -> impl Iterator<Item = (u32, u32, Vec<DVec3>)> + '_ {
        (0..self.len() as u32).flat_map(move |a| {
            self.cells[a as usize].edges().filter_map(move |edge| {
                let b = self.id(edge.destination())?;
                (a < b).then(|| {
                    (
                        a,
                        b,
                        edge.boundary().iter().map(|&ll| to_unit(ll)).collect(),
                    )
                })
            })
        })
    }
}

/// The H3 cell containing `p`. Note H3 takes (lat, lng), the reverse of our
/// GeoJSON order.
pub fn h3_cell(p: LonLat, resolution: Resolution) -> CellIndex {
    LatLng::new(p.lat, p.lon)
        .expect("finite coordinates")
        .to_cell(resolution)
}

fn to_unit(ll: LatLng) -> DVec3 {
    LonLat::new(ll.lng(), ll.lat()).to_unit()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::signed_area;

    fn grid() -> Grid {
        Grid::new(Resolution::Three)
    }

    #[test]
    fn counts() {
        let g = grid();
        assert_eq!(g.len(), 41_162);
        let pentagons = (0..g.len() as u32)
            .filter(|&i| g.neighbors_of(i).count() == 5)
            .count();
        assert_eq!(pentagons, 12);
        assert!((0..g.len() as u32).all(|i| g.neighbors_of(i).count() >= 5));
    }

    #[test]
    fn neighbors_are_mutual() {
        let g = grid();
        for a in 0..g.len() as u32 {
            for b in g.neighbors_of(a) {
                assert!(g.neighbors_of(b).any(|n| n == a), "{a} -> {b}");
            }
        }
    }

    #[test]
    fn center_maps_back_to_its_cell() {
        // Also proves our lon/lat axis convention agrees with H3's.
        let g = grid();
        for i in (0..g.len() as u32).step_by(97) {
            let p = LonLat::from_vec(g.centers[i as usize]);
            assert_eq!(g.cell_at(p), i);
        }
    }

    #[test]
    fn matches_h3_reference_example() {
        // The example from the H3 documentation.
        let p = LonLat::new(-122.0553238, 37.3615593);
        assert_eq!(h3_cell(p, Resolution::Five).to_string(), "85283473fffffff");
    }

    #[test]
    fn boundary_is_ccw_from_outside() {
        let g = grid();
        for i in (0..g.len() as u32).step_by(101) {
            assert!(signed_area(&g.boundary(i)) > 0.0, "cell {i}");
        }
    }

    #[test]
    fn shared_edges_cover_every_adjacency_once() {
        let g = grid();
        let links: usize = (0..g.len() as u32).map(|i| g.neighbors_of(i).count()).sum();
        let edges: Vec<_> = g.shared_edges().collect();
        assert_eq!(edges.len() * 2, links);
        assert!(edges.iter().all(|(_, _, path)| path.len() >= 2));
    }
}
