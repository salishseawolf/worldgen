use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use crate::grid::Grid;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

/// Globe vertex: a point on the unit sphere and the grid cell it belongs to.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CellVertex {
    pub position: [f32; 3],
    pub cell: u32,
}

/// The globe as one triangle fan per grid cell (center plus boundary), so
/// each fragment knows its cell and can read per-cell data from a buffer.
pub fn cell_mesh(grid: &Grid) -> (Vec<CellVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for id in 0..grid.len() as u32 {
        let vertex = |p: DVec3| CellVertex {
            position: p.as_vec3().to_array(),
            cell: id,
        };
        let center = vertices.len() as u32;
        vertices.push(vertex(grid.centers[id as usize]));
        let boundary = grid.boundary(id);
        vertices.extend(boundary.iter().map(|&p| vertex(p)));
        let n = boundary.len() as u32;
        for k in 0..n {
            indices.extend([center, center + 1 + k, center + 1 + (k + 1) % n]);
        }
    }
    (vertices, indices)
}

/// Unit cylinder around +Y with radius 1, from slightly below 0 (so it sinks
/// into the surface instead of floating over globe facets) up to 1, with a
/// top cap. The bottom is never visible, so it stays open.
pub fn cylinder(sides: u32) -> Mesh {
    const BOTTOM: f32 = -0.05;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..sides {
        let angle = |k: u32| k as f32 / sides as f32 * std::f32::consts::TAU;
        let (s0, c0) = angle(i).sin_cos();
        let (s1, c1) = angle(i + 1).sin_cos();
        let base = vertices.len() as u32;
        for (x, z) in [(c0, s0), (c1, s1)] {
            vertices.push(Vertex {
                position: [x, BOTTOM, z],
                normal: [x, 0.0, z],
            });
            vertices.push(Vertex {
                position: [x, 1.0, z],
                normal: [x, 0.0, z],
            });
        }
        indices.extend([base, base + 1, base + 3, base, base + 3, base + 2]);
        let cap = vertices.len() as u32;
        vertices.push(Vertex {
            position: [0.0, 1.0, 0.0],
            normal: [0.0, 1.0, 0.0],
        });
        vertices.push(Vertex {
            position: [c0, 1.0, s0],
            normal: [0.0, 1.0, 0.0],
        });
        vertices.push(Vertex {
            position: [c1, 1.0, s1],
            normal: [0.0, 1.0, 0.0],
        });
        indices.extend([cap, cap + 2, cap + 1]);
    }
    Mesh { vertices, indices }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;
    use h3o::Resolution;

    #[test]
    fn cell_mesh_faces_outward() {
        let grid = Grid::new(Resolution::Three);
        let (vertices, indices) = cell_mesh(&grid);
        let pos = |i: u32| Vec3::from_array(vertices[i as usize].position);
        for tri in indices.chunks(3).step_by(37) {
            let (a, b, c) = (pos(tri[0]), pos(tri[1]), pos(tri[2]));
            assert!((b - a).cross(c - a).dot(a) > 0.0, "front faces are CCW");
            let cells: Vec<u32> = tri.iter().map(|&i| vertices[i as usize].cell).collect();
            assert!(cells[0] == cells[1] && cells[1] == cells[2]);
        }
    }
}
