use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

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

/// Unit icosphere: an icosahedron with each triangle split into four,
/// `subdivisions` times, and every vertex pushed out onto the sphere.
pub fn icosphere(subdivisions: u32) -> Mesh {
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let mut positions: Vec<Vec3> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|p| Vec3::from_array(*p).normalize())
    .collect();

    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    for _ in 0..subdivisions {
        // Shared edges must reuse the same midpoint vertex.
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32| -> u32 {
            let key = (a.min(b), a.max(b));
            *midpoints.entry(key).or_insert_with(|| {
                let p = (positions[a as usize] + positions[b as usize]).normalize();
                positions.push(p);
                (positions.len() - 1) as u32
            })
        };
        faces = faces
            .iter()
            .flat_map(|&[a, b, c]| {
                let ab = midpoint(a, b);
                let bc = midpoint(b, c);
                let ca = midpoint(c, a);
                [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
            })
            .collect();
    }

    Mesh {
        // On a unit sphere the normal is the position.
        vertices: positions
            .iter()
            .map(|p| Vertex {
                position: p.to_array(),
                normal: p.to_array(),
            })
            .collect(),
        indices: faces.into_iter().flatten().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdivision_counts() {
        // V = 10 * 4^n + 2, F = 20 * 4^n
        for n in 0..4 {
            let m = icosphere(n);
            assert_eq!(m.vertices.len(), 10 * 4usize.pow(n) + 2);
            assert_eq!(m.indices.len(), 3 * 20 * 4usize.pow(n));
        }
    }

    #[test]
    fn vertices_on_unit_sphere() {
        for v in icosphere(3).vertices {
            assert!((Vec3::from_array(v.position).length() - 1.0).abs() < 1e-5);
        }
    }
}
