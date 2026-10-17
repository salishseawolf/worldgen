//! Turns app state into per-frame GPU instance data.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use crate::draw::Drawing;
use crate::geo::{LonLat, great_circle_arc};
use crate::grid::Grid;
use crate::plates::Plates;

/// Outlines float this far above the surface (fraction of radius) so they
/// don't z-fight with the globe's facets.
const LINE_LIFT: f64 = 1.0005;
/// Densify edges so each drawn segment spans at most this many degrees.
const ARC_STEP_DEG: f64 = 0.5;

const CURSOR_COLOR: [f32; 4] = [0.9, 0.1, 0.1, 1.0];
const VERTEX_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 1.0];
const DRAWING_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 1.0];
const RUBBER_BAND_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 0.5];
const POLYGON_COLOR: [f32; 4] = [1.0, 0.8, 0.2, 1.0];
const PLATE_BOUNDARY_COLOR: [f32; 4] = [0.05, 0.05, 0.08, 0.85];

/// A screen-space-width line segment between two world points.
/// Fields are tightly packed to match `vertex_attr_array!` offsets.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Segment {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub color: [f32; 4],
}

/// A cylinder standing on the surface along its normal. Tightly packed, as
/// for [`Segment`].
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Column {
    pub base: [f32; 3],
    pub height: f32,
    pub normal: [f32; 3],
    pub width: f32,
    pub color: [f32; 4],
}

#[derive(Default)]
pub struct Scene {
    pub segments: Vec<Segment>,
    pub columns: Vec<Column>,
}

impl Scene {
    /// Cursor column width as a fraction of `marker_scale`.
    pub const CURSOR_WIDTH: f64 = 0.004;

    /// `marker_scale` (meters) sizes the column markers; callers vary it with
    /// zoom so markers keep a steady on-screen size.
    pub fn build(drawing: &Drawing, hover: Option<LonLat>, radius: f64, marker_scale: f64) -> Self {
        let mut scene = Scene::default();

        for polygon in &drawing.polygons {
            let ring: Vec<DVec3> = polygon.exterior.iter().map(|p| p.to_unit()).collect();
            scene.add_path(&ring, true, radius, POLYGON_COLOR);
        }

        let current: Vec<DVec3> = drawing.current.iter().map(|p| p.to_unit()).collect();
        scene.add_path(&current, false, radius, DRAWING_COLOR);
        if let (Some(&last), Some(h)) = (current.last(), hover) {
            scene.add_path(&[last, h.to_unit()], false, radius, RUBBER_BAND_COLOR);
        }

        let marker = marker_scale;
        for &p in &current {
            scene.add_column(p, radius, marker * 0.02, marker * 0.003, VERTEX_COLOR);
        }
        if let Some(h) = hover {
            scene.add_column(
                h.to_unit(),
                radius,
                marker * 0.12,
                marker * Self::CURSOR_WIDTH,
                CURSOR_COLOR,
            );
        }
        scene
    }

    /// Outlines along every cell edge where two plates meet. These only
    /// change when the plates do, so they are built once, not per frame.
    pub fn plate_boundaries(grid: &Grid, plates: &Plates, radius: f64) -> Vec<Segment> {
        let mut scene = Scene::default();
        for (a, b, path) in grid.shared_edges() {
            if plates.cell_plate[a as usize] != plates.cell_plate[b as usize] {
                scene.add_path(&path, false, radius, PLATE_BOUNDARY_COLOR);
            }
        }
        scene.segments
    }

    fn add_path(&mut self, points: &[DVec3], closed: bool, radius: f64, color: [f32; 4]) {
        if points.len() < 2 {
            return;
        }
        let edges = points.len() - usize::from(!closed);
        let mut dense: Vec<DVec3> = (0..edges)
            .flat_map(|i| {
                great_circle_arc(
                    points[i],
                    points[(i + 1) % points.len()],
                    ARC_STEP_DEG.to_radians(),
                )
            })
            .collect();
        dense.push(if closed {
            points[0]
        } else {
            points[points.len() - 1]
        });

        let to_world = |v: DVec3| (v * radius * LINE_LIFT).as_vec3().to_array();
        self.segments.extend(dense.windows(2).map(|w| Segment {
            a: to_world(w[0]),
            b: to_world(w[1]),
            color,
        }));
    }

    fn add_column(&mut self, unit: DVec3, radius: f64, height: f64, width: f64, color: [f32; 4]) {
        self.columns.push(Column {
            base: (unit * radius).as_vec3().to_array(),
            height: height as f32,
            normal: unit.as_vec3().to_array(),
            width: width as f32,
            color,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::EARTH_RADIUS_M;

    #[test]
    fn closed_polygon_outline_is_continuous_loop() {
        let mut d = Drawing::default();
        for p in [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)] {
            d.add_vertex(LonLat::new(p.0, p.1));
        }
        assert!(d.close());
        let s = Scene::build(&d, None, EARTH_RADIUS_M, EARTH_RADIUS_M);
        assert!(s.segments.len() > 3, "edges are densified");
        for w in s.segments.windows(2) {
            assert_eq!(w[0].b, w[1].a);
        }
        assert_eq!(s.segments.last().unwrap().b, s.segments[0].a);
        assert!(s.columns.is_empty());
    }

    #[test]
    fn hover_adds_cursor_column_and_rubber_band() {
        let mut d = Drawing::default();
        d.add_vertex(LonLat::new(0.0, 0.0));
        let s = Scene::build(
            &d,
            Some(LonLat::new(5.0, 5.0)),
            EARTH_RADIUS_M,
            EARTH_RADIUS_M,
        );
        assert_eq!(s.columns.len(), 2, "one vertex marker plus the cursor");
        assert_eq!(s.columns[1].color, CURSOR_COLOR);
        assert!(!s.segments.is_empty());
    }
}
