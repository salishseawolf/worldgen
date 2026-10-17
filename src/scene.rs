//! Turns app state into per-frame GPU instance data.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use crate::geo::{LonLat, great_circle_arc};
use crate::mantle::CM_PER_YEAR;
use crate::measure::Measure;
use crate::tectonics::{Boundary, BoundaryKind, Edge};

/// Outlines float this far above the surface (fraction of radius) so they
/// don't z-fight with the globe's facets.
const LINE_LIFT: f64 = 1.0005;
/// Densify edges so each drawn segment spans at most this many degrees.
const ARC_STEP_DEG: f64 = 0.5;

const CURSOR_COLOR: [f32; 4] = [0.9, 0.1, 0.1, 1.0];
const MEASURE_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 1.0];
/// The line while it still follows the cursor.
const MEASURING_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 0.6];
/// Arrow length per cm/yr of speed (degrees of arc), and the cap.
pub const ARROW_DEG_PER_CM_YEAR: f64 = 0.45;
const ARROW_MAX_DEG: f64 = 7.0;
pub const PLATE_MOTION_COLOR: [f32; 4] = [0.95, 0.95, 0.95, 0.9];
pub const MANTLE_FLOW_COLOR: [f32; 4] = [1.0, 0.55, 0.85, 0.9];

/// Line color for each boundary type; index.html shows the same as a key.
pub fn boundary_color(kind: BoundaryKind) -> [f32; 4] {
    match kind {
        BoundaryKind::Ridge => [1.0, 0.18, 0.1, 1.0],
        BoundaryKind::Rift => [1.0, 0.5, 0.05, 1.0],
        BoundaryKind::Subduction => [0.7, 0.1, 0.75, 1.0],
        BoundaryKind::Collision => [1.0, 0.85, 0.1, 1.0],
        BoundaryKind::Transform => [0.15, 0.85, 0.3, 1.0],
    }
}

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
    pub fn build(measure: &Measure, hover: Option<LonLat>, radius: f64, marker_scale: f64) -> Self {
        let mut scene = Scene::default();
        let marker = marker_scale;

        // The measurement: a great-circle line, with a marker at each end
        // that is fixed (the cursor column marks a live end).
        if let Some((a, b)) = measure.span(hover) {
            let color = if measure.is_finished() {
                MEASURE_COLOR
            } else {
                MEASURING_COLOR
            };
            scene.add_path(&[a.to_unit(), b.to_unit()], false, radius, color);
        }
        for p in [measure.source, measure.target].into_iter().flatten() {
            scene.add_column(
                p.to_unit(),
                radius,
                marker * 0.03,
                marker * 0.003,
                MEASURE_COLOR,
            );
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

    /// Plate boundary lines colored by type. These only change when the
    /// world does, so they are built once, not per frame.
    pub fn boundary_lines(edges: &[Edge], boundaries: &[Boundary], radius: f64) -> Vec<Segment> {
        let mut scene = Scene::default();
        for b in boundaries {
            let path = &edges[b.edge as usize].path;
            scene.add_path(path, false, radius, boundary_color(b.kind));
        }
        scene.segments
    }

    /// An arrow at each of `sites` (unit vectors) along `velocity` there
    /// (rad/Myr, tangent), with length proportional to speed.
    pub fn arrows(
        sites: &[DVec3],
        velocity: impl Fn(DVec3) -> DVec3,
        radius: f64,
        color: [f32; 4],
    ) -> Vec<Segment> {
        let mut scene = Scene::default();
        for &p in sites {
            let v = velocity(p);
            let speed = v.length() / CM_PER_YEAR;
            if speed < 0.05 {
                continue;
            }
            let length = (speed * ARROW_DEG_PER_CM_YEAR)
                .min(ARROW_MAX_DEG)
                .to_radians();
            let dir = v.normalize();
            let tip = p * length.cos() + dir * length.sin();
            scene.add_path(&[p, tip], false, radius, color);
            // Two barbs swept back from the tip.
            let back = (p - tip * p.dot(tip)).normalize();
            let side = tip.cross(back);
            let barb = 0.35 * length;
            for s in [-1.0, 1.0] {
                let d = (back * 0.9 + side * (0.45 * s)).normalize();
                let end = tip * barb.cos() + d * barb.sin();
                scene.add_path(&[tip, end], false, radius, color);
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
    fn measurement_line_follows_cursor_then_stays() {
        let mut m = Measure::default();
        m.click(LonLat::new(0.0, 0.0));
        let hover = Some(LonLat::new(10.0, 0.0));
        let s = Scene::build(&m, hover, EARTH_RADIUS_M, EARTH_RADIUS_M);
        assert!(s.segments.len() > 3, "great circle is densified");
        assert!(s.segments.iter().all(|seg| seg.color == MEASURING_COLOR));
        for w in s.segments.windows(2) {
            assert_eq!(w[0].b, w[1].a, "continuous");
        }
        // Source marker plus the cursor.
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[1].color, CURSOR_COLOR);

        m.click(LonLat::new(10.0, 0.0));
        let s = Scene::build(&m, None, EARTH_RADIUS_M, EARTH_RADIUS_M);
        assert!(s.segments.iter().all(|seg| seg.color == MEASURE_COLOR));
        assert_eq!(s.columns.len(), 2, "both ends marked");
    }

    #[test]
    fn nothing_without_measurement_or_cursor() {
        let s = Scene::build(&Measure::default(), None, EARTH_RADIUS_M, EARTH_RADIUS_M);
        assert!(s.segments.is_empty() && s.columns.is_empty());
    }
}
