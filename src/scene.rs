//! Turns app state into per-frame GPU instance data.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use crate::crust::CrustProvince;
use crate::geo::{LonLat, great_circle_arc};
use crate::mantle::CM_PER_YEAR;
use crate::mantle::Province;
use crate::measure::Measure;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::tectonics::{
    Boundary, BoundaryForce, BoundaryKind, Edge, ForceKind, RADIUS_KM, SLAB_PULL,
};
use crate::world::{Hotspot, PlumeKind};

/// Outlines float this far above the surface (fraction of radius) so they
/// don't z-fight with the globe's facets.
const LINE_LIFT: f64 = 1.0005;
/// Densify edges so each drawn segment spans at most this many degrees.
const ARC_STEP_DEG: f64 = 0.5;

const CURSOR_COLOR: [f32; 4] = [0.9, 0.1, 0.1, 1.0];
const MEASURE_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 1.0];
/// Where the cursor is along the measurement, from the cross-section.
const PROBE_COLOR: [f32; 4] = [1.0, 0.85, 0.2, 1.0];
/// The line while it still follows the cursor.
const MEASURING_COLOR: [f32; 4] = [0.2, 0.9, 1.0, 0.6];
/// Arrow length per cm/yr of speed (degrees of arc), and the cap.
pub const ARROW_DEG_PER_CM_YEAR: f64 = 0.45;
const ARROW_MAX_DEG: f64 = 7.0;
pub const PLATE_MOTION_COLOR: [f32; 4] = [0.95, 0.95, 0.95, 0.9];
pub const MANTLE_FLOW_COLOR: [f32; 4] = [1.0, 0.55, 0.85, 0.9];
pub const PROVINCE_COLOR: [f32; 4] = [0.6, 0.2, 1.0, 1.0];
pub const CRATON_COLOR: [f32; 4] = [0.75, 0.95, 0.2, 1.0];
pub const SLAB_PULL_COLOR: [f32; 4] = [1.0, 0.3, 0.55, 1.0];
pub const RIDGE_PUSH_COLOR: [f32; 4] = [0.45, 0.85, 1.0, 1.0];
/// Force arrows: length (degrees of arc) for the summed pull of 1,000 km of
/// trench sinking 100 Myr seafloor; the shortest and longest drawn.
pub const FORCE_ARROW_DEG_PER_1000_KM: f64 = 4.0;
const MIN_FORCE_ARROW_DEG: f64 = 0.5;
const MAX_FORCE_ARROW_DEG: f64 = 15.0;
/// Forces along a run that mostly cancel (net under this share of their
/// total) draw no arrow.
const MIN_NET_SHARE: f64 = 0.05;
/// Hotspot rings and tracks by plume kind, on a heat scale: the deepest,
/// hottest kind brightest. Ring radius (km) shrinks with kind too.
pub fn hotspot_color(kind: PlumeKind) -> [f32; 4] {
    match kind {
        PlumeKind::Primary => [1.0, 0.9, 0.55, 1.0],
        PlumeKind::Secondary => [1.0, 0.4, 0.0, 1.0],
        PlumeKind::Tertiary => [0.65, 0.04, 0.02, 1.0],
    }
}

fn ring_km(kind: PlumeKind) -> f64 {
    match kind {
        PlumeKind::Primary => 250.0,
        PlumeKind::Secondary => 170.0,
        PlumeKind::Tertiary => 110.0,
    }
}
/// Track ticks: one every this many Myr, this long either side (km).
pub const TRACK_TICK_MYR: usize = 10;
const TRACK_TICK_KM: f64 = 60.0;

/// Force kind, acting plate and other plate: forces in one run share these.
type RunGroup = (u8, u32, u32);

/// Groups forces into uninterrupted runs: same kind, acting on the same
/// plate against the same other plate, with edges sharing endpoints.
fn force_runs<'f>(
    forces: &[&'f BoundaryForce],
    edges: &[Edge],
    cell_plate: &[u32],
) -> Vec<Vec<&'f BoundaryForce>> {
    let group = |f: &BoundaryForce| -> RunGroup {
        let e = &edges[f.edge as usize];
        let other = if f.cell == e.a { e.b } else { e.a };
        (
            f.kind as u8,
            cell_plate[f.cell as usize],
            cell_plate[other as usize],
        )
    };
    // Edge endpoints, rounded so shared vertices compare equal.
    let key = |p: DVec3| (p * 1e6).round().as_i64vec3().to_array();
    let mut parent: Vec<usize> = (0..forces.len()).collect();
    fn root(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut seen: HashMap<(RunGroup, [i64; 3]), usize> = HashMap::new();
    for (i, f) in forces.iter().enumerate() {
        let path = &edges[f.edge as usize].path;
        for end in [path[0], path[path.len() - 1]] {
            match seen.entry((group(f), key(end))) {
                Entry::Occupied(o) => {
                    let (a, b) = (root(&mut parent, i), root(&mut parent, *o.get()));
                    parent[a] = b;
                }
                Entry::Vacant(v) => {
                    v.insert(i);
                }
            }
        }
    }
    let mut runs: HashMap<usize, Vec<&'f BoundaryForce>> = HashMap::new();
    for (i, &f) in forces.iter().enumerate() {
        runs.entry(root(&mut parent, i)).or_default().push(f);
    }
    // Deterministic order (by first edge) for stable drawing.
    let mut runs: Vec<Vec<&BoundaryForce>> = runs.into_values().collect();
    runs.sort_by_key(|r| r.iter().map(|f| f.edge).min());
    runs
}

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
    ///
    /// `probe` marks a point along the finished measurement (where the
    /// cursor is in the cross-section), highlighting the line up to it.
    pub fn build(
        measure: &Measure,
        hover: Option<LonLat>,
        probe: Option<LonLat>,
        radius: f64,
        marker_scale: f64,
    ) -> Self {
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

        if let (Some(p), Some(source)) = (probe, measure.source) {
            scene.add_path(&[source.to_unit(), p.to_unit()], false, radius, PROBE_COLOR);
            scene.add_column(
                p.to_unit(),
                radius,
                marker * 0.08,
                marker * 0.004,
                PROBE_COLOR,
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

    /// One arrow per uninterrupted stretch of boundary: slab pull or ridge
    /// push edges acting on the same plate, against the same other plate,
    /// joined end to end. The arrow sits at the stretch's middle edge and
    /// shows the stretch's summed force, its length growing with the total.
    pub fn force_arrows(
        forces: &[BoundaryForce],
        edges: &[Edge],
        cell_plate: &[u32],
        radius: f64,
    ) -> Vec<Segment> {
        let shown: Vec<&BoundaryForce> = forces
            .iter()
            .filter(|f| {
                matches!(f.kind, ForceKind::SlabPull | ForceKind::RidgePush) && f.strength > 0.0
            })
            .collect();
        let mut scene = Scene::default();
        for run in force_runs(&shown, edges, cell_plate) {
            let net: DVec3 = run.iter().map(|f| f.dir * f.strength * f.length_km).sum();
            let total: f64 = run.iter().map(|f| f.strength * f.length_km).sum();
            // Anchor at the edge nearest the run's middle.
            let center = run.iter().map(|f| f.at).sum::<DVec3>().normalize_or_zero();
            let anchor = run
                .iter()
                .max_by(|a, b| a.at.dot(center).total_cmp(&b.at.dot(center)))
                .map_or(center, |f| f.at);
            let flat = net - anchor * anchor.dot(net);
            if flat.length() < MIN_NET_SHARE * total {
                continue;
            }
            let deg = (FORCE_ARROW_DEG_PER_1000_KM * flat.length() / SLAB_PULL / 1000.0)
                .clamp(MIN_FORCE_ARROW_DEG, MAX_FORCE_ARROW_DEG);
            let color = match run[0].kind {
                ForceKind::SlabPull => SLAB_PULL_COLOR,
                _ => RIDGE_PUSH_COLOR,
            };
            scene.add_arrow(anchor, flat.normalize(), deg.to_radians(), radius, color);
        }
        scene.segments
    }

    /// Outlines of the Archean cratons: every cell edge between craton and
    /// non-craton crust.
    pub fn craton_outlines(
        edges: &[Edge],
        province: &[CrustProvince],
        radius: f64,
    ) -> Vec<Segment> {
        let mut scene = Scene::default();
        let craton = |c: u32| province[c as usize] == CrustProvince::Archean;
        for e in edges.iter().filter(|e| craton(e.a) != craton(e.b)) {
            scene.add_path(&e.path, false, radius, CRATON_COLOR);
        }
        scene.segments
    }

    /// Outlines of the deep mantle provinces.
    pub fn province_outlines(provinces: &[Province], radius: f64) -> Vec<Segment> {
        let mut scene = Scene::default();
        for p in provinces {
            scene.add_path(&p.outline(96), true, radius, PROVINCE_COLOR);
        }
        scene.segments
    }

    /// A ring around each plume, colored and sized by its kind.
    pub fn hotspot_rings(hotspots: &[Hotspot], radius: f64) -> Vec<Segment> {
        let mut scene = Scene::default();
        for h in hotspots {
            let color = hotspot_color(h.kind);
            let r = ring_km(h.kind) / RADIUS_KM;
            let u = h.position.any_orthonormal_vector();
            let v = h.position.cross(u);
            let ring: Vec<DVec3> = (0..32)
                .map(|i| {
                    let a = std::f64::consts::TAU * i as f64 / 32.0;
                    h.position * r.cos() + (u * a.cos() + v * a.sin()) * r.sin()
                })
                .collect();
            scene.add_path(&ring, true, radius, color);
        }
        scene.segments
    }

    /// Each hotspot track (points one Myr apart) as a line with a tick
    /// across it every [`TRACK_TICK_MYR`].
    pub fn hotspot_tracks(
        hotspots: &[Hotspot],
        tracks: &[Vec<DVec3>],
        radius: f64,
    ) -> Vec<Segment> {
        let mut scene = Scene::default();
        let tick = TRACK_TICK_KM / RADIUS_KM;
        for (h, track) in hotspots.iter().zip(tracks) {
            let color = hotspot_color(h.kind);
            scene.add_path(track, false, radius, color);
            for i in (TRACK_TICK_MYR..track.len()).step_by(TRACK_TICK_MYR) {
                let p = track[i];
                let along = track[i] - track[i - 1];
                let across = p.cross(along).normalize_or_zero();
                if across == DVec3::ZERO {
                    continue;
                }
                let a = p * tick.cos() + across * tick.sin();
                let b = p * tick.cos() - across * tick.sin();
                scene.add_path(&[a, b], false, radius, color);
            }
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
            scene.add_arrow(p, v.normalize(), length, radius, color);
        }
        scene.segments
    }

    /// An arrow from `p` along tangent `dir`, `length` radians long.
    fn add_arrow(&mut self, p: DVec3, dir: DVec3, length: f64, radius: f64, color: [f32; 4]) {
        let tip = p * length.cos() + dir * length.sin();
        self.add_path(&[p, tip], false, radius, color);
        // Two barbs swept back from the tip.
        let back = (p - tip * p.dot(tip)).normalize();
        let side = tip.cross(back);
        let barb = 0.35 * length;
        for s in [-1.0, 1.0] {
            let d = (back * 0.9 + side * (0.45 * s)).normalize();
            let end = tip * barb.cos() + d * barb.sin();
            self.add_path(&[tip, end], false, radius, color);
        }
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
        let s = Scene::build(&m, hover, None, EARTH_RADIUS_M, EARTH_RADIUS_M);
        assert!(s.segments.len() > 3, "great circle is densified");
        assert!(s.segments.iter().all(|seg| seg.color == MEASURING_COLOR));
        for w in s.segments.windows(2) {
            assert_eq!(w[0].b, w[1].a, "continuous");
        }
        // Source marker plus the cursor.
        assert_eq!(s.columns.len(), 2);
        assert_eq!(s.columns[1].color, CURSOR_COLOR);

        m.click(LonLat::new(10.0, 0.0));
        let s = Scene::build(&m, None, None, EARTH_RADIUS_M, EARTH_RADIUS_M);
        // A probe along the line adds its marker and highlight.
        let probe = Scene::build(
            &m,
            None,
            Some(LonLat::new(5.0, 0.0)),
            EARTH_RADIUS_M,
            EARTH_RADIUS_M,
        );
        assert_eq!(probe.columns.len(), 3);
        assert!(probe.segments.iter().any(|seg| seg.color == PROBE_COLOR));
        assert!(s.segments.iter().all(|seg| seg.color == MEASURE_COLOR));
        assert_eq!(s.columns.len(), 2, "both ends marked");
    }

    #[test]
    fn hotspot_track_has_a_tick_every_ten_myr() {
        let h = Hotspot {
            kind: PlumeKind::Primary,
            position: DVec3::X,
            strength: 1.0,
        };
        // A track heading east along the equator, 0.1° per Myr.
        let track = |myr: usize| -> Vec<DVec3> {
            (0..=myr)
                .map(|t| LonLat::new(0.1 * t as f64, 0.0).to_unit())
                .collect()
        };
        // Ticks are the only segments that cross the equator.
        let ticks = |segments: Vec<Segment>| {
            segments
                .iter()
                .filter(|s| (s.a[1] > 0.0) != (s.b[1] > 0.0))
                .count()
        };
        assert_eq!(
            ticks(Scene::hotspot_tracks(&[h], &[track(9)], EARTH_RADIUS_M)),
            0
        );
        assert_eq!(
            ticks(Scene::hotspot_tracks(&[h], &[track(25)], EARTH_RADIUS_M)),
            2
        );
    }

    /// Subduction all along the equator, the south plate sinking north.
    fn equator_trench() -> (Vec<Edge>, Vec<Boundary>) {
        let grid = crate::grid::Grid::new(h3o::Resolution::Two);
        let edges = crate::tectonics::edges(&grid);
        let south = |c: u32| grid.centers[c as usize].y < 0.0;
        let boundaries = edges
            .iter()
            .enumerate()
            .filter(|(_, e)| south(e.a) != south(e.b))
            .map(|(i, e)| Boundary {
                edge: i as u32,
                kind: BoundaryKind::Subduction,
                rate: 5.0,
                subducting: if south(e.a) { e.a } else { e.b },
                normal: e.normal,
            })
            .collect();
        (edges, boundaries)
    }

    /// Slab pull north across every edge of `trench`, from the south plate.
    fn pulls(
        edges: &[Edge],
        trench: &[Boundary],
        south: impl Fn(u32) -> bool,
    ) -> Vec<BoundaryForce> {
        trench
            .iter()
            .map(|b| {
                let e = &edges[b.edge as usize];
                BoundaryForce {
                    kind: ForceKind::SlabPull,
                    edge: b.edge,
                    cell: if south(e.a) { e.a } else { e.b },
                    at: e.mid,
                    dir: (DVec3::Y - e.mid * e.mid.y).normalize(),
                    strength: SLAB_PULL,
                    length: 1.0,
                    length_km: e.length * RADIUS_KM,
                }
            })
            .collect()
    }

    #[test]
    fn one_arrow_per_uninterrupted_run() {
        let grid = crate::grid::Grid::new(h3o::Resolution::Two);
        let (edges, boundaries) = equator_trench();
        let cell_plate: Vec<u32> = grid.centers.iter().map(|c| u32::from(c.y < 0.0)).collect();
        let south = |c: u32| grid.centers[c as usize].y < 0.0;
        let arrows = |trench: Vec<Boundary>| {
            Scene::force_arrows(&pulls(&edges, &trench, south), &edges, &cell_plate, 1.0)
        };
        let on = |x: f64| -> Vec<Boundary> {
            boundaries
                .iter()
                .copied()
                .filter(|b| edges[b.edge as usize].mid.x * x.signum() > x.abs())
                .collect()
        };
        // Two separate stretches of trench give two arrows, each pointing
        // north (the shaft's far end is the northernmost point drawn).
        let one = arrows(on(0.5));
        let two = arrows([on(0.5), on(-0.5)].concat());
        assert!(!one.is_empty());
        assert_eq!(two.len(), 2 * one.len());
        let tip = one.iter().map(|s| s.b[1]).fold(f32::MIN, f32::max);
        assert!(tip > 0.01, "points north: {tip}");
        // Pulls that oppose each other along one stretch cancel: no arrow.
        let mut opposed = pulls(&edges, &on(0.5), south);
        for (i, f) in opposed.iter_mut().enumerate() {
            if i % 2 == 1 {
                f.dir = -f.dir;
            }
        }
        if opposed.len() % 2 == 1 {
            opposed.pop();
        }
        assert!(Scene::force_arrows(&opposed, &edges, &cell_plate, 1.0).is_empty());
    }

    #[test]
    fn nothing_without_measurement_or_cursor() {
        let s = Scene::build(
            &Measure::default(),
            None,
            None,
            EARTH_RADIUS_M,
            EARTH_RADIUS_M,
        );
        assert!(s.segments.is_empty() && s.columns.is_empty());
    }
}
