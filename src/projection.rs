//! Flat map projections from each classic family: cylindrical
//! (equirectangular and Mercator), conic (Albers equal-area) and azimuthal
//! (azimuthal equidistant, as twin hemispheres). Map coordinates are in
//! planet radii.
//!
//! The same math runs in `common.wgsl`: shaders inverse-project every map
//! pixel and forward-project lines. This copy serves picking and tests, so
//! keep the two in step.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

use glam::{DMat2, DVec2, DVec3};

use crate::geo::LonLat;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    Globe,
    Equirectangular,
    Mercator,
    Albers,
    AzimuthalEquidistant,
}

impl Projection {
    /// Id shared with the `PROJ_*` constants in common.wgsl.
    pub fn id(self) -> u32 {
        match self {
            Projection::Globe => 0,
            Projection::Equirectangular => 1,
            Projection::Mercator => 2,
            Projection::Albers => 3,
            Projection::AzimuthalEquidistant => 4,
        }
    }

    /// How far north or south (degrees) the focus may move.
    fn focus_limit(self) -> f64 {
        match self {
            Projection::Mercator => MERCATOR_MAX_LAT,
            Projection::Albers => 89.0,
            _ => 90.0,
        }
    }
}

/// Mercator runs to infinity at the poles, so it stops here.
const MERCATOR_MAX_LAT: f64 = 89.0;
/// The classic square web-map Mercator stops at y = ±π (about 85.05°); new
/// views frame that much, and the rest is a drag away.
const MERCATOR_FIT_Y: f64 = PI;
/// Each azimuthal hemisphere is a disc of radius π/2; their centers sit
/// this far either side of x = 0, leaving a small gap between the rims.
const DUAL_OFFSET: f64 = FRAC_PI_2 * 1.04;
/// Albers standard parallels (degrees), mirrored for a southern focus.
const ALBERS_PARALLELS: (f64, f64) = (20.0, 50.0);

/// A flat projection focused on `center`, which sits at the screen center:
/// its meridian runs straight up the middle of the map.
#[derive(Clone, Copy, Debug)]
pub struct MapProjection {
    pub kind: Projection,
    pub center: LonLat,
    /// Albers standard parallels (radians). Chosen from the hemisphere of
    /// the first focus and then fixed, so dragging across the equator
    /// doesn't make the map jump.
    parallels: (f64, f64),
}

impl MapProjection {
    pub fn new(kind: Projection, center: LonLat) -> Self {
        let sign = if center.lat < 0.0 { -1.0 } else { 1.0 };
        Self {
            kind,
            center,
            parallels: (
                (sign * ALBERS_PARALLELS.0).to_radians(),
                (sign * ALBERS_PARALLELS.1).to_radians(),
            ),
        }
    }

    /// (center lon, center lat, parallel 1, parallel 2), radians, for the
    /// shader uniform.
    pub fn params(&self) -> [f32; 4] {
        [
            self.center.lon.to_radians() as f32,
            self.center.lat.to_radians() as f32,
            self.parallels.0 as f32,
            self.parallels.1 as f32,
        ]
    }

    /// Albers constants n, C, rho0 (Snyder, Map Projections: A Working
    /// Manual, eq. 14-3 to 14-6, on the unit sphere).
    fn albers(&self) -> (f64, f64, f64) {
        let (p1, p2) = self.parallels;
        let n = (p1.sin() + p2.sin()) / 2.0;
        let c = p1.cos().powi(2) + 2.0 * n * p1.sin();
        let rho0 = (c - 2.0 * n * self.center.lat.to_radians().sin()).sqrt() / n;
        (n, c, rho0)
    }

    pub fn forward(&self, p: DVec3) -> Option<DVec2> {
        let ll = LonLat::from_vec(p);
        let (lat, dlon) = (
            ll.lat.to_radians(),
            wrap_pi((ll.lon - self.center.lon).to_radians()),
        );
        match self.kind {
            Projection::Globe => None,
            Projection::Equirectangular => Some(DVec2::new(dlon, lat)),
            Projection::Mercator => Some(DVec2::new(dlon, mercator_y(lat))),
            Projection::Albers => {
                let (n, c, rho0) = self.albers();
                let rho = (c - 2.0 * n * lat.sin()).max(0.0).sqrt() / n;
                let theta = n * dlon;
                Some(DVec2::new(rho * theta.sin(), rho0 - rho * theta.cos()))
            }
            Projection::AzimuthalEquidistant => {
                // Focus hemisphere on the left, the far one on the right.
                Some(if p.dot(self.center.to_unit()) >= 0.0 {
                    azimuthal(self.center, p) - DVec2::new(DUAL_OFFSET, 0.0)
                } else {
                    azimuthal(antipode(self.center), p) + DVec2::new(DUAL_OFFSET, 0.0)
                })
            }
        }
    }

    /// The unit vector shown at map point `m`, or None off the map.
    pub fn inverse(&self, m: DVec2) -> Option<DVec3> {
        let lonlat = |dlon: f64, lat: f64| {
            LonLat::new(self.center.lon + dlon.to_degrees(), lat.to_degrees()).to_unit()
        };
        match self.kind {
            Projection::Globe => None,
            Projection::Equirectangular => {
                (m.x.abs() <= PI && m.y.abs() <= FRAC_PI_2).then(|| lonlat(m.x, m.y))
            }
            Projection::Mercator => (m.x.abs() <= PI
                && m.y.abs() <= mercator_y(MERCATOR_MAX_LAT.to_radians()))
            .then(|| lonlat(m.x, m.y.sinh().atan())),
            Projection::Albers => {
                let (n, c, rho0) = self.albers();
                let (x, y) = (m.x * n.signum(), (rho0 - m.y) * n.signum());
                let rho = x.hypot(y) * n.signum();
                let theta = x.atan2(y);
                let sin_lat = (c - (rho * n).powi(2)) / (2.0 * n);
                (theta.abs() <= PI * n.abs() && sin_lat.abs() <= 1.0)
                    .then(|| lonlat(theta / n, sin_lat.asin()))
            }
            Projection::AzimuthalEquidistant => {
                let left = m + DVec2::new(DUAL_OFFSET, 0.0);
                let right = m - DVec2::new(DUAL_OFFSET, 0.0);
                // A hair of slack so points exactly on the rim round-trip.
                let rim = FRAC_PI_2 + 1e-9;
                if left.length() <= rim {
                    Some(azimuthal_inverse(self.center, left))
                } else if right.length() <= rim {
                    Some(azimuthal_inverse(antipode(self.center), right))
                } else {
                    None
                }
            }
        }
    }

    /// Map point of the focus, which is drawn at the screen center.
    pub fn origin(&self) -> DVec2 {
        self.forward(self.center.to_unit()).unwrap_or_default()
    }

    /// Bounding box (min, max) of the whole map, found by sampling.
    pub fn extent(&self) -> (DVec2, DVec2) {
        let mut min = DVec2::splat(f64::INFINITY);
        let mut max = DVec2::splat(f64::NEG_INFINITY);
        for i in 0..=72 {
            for j in 0..=36 {
                // Stay a hair inside the seam, which maps to both edges.
                let dlon = -179.999 + 359.998 * i as f64 / 72.0;
                let lat = -90.0 + 180.0 * j as f64 / 36.0;
                let p = LonLat::new(self.center.lon + dlon, lat).to_unit();
                if let Some(m) = self.forward(p) {
                    min = min.min(m);
                    max = max.max(m);
                }
            }
        }
        (min, max)
    }

    /// Moves the focus over the globe so that `grabbed` (a unit vector)
    /// lands `offset` map units from the screen center. Solved by Newton's
    /// method on (lon, lat); where that can't be met exactly (past a pole
    /// limit), it settles as close as the limits allow.
    fn refocus(&mut self, grabbed: DVec3, offset: DVec2) {
        const H: f64 = 1e-5; // degrees
        const MAX_STEP: f64 = 20.0; // degrees
        let limit = self.kind.focus_limit();
        let residual = |center: LonLat| {
            let p = Self { center, ..*self };
            p.forward(grabbed).map(|m| m - p.origin() - offset)
        };
        let mut c = self.center;
        for _ in 0..12 {
            let Some(r) = residual(c) else { return };
            if r.length() < 1e-12 {
                break;
            }
            // Step latitude away from the limit so the probe stays valid.
            let h_lat = if c.lat + H > limit { -H } else { H };
            let (Some(r_lon), Some(r_lat)) = (
                residual(LonLat::new(c.lon + H, c.lat)),
                residual(LonLat::new(c.lon, c.lat + h_lat)),
            ) else {
                return;
            };
            let jacobian = DMat2::from_cols((r_lon - r) / H, (r_lat - r) / h_lat);
            if jacobian.determinant().abs() < 1e-12 {
                break;
            }
            let step = -(jacobian.inverse() * r);
            let step = step * (MAX_STEP / step.length()).min(1.0);
            c = LonLat::new(
                wrap_pi((c.lon + step.x).to_radians()).to_degrees(),
                (c.lat + step.y).clamp(-limit, limit),
            );
        }
        self.center = c;
    }

    /// True if the straight map segment between `a` and `b` would jump
    /// across the map's cut (the antimeridian of the center, or for the
    /// azimuthal map the rim between its hemispheres). Only the shader needs
    /// this; it is here as the tested reference.
    #[cfg(test)]
    pub fn crosses_seam(&self, a: DVec3, b: DVec3) -> bool {
        match self.kind {
            Projection::Globe => false,
            Projection::AzimuthalEquidistant => {
                let c = self.center.to_unit();
                (a.dot(c) >= 0.0) != (b.dot(c) >= 0.0)
            }
            _ => {
                let rel =
                    |p: DVec3| wrap_pi((LonLat::from_vec(p).lon - self.center.lon).to_radians());
                (rel(a) - rel(b)).abs() > PI
            }
        }
    }
}

/// Azimuthal equidistant map point of `p` around `center`, north up.
fn azimuthal(center: LonLat, p: DVec3) -> DVec2 {
    let (east, north) = center.east_north();
    let local = DVec2::new(p.dot(east), p.dot(north));
    let distance = p.normalize().dot(center.to_unit()).clamp(-1.0, 1.0).acos();
    let len = local.length();
    if len < 1e-12 {
        DVec2::ZERO
    } else {
        local / len * distance
    }
}

fn azimuthal_inverse(center: LonLat, m: DVec2) -> DVec3 {
    let r = m.length();
    let (east, north) = center.east_north();
    let dir = if r < 1e-12 { DVec2::ZERO } else { m / r };
    center.to_unit() * r.cos() + (east * dir.x + north * dir.y) * r.sin()
}

/// The point on the opposite side of the planet.
fn antipode(p: LonLat) -> LonLat {
    LonLat::new(wrap_pi((p.lon + 180.0).to_radians()).to_degrees(), -p.lat)
}

fn mercator_y(lat: f64) -> f64 {
    let max = MERCATOR_MAX_LAT.to_radians();
    (FRAC_PI_4 + lat.clamp(-max, max) / 2.0).tan().ln()
}

fn wrap_pi(a: f64) -> f64 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

/// A flat map on screen: the projection, focused on the screen center, and
/// the zoom.
#[derive(Clone, Copy, Debug)]
pub struct MapView {
    pub projection: MapProjection,
    /// Map units from the focus to the screen center. Zero except on the
    /// twin-hemisphere map, which zooms by panning instead of turning the
    /// globe, so either hemisphere can fill the screen.
    pub pan: DVec2,
    /// Map units from the screen center to its top edge.
    pub half_height: f64,
    /// `half_height` when the whole map fit, the reference for zoom limits.
    fitted: f64,
}

/// Zoom limits relative to the fitted view.
const MIN_ZOOM: f64 = 0.005;
const MAX_ZOOM: f64 = 2.0;

impl MapView {
    /// Zoomed so the whole map would fit on screen, with a small margin.
    pub fn new(projection: MapProjection, aspect: f64) -> Self {
        let (mut min, mut max) = projection.extent();
        if projection.kind == Projection::Mercator {
            min.y = min.y.max(-MERCATOR_FIT_Y);
            max.y = max.y.min(MERCATOR_FIT_Y);
        }
        let size = max - min;
        // The globe has no flat extent; its zoom is unused.
        let fitted = if size.is_finite() {
            0.53 * size.y.max(size.x / aspect)
        } else {
            1.0
        };
        // Center the twin hemispheres, not the focus.
        let pan = if projection.kind == Projection::AzimuthalEquidistant {
            DVec2::new(DUAL_OFFSET, 0.0)
        } else {
            DVec2::ZERO
        };
        Self {
            projection,
            pan,
            half_height: fitted,
            fitted,
        }
    }

    /// Map units from the focus to a point in normalized device coordinates.
    fn offset(&self, ndc_x: f64, ndc_y: f64, aspect: f64) -> DVec2 {
        self.pan + DVec2::new(ndc_x * aspect, ndc_y) * self.half_height
    }

    /// Map point drawn at the screen center.
    pub fn screen_center(&self) -> DVec2 {
        self.projection.origin() + self.pan
    }

    /// The unit vector under a point in normalized device coordinates.
    pub fn surface_at(&self, ndc_x: f64, ndc_y: f64, aspect: f64) -> Option<DVec3> {
        let p = &self.projection;
        p.inverse(p.origin() + self.offset(ndc_x, ndc_y, aspect))
    }

    /// Turns the globe under the map so `grabbed` (a unit vector) sits
    /// under the cursor, moving the focus as a globe drag would.
    pub fn drag_to(&mut self, grabbed: DVec3, ndc_x: f64, ndc_y: f64, aspect: f64) {
        let offset = self.offset(ndc_x, ndc_y, aspect);
        self.projection.refocus(grabbed, offset);
    }

    /// Fallback for drags that start off the map: move the focus by the
    /// pointer motion in CSS pixels, given the canvas height in CSS pixels.
    pub fn drag_by(&mut self, dx: f64, dy: f64, height_px: f64) {
        // Near the focus, one map unit is about one radian.
        let deg_per_px = (2.0 * self.half_height / height_px).to_degrees();
        let c = self.projection.center;
        let limit = self.projection.kind.focus_limit();
        self.projection.center = LonLat::new(
            wrap_pi((c.lon - dx * deg_per_px).to_radians()).to_degrees(),
            (c.lat + dy * deg_per_px).clamp(-limit, limit),
        );
    }

    /// Zooms by `amount` (positive zooms out) keeping the point under the
    /// cursor fixed.
    pub fn zoom_at(&mut self, amount: f64, ndc_x: f64, ndc_y: f64, aspect: f64) {
        let before = self.offset(ndc_x, ndc_y, aspect);
        let anchor = self.surface_at(ndc_x, ndc_y, aspect);
        self.half_height =
            (self.half_height * amount.exp()).clamp(self.fitted * MIN_ZOOM, self.fitted * MAX_ZOOM);
        if self.projection.kind == Projection::AzimuthalEquidistant {
            // Plain 2D zoom about the cursor: the map point under it stays.
            self.pan += before - self.offset(ndc_x, ndc_y, aspect);
        } else if let Some(anchor) = anchor {
            self.drag_to(anchor, ndc_x, ndc_y, aspect);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projections() -> Vec<MapProjection> {
        let mut out = Vec::new();
        for kind in [
            Projection::Equirectangular,
            Projection::Mercator,
            Projection::Albers,
            Projection::AzimuthalEquidistant,
        ] {
            for center in [
                LonLat::new(0.0, 0.0),
                LonLat::new(-123.0, 48.5),
                LonLat::new(150.0, -35.0),
            ] {
                out.push(MapProjection::new(kind, center));
            }
        }
        out
    }

    #[test]
    fn inverse_undoes_forward() {
        for proj in projections() {
            for lon in (-170..=170).step_by(20) {
                for lat in (-80..=80).step_by(20) {
                    let p = LonLat::new(lon as f64, lat as f64).to_unit();
                    let m = proj.forward(p).unwrap();
                    let back = proj
                        .inverse(m)
                        .unwrap_or_else(|| panic!("{proj:?} {lon},{lat}"));
                    assert!(back.abs_diff_eq(p, 1e-9), "{proj:?} {lon},{lat}: {back}");
                }
            }
        }
    }

    #[test]
    fn center_maps_to_origin() {
        for proj in projections() {
            let m = proj.origin();
            // Cylindrical maps keep the equator at y = 0, and the focus
            // hemisphere is the left of the azimuthal pair.
            let expected = match proj.kind {
                Projection::Equirectangular => DVec2::new(0.0, proj.center.lat.to_radians()),
                Projection::Mercator => DVec2::new(0.0, mercator_y(proj.center.lat.to_radians())),
                Projection::AzimuthalEquidistant => DVec2::new(-DUAL_OFFSET, 0.0),
                _ => DVec2::ZERO,
            };
            assert!(m.abs_diff_eq(expected, 1e-9), "{proj:?}: {m}");
        }
    }

    #[test]
    fn north_is_up_on_center_meridian() {
        for proj in projections() {
            let c = proj.center;
            let below = proj
                .forward(LonLat::new(c.lon, c.lat - 5.0).to_unit())
                .unwrap();
            let above = proj
                .forward(LonLat::new(c.lon, c.lat + 5.0).to_unit())
                .unwrap();
            assert!(
                above.y > below.y && (above.x - below.x).abs() < 1e-9,
                "{proj:?}"
            );
            let east = proj
                .forward(LonLat::new(c.lon + 5.0, c.lat).to_unit())
                .unwrap();
            assert!(east.x > proj.origin().x, "{proj:?}: east is right");
        }
    }

    #[test]
    fn albers_is_equal_area() {
        // Equal lon/lat boxes at one latitude band keep the same map area,
        // and a box's map area matches its area on the unit sphere.
        let proj = MapProjection::new(Projection::Albers, LonLat::new(-100.0, 40.0));
        let quad_area = |lon: f64, lat: f64| {
            let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(dx, dy)| {
                proj.forward(LonLat::new(lon + dx, lat + dy).to_unit())
                    .unwrap()
            });
            let mut a = 0.0;
            for i in 0..4 {
                a += corners[i].perp_dot(corners[(i + 1) % 4]);
            }
            a.abs() / 2.0
        };
        let sphere = |lat: f64| {
            1f64.to_radians() * ((lat + 1.0).to_radians().sin() - lat.to_radians().sin())
        };
        for (lon, lat) in [(-100.0, 40.0), (-60.0, 10.0), (-140.0, 65.0)] {
            let ratio = quad_area(lon, lat) / sphere(lat);
            assert!((ratio - 1.0).abs() < 1e-3, "{lon},{lat}: {ratio}");
        }
    }

    #[test]
    fn off_map_points_have_no_inverse() {
        let at = |kind| MapProjection::new(kind, LonLat::new(0.0, 0.0));
        let eq = at(Projection::Equirectangular);
        assert!(eq.inverse(DVec2::new(3.2, 0.0)).is_none());
        assert!(eq.inverse(DVec2::new(0.0, 1.6)).is_none());
        let mercator = at(Projection::Mercator);
        assert!(mercator.inverse(DVec2::new(0.0, 3.2)).is_some(), "past 85°");
        assert!(mercator.inverse(DVec2::new(0.0, 4.8)).is_none(), "past 89°");
        let az = at(Projection::AzimuthalEquidistant);
        assert!(az.inverse(DVec2::ZERO).is_none(), "gap between hemispheres");
        assert!(az.inverse(DVec2::new(3.0, 1.0)).is_none());
        let albers = at(Projection::Albers);
        let (min, _) = albers.extent();
        assert!(albers.inverse(min - DVec2::splat(0.5)).is_none());
    }

    #[test]
    fn seam_detection() {
        let proj = MapProjection::new(Projection::Equirectangular, LonLat::new(-123.0, 0.0));
        // The cut is at lon 57 E.
        let west = LonLat::new(56.9, 0.0).to_unit();
        let east = LonLat::new(57.1, 0.0).to_unit();
        assert!(proj.crosses_seam(west, east));
        let near = LonLat::new(-122.0, 0.0).to_unit();
        assert!(!proj.crosses_seam(near, LonLat::new(-124.0, 0.0).to_unit()));
    }

    #[test]
    fn mercator_is_conformal() {
        // Map scale is the same east and north. A step of one degree east
        // is cos(lat) as long on the sphere as one degree north.
        let proj = MapProjection::new(Projection::Mercator, LonLat::new(0.0, 0.0));
        for lat in [0.0, 30.0, 60.0, 80.0] {
            let at = |dlon: f64, dlat: f64| {
                proj.forward(LonLat::new(dlon, lat + dlat).to_unit())
                    .unwrap()
            };
            let base = at(0.0, 0.0);
            let east = (at(1e-4, 0.0) - base).length() / lat.to_radians().cos();
            let north = (at(0.0, 1e-4) - base).length();
            assert!(
                (east / north - 1.0).abs() < 1e-3,
                "{lat}: {east} vs {north}"
            );
        }
    }

    #[test]
    fn drag_keeps_grabbed_point_under_cursor() {
        for proj in projections() {
            let mut view = MapView::new(proj, 1.5);
            view.half_height *= 0.3;
            // Both within the right-hand azimuthal hemisphere.
            let (from, to) = ((0.3, -0.2), (0.5, 0.1));
            let grabbed = view.surface_at(from.0, from.1, 1.5).unwrap();
            view.drag_to(grabbed, to.0, to.1, 1.5);
            let under = view.surface_at(to.0, to.1, 1.5).unwrap();
            assert!(
                under.angle_between(grabbed) < 1e-6,
                "{proj:?}: off by {} rad",
                under.angle_between(grabbed)
            );
        }
    }

    #[test]
    fn horizontal_drag_wraps_around_the_world() {
        // Dragging a cylindrical map sideways turns the globe: the focus
        // longitude moves and wraps instead of the map sliding off screen.
        let proj = MapProjection::new(Projection::Mercator, LonLat::new(170.0, 0.0));
        let mut view = MapView::new(proj, 1.0);
        // About 38 degrees of longitude.
        let grabbed = view.surface_at(0.1, 0.0, 1.0).unwrap();
        view.drag_to(grabbed, -0.1, 0.0, 1.0);
        let lon = view.projection.center.lon;
        assert!(
            (-160.0..-140.0).contains(&lon),
            "focus moved east across 180: {lon}"
        );
        assert!(
            view.surface_at(-0.1, 0.0, 1.0)
                .unwrap()
                .angle_between(grabbed)
                < 1e-6
        );
    }

    #[test]
    fn drag_past_pole_stops_at_limit() {
        let proj = MapProjection::new(Projection::Equirectangular, LonLat::new(0.0, 80.0));
        let mut view = MapView::new(proj, 1.0);
        let grabbed = view.surface_at(0.0, 0.0, 1.0).unwrap();
        view.drag_to(grabbed, 0.0, -1.0, 1.0);
        assert!(view.projection.center.lat <= 90.0);
        assert!(view.projection.center.lat > 80.0);
    }

    #[test]
    fn hemispheres_meet_at_their_rims() {
        let proj = MapProjection::new(Projection::AzimuthalEquidistant, LonLat::new(-123.0, 48.5));
        let c = proj.center.to_unit();
        let (east, _) = proj.center.east_north();
        // A point due east, a hair either side of the dividing great circle.
        let near = (c * 1e-6 + east).normalize();
        let far = (-c * 1e-6 + east).normalize();
        let (a, b) = (proj.forward(near).unwrap(), proj.forward(far).unwrap());
        assert!(
            a.abs_diff_eq(DVec2::new(FRAC_PI_2 - DUAL_OFFSET, 0.0), 1e-5),
            "{a}"
        );
        assert!(
            b.abs_diff_eq(DVec2::new(DUAL_OFFSET - FRAC_PI_2, 0.0), 1e-5),
            "{b}"
        );
    }

    #[test]
    fn dual_zoom_pans_without_turning_the_globe() {
        let proj = MapProjection::new(Projection::AzimuthalEquidistant, LonLat::new(-123.0, 48.5));
        let mut view = MapView::new(proj, 1.5);
        let before = view.surface_at(0.4, 0.1, 1.5).unwrap();
        for _ in 0..5 {
            view.zoom_at(-0.5, 0.4, 0.1, 1.5);
        }
        assert_eq!(view.projection.center, proj.center);
        assert!(
            view.surface_at(0.4, 0.1, 1.5)
                .unwrap()
                .angle_between(before)
                < 1e-9
        );
        // Zoomed into the far hemisphere: it fills the screen.
        assert!(
            view.surface_at(0.0, 0.0, 1.5)
                .unwrap()
                .dot(proj.center.to_unit())
                < 0.0
        );
    }

    #[test]
    fn view_zoom_keeps_cursor_point() {
        let proj = MapProjection::new(Projection::Albers, LonLat::new(-123.0, 48.5));
        let mut view = MapView::new(proj, 1.5);
        let before = view.surface_at(0.3, -0.2, 1.5).unwrap();
        view.zoom_at(-0.7, 0.3, -0.2, 1.5);
        assert!(view.half_height < view.fitted);
        let after = view.surface_at(0.3, -0.2, 1.5).unwrap();
        assert!(after.angle_between(before) < 1e-6);
    }
}
