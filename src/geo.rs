//! Spherical GIS math.
//!
//! Coordinates are WGS84 longitude/latitude in degrees (EPSG:4326), stored in
//! GeoJSON axis order (lon, lat). The planet is modeled as a sphere.
//!
//! Render-space axes are ECEF renamed for Y-up:
//! north pole = +Y, (lon 0, lat 0) = +X, (lon 90°E, lat 0) = -Z.

use glam::DVec3;

/// Mean Earth radius (IUGG R1) in meters.
pub const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Below this angle (radians) two points count as the same vertex.
const SAME_POINT_RAD: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LonLat {
    /// Degrees east, in [-180, 180].
    pub lon: f64,
    /// Degrees north, in [-90, 90].
    pub lat: f64,
}

impl LonLat {
    pub const fn new(lon: f64, lat: f64) -> Self {
        Self { lon, lat }
    }

    pub fn to_unit(self) -> DVec3 {
        let (lon, lat) = (self.lon.to_radians(), self.lat.to_radians());
        DVec3::new(lat.cos() * lon.cos(), lat.sin(), -lat.cos() * lon.sin())
    }

    /// `v` need not be normalized, only non-zero.
    pub fn from_vec(v: DVec3) -> Self {
        let v = v.normalize();
        Self {
            lon: (-v.z).atan2(v.x).to_degrees(),
            lat: v.y.clamp(-1.0, 1.0).asin().to_degrees(),
        }
    }

    /// Human-readable form, e.g. `47.6062° N, 122.3321° W`.
    pub fn display(self) -> String {
        let ns = if self.lat >= 0.0 { 'N' } else { 'S' };
        let ew = if self.lon >= 0.0 { 'E' } else { 'W' };
        format!("{:.4}° {ns}, {:.4}° {ew}", self.lat.abs(), self.lon.abs())
    }
}

/// Points along the great-circle arc from `a` to `b` (unit vectors), spaced at
/// most `max_step` radians apart. Includes `a`, excludes `b`, so consecutive
/// arcs chain without duplicates. Antipodal endpoints have no unique arc and
/// yield just `a`.
pub fn great_circle_arc(a: DVec3, b: DVec3, max_step: f64) -> Vec<DVec3> {
    let angle = a.angle_between(b);
    if angle < SAME_POINT_RAD || (std::f64::consts::PI - angle) < 1e-6 {
        return vec![a];
    }
    let steps = (angle / max_step).ceil().max(1.0) as usize;
    let sin = angle.sin();
    (0..steps)
        .map(|i| {
            let t = i as f64 / steps as f64;
            (a * ((1.0 - t) * angle).sin() + b * (t * angle).sin()) / sin
        })
        .collect()
}

/// Signed area in steradians of the spherical polygon with great-circle edges
/// through `ring` (unit vectors, open ring). Positive when the vertices run
/// counterclockwise seen from outside the sphere, i.e. the interior is on the
/// left. Fan-triangulates from vertex 0 using the Van Oosterom-Strackee
/// triangle formula, so it is exact for any ring smaller than a hemisphere.
pub fn signed_area(ring: &[DVec3]) -> f64 {
    if ring.len() < 3 {
        return 0.0;
    }
    let a = ring[0];
    ring[1..]
        .windows(2)
        .map(|w| {
            let (b, c) = (w[0], w[1]);
            let numerator = a.dot(b.cross(c));
            let denominator = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
            2.0 * numerator.atan2(denominator)
        })
        .sum()
}

/// A polygon with great-circle edges, shaped for a direct GeoJSON mapping.
///
/// `exterior` is an open ring (the closing point is not repeated) wound
/// counterclockwise, matching the RFC 7946 right-hand rule for exterior rings.
/// GeoJSON export needs only to append `exterior[0]`, densify edges with
/// [`great_circle_arc`] (GeoJSON edges are straight in lon/lat), and split
/// rings that cross the antimeridian.
#[derive(Clone, Debug, PartialEq)]
pub struct Polygon {
    pub exterior: Vec<LonLat>,
}

impl Polygon {
    /// Builds a polygon from at least 3 distinct vertices in either winding.
    /// The region taken as the interior is the smaller of the two the ring
    /// divides the sphere into.
    pub fn from_ring(mut ring: Vec<LonLat>) -> Option<Self> {
        if ring.len() > 1 && same_point(ring[0], ring[ring.len() - 1]) {
            ring.pop();
        }
        if ring.len() < 3 {
            return None;
        }
        let units: Vec<DVec3> = ring.iter().map(|p| p.to_unit()).collect();
        if signed_area(&units) < 0.0 {
            ring.reverse();
        }
        Some(Self { exterior: ring })
    }

    pub fn area_m2(&self, radius: f64) -> f64 {
        let units: Vec<DVec3> = self.exterior.iter().map(|p| p.to_unit()).collect();
        signed_area(&units).abs() * radius * radius
    }
}

pub fn same_point(a: LonLat, b: LonLat) -> bool {
    a.to_unit().angle_between(b.to_unit()) < SAME_POINT_RAD
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn axis_convention() {
        assert!(LonLat::new(0.0, 0.0).to_unit().abs_diff_eq(DVec3::X, 1e-12));
        assert!(
            LonLat::new(0.0, 90.0)
                .to_unit()
                .abs_diff_eq(DVec3::Y, 1e-12)
        );
        assert!(
            LonLat::new(90.0, 0.0)
                .to_unit()
                .abs_diff_eq(-DVec3::Z, 1e-12)
        );
    }

    #[test]
    fn round_trip() {
        for &(lon, lat) in &[
            (0.0, 0.0),
            (-122.3321, 47.6062),
            (179.999, -33.0),
            (-179.999, 12.0),
            (45.0, 89.9),
            (-90.0, -89.9),
        ] {
            let p = LonLat::from_vec(LonLat::new(lon, lat).to_unit());
            assert!(
                close(p.lon, lon, 1e-9) && close(p.lat, lat, 1e-9),
                "{lon},{lat} -> {p:?}"
            );
        }
    }

    #[test]
    fn poles_keep_latitude() {
        assert!(close(LonLat::from_vec(DVec3::Y).lat, 90.0, 1e-12));
        assert!(close(LonLat::from_vec(-DVec3::Y).lat, -90.0, 1e-12));
    }

    #[test]
    fn arc_stays_on_sphere_and_respects_step() {
        let a = LonLat::new(-120.0, 10.0).to_unit();
        let b = LonLat::new(60.0, 70.0).to_unit();
        let step = 1f64.to_radians();
        let mut pts = great_circle_arc(a, b, step);
        assert_eq!(pts[0], a);
        pts.push(b);
        for w in pts.windows(2) {
            assert!(close(w[0].length(), 1.0, 1e-12));
            assert!(w[0].angle_between(w[1]) <= step + 1e-12);
        }
    }

    #[test]
    fn octant_area() {
        // One eighth of the sphere: 4π / 8 steradians.
        let ring = vec![
            LonLat::new(0.0, 0.0),
            LonLat::new(-90.0, 0.0),
            LonLat::new(0.0, 90.0),
        ];
        let units: Vec<DVec3> = ring.iter().map(|p| p.to_unit()).collect();
        assert!(close(signed_area(&units).abs(), PI / 2.0, 1e-12));
        let poly = Polygon::from_ring(ring).unwrap();
        let expected = 4.0 * PI * EARTH_RADIUS_M.powi(2) / 8.0;
        assert!(close(poly.area_m2(EARTH_RADIUS_M) / expected, 1.0, 1e-12));
    }

    #[test]
    fn winding_normalized_to_ccw() {
        let cw = vec![
            LonLat::new(0.0, 0.0),
            LonLat::new(0.0, 10.0),
            LonLat::new(10.0, 10.0),
            LonLat::new(10.0, 0.0),
        ];
        let units: Vec<DVec3> = cw.iter().map(|p| p.to_unit()).collect();
        assert!(signed_area(&units) < 0.0);
        let poly = Polygon::from_ring(cw.clone()).unwrap();
        let units: Vec<DVec3> = poly.exterior.iter().map(|p| p.to_unit()).collect();
        assert!(signed_area(&units) > 0.0);
        // Already CCW stays as drawn.
        let ccw: Vec<LonLat> = cw.into_iter().rev().collect();
        assert_eq!(Polygon::from_ring(ccw.clone()).unwrap().exterior, ccw);
    }

    #[test]
    fn polygon_across_antimeridian_and_around_pole() {
        let across = Polygon::from_ring(vec![
            LonLat::new(170.0, -5.0),
            LonLat::new(-170.0, -5.0),
            LonLat::new(-170.0, 5.0),
            LonLat::new(170.0, 5.0),
        ])
        .unwrap();
        // About 20° x 10° near the equator, not the complementary region.
        let area_sr = across.area_m2(1.0);
        assert!(area_sr > 0.05 && area_sr < 0.07, "{area_sr}");

        let cap = Polygon::from_ring(
            (0..6)
                .map(|i| LonLat::new(-180.0 + 60.0 * i as f64, 80.0))
                .collect(),
        )
        .unwrap();
        assert!(cap.area_m2(1.0) < 0.2);
    }

    #[test]
    fn rejects_degenerate_and_drops_closing_point() {
        assert!(Polygon::from_ring(vec![LonLat::new(0.0, 0.0), LonLat::new(1.0, 1.0)]).is_none());
        let closed = vec![
            LonLat::new(0.0, 0.0),
            LonLat::new(10.0, 0.0),
            LonLat::new(10.0, 10.0),
            LonLat::new(0.0, 0.0),
        ];
        assert_eq!(Polygon::from_ring(closed).unwrap().exterior.len(), 3);
    }
}
