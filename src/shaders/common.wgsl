// Prepended to every shader.
struct Globals {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    // x, y: viewport size in physical pixels; z: planet radius (m); w: line width (px)
    params: vec4<f32>,
    // x: bitmask of visible LAYER_* flags; y: projection (PROJ_*);
    // z: what colors the surface (COLOR_*)
    flags: vec4<u32>,
    // Flat maps: center lon, center lat, Albers parallels 1 and 2 (radians)
    projection: vec4<f32>,
    // Flat maps: map point at the screen center (xy); map units to the top
    // edge; aspect
    map: vec4<f32>,
};

// Must match the constants in view.rs.
const LAYER_BOUNDARIES: u32 = 2u;
const LAYER_GRATICULE: u32 = 4u;
const COLOR_PLATES: u32 = 0u;
const COLOR_ELEVATION: u32 = 1u;
const COLOR_AGE: u32 = 2u;
const COLOR_CRUST: u32 = 3u;
const COLOR_MANTLE: u32 = 4u;
const COLOR_ROCK: u32 = 5u;
const COLOR_COMPOSITION: u32 = 6u;
const COLOR_LITHOSPHERE: u32 = 7u;
const COLOR_BUOYANCY: u32 = 8u;
const COLOR_PROVINCE: u32 = 9u;

// Must match Projection::id in projection.rs, which also holds the
// reference (and tested) copy of the projection math below.
const PROJ_GLOBE: u32 = 0u;
const PROJ_EQUIRECTANGULAR: u32 = 1u;
const PROJ_MERCATOR: u32 = 2u;
const PROJ_ALBERS: u32 = 3u;
const PROJ_AZIMUTHAL: u32 = 4u;

const PI: f32 = 3.14159265358979;
const HALF_PI: f32 = 1.57079632679490;
// Mercator stops at 89°, where y is about ±4.74.
const MERCATOR_MAX_LAT: f32 = 1.55334303427495;
const MERCATOR_MAX_Y: f32 = 4.74134876036470;
// Twin azimuthal hemispheres: disc centers sit this far either side of x = 0.
const DUAL_OFFSET: f32 = 1.63362817986669;

@group(0) @binding(0) var<uniform> globals: Globals;

fn lambert(n: vec3<f32>) -> f32 {
    return max(dot(normalize(n), globals.light_dir.xyz), 0.0);
}

fn is_map() -> bool {
    return globals.flags.y != PROJ_GLOBE;
}

// Longitude and latitude (radians) of unit vector p.
fn lonlat_of(p: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(-p.z, p.x), asin(clamp(p.y, -1.0, 1.0)));
}

fn unit_at(lon: f32, lat: f32) -> vec3<f32> {
    return vec3<f32>(cos(lat) * cos(lon), sin(lat), -cos(lat) * sin(lon));
}

fn wrap_pi(a: f32) -> f32 {
    return a - 2.0 * PI * floor((a + PI) / (2.0 * PI));
}

// Albers constants n, C, rho0 on the unit sphere.
fn albers() -> vec3<f32> {
    let p1 = globals.projection.z;
    let p2 = globals.projection.w;
    let n = (sin(p1) + sin(p2)) / 2.0;
    let c = cos(p1) * cos(p1) + 2.0 * n * sin(p1);
    let rho0 = sqrt(c - 2.0 * n * sin(globals.projection.y)) / n;
    return vec3<f32>(n, c, rho0);
}

// Azimuthal equidistant map point of unit vector p around `center` (lon,
// lat), north up.
fn azimuthal(p: vec3<f32>, center: vec2<f32>) -> vec2<f32> {
    let c = unit_at(center.x, center.y);
    let east = vec3<f32>(-sin(center.x), 0.0, -cos(center.x));
    let north = cross(c, east);
    let local = vec2<f32>(dot(p, east), dot(p, north));
    let len = length(local);
    let distance = acos(clamp(dot(p, c), -1.0, 1.0));
    return select(vec2<f32>(0.0), local / len * distance, len > 1e-7);
}

fn azimuthal_inverse(m: vec2<f32>, center: vec2<f32>) -> vec3<f32> {
    let r = length(m);
    let c = unit_at(center.x, center.y);
    let east = vec3<f32>(-sin(center.x), 0.0, -cos(center.x));
    let north = cross(c, east);
    let dir = select(vec2<f32>(0.0), m / r, r > 1e-7);
    return c * cos(r) + (east * dir.x + north * dir.y) * sin(r);
}

fn antipode(lonlat: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(lonlat.x + PI, -lonlat.y);
}

// Map coordinates (planet radii) of unit vector p.
fn project(p: vec3<f32>) -> vec2<f32> {
    let center = globals.projection.xy;
    let ll = lonlat_of(p);
    let dlon = wrap_pi(ll.x - center.x);
    switch globals.flags.y {
        case PROJ_MERCATOR: {
            let lat = clamp(ll.y, -MERCATOR_MAX_LAT, MERCATOR_MAX_LAT);
            return vec2<f32>(dlon, log(tan(PI / 4.0 + lat / 2.0)));
        }
        case PROJ_ALBERS: {
            let k = albers();
            let rho = sqrt(max(k.y - 2.0 * k.x * sin(ll.y), 0.0)) / k.x;
            let theta = k.x * dlon;
            return vec2<f32>(rho * sin(theta), k.z - rho * cos(theta));
        }
        case PROJ_AZIMUTHAL: {
            // Focus hemisphere on the left, the far one on the right.
            if dot(p, unit_at(center.x, center.y)) >= 0.0 {
                return azimuthal(p, center) - vec2<f32>(DUAL_OFFSET, 0.0);
            }
            return azimuthal(p, antipode(center)) + vec2<f32>(DUAL_OFFSET, 0.0);
        }
        default: {
            return vec2<f32>(dlon, ll.y);
        }
    }
}

// Unit vector shown at map point m (xyz), and whether m is on the map (w).
fn unproject(m: vec2<f32>) -> vec4<f32> {
    let center = globals.projection.xy;
    switch globals.flags.y {
        case PROJ_MERCATOR: {
            let ok = abs(m.x) <= PI && abs(m.y) <= MERCATOR_MAX_Y;
            return vec4<f32>(unit_at(center.x + m.x, atan(sinh(m.y))), select(0.0, 1.0, ok));
        }
        case PROJ_ALBERS: {
            let k = albers();
            let s = sign(k.x);
            let x = m.x * s;
            let y = (k.z - m.y) * s;
            let rho = length(vec2<f32>(x, y)) * s;
            let theta = atan2(x, y);
            let sin_lat = (k.y - (rho * k.x) * (rho * k.x)) / (2.0 * k.x);
            let ok = abs(theta) <= PI * abs(k.x) && abs(sin_lat) <= 1.0;
            let lat = asin(clamp(sin_lat, -1.0, 1.0));
            return vec4<f32>(unit_at(center.x + theta / k.x, lat), select(0.0, 1.0, ok));
        }
        case PROJ_AZIMUTHAL: {
            let left = m + vec2<f32>(DUAL_OFFSET, 0.0);
            let right = m - vec2<f32>(DUAL_OFFSET, 0.0);
            if length(left) <= HALF_PI {
                return vec4<f32>(azimuthal_inverse(left, center), 1.0);
            }
            return vec4<f32>(azimuthal_inverse(right, antipode(center)), select(0.0, 1.0, length(right) <= HALF_PI));
        }
        default: {
            let ok = abs(m.x) <= PI && abs(m.y) <= HALF_PI;
            return vec4<f32>(unit_at(center.x + m.x, m.y), select(0.0, 1.0, ok));
        }
    }
}

// True if the straight map segment from a to b would jump across the cut.
fn crosses_seam(a: vec3<f32>, b: vec3<f32>) -> bool {
    let center = globals.projection.xy;
    if globals.flags.y == PROJ_AZIMUTHAL {
        // Ends in different hemispheres sit on different discs.
        let c = unit_at(center.x, center.y);
        return (dot(a, c) >= 0.0) != (dot(b, c) >= 0.0);
    }
    let la = wrap_pi(lonlat_of(normalize(a)).x - center.x);
    let lb = wrap_pi(lonlat_of(normalize(b)).x - center.x);
    return abs(la - lb) > PI;
}

// Map point under a position in normalized device coordinates.
fn map_point(ndc: vec2<f32>) -> vec2<f32> {
    return globals.map.xy + ndc * vec2<f32>(globals.map.w, 1.0) * globals.map.z;
}

// Clip position of a world-space point (meters) in the current projection.
// Flat maps put everything at one depth; draw order sorts it out.
fn world_to_clip(world: vec3<f32>) -> vec4<f32> {
    if !is_map() {
        return globals.view_proj * vec4<f32>(world, 1.0);
    }
    let m = (project(normalize(world)) - globals.map.xy) / globals.map.z;
    return vec4<f32>(m.x / globals.map.w, m.y, 0.5, 1.0);
}

// Coverage of a line `dist` degrees away, about 1.5 px wide.
// `deriv` is how many degrees one pixel spans.
fn line_at(dist: f32, deriv: f32) -> f32 {
    return 1.0 - smoothstep(0.5 * deriv, 1.5 * deriv, dist);
}

// Coverage of grid lines every `spacing` degrees.
fn grid_line(coord: f32, spacing: f32, deriv: f32) -> f32 {
    return line_at(abs(fract(coord / spacing + 0.5) - 0.5) * spacing, deriv);
}

// Graticule drawn over `color` at unit vector p, if that layer is on.
// Fragment stage only (uses derivatives).
fn with_graticule(color: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let lat = degrees(asin(clamp(p.y, -1.0, 1.0)));
    let lon = degrees(atan2(-p.z, p.x));
    // Same longitude shifted 180°, so its seam sits at the prime meridian.
    // Taking the smaller derivative hides the jump at the antimeridian.
    let lon_shifted = degrees(atan2(p.z, -p.x));
    let d_lon = min(fwidth(lon), fwidth(lon_shifted));
    let d_lat = fwidth(lat);

    // Meridians converge at the poles; fade them out before they smear.
    let pole_fade = 1.0 - smoothstep(80.0, 88.0, abs(lat));
    let minor = max(grid_line(lat, 10.0, d_lat), grid_line(lon, 10.0, d_lon) * pole_fade);
    // Equator and prime meridian.
    let major = max(line_at(abs(lat), d_lat), line_at(abs(lon), d_lon) * pole_fade);

    // Blends are in linear light, where small amounts already look strong.
    var out = mix(color, vec3<f32>(0.85, 0.9, 1.0), minor * 0.12);
    out = mix(out, vec3<f32>(1.0, 1.0, 1.0), major * 0.4);
    return select(color, out, (globals.flags.x & LAYER_GRATICULE) != 0u);
}
