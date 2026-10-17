// Per-cell surface data, shared by the globe and flat-map shaders.
// Prepended after common.wgsl.

// Must match CellData in renderer.rs.
struct CellData {
    plate: u32,
    elevation_m: f32,
    // Seafloor age (Myr), or negative for continental crust.
    age_myr: f32,
    thickness_km: f32,
    // Elevation from vertical mantle flow (m).
    dynamic_m: f32,
};

@group(1) @binding(0) var<storage, read> cells: array<CellData>;
@group(1) @binding(1) var<storage, read> plate_color: array<vec4<f32>>;
// Grid cell id for each direction, baked by cells.wgsl (see CUBE_FACES).
@group(1) @binding(2) var cell_ids: texture_2d_array<u32>;

const NO_CELL: u32 = 0xffffffffu;

// Color ramps from palette.rs: entries are (linear rgb, value). Offsets
// must match the *_AT constants there.
@group(1) @binding(3) var<storage, read> palettes: array<vec4<f32>>;
const RAMP_LEN: u32 = 12u;
const ELEVATION_AT: u32 = 0u;
const AGE_AT: u32 = 12u;
const THICKNESS_AT: u32 = 24u;
const MANTLE_AT: u32 = 36u;
const NO_AGE_AT: u32 = 48u;
const OCEANIC_CRUST_AT: u32 = 49u;

// Color of `value` along the ramp starting at entry `at`, interpolating
// between stops and clamping at the ends.
fn ramp(at: u32, value: f32) -> vec3<f32> {
    var c = palettes[at].rgb;
    for (var i = 1u; i < RAMP_LEN; i++) {
        let a = palettes[at + i - 1u];
        let b = palettes[at + i];
        if value > a.w {
            // Padding repeats the last stop, so guard the zero-width span.
            let t = select(1.0, clamp((value - a.w) / (b.w - a.w), 0.0, 1.0), b.w > a.w);
            c = mix(a.rgb, b.rgb, t);
        }
    }
    return c;
}

fn surface_albedo(cell: u32) -> vec3<f32> {
    let d = cells[cell];
    switch globals.flags.z {
        case COLOR_ELEVATION: {
            return ramp(ELEVATION_AT, d.elevation_m);
        }
        case COLOR_AGE: {
            if d.age_myr < 0.0 {
                return palettes[NO_AGE_AT].rgb;
            }
            return ramp(AGE_AT, d.age_myr);
        }
        case COLOR_CRUST: {
            if d.age_myr >= 0.0 {
                return palettes[OCEANIC_CRUST_AT].rgb;
            }
            return ramp(THICKNESS_AT, d.thickness_km);
        }
        case COLOR_MANTLE: {
            return ramp(MANTLE_AT, d.dynamic_m);
        }
        default: {
            return plate_color[d.plate].rgb;
        }
    }
}

// Grid cell id in direction d, from the baked cube map.
fn cell_toward(d: vec3<f32>) -> u32 {
    var face = 0u;
    var best = -2.0;
    for (var f = 0u; f < 6u; f++) {
        let s = dot(d, CUBE_FACES[f][2]);
        if s > best {
            best = s;
            face = f;
        }
    }
    let axes = CUBE_FACES[face];
    let uv = vec2<f32>(dot(d, axes[0]), dot(d, axes[1])) / best;
    let size = i32(textureDimensions(cell_ids).x);
    let texel = clamp(
        vec2<i32>(vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5) * f32(size)),
        vec2<i32>(0),
        vec2<i32>(size - 1),
    );
    let id = textureLoad(cell_ids, texel, face, 0).r;
    if id != NO_CELL {
        return id;
    }
    // A pixel can slip between two cells' triangles; borrow a neighbor's.
    return textureLoad(cell_ids, min(texel + vec2<i32>(1, 0), vec2<i32>(size - 1)), face, 0).r;
}
