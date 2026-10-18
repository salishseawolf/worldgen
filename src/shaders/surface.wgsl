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
    // Crust composition, 0 mafic to 1 felsic.
    felsic: f32,
    // Whole lithosphere thickness (km).
    lithosphere_km: f32,
    // Column density minus the asthenosphere's (kg/m³).
    density_excess: f32,
    // Index into the surface rock colors.
    rock: u32,
    // Index into the age province colors.
    province: u32,
};

@group(1) @binding(0) var<storage, read> cells: array<CellData>;
@group(1) @binding(1) var<storage, read> plate_color: array<vec4<f32>>;
// Grid cell id for each direction, baked by cells.wgsl (see CUBE_FACES).
@group(1) @binding(2) var cell_ids: texture_2d_array<u32>;


// Color ramps from palette.rs: entries are (linear rgb, value). Offsets
// must match the *_AT constants there.
@group(1) @binding(3) var<storage, read> palettes: array<vec4<f32>>;
const RAMP_LEN: u32 = 12u;
const ELEVATION_AT: u32 = 0u;
const AGE_AT: u32 = 12u;
const THICKNESS_AT: u32 = 24u;
const MANTLE_AT: u32 = 36u;
const COMPOSITION_AT: u32 = 48u;
const LITHOSPHERE_AT: u32 = 60u;
const BUOYANCY_AT: u32 = 72u;
const NO_AGE_AT: u32 = 84u;
const OCEANIC_CRUST_AT: u32 = 85u;
const ROCK_AT: u32 = 86u;
const PROVINCE_AT: u32 = 93u;

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
        case COLOR_ROCK: {
            return palettes[ROCK_AT + d.rock].rgb;
        }
        case COLOR_COMPOSITION: {
            return ramp(COMPOSITION_AT, d.felsic);
        }
        case COLOR_LITHOSPHERE: {
            return ramp(LITHOSPHERE_AT, d.lithosphere_km);
        }
        case COLOR_BUOYANCY: {
            return ramp(BUOYANCY_AT, d.density_excess);
        }
        case COLOR_PROVINCE: {
            return palettes[PROVINCE_AT + d.province].rgb;
        }
        default: {
            return plate_color[d.plate].rgb;
        }
    }
}
