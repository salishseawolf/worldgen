//! Color ramps for the surface color modes.
//!
//! This is the one source for both the shader, which receives
//! [`gpu_table`] as a storage buffer, and the on-page color key, which
//! draws the same stops as CSS gradients.

/// A color (sRGB, 0..1) at a value along a ramp.
#[derive(Clone, Copy, Debug)]
pub struct Stop {
    pub value: f64,
    pub srgb: [f64; 3],
}

const fn stop(value: f64, r: f64, g: f64, b: f64) -> Stop {
    Stop {
        value,
        srgb: [r, g, b],
    }
}

/// Elevation (m): ocean in blues by depth, land in greys by height. Plain
/// grey on land, so no height reads as a biome. The jump at sea level keeps
/// coastlines crisp.
pub const ELEVATION: &[Stop] = &[
    stop(-8000.0, 0.03, 0.05, 0.16),
    stop(-5500.0, 0.07, 0.15, 0.36),
    stop(-3500.0, 0.13, 0.28, 0.55),
    stop(-1500.0, 0.26, 0.47, 0.73),
    stop(-200.0, 0.50, 0.70, 0.86),
    stop(0.0, 0.62, 0.80, 0.90),
    stop(0.5, 0.20, 0.20, 0.20),
    stop(1000.0, 0.42, 0.42, 0.42),
    stop(2500.0, 0.64, 0.64, 0.64),
    stop(5000.0, 0.88, 0.88, 0.88),
    stop(8000.0, 1.0, 1.0, 1.0),
];

/// Seafloor age (Myr): the usual red-to-violet of seafloor age maps.
pub const AGE: &[Stop] = &[
    stop(0.0, 0.85, 0.10, 0.10),
    stop(25.0, 0.95, 0.50, 0.10),
    stop(50.0, 0.95, 0.85, 0.20),
    stop(80.0, 0.30, 0.75, 0.30),
    stop(110.0, 0.20, 0.70, 0.80),
    stop(150.0, 0.20, 0.35, 0.85),
    stop(200.0, 0.50, 0.25, 0.75),
];
/// Continents have no seafloor age.
pub const NO_AGE: [f64; 3] = [0.55, 0.55, 0.55];

/// Continental crust thickness (km), pale to dark brown.
pub const THICKNESS: &[Stop] = &[
    stop(20.0, 0.92, 0.85, 0.68),
    stop(45.0, 0.70, 0.52, 0.33),
    stop(70.0, 0.40, 0.24, 0.12),
];
/// Dynamic topography (m): blue where sinking mantle drags the surface
/// down, red where rising mantle lifts it.
pub const MANTLE: &[Stop] = &[
    stop(-800.0, 0.13, 0.30, 0.70),
    stop(-400.0, 0.45, 0.60, 0.85),
    stop(0.0, 0.80, 0.80, 0.80),
    stop(400.0, 0.92, 0.58, 0.42),
    stop(800.0, 0.75, 0.15, 0.10),
];

/// Crust composition (felsic fraction): dark basalt to pale pink granite.
pub const COMPOSITION: &[Stop] = &[
    stop(0.0, 0.20, 0.22, 0.27),
    stop(0.5, 0.62, 0.55, 0.60),
    stop(1.0, 0.96, 0.75, 0.75),
];

/// Lithosphere thickness (km): pale and thin to dark and deep-rooted.
pub const LITHOSPHERE: &[Stop] = &[
    stop(0.0, 0.95, 0.92, 0.80),
    stop(50.0, 0.80, 0.70, 0.45),
    stop(100.0, 0.55, 0.45, 0.30),
    stop(160.0, 0.35, 0.28, 0.25),
    stop(250.0, 0.15, 0.12, 0.15),
];

/// Column density minus the asthenosphere's (kg/m³): purple floats,
/// orange is heavy enough to sink. Purple and orange keep clear of the
/// mantle view's blue and red.
pub const BUOYANCY: &[Stop] = &[
    stop(-500.0, 0.35, 0.15, 0.55),
    stop(-200.0, 0.65, 0.55, 0.80),
    stop(0.0, 0.80, 0.80, 0.80),
    stop(50.0, 0.95, 0.70, 0.40),
    stop(100.0, 0.85, 0.40, 0.05),
];

/// Surface rock colors, in the order of `SurfaceRock::ALL`: muted, after
/// geologic-map convention (granite pink), and no biome greens.
pub const ROCK: [[f64; 3]; 7] = [
    [0.62, 0.70, 0.78],
    [0.25, 0.27, 0.32],
    [0.55, 0.48, 0.62],
    [0.93, 0.62, 0.62],
    [0.55, 0.38, 0.45],
    [0.85, 0.72, 0.42],
    [0.45, 0.22, 0.18],
];

/// Crustal age provinces, in the order of `CrustProvince::ALL`: oceanic,
/// then Archean (deep red-brown) to Phanerozoic (pale), older darker.
pub const PROVINCE: [[f64; 3]; 4] = [
    [0.25, 0.38, 0.55],
    [0.62, 0.25, 0.28],
    [0.85, 0.58, 0.35],
    [0.90, 0.86, 0.58],
];

/// Oceanic crust in the crust view.
pub const OCEANIC_CRUST: [f64; 3] = [0.20, 0.36, 0.62];

/// Entries per ramp in the GPU table; shorter ramps repeat their last stop.
pub const RAMP_LEN: usize = 12;

/// Layout of the GPU table, in entries: must match surface.wgsl.
pub const ELEVATION_AT: usize = 0;
pub const AGE_AT: usize = RAMP_LEN;
pub const THICKNESS_AT: usize = 2 * RAMP_LEN;
pub const MANTLE_AT: usize = 3 * RAMP_LEN;
pub const COMPOSITION_AT: usize = 4 * RAMP_LEN;
pub const LITHOSPHERE_AT: usize = 5 * RAMP_LEN;
pub const BUOYANCY_AT: usize = 6 * RAMP_LEN;
pub const NO_AGE_AT: usize = 7 * RAMP_LEN;
pub const OCEANIC_CRUST_AT: usize = 7 * RAMP_LEN + 1;
pub const ROCK_AT: usize = 7 * RAMP_LEN + 2;
pub const PROVINCE_AT: usize = ROCK_AT + ROCK.len();

/// All ramps and fixed colors as `[r, g, b, value]` in linear RGB, for the
/// shader.
pub fn gpu_table() -> Vec<[f32; 4]> {
    let entry = |srgb: [f64; 3], value: f64| {
        let [r, g, b] = srgb.map(srgb_to_linear);
        [r as f32, g as f32, b as f32, value as f32]
    };
    let mut table = vec![[0.0; 4]; PROVINCE_AT + PROVINCE.len()];
    for (at, ramp) in [
        (ELEVATION_AT, ELEVATION),
        (AGE_AT, AGE),
        (THICKNESS_AT, THICKNESS),
        (MANTLE_AT, MANTLE),
        (COMPOSITION_AT, COMPOSITION),
        (LITHOSPHERE_AT, LITHOSPHERE),
        (BUOYANCY_AT, BUOYANCY),
    ] {
        assert!(ramp.len() <= RAMP_LEN);
        let last = ramp[ramp.len() - 1];
        for i in 0..RAMP_LEN {
            let s = ramp.get(i).copied().unwrap_or(last);
            table[at + i] = entry(s.srgb, s.value);
        }
    }
    table[NO_AGE_AT] = entry(NO_AGE, 0.0);
    table[OCEANIC_CRUST_AT] = entry(OCEANIC_CRUST, 0.0);
    for (i, &color) in ROCK.iter().enumerate() {
        table[ROCK_AT + i] = entry(color, 0.0);
    }
    for (i, &color) in PROVINCE.iter().enumerate() {
        table[PROVINCE_AT + i] = entry(color, 0.0);
    }
    table
}

pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// CSS hex color for an sRGB color.
pub fn css(srgb: [f64; 3]) -> String {
    let [r, g, b] = srgb.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    format!("#{r:02x}{g:02x}{b:02x}")
}

/// CSS hex color for a linear RGB color.
pub fn css_linear(linear: [f32; 3]) -> String {
    css(linear.map(|c| linear_to_srgb(c as f64)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ramps_ascend() {
        for ramp in [
            ELEVATION,
            AGE,
            THICKNESS,
            MANTLE,
            COMPOSITION,
            LITHOSPHERE,
            BUOYANCY,
        ] {
            assert!(ramp.windows(2).all(|w| w[0].value < w[1].value));
        }
    }

    #[test]
    fn table_layout() {
        let t = gpu_table();
        assert_eq!(t.len(), PROVINCE_AT + PROVINCE.len());
        assert_eq!(t[ELEVATION_AT][3], -8000.0);
        // Short ramps pad with their last stop.
        assert_eq!(t[THICKNESS_AT + RAMP_LEN - 1], t[THICKNESS_AT + 2]);
    }

    #[test]
    fn shader_offsets_match() {
        let wgsl = include_str!("shaders/surface.wgsl");
        for (name, at) in [
            ("ELEVATION_AT", ELEVATION_AT),
            ("AGE_AT", AGE_AT),
            ("THICKNESS_AT", THICKNESS_AT),
            ("MANTLE_AT", MANTLE_AT),
            ("COMPOSITION_AT", COMPOSITION_AT),
            ("LITHOSPHERE_AT", LITHOSPHERE_AT),
            ("BUOYANCY_AT", BUOYANCY_AT),
            ("ROCK_AT", ROCK_AT),
            ("PROVINCE_AT", PROVINCE_AT),
            ("NO_AGE_AT", NO_AGE_AT),
            ("OCEANIC_CRUST_AT", OCEANIC_CRUST_AT),
            ("RAMP_LEN", RAMP_LEN),
        ] {
            let decl = format!("const {name}: u32 = {at}u;");
            assert!(wgsl.contains(&decl), "surface.wgsl lacks `{decl}`");
        }
    }

    #[test]
    fn land_is_grey() {
        for s in ELEVATION.iter().filter(|s| s.value > 0.0) {
            let [r, g, b] = s.srgb;
            assert!(r == g && g == b, "{s:?}");
        }
    }

    #[test]
    fn css_round_trips_linear() {
        assert_eq!(css([1.0, 0.0, 0.5]), "#ff0080");
        assert_eq!(css_linear([srgb_to_linear(0.5) as f32; 3]), "#808080");
    }
}
