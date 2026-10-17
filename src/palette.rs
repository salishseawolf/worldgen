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
/// Oceanic crust in the crust view.
pub const OCEANIC_CRUST: [f64; 3] = [0.20, 0.36, 0.62];

/// Entries per ramp in the GPU table; shorter ramps repeat their last stop.
pub const RAMP_LEN: usize = 12;

/// Layout of the GPU table, in entries: must match surface.wgsl.
pub const ELEVATION_AT: usize = 0;
pub const AGE_AT: usize = RAMP_LEN;
pub const THICKNESS_AT: usize = 2 * RAMP_LEN;
pub const NO_AGE_AT: usize = 3 * RAMP_LEN;
pub const OCEANIC_CRUST_AT: usize = 3 * RAMP_LEN + 1;

/// All ramps and fixed colors as `[r, g, b, value]` in linear RGB, for the
/// shader.
pub fn gpu_table() -> Vec<[f32; 4]> {
    let entry = |srgb: [f64; 3], value: f64| {
        let [r, g, b] = srgb.map(srgb_to_linear);
        [r as f32, g as f32, b as f32, value as f32]
    };
    let mut table = vec![[0.0; 4]; OCEANIC_CRUST_AT + 1];
    for (at, ramp) in [
        (ELEVATION_AT, ELEVATION),
        (AGE_AT, AGE),
        (THICKNESS_AT, THICKNESS),
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
        for ramp in [ELEVATION, AGE, THICKNESS] {
            assert!(ramp.windows(2).all(|w| w[0].value < w[1].value));
        }
    }

    #[test]
    fn table_layout() {
        let t = gpu_table();
        assert_eq!(t.len(), OCEANIC_CRUST_AT + 1);
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
