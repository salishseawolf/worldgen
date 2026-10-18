//! What the viewer has chosen to see: the surface color mode and the
//! overlay layers. Shared by the renderer, the color key and the shaders
//! (common.wgsl repeats these values).

/// Bits of the overlay layer mask.
pub const LAYER_BOUNDARIES: u32 = 2;
pub const LAYER_GRATICULE: u32 = 4;
pub const LAYER_MOTION: u32 = 8;
pub const LAYER_MANTLE: u32 = 16;
pub const LAYER_PROVINCES: u32 = 32;
pub const LAYER_HOTSPOTS: u32 = 64;
pub const LAYER_TRACKS: u32 = 128;
pub const LAYER_CRATONS: u32 = 256;
pub const LAYER_FORCES: u32 = 512;

/// What colors the surface.
pub const COLOR_PLATES: u32 = 0;
pub const COLOR_ELEVATION: u32 = 1;
pub const COLOR_AGE: u32 = 2;
pub const COLOR_CRUST: u32 = 3;
pub const COLOR_MANTLE: u32 = 4;
pub const COLOR_ROCK: u32 = 5;
pub const COLOR_COMPOSITION: u32 = 6;
pub const COLOR_LITHOSPHERE: u32 = 7;
pub const COLOR_BUOYANCY: u32 = 8;
pub const COLOR_PROVINCE: u32 = 9;
