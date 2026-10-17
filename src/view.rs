//! What the viewer has chosen to see: the surface color mode and the
//! overlay layers. Shared by the renderer, the color key and the shaders
//! (common.wgsl repeats these values).

/// Bits of the overlay layer mask.
pub const LAYER_BOUNDARIES: u32 = 2;
pub const LAYER_GRATICULE: u32 = 4;
pub const LAYER_MOTION: u32 = 8;
pub const LAYER_MANTLE: u32 = 16;

/// What colors the surface.
pub const COLOR_PLATES: u32 = 0;
pub const COLOR_ELEVATION: u32 = 1;
pub const COLOR_AGE: u32 = 2;
pub const COLOR_CRUST: u32 = 3;
