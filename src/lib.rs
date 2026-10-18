// Only the pure math and state modules build natively (for tests); the
// renderer and browser glue target wasm.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod camera;
mod column_view;
mod cross_section;
mod crust;
mod geo;
mod grid;
mod legend;
mod mantle;
mod measure;
mod mesh;
mod palette;
mod perf;
mod plates;
mod projection;
#[cfg(target_arch = "wasm32")]
mod renderer;
mod rng;
mod scene;
mod tectonics;
mod view;
#[cfg(target_arch = "wasm32")]
mod web;
mod world;
