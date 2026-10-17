// Only the pure math and state modules build natively (for tests); the
// renderer and browser glue target wasm.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod camera;
mod draw;
mod geo;
mod mesh;
#[cfg(target_arch = "wasm32")]
mod renderer;
mod scene;
#[cfg(target_arch = "wasm32")]
mod web;
