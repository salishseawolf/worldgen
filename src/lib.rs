// Only the mesh tests build natively; everything else targets the browser.
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]

mod camera;
mod mesh;
#[cfg(target_arch = "wasm32")]
mod renderer;
#[cfg(target_arch = "wasm32")]
mod web;
