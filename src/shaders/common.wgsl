// Prepended to every shader.
struct Globals {
    view_proj: mat4x4<f32>,
    light_dir: vec4<f32>,
    // x, y: viewport size in physical pixels; z: planet radius (m); w: line width (px)
    params: vec4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

fn lambert(n: vec3<f32>) -> f32 {
    return max(dot(normalize(n), globals.light_dir.xyz), 0.0);
}
