struct Camera {
    view_proj: mat4x4<f32>,
    model: mat4x4<f32>,
    light_dir: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world = camera.model * vec4<f32>(in.position, 1.0);
    out.clip = camera.view_proj * world;
    // Model is rotation-only, so it transforms normals directly.
    out.normal = (camera.model * vec4<f32>(in.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let diffuse = max(dot(n, normalize(camera.light_dir.xyz)), 0.0);
    let base = vec3<f32>(0.25, 0.55, 0.85);
    let color = base * (0.08 + 0.92 * diffuse);
    return vec4<f32>(color, 1.0);
}
