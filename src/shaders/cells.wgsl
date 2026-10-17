// Bakes the grid cell id seen in every direction into one cube face (the
// instance index picks the face), as seen from the planet's center.

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(flat) cell: u32,
};

@vertex
fn vs_main(
    @builtin(instance_index) face: u32,
    @location(0) position: vec3<f32>,
    @location(1) cell: u32,
) -> VertexOut {
    let axes = CUBE_FACES[face];
    let w = dot(position, axes[2]);
    var out: VertexOut;
    out.clip = vec4<f32>(dot(position, axes[0]), dot(position, axes[1]), 0.5 * w, w);
    out.cell = cell;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) u32 {
    return in.cell;
}
