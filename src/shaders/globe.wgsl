struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) unit: vec3<f32>,
    @location(1) @interpolate(flat) cell: u32,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) cell: u32) -> VertexOut {
    var out: VertexOut;
    out.clip = globals.view_proj * vec4<f32>(position * globals.params.z, 1.0);
    out.unit = position;
    out.cell = cell;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = normalize(in.unit);
    let base = surface_albedo(in.cell) * (0.12 + 0.88 * lambert(p));
    return vec4<f32>(with_graticule(base, p), 1.0);
}
