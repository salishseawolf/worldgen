// Flat maps: a full-screen triangle where each pixel inverse-projects to a
// point on the planet and looks up its cell.

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    let ndc = vec2<f32>(f32(vi & 1u) * 4.0 - 1.0, f32(vi >> 1u) * 4.0 - 1.0);
    var out: VertexOut;
    out.clip = vec4<f32>(ndc, 0.9, 1.0);
    out.ndc = ndc;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let hit = unproject(map_point(in.ndc));
    let p = normalize(hit.xyz);
    let color = with_graticule(surface_albedo(cell_toward(p)), p);
    // After the derivatives in with_graticule, which need every pixel.
    if hit.w == 0.0 {
        discard;
    }
    return vec4<f32>(color, 1.0);
}
