// Point-to-cell lookup through the baked cube map. Needs `cell_ids`
// (texture_2d_array<u32>) declared by the including shader, and cube.wgsl.

const NO_CELL: u32 = 0xffffffffu;

// Grid cell id in direction d, from the baked cube map.
fn cell_toward(d: vec3<f32>) -> u32 {
    var face = 0u;
    var best = -2.0;
    for (var f = 0u; f < 6u; f++) {
        let s = dot(d, CUBE_FACES[f][2]);
        if s > best {
            best = s;
            face = f;
        }
    }
    let axes = CUBE_FACES[face];
    let uv = vec2<f32>(dot(d, axes[0]), dot(d, axes[1])) / best;
    let size = i32(textureDimensions(cell_ids).x);
    let texel = clamp(
        vec2<i32>(vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5) * f32(size)),
        vec2<i32>(0),
        vec2<i32>(size - 1),
    );
    let id = textureLoad(cell_ids, texel, face, 0).r;
    if id != NO_CELL {
        return id;
    }
    // A pixel can slip between two cells' triangles; borrow a neighbor's.
    return textureLoad(cell_ids, min(texel + vec2<i32>(1, 0), vec2<i32>(size - 1)), face, 0).r;
}
