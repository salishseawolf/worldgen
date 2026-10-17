// Per-cell surface data, shared by the globe and flat-map shaders.
// Prepended after common.wgsl.

@group(1) @binding(0) var<storage, read> cell_plate: array<u32>;
@group(1) @binding(1) var<storage, read> plate_color: array<vec4<f32>>;
// Grid cell id for each direction, baked by cells.wgsl (see CUBE_FACES).
@group(1) @binding(2) var cell_ids: texture_2d_array<u32>;

const NO_CELL: u32 = 0xffffffffu;

fn surface_albedo(cell: u32) -> vec3<f32> {
    if (globals.flags.x & LAYER_PLATES) != 0u {
        return plate_color[cell_plate[cell]].rgb;
    }
    return vec3<f32>(0.25, 0.55, 0.85);
}

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
