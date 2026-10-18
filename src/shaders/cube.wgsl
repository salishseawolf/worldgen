// Prepended after common.wgsl wherever the cell-id cube map is used.

// Cube-map faces used to store per-pixel cell ids: columns are the face's
// right, up and forward axes. cells.wgsl renders with these and
// surface.wgsl reads back with them.
const CUBE_FACES = array<mat3x3<f32>, 6>(
    mat3x3<f32>(vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0)),
    mat3x3<f32>(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0)),
    mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, -1.0), vec3<f32>(0.0, 1.0, 0.0)),
    mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, -1.0, 0.0)),
    mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0)),
    mat3x3<f32>(vec3<f32>(-1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, -1.0)),
);
