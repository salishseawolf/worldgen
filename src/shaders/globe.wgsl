struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) unit: vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>) -> VertexOut {
    var out: VertexOut;
    out.clip = globals.view_proj * vec4<f32>(position * globals.params.z, 1.0);
    out.unit = position;
    return out;
}

// Coverage of a line `dist` degrees away, about 1.5 px wide.
// `deriv` is how many degrees one pixel spans.
fn line_at(dist: f32, deriv: f32) -> f32 {
    return 1.0 - smoothstep(0.5 * deriv, 1.5 * deriv, dist);
}

// Coverage of grid lines every `spacing` degrees.
fn grid_line(coord: f32, spacing: f32, deriv: f32) -> f32 {
    return line_at(abs(fract(coord / spacing + 0.5) - 0.5) * spacing, deriv);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let p = normalize(in.unit);
    let lat = degrees(asin(clamp(p.y, -1.0, 1.0)));
    let lon = degrees(atan2(-p.z, p.x));
    // Same longitude shifted 180°, so its seam sits at the prime meridian.
    // Taking the smaller derivative hides the jump at the antimeridian.
    let lon_shifted = degrees(atan2(p.z, -p.x));
    let d_lon = min(fwidth(lon), fwidth(lon_shifted));
    let d_lat = fwidth(lat);

    // Meridians converge at the poles; fade them out before they smear.
    let pole_fade = 1.0 - smoothstep(80.0, 88.0, abs(lat));
    let minor = max(grid_line(lat, 10.0, d_lat), grid_line(lon, 10.0, d_lon) * pole_fade);
    // Equator and prime meridian.
    let major = max(line_at(abs(lat), d_lat), line_at(abs(lon), d_lon) * pole_fade);

    let base = vec3<f32>(0.25, 0.55, 0.85) * (0.12 + 0.88 * lambert(p));
    var color = mix(base, vec3<f32>(0.85, 0.9, 1.0), minor * 0.35);
    color = mix(color, vec3<f32>(1.0, 1.0, 1.0), major * 0.8);
    return vec4<f32>(color, 1.0);
}
