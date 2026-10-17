//! Browsers validate WGSL only at runtime, so check every shader here.
//! Each is prefixed the same way the renderer builds them.

const COMMON: &str = include_str!("../src/shaders/common.wgsl");
const SURFACE: &str = include_str!("../src/shaders/surface.wgsl");

fn validate(name: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
}

#[test]
fn globe() {
    let src = include_str!("../src/shaders/globe.wgsl");
    validate("globe.wgsl", &format!("{COMMON}{SURFACE}{src}"));
}

#[test]
fn map() {
    let src = include_str!("../src/shaders/map.wgsl");
    validate("map.wgsl", &format!("{COMMON}{SURFACE}{src}"));
}

#[test]
fn cells() {
    let src = include_str!("../src/shaders/cells.wgsl");
    validate("cells.wgsl", &format!("{COMMON}{src}"));
}

#[test]
fn lines() {
    let src = include_str!("../src/shaders/lines.wgsl");
    validate("lines.wgsl", &format!("{COMMON}{src}"));
}

#[test]
fn columns() {
    let src = include_str!("../src/shaders/columns.wgsl");
    validate("columns.wgsl", &format!("{COMMON}{src}"));
}
