//! Browsers validate WGSL only at runtime, so check every shader here.
//! Each is prefixed with common.wgsl, the same way the renderer builds them.

const COMMON: &str = include_str!("../src/shaders/common.wgsl");

fn validate(name: &str, source: &str) {
    let full = format!("{COMMON}{source}");
    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&full)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&full)));
}

#[test]
fn globe() {
    validate("globe.wgsl", include_str!("../src/shaders/globe.wgsl"));
}

#[test]
fn lines() {
    validate("lines.wgsl", include_str!("../src/shaders/lines.wgsl"));
}

#[test]
fn columns() {
    validate("columns.wgsl", include_str!("../src/shaders/columns.wgsl"));
}
