//! The column inspector: an SVG drawing of the lithosphere column under the
//! cursor, its layers to scale with their thickness and density.

use std::fmt::Write;

use crate::crust::{
    Column, RHO_ASTHENOSPHERE, RHO_SEDIMENT, crust_density, mantle_lithosphere_density,
};
use crate::palette::{self, Stop};

/// The bar shows this much depth (km); deeper roots run off the bottom.
const DEPTH_KM: f64 = 260.0;
const WIDTH: f64 = 236.0;
const BAR_X: f64 = 6.0;
const BAR_W: f64 = 34.0;
const TOP: f64 = 6.0;
const BAR_H: f64 = 150.0;
/// Thin layers (sediment, oceanic crust) still get this many pixels.
const MIN_BAND_PX: f64 = 3.0;

const SEDIMENT_COLOR: [f64; 3] = [0.85, 0.72, 0.42];
const MANTLE_COLOR: [f64; 3] = [0.42, 0.34, 0.30];
const DEPLETED_COLOR: [f64; 3] = [0.62, 0.55, 0.45];
const ASTHENOSPHERE_COLOR: [f64; 3] = [0.72, 0.30, 0.18];

pub fn svg(col: &Column) -> String {
    let px = BAR_H / DEPTH_KM;
    // Layers top to bottom: (label, thickness km, density, color).
    let layers = [
        (
            "Sediment",
            col.sediment_km,
            RHO_SEDIMENT,
            palette::css(SEDIMENT_COLOR),
        ),
        (
            "Crust",
            col.crust_km,
            crust_density(col.felsic),
            palette::css(ramp(palette::COMPOSITION, col.felsic)),
        ),
        (
            "Mantle lithosphere",
            col.mantle_km,
            mantle_lithosphere_density(col.depletion),
            palette::css(mix(MANTLE_COLOR, DEPLETED_COLOR, col.depletion)),
        ),
    ];

    let height = TOP + BAR_H + 4.0;
    let mut out = format!(
        "<svg viewBox=\"0 0 {WIDTH} {height}\" width=\"{WIDTH}\" height=\"{height}\" \
         role=\"img\" aria-label=\"Lithosphere column\">"
    );
    // Asthenosphere fills the bar; layers paint over it from the top.
    let _ = write!(
        out,
        "<rect x=\"{BAR_X}\" y=\"{TOP}\" width=\"{BAR_W}\" height=\"{BAR_H}\" fill=\"{}\"/>",
        palette::css(ASTHENOSPHERE_COLOR)
    );
    let mut y = TOP;
    let mut rows = Vec::new();
    for (label, km, density, color) in layers {
        if km <= 0.0 {
            continue;
        }
        let h = (km * px).max(MIN_BAND_PX).min(TOP + BAR_H - y);
        let _ = write!(
            out,
            "<rect x=\"{BAR_X}\" y=\"{y:.1}\" width=\"{BAR_W}\" height=\"{h:.1}\" fill=\"{color}\"/>"
        );
        y += h;
        let detail = if label == "Crust" {
            format!(", {:.0}% felsic", 100.0 * col.felsic)
        } else {
            String::new()
        };
        rows.push((
            color,
            format!(
                "{label} {} · {:.2} g/cm³{detail}",
                km_text(km),
                density / 1000.0
            ),
        ));
    }
    rows.push((
        palette::css(ASTHENOSPHERE_COLOR),
        format!("Asthenosphere · {:.2} g/cm³", RHO_ASTHENOSPHERE / 1000.0),
    ));

    // Legend rows beside the bar.
    let text_x = BAR_X + BAR_W + 10.0;
    for (i, (color, text)) in rows.iter().enumerate() {
        let ty = TOP + 12.0 + 17.0 * i as f64;
        let _ = write!(
            out,
            "<rect x=\"{text_x}\" y=\"{:.1}\" width=\"8\" height=\"8\" fill=\"{color}\"/>\
             <text x=\"{:.1}\" y=\"{ty:.1}\">{text}</text>",
            ty - 8.0,
            text_x + 12.0
        );
    }
    let total = col.thickness_km();
    let _ = write!(
        out,
        "<text x=\"{text_x}\" y=\"{:.1}\" class=\"dim\">Lithosphere {} total</text>\
         <text x=\"{text_x}\" y=\"{:.1}\" class=\"dim\">Bar depth {DEPTH_KM:.0} km</text>",
        TOP + 12.0 + 17.0 * rows.len() as f64 + 4.0,
        km_text(total),
        TOP + BAR_H,
    );
    out += "</svg>";
    out
}

/// `0.4 km` for thin layers, `38 km` otherwise.
fn km_text(km: f64) -> String {
    if km < 10.0 {
        format!("{km:.1} km")
    } else {
        format!("{km:.0} km")
    }
}

fn mix(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// sRGB color at `value` along a palette ramp (as the shader does it).
fn ramp(stops: &[Stop], value: f64) -> [f64; 3] {
    let mut c = stops[0].srgb;
    for w in stops.windows(2) {
        if value > w[0].value {
            let t = ((value - w[0].value) / (w[1].value - w[0].value)).clamp(0.0, 1.0);
            c = mix(w[0].srgb, w[1].srgb, t);
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_each_layer_with_density() {
        let s = svg(&Column::oceanic(50.0, 0.4));
        assert!(s.starts_with("<svg") && s.ends_with("</svg>"));
        assert!(s.contains("Sediment 0.4 km · 2.20 g/cm³"));
        assert!(s.contains("Crust 7.0 km · 2.90 g/cm³, 0% felsic"));
        assert!(s.contains("Mantle lithosphere"));
        assert!(s.contains("Asthenosphere · 3.30 g/cm³"));
    }

    #[test]
    fn skips_missing_layers_and_clips_deep_roots() {
        let s = svg(&crate::crust::CrustProvince::Proterozoic.column().unwrap());
        assert!(!s.contains("Sediment"));
        // A root deeper than the bar stops at its bottom.
        let deep = Column {
            mantle_km: 400.0,
            ..crate::crust::CrustProvince::Proterozoic.column().unwrap()
        };
        let s = svg(&deep);
        let max_y = TOP + BAR_H + 0.1;
        for part in s.split("y=\"").skip(1) {
            let y: f64 = part.split('"').next().unwrap().parse().unwrap();
            assert!(y <= max_y, "{y}");
        }
    }

    #[test]
    fn ramp_matches_palette_ends() {
        assert_eq!(
            ramp(palette::COMPOSITION, -1.0),
            palette::COMPOSITION[0].srgb
        );
        assert_eq!(
            ramp(palette::COMPOSITION, 2.0),
            palette::COMPOSITION[2].srgb
        );
    }
}
