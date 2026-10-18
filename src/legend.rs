//! The color key panel: HTML describing the current color mode and the
//! overlays that are switched on.

use std::fmt::Write;

use crate::crust::{CrustProvince, SurfaceRock};
use crate::palette::{self, Stop};
use crate::scene::{
    ARROW_DEG_PER_CM_YEAR, CRATON_COLOR, FORCE_ARROW_DEG_PER_1000_KM, MANTLE_FLOW_COLOR,
    PLATE_MOTION_COLOR, PROVINCE_COLOR, RIDGE_PUSH_COLOR, SLAB_PULL_COLOR, TRACK_TICK_MYR,
    boundary_color, hotspot_color,
};
use crate::tectonics::BoundaryKind;
use crate::view::{
    COLOR_AGE, COLOR_BUOYANCY, COLOR_COMPOSITION, COLOR_CRUST, COLOR_ELEVATION, COLOR_LITHOSPHERE,
    COLOR_MANTLE, COLOR_PLATES, COLOR_PROVINCE, COLOR_ROCK, LAYER_BOUNDARIES, LAYER_CRATONS,
    LAYER_FORCES, LAYER_GRATICULE, LAYER_HOTSPOTS, LAYER_MANTLE, LAYER_MOTION, LAYER_PROVINCES,
    LAYER_TRACKS,
};
use crate::world::PlumeKind;

/// What the key shows for one plate.
pub struct PlateKey {
    /// Linear RGB, as rendered.
    pub color: [f32; 3],
    /// Mean surface speed over the plate (cm/yr).
    pub speed_cm_year: f64,
}

pub fn html(color_mode: u32, layers: u32, plates: &[PlateKey]) -> String {
    let mut out = String::new();
    match color_mode {
        COLOR_ELEVATION => {
            out += "<h2>Elevation</h2>";
            out += &ramp(
                palette::ELEVATION,
                -8000.0,
                8000.0,
                &[-8000.0, -4000.0, 0.0, 4000.0, 8000.0],
                true,
                |v| format!("{} km", signed(v / 1000.0)),
            );
            out += "<p>Sea level at 0. Ocean depth in blues, land height in greys.</p>";
        }
        COLOR_AGE => {
            out += "<h2>Seafloor age</h2>";
            out += &ramp(
                palette::AGE,
                0.0,
                200.0,
                &[0.0, 50.0, 100.0, 150.0, 200.0],
                false,
                |v| format!("{v:.0}"),
            );
            out += "<p>Million years since the seafloor formed at a ridge.</p>";
            out += &swatch(&palette::css(palette::NO_AGE), "Continental crust");
        }
        COLOR_CRUST => {
            out += "<h2>Crust</h2>";
            out += &swatch(&palette::css(palette::OCEANIC_CRUST), "Oceanic crust");
            out += "<p>Continental crust by thickness:</p>";
            out += &ramp(
                palette::THICKNESS,
                20.0,
                70.0,
                &[20.0, 45.0, 70.0],
                true,
                |v| format!("{v:.0} km"),
            );
        }
        COLOR_MANTLE => {
            out += "<h2>Mantle (rising / sinking)</h2>";
            out += &ramp(
                palette::MANTLE,
                -800.0,
                800.0,
                &[-800.0, -400.0, 0.0, 400.0, 800.0],
                false,
                |v| format!("{} m", signed(v)),
            );
            out += "<p>How far vertical mantle flow lifts the surface (red, over \
                    rising mantle) or drags it down (blue, over sinking mantle and \
                    slabs under trenches).</p>";
        }
        COLOR_ROCK => {
            out += "<h2>Surface rock</h2>";
            for (rock, color) in SurfaceRock::ALL.iter().zip(palette::ROCK) {
                out += &swatch(
                    &palette::css(color),
                    &format!("{}: {}", rock.name(), rock.origin()),
                );
            }
        }
        COLOR_PROVINCE => {
            out += "<h2>Crustal age province</h2>";
            for (p, color) in CrustProvince::ALL.iter().zip(palette::PROVINCE) {
                let note = match p.column() {
                    Some(c) => format!(
                        "{}: lithosphere {:.0} km, crust {:.0} km",
                        p.name(),
                        c.thickness_km(),
                        c.crust_km
                    ),
                    None => format!("{}: thin, young lithosphere", p.name()),
                };
                out += &swatch(&palette::css(color), &note);
            }
            out += "<p>Continents grew outward from ancient cores; crust at \
                    active boundaries is reworked into young lithosphere.</p>";
        }
        COLOR_COMPOSITION => {
            out += "<h2>Crust composition</h2>";
            out += &ramp(
                palette::COMPOSITION,
                0.0,
                1.0,
                &[0.0, 0.5, 1.0],
                false,
                |v| match v {
                    v if v < 0.25 => "basaltic".into(),
                    v if v < 0.75 => "andesitic".into(),
                    _ => "granitic".into(),
                },
            );
            out += "<p>Felsic fraction, from mafic (about 50% silica, 2.90 g/cm³) \
                    to felsic (about 70%, 2.70 g/cm³). Ocean crust is basaltic; \
                    continents average andesitic.</p>";
        }
        COLOR_LITHOSPHERE => {
            out += "<h2>Lithosphere thickness</h2>";
            out += &ramp(
                palette::LITHOSPHERE,
                0.0,
                250.0,
                &[0.0, 100.0, 250.0],
                true,
                |v| format!("{v:.0} km"),
            );
            out += "<p>Crust plus cold mantle lithosphere. Thin at ridges, \
                    thickening as seafloor ages; deepest under continents.</p>";
        }
        COLOR_BUOYANCY => {
            out += "<h2>Buoyancy</h2>";
            out += &ramp(
                palette::BUOYANCY,
                -500.0,
                100.0,
                &[-500.0, -200.0, 0.0, 100.0],
                true,
                |v| format!("{} kg/m³", signed(v)),
            );
            out += "<p>Column density minus the mantle below it. Purple floats; \
                    orange is heavy enough to sink, as seafloor becomes after \
                    about 12 Myr.</p>";
        }
        COLOR_PLATES => {
            out += "<h2>Plates</h2><p>Plate number and mean speed (cm/yr).</p>";
            out += "<div class=\"plates\">";
            for (i, p) in plates.iter().enumerate() {
                let _ = write!(
                    out,
                    "<span><i style=\"background:{}\"></i>{} · {:.1}</span>",
                    palette::css_linear(p.color),
                    i + 1,
                    p.speed_cm_year
                );
            }
            out += "</div>";
        }
        _ => {}
    }

    if layers & LAYER_BOUNDARIES != 0 {
        out += "<h2>Plate boundaries</h2>";
        for (kind, label) in [
            (BoundaryKind::Ridge, "Spreading ridge"),
            (BoundaryKind::Rift, "Continental rift"),
            (BoundaryKind::Subduction, "Subduction zone"),
            (BoundaryKind::Collision, "Collision"),
            (BoundaryKind::Transform, "Transform fault"),
        ] {
            out += &line(&rgba_css(boundary_color(kind)), label);
        }
    }
    if layers & LAYER_FORCES != 0 {
        out += "<h2>Plate forces</h2>";
        out += &arrow(
            &rgba_css(SLAB_PULL_COLOR),
            "Slab pull: a sinking slab pulls its plate",
        );
        out += &arrow(
            &rgba_css(RIDGE_PUSH_COLOR),
            "Ridge push: about a tenth as strong",
        );
        let _ = write!(
            out,
            "<p>One arrow per uninterrupted stretch of boundary between two plates, \
             showing the stretch's summed force: 1,000 km of trench sinking 100 Myr \
             seafloor spans {FORCE_ARROW_DEG_PER_1000_KM:.0}° of arc.</p>"
        );
    }
    let arrow_note = format!(
        "<p>Length grows with speed: 10 cm/yr spans {:.1}° of arc.</p>",
        10.0 * ARROW_DEG_PER_CM_YEAR
    );
    if layers & LAYER_MOTION != 0 {
        out += "<h2>Plate motion</h2>";
        out += &arrow(
            &rgba_css(PLATE_MOTION_COLOR),
            "Surface velocity of the plate",
        );
        out += &arrow_note;
    }
    if layers & LAYER_MANTLE != 0 {
        out += "<h2>Mantle flow</h2>";
        out += &arrow(&rgba_css(MANTLE_FLOW_COLOR), "Convection under the plates");
        out += &arrow_note;
    }
    if layers & LAYER_CRATONS != 0 {
        out += "<h2>Craton outlines</h2>";
        out += &line(&rgba_css(CRATON_COLOR), "Edge of Archean craton crust");
    }
    if layers & LAYER_PROVINCES != 0 {
        out += "<h2>Deep mantle provinces</h2>";
        out += &line(
            &rgba_css(PROVINCE_COLOR),
            "Hot regions at the base of the mantle",
        );
        out += "<p>Strong plumes rise from their edges.</p>";
    }
    // Hotspot kinds share colors between the rings and the tracks.
    let kinds = [
        (PlumeKind::Primary, 14, "Primary: deep, from province edges"),
        (
            PlumeKind::Secondary,
            11,
            "Secondary: from domes over provinces",
        ),
        (
            PlumeKind::Tertiary,
            8,
            "Tertiary: shallow, near ridges and rifts",
        ),
    ];
    if layers & LAYER_HOTSPOTS != 0 {
        out += "<h2>Hotspots</h2>";
        for (kind, size, label) in kinds {
            out += &ring(&rgba_css(hotspot_color(kind)), size, label);
        }
    }
    if layers & LAYER_TRACKS != 0 {
        out += "<h2>Hotspot tracks</h2>";
        for (kind, _, label) in kinds {
            out += &line(
                &rgba_css(hotspot_color(kind)),
                label.split(':').next().unwrap_or(label),
            );
        }
        let _ = write!(
            out,
            "<p>Volcanoes carried away from the plume; ticks every {TRACK_TICK_MYR} Myr of age.</p>"
        );
    }
    if layers & LAYER_GRATICULE != 0 {
        out += "<h2>Graticule</h2>";
        out += "<p>Lines every 10°; the equator and prime meridian are brighter.</p>";
    }
    out
}

/// A horizontal gradient bar over `lo..hi` with labeled ticks. With
/// `open_ends`, values can run past the bar (the map shows them in the end
/// colors), so the end ticks read "≤ lo" and "≥ hi".
fn ramp(
    stops: &[Stop],
    lo: f64,
    hi: f64,
    ticks: &[f64],
    open_ends: bool,
    label: impl Fn(f64) -> String,
) -> String {
    let at = |v: f64| 100.0 * (v - lo) / (hi - lo);
    let gradient: Vec<String> = stops
        .iter()
        .map(|s| format!("{} {:.2}%", palette::css(s.srgb), at(s.value)))
        .collect();
    let mut out = format!(
        "<div class=\"bar\" style=\"background:linear-gradient(to right, {})\"></div>",
        gradient.join(", ")
    );
    out += "<div class=\"ticks\">";
    for &t in ticks {
        let prefix = match t {
            t if open_ends && t <= lo => "\u{2264}",
            t if open_ends && t >= hi => "\u{2265}",
            _ => "",
        };
        let _ = write!(
            out,
            "<span style=\"left:{:.2}%\">{prefix}{}</span>",
            at(t),
            label(t)
        );
    }
    out += "</div>";
    out
}

fn swatch(css: &str, label: &str) -> String {
    format!("<div class=\"row\"><i class=\"swatch\" style=\"background:{css}\"></i>{label}</div>")
}

fn line(css: &str, label: &str) -> String {
    format!("<div class=\"row\"><i class=\"line\" style=\"background:{css}\"></i>{label}</div>")
}

fn ring(css: &str, size_px: u32, label: &str) -> String {
    format!(
        "<div class=\"row\"><i class=\"ring\" style=\"width:{size_px}px;height:{size_px}px;\
         border-color:{css}\"></i>{label}</div>"
    )
}

fn arrow(css: &str, label: &str) -> String {
    format!("<div class=\"row\"><b class=\"arrow\" style=\"color:{css}\">→</b>{label}</div>")
}

fn rgba_css(c: [f32; 4]) -> String {
    palette::css_linear([c[0], c[1], c[2]])
}

/// `+1.5` or `−4` with a real minus sign, trimming `.0`.
fn signed(v: f64) -> String {
    let body = if v.fract() == 0.0 {
        format!("{:.0}", v.abs())
    } else {
        format!("{:.1}", v.abs())
    };
    match v {
        v if v < 0.0 => format!("\u{2212}{body}"),
        v if v > 0.0 => format!("+{body}"),
        _ => body,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shows_mode_and_only_active_overlays() {
        let h = html(COLOR_ELEVATION, LAYER_BOUNDARIES, &[]);
        assert!(h.contains("<h2>Elevation</h2>"));
        assert!(h.contains("Subduction zone"));
        assert!(!h.contains("Plate motion"));
        // Elevation can run past the bar's ends.
        assert!(h.contains("\u{2264}\u{2212}8 km") && h.contains("\u{2265}+8 km"));
    }

    #[test]
    fn province_mode_and_cratons() {
        let h = html(COLOR_PROVINCE, LAYER_CRATONS, &[]);
        for p in CrustProvince::ALL {
            assert!(h.contains(p.name()));
        }
        assert!(h.contains("Craton outlines"));
    }

    #[test]
    fn lithosphere_modes() {
        let rock = html(COLOR_ROCK, 0, &[]);
        for r in SurfaceRock::ALL {
            assert!(rock.contains(r.name()), "{}", r.name());
        }
        assert!(html(COLOR_COMPOSITION, 0, &[]).contains("granitic"));
        assert!(html(COLOR_LITHOSPHERE, 0, &[]).contains("\u{2265}250 km"));
        let b = html(COLOR_BUOYANCY, 0, &[]);
        assert!(b.contains("\u{2264}\u{2212}500 kg/m³") && b.contains("12 Myr"));
    }

    #[test]
    fn deep_earth_layers() {
        let h = html(
            COLOR_MANTLE,
            LAYER_PROVINCES | LAYER_HOTSPOTS | LAYER_TRACKS,
            &[],
        );
        assert!(h.contains("Mantle (rising / sinking)") && h.contains("+800 m"));
        assert!(h.contains("Deep mantle provinces") && h.contains("Primary: deep"));
        assert!(h.contains("ticks every 10 Myr"));
    }

    #[test]
    fn plates_list_colors_and_speeds() {
        let plates = [
            PlateKey {
                color: [1.0, 0.0, 0.0],
                speed_cm_year: 4.25,
            },
            PlateKey {
                color: [0.0, 0.0, 1.0],
                speed_cm_year: 1.0,
            },
        ];
        let h = html(COLOR_PLATES, 0, &plates);
        assert!(h.contains("#ff0000") && h.contains("1 · 4.2") && h.contains("2 · 1.0"));
    }

    #[test]
    fn ramp_positions_stops_by_value() {
        let h = ramp(palette::THICKNESS, 20.0, 70.0, &[45.0], true, |v| {
            format!("{v}")
        });
        assert!(h.contains("0.00%") && h.contains("50.00%") && h.contains("100.00%"));
        assert!(h.contains("left:50.00%"));
    }
}
