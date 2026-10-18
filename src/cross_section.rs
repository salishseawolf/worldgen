//! A vertical slice of the world along a measured great-circle line: an
//! exaggerated elevation profile above, and the lithosphere layers to true
//! scale below, with markers where the line crosses plate boundaries,
//! craton edges and hotspot volcanoes.

use std::fmt::Write;

use glam::DVec3;

use crate::crust::{Column, CrustProvince, crust_density};
use crate::geo::LonLat;
use crate::grid::Grid;
use crate::palette;
use crate::scene::{CRATON_COLOR, boundary_color, hotspot_color};
use crate::tectonics::{Boundary, BoundaryKind, Edge};
use crate::world::World;

/// Samples along the line.
pub const SAMPLES: usize = 240;
/// Drawing size (SVG units) and layout.
const WIDTH: f64 = 700.0;
const LEFT: f64 = 46.0;
const RIGHT: f64 = 8.0;
const PROFILE_TOP: f64 = 6.0;
const PROFILE_H: f64 = 70.0;
const SECTION_TOP: f64 = 86.0;
const SECTION_H: f64 = 140.0;
const HEIGHT: f64 = SECTION_TOP + SECTION_H + 18.0;
/// The elevation strip spans at least this far either side of sea level
/// (m), more if the line goes higher or deeper.
const MIN_ELEVATION_RANGE_M: f64 = 500.0;
/// Ids of the clip paths that keep drawing inside each panel.
const STRIP_CLIP: &str = "strip-clip";
const SECTION_CLIP: &str = "section-clip";
/// The section runs from this far above sea level down to `DEPTH_KM` below
/// (km), each column hanging from its true elevation.
const ABOVE_SEA_KM: f64 = 10.0;
const DEPTH_KM: f64 = 300.0;

const WATER: [f64; 3] = [0.20, 0.36, 0.62];
const GROUND: [f64; 3] = [0.55, 0.53, 0.50];
const SEDIMENT: [f64; 3] = [0.85, 0.72, 0.42];
const MANTLE: [f64; 3] = [0.42, 0.34, 0.30];
const DEPLETED: [f64; 3] = [0.62, 0.55, 0.45];
const ASTHENOSPHERE: [f64; 3] = [0.72, 0.30, 0.18];

/// The world at one point along the line.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    /// Distance from the line's start (km).
    pub km: f64,
    pub elevation_m: f64,
    pub column: Column,
}

/// Something the line crosses, drawn as a colored mark at `km`.
pub struct Marker {
    pub km: f64,
    /// Linear RGB, as for map lines.
    pub color: [f32; 4],
    pub label: String,
    /// At a subduction zone, the slab to sketch under it.
    pub slab: Option<Slab>,
}

/// A schematic slab: the sinking plate's lithosphere dipping into the
/// mantle from the trench. Schematic because nothing yet tracks slabs
/// through time; its dip follows the rule of thumb that old, cold slabs
/// sink more steeply.
#[derive(Clone, Copy, Debug)]
pub struct Slab {
    /// Toward increasing distance along the line (true) or back (false).
    pub dips_forward: bool,
    pub dip_deg: f64,
    pub thickness_km: f64,
}

/// Slab dip (degrees) from young to old (100 Myr and up) seafloor.
const SLAB_DIP_DEG: (f64, f64) = (30.0, 60.0);

/// Samples the world along the great circle from `a` to `b`, with markers
/// for what the line crosses. With `km`, samples just that stretch (distance
/// from `a`), so zooming in shows finer detail.
pub fn sample(
    world: &World,
    grid: &Grid,
    edges: &[Edge],
    (a, b): (LonLat, LonLat),
    radius_km: f64,
    km: Option<(f64, f64)>,
) -> (Vec<Sample>, Vec<Marker>) {
    let (pa, pb) = (a.to_unit(), b.to_unit());
    let angle = pa.angle_between(pb);
    let total_km = angle * radius_km;
    let (t0, t1) = km.map_or((0.0, 1.0), |(k0, k1)| {
        if total_km < 1e-9 {
            (0.0, 1.0)
        } else {
            (
                (k0 / total_km).clamp(0.0, 1.0),
                (k1 / total_km).clamp(0.0, 1.0),
            )
        }
    });
    let point = |t: f64| -> DVec3 {
        if angle < 1e-9 {
            pa
        } else {
            (pa * ((1.0 - t) * angle).sin() + pb * (t * angle).sin()) / angle.sin()
        }
    };
    let crust = &world.crust;
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut markers = Vec::new();
    let mut last: Option<(u32, DVec3)> = None;
    for i in 0..SAMPLES {
        let t = t0 + (t1 - t0) * i as f64 / (SAMPLES - 1) as f64;
        let p = point(t);
        let c = grid.cell_at(LonLat::from_vec(p));
        let km = t * angle * radius_km;
        samples.push(Sample {
            km,
            elevation_m: crust.elevation_m[c as usize],
            column: crust.column[c as usize],
        });
        if let Some(mark) = world.hotspot_marks[c as usize]
            && last.is_none_or(|(prev, _)| world.hotspot_marks[prev as usize].is_none())
        {
            let h = world.hotspots[mark.hotspot as usize];
            markers.push(Marker {
                km,
                color: hotspot_color(h.kind),
                label: format!(
                    "Hotspot volcano, {:.0} Myr ({})",
                    mark.age_myr,
                    h.kind.name()
                ),
                slab: None,
            });
        }
        if let Some((prev, prev_p)) = last {
            let plates = &world.plates.cell_plate;
            if plates[prev as usize] != plates[c as usize] {
                let mid = (prev_p + p).normalize();
                if let Some(b) = nearest_boundary(world, edges, mid) {
                    let slab = (b.kind == BoundaryKind::Subduction).then(|| {
                        let sinking = b.subducting as usize;
                        let young_old = (crust.age_myr[sinking] / 100.0).clamp(0.0, 1.0);
                        Slab {
                            // The slab dips under the overriding plate: forward
                            // if the sinking plate is the one behind us.
                            dips_forward: plates[sinking] == plates[prev as usize],
                            dip_deg: SLAB_DIP_DEG.0 + (SLAB_DIP_DEG.1 - SLAB_DIP_DEG.0) * young_old,
                            thickness_km: crust.column[sinking].thickness_km(),
                        }
                    });
                    markers.push(Marker {
                        km,
                        color: boundary_color(b.kind),
                        label: boundary_label(b.kind).into(),
                        slab,
                    });
                }
            }
            let craton = |c: u32| crust.province[c as usize] == CrustProvince::Archean;
            if craton(prev) != craton(c) {
                markers.push(Marker {
                    km,
                    color: CRATON_COLOR,
                    label: "Craton edge".into(),
                    slab: None,
                });
            }
        }
        last = Some((c, p));
    }
    (samples, markers)
}

/// The point `km` along the great circle from `a` to `b`.
pub fn point_at((a, b): (LonLat, LonLat), km: f64, radius_km: f64) -> LonLat {
    let (pa, pb) = (a.to_unit(), b.to_unit());
    let angle = pa.angle_between(pb);
    if angle < 1e-9 {
        return a;
    }
    let t = (km / radius_km / angle).clamp(0.0, 1.0);
    LonLat::from_vec((pa * ((1.0 - t) * angle).sin() + pb * (t * angle).sin()) / angle.sin())
}

/// The sample nearest `km` along the line.
pub fn nearest_sample(samples: &[Sample], km: f64) -> Option<&Sample> {
    samples
        .iter()
        .min_by(|a, b| (a.km - km).abs().total_cmp(&(b.km - km).abs()))
}

/// What is at `depth_km` below sea level (negative above) in a sample's
/// column, which hangs from its surface.
pub fn layer_at(sample: &Sample, depth_km: f64) -> &'static str {
    let surface = -sample.elevation_m / 1000.0;
    let col = &sample.column;
    if depth_km < surface.min(0.0) {
        return "air";
    }
    if depth_km < surface {
        return "water";
    }
    let below = depth_km - surface;
    [
        (col.sediment_km, "sediment"),
        (col.crust_km, "crust"),
        (col.mantle_km, "mantle lithosphere"),
    ]
    .iter()
    .scan(0.0, |top, &(km, name)| {
        *top += km;
        Some((*top, name))
    })
    .find(|&(bottom, _)| below < bottom)
    .map_or("asthenosphere", |(_, name)| name)
}

fn nearest_boundary<'w>(world: &'w World, edges: &[Edge], p: DVec3) -> Option<&'w Boundary> {
    world.boundaries.iter().max_by(|x, y| {
        let dx = edges[x.edge as usize].mid.dot(p);
        let dy = edges[y.edge as usize].mid.dot(p);
        dx.total_cmp(&dy)
    })
}

fn boundary_label(kind: BoundaryKind) -> &'static str {
    match kind {
        BoundaryKind::Ridge => "Spreading ridge",
        BoundaryKind::Rift => "Continental rift",
        BoundaryKind::Subduction => "Subduction zone",
        BoundaryKind::Collision => "Collision",
        BoundaryKind::Transform => "Transform fault",
    }
}

/// User zoom: the ranges picked by dragging, or None for the default view.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Zoom {
    /// Distance along the line (km).
    pub km: Option<(f64, f64)>,
    /// Elevation strip range, low to high (m).
    pub elevation_m: Option<(f64, f64)>,
    /// Section depth below sea level, shallow to deep (km; negative is
    /// above sea level).
    pub depth_km: Option<(f64, f64)>,
}

impl Zoom {
    pub fn is_zoomed(&self) -> bool {
        *self != Zoom::default()
    }
}

/// Smallest ranges a drag can zoom to: distance (km), elevation (m) and
/// depth (km).
const MIN_SPAN_KM: f64 = 1.0;
const MIN_SPAN_M: f64 = 50.0;
const MIN_SPAN_DEPTH_KM: f64 = 0.5;

/// Which part of the chart a point is in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Panel {
    Elevation,
    Section,
}

/// The ranges a drawing actually shows, and the mapping between them and
/// SVG coordinates (for turning a dragged box back into ranges).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub km: (f64, f64),
    pub elevation_m: (f64, f64),
    pub depth_km: (f64, f64),
}

impl Frame {
    /// SVG drawing width (the drawing scales to the panel's width).
    pub const WIDTH: f64 = WIDTH;

    fn plot_w() -> f64 {
        WIDTH - LEFT - RIGHT
    }

    fn x(&self, km: f64) -> f64 {
        LEFT + (km - self.km.0) / (self.km.1 - self.km.0) * Self::plot_w()
    }

    fn elevation_y(&self, m: f64) -> f64 {
        let (lo, hi) = self.elevation_m;
        PROFILE_TOP + (hi - m) / (hi - lo) * PROFILE_H
    }

    fn depth_y(&self, km: f64) -> f64 {
        let (top, bottom) = self.depth_km;
        SECTION_TOP + (km - top) / (bottom - top) * SECTION_H
    }

    /// The plot's left and right edges (SVG units).
    pub fn plot_x() -> (f64, f64) {
        (LEFT, LEFT + Self::plot_w())
    }

    /// A panel's top and bottom edges (SVG units).
    pub fn panel_y(panel: Panel) -> (f64, f64) {
        match panel {
            Panel::Elevation => (PROFILE_TOP, PROFILE_TOP + PROFILE_H),
            Panel::Section => (SECTION_TOP, SECTION_TOP + SECTION_H),
        }
    }

    /// The panel under an SVG point, if any.
    pub fn panel_at(&self, x: f64, y: f64) -> Option<Panel> {
        if !(LEFT..=LEFT + Self::plot_w()).contains(&x) {
            return None;
        }
        if (PROFILE_TOP..=PROFILE_TOP + PROFILE_H).contains(&y) {
            Some(Panel::Elevation)
        } else if (SECTION_TOP..=SECTION_TOP + SECTION_H).contains(&y) {
            Some(Panel::Section)
        } else {
            None
        }
    }

    /// Distance (km) at SVG x, clamped to the plot.
    pub fn km_at(&self, x: f64) -> f64 {
        let t = ((x - LEFT) / Self::plot_w()).clamp(0.0, 1.0);
        self.km.0 + t * (self.km.1 - self.km.0)
    }

    /// Elevation (m) or depth (km) at SVG y in `panel`, clamped to it.
    pub fn value_at(&self, panel: Panel, y: f64) -> f64 {
        match panel {
            Panel::Elevation => {
                let t = ((y - PROFILE_TOP) / PROFILE_H).clamp(0.0, 1.0);
                self.elevation_m.1 - t * (self.elevation_m.1 - self.elevation_m.0)
            }
            Panel::Section => {
                let t = ((y - SECTION_TOP) / SECTION_H).clamp(0.0, 1.0);
                self.depth_km.0 + t * (self.depth_km.1 - self.depth_km.0)
            }
        }
    }

    /// The zoom for a box dragged from `from` to `to` (SVG points) in
    /// `panel`, on top of `zoom`. A drag mostly along one axis zooms only
    /// that axis (`min_px` sets "mostly"); tiny drags change nothing.
    pub fn zoom_to(
        &self,
        zoom: Zoom,
        panel: Panel,
        from: (f64, f64),
        to: (f64, f64),
        min_px: f64,
    ) -> Zoom {
        let (dx, dy) = ((to.0 - from.0).abs(), (to.1 - from.1).abs());
        let mut out = zoom;
        if dx >= min_px {
            let (a, b) = (self.km_at(from.0), self.km_at(to.0));
            out.km = Some(span(a.min(b), a.max(b), MIN_SPAN_KM));
        }
        if dy >= min_px {
            let (a, b) = (self.value_at(panel, from.1), self.value_at(panel, to.1));
            let (lo, hi) = (a.min(b), a.max(b));
            match panel {
                Panel::Elevation => out.elevation_m = Some(span(lo, hi, MIN_SPAN_M)),
                Panel::Section => out.depth_km = Some(span(lo, hi, MIN_SPAN_DEPTH_KM)),
            }
        }
        out
    }
}

/// `lo..hi` widened about its middle to at least `min`.
fn span(lo: f64, hi: f64, min: f64) -> (f64, f64) {
    if hi - lo >= min {
        (lo, hi)
    } else {
        let mid = (lo + hi) / 2.0;
        (mid - min / 2.0, mid + min / 2.0)
    }
}

/// The frame to draw `samples` (already limited to the zoomed distance)
/// with `zoom`: zoomed ranges where set, defaults elsewhere.
pub fn frame(samples: &[Sample], zoom: &Zoom) -> Frame {
    let km = zoom.km.unwrap_or_else(|| {
        let first = samples.first().map_or(0.0, |s| s.km);
        let last = samples.last().map_or(0.0, |s| s.km);
        (first, last.max(first + 1e-6))
    });
    // By default the strip fits the visible line, tidily rounded, with sea
    // level always in view.
    let elevation_m = zoom.elevation_m.unwrap_or_else(|| {
        let (low, high) = samples.iter().fold((0.0f64, 0.0f64), |(lo, hi), s| {
            (lo.min(s.elevation_m), hi.max(s.elevation_m))
        });
        (-strip_range(-low), strip_range(high))
    });
    let depth_km = zoom.depth_km.unwrap_or((-ABOVE_SEA_KM, DEPTH_KM));
    Frame {
        km,
        elevation_m,
        depth_km,
    }
}

/// Draws `samples` (the zoomed stretch) with `markers` from the whole line:
/// slabs are sketched wherever they reach into view, marker lines only where
/// they fall within it.
pub fn svg(samples: &[Sample], markers: &[Marker], zoom: &Zoom) -> (String, Frame) {
    let f = frame(samples, zoom);
    let plot_w = Frame::plot_w();
    let css = palette::css;
    let right = LEFT + plot_w;
    // How much taller the strip draws elevation than the true-scale section.
    let exaggeration = (PROFILE_H / (f.elevation_m.1 - f.elevation_m.0))
        / (SECTION_H / ((f.depth_km.1 - f.depth_km.0) * 1000.0));

    let mut out = format!(
        "<svg viewBox=\"0 0 {WIDTH} {HEIGHT}\" width=\"100%\" role=\"img\" \
         aria-label=\"Cross-section along the measurement\">\
         <clipPath id=\"{STRIP_CLIP}\"><rect x=\"{LEFT}\" y=\"{PROFILE_TOP}\" \
         width=\"{plot_w}\" height=\"{PROFILE_H}\"/></clipPath>\
         <clipPath id=\"{SECTION_CLIP}\"><rect x=\"{LEFT}\" y=\"{SECTION_TOP}\" \
         width=\"{plot_w}\" height=\"{SECTION_H}\"/></clipPath>"
    );

    // Elevation strip: water up to sea level, ground over it. Drawn to the
    // frame's mapping and clipped, so zoomed ranges just crop.
    let strip_bottom = PROFILE_TOP + PROFILE_H;
    let sea = f.elevation_y(0.0);
    let _ = write!(out, "<g clip-path=\"url(#{STRIP_CLIP})\">");
    if sea < strip_bottom {
        let top = sea.max(PROFILE_TOP);
        let _ = write!(
            out,
            "<rect x=\"{LEFT}\" y=\"{top:.1}\" width=\"{plot_w}\" height=\"{:.1}\" fill=\"{}\"/>",
            strip_bottom - top,
            css(WATER)
        );
    }
    let below = strip_bottom + 1.0;
    let mut ground = format!(
        "{:.1},{below} ",
        f.x(samples.first().map_or(f.km.0, |s| s.km))
    );
    for s in samples {
        let _ = write!(
            ground,
            "{:.1},{:.1} ",
            f.x(s.km),
            f.elevation_y(s.elevation_m)
        );
    }
    let _ = write!(
        ground,
        "{:.1},{below}",
        f.x(samples.last().map_or(f.km.1, |s| s.km))
    );
    let _ = write!(
        out,
        "<polygon points=\"{ground}\" fill=\"{}\"/>",
        css(GROUND)
    );
    let _ = write!(
        out,
        "<line x1=\"{LEFT}\" y1=\"{sea:.1}\" x2=\"{right:.1}\" y2=\"{sea:.1}\" class=\"sea\"/></g>"
    );

    // Section: asthenosphere below sea level, then each sample's water and
    // layers hanging from its true elevation, all to true scale, clipped.
    let _ = write!(out, "<g clip-path=\"url(#{SECTION_CLIP})\">");
    let section_sea = f.depth_y(0.0).max(SECTION_TOP);
    let section_bottom = SECTION_TOP + SECTION_H;
    if section_sea < section_bottom {
        let _ = write!(
            out,
            "<rect x=\"{LEFT}\" y=\"{section_sea:.1}\" width=\"{plot_w}\" height=\"{:.1}\" fill=\"{}\"/>",
            section_bottom - section_sea,
            css(ASTHENOSPHERE)
        );
    }
    for (i, s) in samples.iter().enumerate() {
        // Each sample's stripe runs halfway to its neighbors.
        let prev = if i > 0 { samples[i - 1].km } else { s.km };
        let next = samples.get(i + 1).map_or(s.km, |n| n.km);
        let (x0, x1) = (f.x((prev + s.km) / 2.0), f.x((s.km + next) / 2.0));
        let col = &s.column;
        let surface = -s.elevation_m / 1000.0;
        let layers = [
            (surface.max(0.0), WATER),
            (col.sediment_km, SEDIMENT),
            (col.crust_km, crust_color(col.felsic)),
            (col.mantle_km, mix(MANTLE, DEPLETED, col.depletion)),
        ];
        // Water fills from sea level down to the seafloor; rock from the
        // surface down.
        let mut depth = surface.min(0.0);
        for (km, color) in layers {
            if km <= 0.0 {
                continue;
            }
            let (y0, y1) = (f.depth_y(depth), f.depth_y(depth + km));
            depth += km;
            if y1 < SECTION_TOP || y0 > section_bottom {
                continue;
            }
            let (y0, y1) = (y0.max(SECTION_TOP - 1.0), y1.min(section_bottom + 1.0));
            let _ = write!(
                out,
                "<rect x=\"{x0:.1}\" y=\"{y0:.1}\" width=\"{:.2}\" height=\"{:.2}\" fill=\"{}\"/>",
                x1 - x0 + 0.4,
                (y1 - y0).max(0.6),
                css(color)
            );
        }
    }
    if (SECTION_TOP..=section_bottom).contains(&f.depth_y(0.0)) {
        let y = f.depth_y(0.0);
        let _ = write!(
            out,
            "<line x1=\"{LEFT}\" y1=\"{y:.1}\" x2=\"{right:.1}\" y2=\"{y:.1}\" class=\"sea\"/>"
        );
    }

    // Schematic slabs: a band from the trench down into the mantle, under
    // the overriding side, as thick as the sinking lithosphere.
    for m in markers {
        let Some(slab) = m.slab else { continue };
        let dip = slab.dip_deg.to_radians();
        let sign = if slab.dips_forward { 1.0 } else { -1.0 };
        // Horizontal run to the bottom of the default section, and the
        // band's width measured along the line.
        let run_km = DEPTH_KM / dip.tan();
        let width_km = slab.thickness_km / dip.sin();
        let x_at = |km: f64| f.x(m.km + sign * km);
        let (top, bottom) = (f.depth_y(0.0), f.depth_y(DEPTH_KM));
        let _ = write!(
            out,
            "<g><title>Slab (schematic): dips about {:.0}°, {:.0} km thick</title>\
             <polygon points=\"{:.1},{top:.1} {:.1},{bottom:.1} {:.1},{bottom:.1} {:.1},{top:.1}\" \
             fill=\"{}\" class=\"slab\"/></g>",
            slab.dip_deg,
            slab.thickness_km,
            x_at(0.0),
            x_at(run_km),
            x_at(run_km - width_km),
            x_at(-width_km),
            css(MANTLE),
        );
    }
    out += "</g>";
    let _ = write!(
        out,
        "<text x=\"{:.1}\" y=\"{:.1}\" class=\"axis\" text-anchor=\"end\">×{} vertical</text>",
        right - 4.0,
        PROFILE_TOP + 10.0,
        group_or_decimal(exaggeration)
    );

    // Markers across both panels, where in view.
    for m in markers.iter().filter(|m| (f.km.0..=f.km.1).contains(&m.km)) {
        let mx = f.x(m.km);
        let color = palette::css_linear([m.color[0], m.color[1], m.color[2]]);
        let _ = write!(
            out,
            "<g><title>{}</title><line x1=\"{mx:.1}\" y1=\"{PROFILE_TOP}\" x2=\"{mx:.1}\" \
             y2=\"{section_bottom:.1}\" stroke=\"{color}\" class=\"marker\"/><path d=\"M{mx:.1},{:.1} \
             l5,-6 h-10 z\" fill=\"{color}\"/></g>",
            m.label,
            PROFILE_TOP + 6.0,
        );
    }

    // Axes, labeled for the frame's ranges.
    let label = |out: &mut String, y: f64, text: String| {
        let _ = write!(
            out,
            "<text x=\"{:.1}\" y=\"{:.1}\" class=\"axis\" text-anchor=\"end\">{text}</text>",
            LEFT - 4.0,
            y + 3.5
        );
    };
    let (lo, hi) = f.elevation_m;
    label(&mut out, f.elevation_y(hi), signed_km(hi));
    if lo < 0.0
        && hi > 0.0
        && (f.elevation_y(0.0) - PROFILE_TOP).min(strip_bottom - f.elevation_y(0.0)) > 10.0
    {
        label(&mut out, f.elevation_y(0.0), "0".into());
    }
    label(&mut out, f.elevation_y(lo), signed_km(lo));
    let (top, bottom) = f.depth_km;
    let step = nice_step((bottom - top) / 3.0);
    let mut d = (top / step).ceil() * step;
    while d <= bottom + 1e-9 {
        let text = if d.abs() < 1e-9 {
            "sea".to_string()
        } else if d < 0.0 {
            format!("+{} km", group_or_decimal(-d))
        } else {
            format!("{} km", group_or_decimal(d))
        };
        label(&mut out, f.depth_y(d), text);
        d += step;
    }
    let tick = nice_step((f.km.1 - f.km.0) / 6.0);
    let mut km = (f.km.0 / tick).ceil() * tick;
    while km <= f.km.1 + 1e-6 {
        let _ = write!(
            out,
            "<text x=\"{:.1}\" y=\"{:.1}\" class=\"axis\" text-anchor=\"middle\">{}</text>",
            f.x(km),
            HEIGHT - 4.0,
            group_or_decimal(km)
        );
        km += tick;
    }
    out += "</svg>";
    (out, f)
}

/// `+1.5 km` or `−11 km` for an elevation in meters.
fn signed_km(m: f64) -> String {
    let sign = if m < 0.0 { "\u{2212}" } else { "+" };
    format!("{sign}{} km", group_or_decimal(m.abs() / 1000.0))
}

/// Whole numbers with thousands separators, others with up to 2 decimals.
fn group_or_decimal(v: f64) -> String {
    if (v - v.round()).abs() < 1e-6 {
        let n = v.round();
        let s = crate::geo::group_thousands(n.abs() as u64);
        if n < 0.0 { format!("\u{2212}{s}") } else { s }
    } else {
        let s = format!("{v:.2}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// The elevation strip's extent (m) on one side of sea level for a line
/// reaching `m` there: a little headroom, then rounded up to 500 m, or to
/// whole kilometers past that.
fn strip_range(m: f64) -> f64 {
    let padded = (m * 1.05).max(MIN_ELEVATION_RANGE_M);
    if padded <= 1000.0 {
        nice_step(padded)
    } else {
        (padded / 1000.0).ceil() * 1000.0
    }
}

/// A round step (1, 2 or 5 times a power of ten) at least `raw`.
fn nice_step(raw: f64) -> f64 {
    let raw = raw.max(1e-6);
    let power = 10f64.powf(raw.log10().floor());
    [1.0, 2.0, 5.0, 10.0]
        .into_iter()
        .map(|m| m * power)
        .find(|&s| s >= raw)
        .unwrap_or(10.0 * power)
}

/// The crust color in the composition view (sRGB).
fn crust_color(felsic: f64) -> [f64; 3] {
    let stops = palette::COMPOSITION;
    let mut c = stops[0].srgb;
    for w in stops.windows(2) {
        if felsic > w[0].value {
            let t = ((felsic - w[0].value) / (w[1].value - w[0].value)).clamp(0.0, 1.0);
            c = mix(w[0].srgb, w[1].srgb, t);
        }
    }
    c
}

fn mix(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    [0, 1, 2].map(|i| a[i] + (b[i] - a[i]) * t)
}

/// One-line summary for the panel header: length and mean crust.
pub fn describe(samples: &[Sample]) -> String {
    let Some(last) = samples.last() else {
        return String::new();
    };
    let mean_crust = samples.iter().map(|s| s.column.crust_km).sum::<f64>() / samples.len() as f64;
    let mean_density = samples
        .iter()
        .map(|s| crust_density(s.column.felsic))
        .sum::<f64>()
        / samples.len() as f64;
    format!(
        "{} km · mean crust {:.0} km at {:.2} g/cm³",
        crate::geo::group_thousands(last.km.round() as u64),
        mean_crust,
        mean_density / 1000.0
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::EARTH_RADIUS_M;
    use crate::tectonics::edges;
    use h3o::Resolution;

    fn flat(elevations: &[f64]) -> Vec<Sample> {
        elevations
            .iter()
            .enumerate()
            .map(|(i, &e)| Sample {
                km: 100.0 * i as f64,
                elevation_m: e,
                column: Column::oceanic(50.0, 0.0),
            })
            .collect()
    }

    #[test]
    fn strip_scales_to_the_line() {
        // A line between +0.4 km and -4 km gets a strip from +0.5 to -5
        // km, not a fixed ±8 km.
        let (s, _) = svg(&flat(&[400.0, -4000.0, -3500.0]), &[], &Zoom::default());
        assert!(
            s.contains(">+0.5 km<") && s.contains(">\u{2212}5 km<"),
            "{s}"
        );
        assert!(s.contains(" vertical<"));
        // A deep trench rounds to the next whole km, not up to -20.
        assert_eq!(strip_range(10_229.0), 11_000.0);
        assert_eq!(strip_range(300.0), 500.0);
        assert_eq!(strip_range(800.0), 1000.0);
        // Ocean columns hang below sea level: water is drawn above them.
        assert!(s.contains(&palette::css(WATER)));
    }

    #[test]
    fn dragged_box_zooms_both_axes() {
        let samples = flat(&[400.0, -4000.0, -3500.0, -5000.0, -4500.0]);
        let f = frame(&samples, &Zoom::default());
        assert_eq!(f.km, (0.0, 400.0));
        assert_eq!(f.depth_km, (-ABOVE_SEA_KM, DEPTH_KM));
        // Mapping round-trips through SVG coordinates.
        assert!((f.km_at(f.x(123.0)) - 123.0).abs() < 1e-9);
        assert!((f.value_at(Panel::Section, f.depth_y(42.0)) - 42.0).abs() < 1e-9);
        assert!((f.value_at(Panel::Elevation, f.elevation_y(-2000.0)) + 2000.0).abs() < 1e-9);

        // A box in the section zooms distance and depth.
        let (x0, x1) = (f.x(100.0), f.x(300.0));
        let (y0, y1) = (f.depth_y(20.0), f.depth_y(120.0));
        let z = f.zoom_to(Zoom::default(), Panel::Section, (x0, y0), (x1, y1), 6.0);
        let (k0, k1) = z.km.unwrap();
        let (d0, d1) = z.depth_km.unwrap();
        assert!((k0 - 100.0).abs() < 1e-6 && (k1 - 300.0).abs() < 1e-6);
        assert!((d0 - 20.0).abs() < 1e-6 && (d1 - 120.0).abs() < 1e-6);
        assert!(z.elevation_m.is_none());
        // A sideways drag zooms distance only; a tiny one does nothing.
        let side = f.zoom_to(
            Zoom::default(),
            Panel::Elevation,
            (x0, 40.0),
            (x1, 42.0),
            6.0,
        );
        assert!(side.km.is_some() && side.elevation_m.is_none());
        let tiny = f.zoom_to(
            Zoom::default(),
            Panel::Section,
            (x0, y0),
            (x0 + 2.0, y0 + 2.0),
            6.0,
        );
        assert!(!tiny.is_zoomed());
        // Zooms never collapse below a minimum span.
        let narrow = f.zoom_to(
            Zoom::default(),
            Panel::Section,
            (x0, y0),
            (x0 + 6.0, y0 + 6.0),
            6.0,
        );
        let (d0, d1) = narrow.depth_km.unwrap();
        assert!(d1 - d0 >= MIN_SPAN_DEPTH_KM - 1e-9);

        // Drawing honors the zoom: depth labels for 20 to 120 km.
        let (svg_text, drawn) = svg(&samples, &[], &z);
        assert_eq!(drawn.depth_km, z.depth_km.unwrap());
        assert!(svg_text.contains(">50 km<") && !svg_text.contains(">300 km<"));
    }

    #[test]
    fn slab_shows_in_a_zoom_away_from_its_trench() {
        let samples = flat(&[-4000.0, -4000.0, -4000.0, -4000.0, -4000.0]);
        let trench = Marker {
            km: 300.0,
            color: [1.0; 4],
            label: "Subduction zone".into(),
            slab: Some(Slab {
                dips_forward: false,
                dip_deg: 45.0,
                thickness_km: 90.0,
            }),
        };
        // Zoomed to 100-200 km, short of the trench at 300 km: the slab
        // dipping back toward the start passes under this stretch.
        let zoom = Zoom {
            km: Some((100.0, 200.0)),
            ..Zoom::default()
        };
        let visible: Vec<Sample> = samples
            .into_iter()
            .filter(|s| (100.0..=200.0).contains(&s.km))
            .collect();
        let (s, _) = svg(&visible, &[trench], &zoom);
        assert!(s.contains("Slab (schematic)"));
        // The trench's own marker line is out of view.
        assert!(!s.contains("<title>Subduction zone</title>"));
    }

    #[test]
    fn hover_helpers() {
        let (a, b) = (LonLat::new(0.0, 0.0), LonLat::new(90.0, 0.0));
        let r = EARTH_RADIUS_M / 1000.0;
        let quarter = std::f64::consts::FRAC_PI_2 * r;
        let mid = point_at((a, b), quarter / 2.0, r);
        assert!((mid.lon - 45.0).abs() < 1e-9 && mid.lat.abs() < 1e-9);
        assert_eq!(
            point_at((a, b), 2.0 * quarter, r),
            point_at((a, b), quarter, r)
        );

        let s = Sample {
            km: 0.0,
            elevation_m: -4000.0,
            column: Column::oceanic(50.0, 0.5),
        };
        assert_eq!(layer_at(&s, 2.0), "water");
        assert_eq!(layer_at(&s, 4.2), "sediment");
        assert_eq!(layer_at(&s, 8.0), "crust");
        assert_eq!(layer_at(&s, 40.0), "mantle lithosphere");
        assert_eq!(layer_at(&s, 200.0), "asthenosphere");
        let land = Sample {
            elevation_m: 500.0,
            ..s
        };
        assert_eq!(layer_at(&land, -1.0), "air");
        assert_eq!(layer_at(&land, -0.2), "sediment");
        assert_eq!(
            nearest_sample(&flat(&[0.0, 0.0, 0.0]), 140.0).unwrap().km,
            100.0
        );
    }

    #[test]
    fn nice_steps() {
        assert_eq!(nice_step(0.8), 1.0);
        assert_eq!(nice_step(130.0), 200.0);
        assert_eq!(nice_step(420.0), 500.0);
    }

    #[test]
    fn slice_across_a_world() {
        let grid = Grid::new(Resolution::Three);
        let edges = edges(&grid);
        let world = World::generate(&grid, &edges, 4);
        // Half way round the planet crosses several plates.
        let (a, b) = (LonLat::new(-60.0, 10.0), LonLat::new(100.0, -10.0));
        let radius_km = EARTH_RADIUS_M / 1000.0;
        let (samples, markers) = sample(&world, &grid, &edges, (a, b), radius_km, None);
        assert_eq!(samples.len(), SAMPLES);
        assert!(samples[0].km == 0.0);
        let expected = a.to_unit().angle_between(b.to_unit()) * EARTH_RADIUS_M / 1000.0;
        assert!((samples.last().unwrap().km - expected).abs() < 1e-6);
        assert!(
            markers.iter().any(|m| m.label != "Craton edge"),
            "crosses boundaries"
        );
        assert!(markers.iter().all(|m| (0.0..=expected).contains(&m.km)));
        // Sampling a stretch resamples just that part, in finer steps.
        let (part, _) = sample(
            &world,
            &grid,
            &edges,
            (a, b),
            radius_km,
            Some((1000.0, 2000.0)),
        );
        assert_eq!(part.len(), SAMPLES);
        assert!((part[0].km - 1000.0).abs() < 1e-6);
        assert!((part.last().unwrap().km - 2000.0).abs() < 1e-6);
        // Every subduction crossing sketches a slab, dipping 30 to 60°.
        for m in markers.iter().filter(|m| m.label == "Subduction zone") {
            let slab = m.slab.expect("subduction has a slab");
            assert!((30.0..=60.0).contains(&slab.dip_deg));
            assert!(slab.thickness_km > 7.0);
        }
        assert!(
            markers
                .iter()
                .filter(|m| m.label != "Subduction zone")
                .all(|m| m.slab.is_none())
        );

        let (s, _) = svg(&samples, &markers, &Zoom::default());
        // Slabs are clipped to the section rather than squeezed into it.
        assert!(
            s.contains("clip-path=\"url(#section-clip)\"")
                || !markers.iter().any(|m| m.slab.is_some())
        );
        assert!(s.starts_with("<svg") && s.ends_with("</svg>"));
        assert!(s.contains("<title>"));
        // Nothing drawn below the section's floor.
        for part in s.split(" y=\"").skip(1) {
            let y: f64 = part.split('"').next().unwrap().parse().unwrap();
            assert!(y <= HEIGHT, "{y}");
        }
    }
}
