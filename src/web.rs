use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    Element, HtmlCanvasElement, HtmlInputElement, HtmlSelectElement, KeyboardEvent, PointerEvent,
    WheelEvent,
};

use crate::camera::{OrbitCamera, ray_sphere};
use crate::column_view;
use crate::cross_section;
use crate::geo::{EARTH_RADIUS_M, LonLat, group_thousands};
use crate::grid::{Grid, h3_centers};
use crate::legend::{self, PlateKey};
use crate::mantle::CM_PER_YEAR;
use crate::measure::{Measure, Reading};
use crate::perf::{FrameStats, format_bytes};
use crate::projection::{MapProjection, MapView, Projection};
use crate::renderer::{Globals, Renderer};
use crate::scene::{MANTLE_FLOW_COLOR, PLATE_MOTION_COLOR, Scene};
use crate::tectonics::{BoundaryKind, Edge, edges};
use crate::view::{
    COLOR_AGE, COLOR_BUOYANCY, COLOR_COMPOSITION, COLOR_CRUST, COLOR_ELEVATION, COLOR_LITHOSPHERE,
    COLOR_MANTLE, COLOR_PLATES, COLOR_PROVINCE, COLOR_ROCK, LAYER_BOUNDARIES, LAYER_CRATONS,
    LAYER_FORCES, LAYER_GRATICULE, LAYER_HOTSPOTS, LAYER_MANTLE, LAYER_MOTION, LAYER_PROVINCES,
    LAYER_TRACKS,
};
use crate::world::World;
use glam::DVec3;
use h3o::Resolution;

/// Start over the Salish Sea.
const START_VIEW: LonLat = LonLat::new(-123.0, 48.5);
const START_ALTITUDE_RADII: f64 = 2.0;
/// A press that moves less than this (CSS px) is a click, not a drag.
const CLICK_SLOP_PX: f64 = 4.0;
const LINE_WIDTH_CSS_PX: f64 = 2.0;
/// About 41k cells, each roughly 0.6° across: plenty for plates.
const PLATE_RESOLUTION: Resolution = Resolution::Three;
/// Arrows sit at the centers of this coarser H3 grid (842 cells).
const ARROW_RESOLUTION: Resolution = Resolution::One;
/// Layer tray checkboxes and the layer bit each controls.
const LAYER_TOGGLES: [(&str, u32); 9] = [
    ("layer-boundaries", LAYER_BOUNDARIES),
    ("layer-forces", LAYER_FORCES),
    ("layer-cratons", LAYER_CRATONS),
    ("layer-motion", LAYER_MOTION),
    ("layer-graticule", LAYER_GRATICULE),
    ("layer-mantle", LAYER_MANTLE),
    ("layer-provinces", LAYER_PROVINCES),
    ("layer-hotspots", LAYER_HOTSPOTS),
    ("layer-tracks", LAYER_TRACKS),
];
/// "Color by" dropdown options, by their `value` in index.html.
const COLOR_CHOICES: [(&str, u32); 10] = [
    ("elevation", COLOR_ELEVATION),
    ("rock", COLOR_ROCK),
    ("age", COLOR_AGE),
    ("plates", COLOR_PLATES),
    ("crust", COLOR_CRUST),
    ("composition", COLOR_COMPOSITION),
    ("province", COLOR_PROVINCE),
    ("lithosphere", COLOR_LITHOSPHERE),
    ("buoyancy", COLOR_BUOYANCY),
    ("mantle", COLOR_MANTLE),
];
/// A cross-section drag shorter than this (CSS px) along an axis doesn't
/// zoom that axis.
const SECTION_DRAG_MIN_PX: f64 = 6.0;
/// Dynamic topography below this (m) reads as "little vertical flow".
const MANTLE_STILL_M: f64 = 50.0;
/// Within this distance (m) of a deep province's outline, the HUD calls it
/// the edge, where strong plumes rise.
const PROVINCE_EDGE_M: f64 = 300_000.0;
/// Projection tray radio buttons.
const PROJECTION_CHOICES: [(&str, Projection); 5] = [
    ("proj-globe", Projection::Globe),
    ("proj-equirectangular", Projection::Equirectangular),
    ("proj-mercator", Projection::Mercator),
    ("proj-albers", Projection::Albers),
    ("proj-azimuthal", Projection::AzimuthalEquidistant),
];
/// How often the performance panel refreshes (ms); updating the page every
/// frame would itself cost time.
const PERF_REFRESH_MS: f64 = 250.0;
/// How long the copy button shows its check mark.
const COPIED_MS: i32 = 1500;
/// On flat maps, the cursor column's width in CSS px (see `Scene::build`).
const MAP_MARKER_PX: f64 = 3.0;

type FrameCallback = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    let _ = console_log::init_with_level(log::Level::Info);
    wasm_bindgen_futures::spawn_local(async {
        if let Err(e) = run().await {
            log::error!("{e}");
            show_error(&e);
        }
    });
}

struct Press {
    start: (f64, f64),
    last: (f64, f64),
    dragging: bool,
    /// Surface point (unit vector) under the cursor at press time; dragging
    /// keeps it under the cursor. None if the press started off the planet.
    grabbed: Option<DVec3>,
}

struct App {
    canvas: HtmlCanvasElement,
    renderer: Renderer,
    camera: OrbitCamera,
    /// The flat map, used unless its projection is `Globe`. Globe and maps
    /// share one focus: switching views keeps the same point centered.
    map: MapView,
    measure: Measure,
    /// Last on-planet cursor position while measuring, so the box keeps
    /// the live reading when the cursor leaves to click the copy button.
    measure_end: Option<LonLat>,
    measure_box: MeasureBox,
    grid: Grid,
    edges: Vec<Edge>,
    world: World,
    arrow_sites: Vec<DVec3>,
    /// World seed, mirrored in the URL as `?seed=` so worlds can be shared.
    seed: u64,
    /// Visible `LAYER_*` bits, mirroring the tray checkboxes.
    layers: u32,
    /// `COLOR_*` mode, mirroring the tray radio buttons.
    color: u32,
    /// Cursor in CSS pixels relative to the canvas, if over it.
    cursor: Option<(f64, f64)>,
    press: Option<Press>,
    hud_cursor: Element,
    hud_cell: Element,
    hud_crust: Element,
    hud_deep: Element,
    hud_rock: Element,
    hud_boundary: Element,
    /// For each cell on a plate boundary, one boundary touching it.
    cell_boundary: HashMap<u32, usize>,
    inspector: Inspector,
    section: Section,
    world_seed: Element,
    legend: Element,
    hud_dirty: bool,
    perf: Perf,
}

/// The column inspector card.
struct Inspector {
    card: Element,
    body: Element,
    visible: bool,
    /// The cell last drawn, to redraw only when the cursor changes cell.
    shown: Option<u32>,
}

/// The cross-section panel.
struct Section {
    card: Element,
    title: Element,
    body: Element,
    /// The box drawn while dragging to zoom, and the reset button.
    brush: Element,
    reset: Element,
    visible: bool,
    /// The span last drawn, to redraw only when the measurement or zoom
    /// changes.
    shown: Option<(LonLat, LonLat)>,
    dirty: bool,
    zoom: cross_section::Zoom,
    /// How the last drawing maps to SVG coordinates.
    frame: Option<cross_section::Frame>,
    /// A drag in progress: the panel and where it started (SVG units).
    drag: Option<(cross_section::Panel, (f64, f64))>,
    /// The samples last drawn, for the hover readout.
    samples: Vec<cross_section::Sample>,
    /// The hover guide line and its readout.
    guide: Element,
    readout: Element,
    /// Where the cursor is along the line (km), while over the chart.
    probe_km: Option<f64>,
}

/// The performance panel and the numbers behind it.
struct Perf {
    stats: FrameStats,
    card: Element,
    body: Element,
    visible: bool,
    /// When the panel last refreshed (animation frame time, ms).
    shown_at: f64,
    /// How long the last world took to generate (ms).
    generation_ms: f64,
}

/// The measurement info box.
struct MeasureBox {
    card: Element,
    source: Element,
    bearing: Element,
    distance: Element,
    status: Element,
    copy: Element,
}

impl App {
    /// The point on the planet under the cursor, if any.
    fn hover(&self) -> Option<LonLat> {
        self.surface_at(self.cursor?).map(LonLat::from_vec)
    }

    /// Normalized device coordinates for a position in CSS pixels.
    fn ndc(&self, (x, y): (f64, f64)) -> (f64, f64) {
        let (w, h) = (
            self.canvas.client_width() as f64,
            self.canvas.client_height() as f64,
        );
        (2.0 * x / w - 1.0, 1.0 - 2.0 * y / h)
    }

    fn is_map(&self) -> bool {
        self.map.projection.kind != Projection::Globe
    }

    /// Unit vector of the surface point under a CSS-pixel position.
    fn surface_at(&self, pos: (f64, f64)) -> Option<DVec3> {
        let (x, y) = self.ndc(pos);
        if self.is_map() {
            return self.map.surface_at(x, y, self.camera.aspect);
        }
        ray_sphere(&self.camera.ray(x, y), self.camera.radius).map(DVec3::normalize)
    }

    fn set_projection(&mut self, kind: Projection) {
        let focus = if self.is_map() {
            self.map.projection.center
        } else {
            LonLat::from_vec(self.camera.eye_dir())
        };
        if kind == Projection::Globe {
            self.camera.set_eye_dir(focus.to_unit());
        }
        self.map = MapView::new(MapProjection::new(kind, focus), self.camera.aspect);
    }

    /// Regenerate the world from `seed` and record it in the URL.
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        let start = now_ms();
        self.world = World::generate(&self.grid, &self.edges, seed);
        self.perf.generation_ms = now_ms() - start;
        self.renderer.set_world(&self.world);
        self.update_overlays();
        self.update_legend();
        self.index_boundaries();
        self.inspector.shown = None;
        self.section.shown = None;
        write_seed_to_url(seed);
        self.hud_dirty = true;
    }

    fn set_color(&mut self, mode: u32) {
        self.color = mode;
        self.update_legend();
    }

    /// Redraw the color key for the current mode, overlays and plates.
    fn update_legend(&self) {
        let plates = &self.world.plates;
        let mut sums = vec![(0.0, 0.0); plates.plates.len()];
        for (c, &p) in self.grid.centers.iter().enumerate() {
            let plate = plates.cell_plate[c] as usize;
            sums[plate].0 += plates.plates[plate].omega.cross(p).length() / CM_PER_YEAR;
            sums[plate].1 += 1.0;
        }
        let keys: Vec<PlateKey> = plates
            .plates
            .iter()
            .zip(sums)
            .map(|(p, (sum, n))| PlateKey {
                color: p.color,
                speed_cm_year: sum / f64::max(n, 1.0),
            })
            .collect();
        self.legend
            .set_inner_html(&legend::html(self.color, self.layers, &keys));
    }

    /// Rebuild the line overlays for the current world.
    fn update_overlays(&mut self) {
        let r = self.camera.radius;
        let w = &self.world;
        let grid = &self.grid;
        let boundaries = Scene::boundary_lines(&self.edges, &w.boundaries, r);
        let plate_velocity = |p: DVec3| {
            let cell = grid.cell_at(LonLat::from_vec(p));
            w.plates.plate_at(cell).omega.cross(p)
        };
        let motion = Scene::arrows(&self.arrow_sites, plate_velocity, r, PLATE_MOTION_COLOR);
        let mantle = Scene::arrows(
            &self.arrow_sites,
            |p| w.mantle.velocity(p),
            r,
            MANTLE_FLOW_COLOR,
        );
        self.renderer.set_overlay(LAYER_BOUNDARIES, &boundaries);
        self.renderer.set_overlay(
            LAYER_FORCES,
            &Scene::force_arrows(&w.forces, &self.edges, &w.plates.cell_plate, r),
        );
        self.renderer.set_overlay(
            LAYER_CRATONS,
            &Scene::craton_outlines(&self.edges, &w.crust.province, r),
        );
        self.renderer.set_overlay(LAYER_MOTION, &motion);
        self.renderer.set_overlay(LAYER_MANTLE, &mantle);
        self.renderer.set_overlay(
            LAYER_PROVINCES,
            &Scene::province_outlines(&w.mantle.provinces, r),
        );
        self.renderer
            .set_overlay(LAYER_HOTSPOTS, &Scene::hotspot_rings(&w.hotspots, r));
        self.renderer.set_overlay(
            LAYER_TRACKS,
            &Scene::hotspot_tracks(&w.hotspots, &w.hotspot_tracks, r),
        );
    }

    fn set_layer(&mut self, bit: u32, on: bool) {
        if on {
            self.layers |= bit;
        } else {
            self.layers &= !bit;
        }
        self.update_legend();
    }

    /// One animation frame; `time_ms` is the frame's timestamp.
    fn frame(&mut self, time_ms: f64) {
        let start = now_ms();
        self.draw_frame();
        self.perf.stats.record(time_ms, now_ms() - start);
        if self.perf.visible && time_ms - self.perf.shown_at >= PERF_REFRESH_MS {
            self.perf.shown_at = time_ms;
            self.update_perf_panel();
        }
    }

    fn set_perf_visible(&mut self, visible: bool) {
        self.perf.visible = visible;
        if visible {
            let _ = self.perf.card.remove_attribute("hidden");
            self.update_perf_panel();
        } else {
            let _ = self.perf.card.set_attribute("hidden", "");
        }
    }

    fn update_perf_panel(&self) {
        let row = |label: &str, value: String| format!("<dt>{label}</dt><dd>{value}</dd>");
        let mut html = String::new();
        match self.perf.stats.summary() {
            Some(s) => {
                html += &row("FPS", format!("{:.1}", s.fps));
                html += &row(
                    "Frame time",
                    format!("{:.1} ms (worst {:.1})", s.frame_ms, s.worst_frame_ms),
                );
                html += &row(
                    "CPU per frame",
                    format!("{:.2} ms (worst {:.2})", s.cpu_ms, s.worst_cpu_ms),
                );
            }
            None => html += &row("FPS", "measuring".into()),
        }
        html += &row(
            "World generation",
            format!("{:.0} ms", self.perf.generation_ms),
        );
        let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
        html += &row(
            "Canvas",
            format!(
                "{} × {} px (DPR {dpr})",
                self.canvas.width(),
                self.canvas.height()
            ),
        );
        let c = self.renderer.counts;
        html += &row("Draw calls", c.draw_calls.to_string());
        html += &row("Triangles", group_thousands(c.triangles));
        html += &row("Line segments", group_thousands(c.line_segments));
        html += &row(
            "GPU memory",
            format!("≈ {}", format_bytes(self.renderer.gpu_bytes())),
        );
        self.perf.body.set_inner_html(&format!("<dl>{html}</dl>"));
    }

    fn draw_frame(&mut self) {
        let (width, height) = self.renderer.sync_size();
        self.camera.aspect = width as f64 / height as f64;
        let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());

        let hover = self.hover();
        let screen_center = self.map.screen_center();
        let globals = Globals {
            view_proj: self.camera.view_proj().as_mat4().to_cols_array_2d(),
            light_dir: self.camera.light_dir().as_vec3().extend(0.0).to_array(),
            params: [
                width as f32,
                height as f32,
                self.camera.radius as f32,
                (LINE_WIDTH_CSS_PX * dpr) as f32,
            ],
            flags: [self.layers, self.map.projection.kind.id(), self.color, 0],
            projection: self.map.projection.params(),
            map: [
                screen_center.x as f32,
                screen_center.y as f32,
                self.map.half_height as f32,
                self.camera.aspect as f32,
            ],
        };
        let marker_scale = if self.is_map() {
            // Meters per CSS pixel at the map's nominal scale.
            let m_per_px = self.map.half_height * self.camera.radius
                / (self.canvas.client_height().max(1) as f64 / 2.0);
            MAP_MARKER_PX * m_per_px / Scene::CURSOR_WIDTH
        } else {
            self.camera.altitude.min(self.camera.radius)
        };
        if !self.measure.is_finished() && hover.is_some() {
            self.measure_end = hover;
        }
        let probe = self
            .section
            .probe_km
            .zip(self.measure.span(self.measure_end))
            .map(|(km, line)| cross_section::point_at(line, km, self.camera.radius / 1000.0));
        let scene = Scene::build(
            &self.measure,
            hover,
            probe,
            self.camera.radius,
            marker_scale,
        );
        self.renderer.render(&globals, &scene);
        self.update_hud(hover);
    }

    fn update_hud(&mut self, hover: Option<LonLat>) {
        let off = if self.is_map() {
            "Off the map"
        } else {
            "Off the globe"
        };
        let text = hover.map_or_else(|| off.to_string(), LonLat::display);
        set_text(&self.hud_cursor, &text);
        let (cell, crust) = hover.map_or_else(Default::default, |h| {
            let id = self.grid.cell_at(h);
            let c = id as usize;
            let plates = &self.world.plates;
            let cell = format!(
                "Cell {} · plate {}, {:.1} cm/yr",
                self.grid.cells[c],
                plates.cell_plate[c] + 1,
                plates
                    .plate_at(id)
                    .speed_cm_per_year(h.to_unit(), self.camera.radius)
            );
            let crust = &self.world.crust;
            let elevation = format_meters(crust.elevation_m[c]);
            let crust = if crust.kind[c].is_continental() {
                format!(
                    "Continent ({}), {:.0} km crust, {elevation}",
                    crust.province[c].name(),
                    crust.column[c].crust_km
                )
            } else {
                format!("Ocean floor, {:.0} Myr old, {elevation}", crust.age_myr[c])
            };
            (cell, crust)
        });
        set_text(&self.hud_cell, &cell);
        set_text(&self.hud_crust, &crust);
        let deep = hover.map_or_else(String::new, |h| self.deep_earth_text(h));
        set_text(&self.hud_deep, &deep);
        let cell = hover.map(|h| self.grid.cell_at(h));
        let rock = cell.map_or_else(String::new, |c| self.rock_text(c as usize));
        set_text(&self.hud_rock, &rock);
        let boundary = cell.map_or_else(String::new, |c| self.boundary_text(c));
        set_text(&self.hud_boundary, &boundary);
        self.update_inspector(cell);
        self.update_section();
        self.update_measure_box();
        if !self.hud_dirty {
            return;
        }
        self.hud_dirty = false;
        set_text(&self.world_seed, &format!("Seed {}", self.seed));
    }

    /// Maps boundary cells to a boundary touching them, for the HUD.
    fn index_boundaries(&mut self) {
        self.cell_boundary.clear();
        for (i, b) in self.world.boundaries.iter().enumerate() {
            let e = &self.edges[b.edge as usize];
            self.cell_boundary.entry(e.a).or_insert(i);
            self.cell_boundary.entry(e.b).or_insert(i);
        }
    }

    /// HUD line about the plate boundary at cell `c`, if it is on one.
    fn boundary_text(&self, c: u32) -> String {
        let Some(&i) = self.cell_boundary.get(&c) else {
            return String::new();
        };
        let b = &self.world.boundaries[i];
        let e = &self.edges[b.edge as usize];
        let plate = |cell: u32| self.world.plates.cell_plate[cell as usize] + 1;
        let (pa, pb) = (plate(e.a), plate(e.b));
        let rate = b.rate;
        match b.kind {
            BoundaryKind::Subduction => {
                let over = b.overriding(&self.edges);
                let columns = &self.world.crust.column;
                let denser = columns[b.subducting as usize].density_excess()
                    - columns[over as usize].density_excess();
                format!(
                    "Subduction: plate {} sinks under plate {} at {rate:.1} cm/yr \
                     (its column {denser:.0} kg/m³ denser)",
                    plate(b.subducting),
                    plate(over)
                )
            }
            BoundaryKind::Collision => {
                format!("Collision: plates {pa} and {pb} converge at {rate:.1} cm/yr")
            }
            BoundaryKind::Ridge => {
                format!("Spreading ridge: plates {pa} and {pb} part at {rate:.1} cm/yr")
            }
            BoundaryKind::Rift => {
                format!("Continental rift: plates {pa} and {pb} part at {rate:.1} cm/yr")
            }
            BoundaryKind::Transform => {
                format!("Transform fault: plates {pa} and {pb} slide past at {rate:.1} cm/yr")
            }
        }
    }

    /// HUD line about the surface rock and column of cell `c`.
    fn rock_text(&self, c: usize) -> String {
        let crust = &self.world.crust;
        let col = &crust.column[c];
        format!(
            "{} · crust {:.2} g/cm³, {:.0}% felsic · lithosphere {:.0} km",
            crust.rock[c].name(),
            crate::crust::crust_density(col.felsic) / 1000.0,
            100.0 * col.felsic,
            col.thickness_km()
        )
    }

    fn set_inspector_visible(&mut self, visible: bool) {
        self.inspector.visible = visible;
        self.inspector.shown = None;
        if visible {
            let _ = self.inspector.card.remove_attribute("hidden");
        } else {
            let _ = self.inspector.card.set_attribute("hidden", "");
        }
    }

    /// Redraws the column inspector if it is open and the cell changed.
    fn update_inspector(&mut self, cell: Option<u32>) {
        if !self.inspector.visible || cell == self.inspector.shown {
            return;
        }
        self.inspector.shown = cell;
        let html = cell.map_or_else(
            || "<p class=\"help\">Point at the planet to inspect its column.</p>".into(),
            |c| column_view::svg(&self.world.crust.column[c as usize]),
        );
        self.inspector.body.set_inner_html(&html);
    }

    fn set_section_visible(&mut self, visible: bool) {
        self.section.visible = visible;
        self.section.shown = None;
        if visible {
            let _ = self.section.card.remove_attribute("hidden");
            self.update_section();
        } else {
            let _ = self.section.card.set_attribute("hidden", "");
            self.section_leave();
        }
    }

    /// Redraws the cross-section if it is open and the measurement changed
    /// (it follows the cursor while measuring).
    fn update_section(&mut self) {
        if !self.section.visible {
            return;
        }
        let span = self.measure.span(self.measure_end);
        if span != self.section.shown {
            // A new line starts unzoomed.
            self.section.zoom = cross_section::Zoom::default();
        } else if span.is_some() && !self.section.dirty {
            return;
        }
        self.section.shown = span;
        self.section.dirty = false;
        if self.section.zoom.is_zoomed() {
            let _ = self.section.reset.remove_attribute("hidden");
        } else {
            let _ = self.section.reset.set_attribute("hidden", "");
        }
        match span {
            Some(line) => {
                let radius_km = self.camera.radius / 1000.0;
                let zoom = self.section.zoom;
                let (samples, _) = cross_section::sample(
                    &self.world,
                    &self.grid,
                    &self.edges,
                    line,
                    radius_km,
                    zoom.km,
                );
                // Markers come from the whole line: a slab dips far from its
                // trench, so it can cross a zoomed view whose stretch doesn't
                // include the trench itself. The header describes the whole
                // line too.
                let (whole, markers) = cross_section::sample(
                    &self.world,
                    &self.grid,
                    &self.edges,
                    line,
                    radius_km,
                    None,
                );
                set_text(
                    &self.section.title,
                    &format!("Cross-section · {}", cross_section::describe(&whole)),
                );
                let (svg, frame) = cross_section::svg(&samples, &markers, &zoom);
                self.section.body.set_inner_html(&svg);
                self.section.frame = Some(frame);
                self.section.samples = samples;
            }
            None => {
                self.section.frame = None;
                set_text(&self.section.title, "Cross-section");
                self.section.body.set_inner_html(
                    "<p class=\"help\">Click two points on the planet to measure a \
                     line; its cross-section appears here.</p>",
                );
            }
        }
    }

    /// A pointer position as SVG coordinates of the cross-section chart,
    /// and the scale from SVG units to CSS pixels.
    fn section_point(&self, e: &web_sys::MouseEvent) -> ((f64, f64), f64) {
        let rect = self.section.body.get_bounding_client_rect();
        let scale = cross_section::Frame::WIDTH / rect.width().max(1.0);
        (
            (
                (e.client_x() as f64 - rect.left()) * scale,
                (e.client_y() as f64 - rect.top()) * scale,
            ),
            scale,
        )
    }

    fn section_down(&mut self, e: &PointerEvent) {
        let Some(frame) = self.section.frame else {
            return;
        };
        let (p, _) = self.section_point(e);
        if e.button() == 0
            && let Some(panel) = frame.panel_at(p.0, p.1)
        {
            self.section.drag = Some((panel, p));
            let _ = self.section.body.set_pointer_capture(e.pointer_id());
        }
    }

    /// Draws the zoom box from the drag's start to the pointer, kept within
    /// the panel the drag started in.
    fn section_move(&mut self, e: &PointerEvent) {
        let Some((panel, from)) = self.section.drag else {
            self.section_hover(e);
            return;
        };
        let (to, scale) = self.section_point(e);
        let (x0, x1) = cross_section::Frame::plot_x();
        let (y0, y1) = cross_section::Frame::panel_y(panel);
        let to = (to.0.clamp(x0, x1), to.1.clamp(y0, y1));
        let (left, top) = (from.0.min(to.0) / scale, from.1.min(to.1) / scale);
        let (w, h) = ((to.0 - from.0).abs() / scale, (to.1 - from.1).abs() / scale);
        let _ = self.section.brush.set_attribute(
            "style",
            &format!("left:{left:.1}px;top:{top:.1}px;width:{w:.1}px;height:{h:.1}px"),
        );
        let _ = self.section.brush.remove_attribute("hidden");
    }

    fn section_up(&mut self, e: &PointerEvent) {
        let Some((panel, from)) = self.section.drag.take() else {
            return;
        };
        let _ = self.section.brush.set_attribute("hidden", "");
        let Some(frame) = self.section.frame else {
            return;
        };
        let (to, scale) = self.section_point(e);
        let zoom = frame.zoom_to(
            self.section.zoom,
            panel,
            from,
            to,
            SECTION_DRAG_MIN_PX * scale,
        );
        if zoom != self.section.zoom {
            self.section.zoom = zoom;
            self.section.dirty = true;
            self.update_section();
        }
    }

    /// Follows the cursor over the chart: a guide line, a readout, and the
    /// matching point marked on the globe.
    fn section_hover(&mut self, e: &PointerEvent) {
        let (Some(frame), Some(line)) = (self.section.frame, self.measure.span(self.measure_end))
        else {
            return self.section_leave();
        };
        let (p, scale) = self.section_point(e);
        let (x0, x1) = cross_section::Frame::plot_x();
        if !(x0..=x1).contains(&p.0) {
            return self.section_leave();
        }
        let km = frame.km_at(p.0);
        self.section.probe_km = Some(km);
        let _ = self
            .section
            .guide
            .set_attribute("style", &format!("left:{:.1}px", p.0 / scale));
        let _ = self.section.guide.remove_attribute("hidden");

        let radius_km = self.camera.radius / 1000.0;
        let total = line.0.to_unit().angle_between(line.1.to_unit()) * radius_km;
        let at = cross_section::point_at(line, km, radius_km);
        let mut text = format!(
            "{} of {} km · {}",
            format_km(km),
            format_km(total),
            at.display()
        );
        if let Some(s) = cross_section::nearest_sample(&self.section.samples, km) {
            text += &format!(" · {}", format_meters(s.elevation_m));
            if frame.panel_at(p.0, p.1) == Some(cross_section::Panel::Section) {
                let depth = frame.value_at(cross_section::Panel::Section, p.1);
                let place = if depth < 0.0 {
                    format!("{} km above sea level", format_km(-depth))
                } else {
                    format!("{} km below sea level", format_km(depth))
                };
                text += &format!(" · {place}: {}", cross_section::layer_at(s, depth));
            }
        }
        set_text(&self.section.readout, &text);
    }

    fn section_leave(&mut self) {
        self.section.probe_km = None;
        let _ = self.section.guide.set_attribute("hidden", "");
        set_text(
            &self.section.readout,
            "Hover the chart to find the point on the globe.",
        );
    }

    fn reset_section_zoom(&mut self) {
        if self.section.zoom.is_zoomed() {
            self.section.zoom = cross_section::Zoom::default();
            self.section.dirty = true;
            self.update_section();
        }
    }

    /// HUD line about the mantle and any hotspot volcano under `h`.
    fn deep_earth_text(&self, h: LonLat) -> String {
        let c = self.grid.cell_at(h) as usize;
        let d = self.world.dynamic_m[c];
        let mut text = if d.abs() < MANTLE_STILL_M {
            "Mantle: little vertical flow".to_string()
        } else if d > 0.0 {
            format!("Mantle rising, {}", format_meters(d))
        } else {
            format!("Mantle sinking, {}", format_meters(d))
        };
        let to_edge = self
            .world
            .mantle
            .provinces
            .iter()
            .map(|p| p.edge_distance(h.to_unit()))
            .fold(f64::MAX, f64::min);
        if to_edge.abs() * self.camera.radius < PROVINCE_EDGE_M {
            text += " (deep province edge)";
        } else if to_edge < 0.0 {
            text += " (over a deep province)";
        }
        if let Some(mark) = self.world.hotspot_marks[c] {
            let plume = format!(
                "{} plume",
                self.world.hotspots[mark.hotspot as usize].kind.name()
            );
            if mark.age_myr < 1.0 {
                text += &format!(" · Active hotspot volcano ({plume})");
            } else {
                text += &format!(" · Hotspot volcano, {:.0} Myr old ({plume})", mark.age_myr);
            }
        }
        text
    }

    /// The current measurement, finished or still following the cursor.
    fn reading(&self) -> Option<Reading> {
        let (a, b) = self.measure.span(self.measure_end)?;
        Some(Reading::new(a, b, self.camera.radius))
    }

    fn update_measure_box(&self) {
        let b = &self.measure_box;
        let Some(source) = self.measure.source else {
            let _ = b.card.set_attribute("hidden", "");
            return;
        };
        let _ = b.card.remove_attribute("hidden");
        set_text(&b.source, &source.display());
        let reading = self.reading();
        set_text(
            &b.bearing,
            &reading
                .as_ref()
                .map_or_else(String::new, Reading::bearing_text),
        );
        set_text(
            &b.distance,
            &reading
                .as_ref()
                .map_or_else(String::new, Reading::distance_text),
        );
        set_text(
            &b.status,
            if self.measure.is_finished() {
                "Click to start a new measurement · Esc: clear"
            } else {
                "Click to set the end point · Esc: clear"
            },
        );
    }

    fn copy_measurement(&self) {
        if let Some(r) = self.reading() {
            copy_to_clipboard(&r.clipboard_text(), &self.measure_box.copy);
        }
    }

    fn pointer_down(&mut self, e: &PointerEvent) {
        if e.button() != 0 {
            return;
        }
        let _ = self.canvas.set_pointer_capture(e.pointer_id());
        let pos = (e.offset_x() as f64, e.offset_y() as f64);
        self.press = Some(Press {
            start: pos,
            last: pos,
            dragging: false,
            grabbed: self.surface_at(pos),
        });
    }

    fn pointer_move(&mut self, e: &PointerEvent) {
        let pos = (e.offset_x() as f64, e.offset_y() as f64);
        self.cursor = Some(pos);
        let (x, y) = self.ndc(pos);
        if let Some(press) = &mut self.press {
            let moved = (pos.0 - press.start.0).hypot(pos.1 - press.start.1);
            press.dragging |= moved > CLICK_SLOP_PX;
            if press.dragging && self.map.projection.kind != Projection::Globe {
                // Turn the globe under the map, as on the globe itself.
                match press.grabbed {
                    Some(g) => self.map.drag_to(g, x, y, self.camera.aspect),
                    None => self.map.drag_by(
                        pos.0 - press.last.0,
                        pos.1 - press.last.1,
                        self.canvas.client_height().max(1) as f64,
                    ),
                }
            } else if press.dragging {
                // Push the globe by the grabbed point; fall back to pixel
                // rotation when the press or the cursor is off the globe.
                let grabbed = press.grabbed.is_some_and(|g| self.camera.drag_to(g, x, y));
                if !grabbed {
                    self.camera.drag(pos.0 - press.last.0, pos.1 - press.last.1);
                }
            }
            press.last = pos;
        }
    }

    fn pointer_up(&mut self, e: &PointerEvent) {
        if e.button() != 0 {
            return;
        }
        if let Some(press) = self.press.take()
            && !press.dragging
            && let Some(p) = self.hover()
        {
            if self.measure.is_finished() || self.measure.source.is_none() {
                self.measure_end = None;
            }
            self.measure.click(p);
        }
    }

    fn key_down(&mut self, e: &KeyboardEvent) {
        // Leave browser shortcuts alone, and keys aimed at controls.
        let on_control = e
            .target()
            .and_then(|t| t.dyn_into::<Element>().ok())
            .is_some_and(|el| matches!(el.tag_name().as_str(), "INPUT" | "BUTTON"));
        if on_control || e.ctrl_key() || e.meta_key() || e.alt_key() {
            return;
        }
        if e.key() == "Escape" {
            self.measure.clear();
            self.measure_end = None;
            e.prevent_default();
        }
    }

    fn wheel(&mut self, e: &WheelEvent) {
        e.prevent_default();
        // deltaMode 1 is lines; treat a line as about 16 px.
        let px = e.delta_y() * if e.delta_mode() == 1 { 16.0 } else { 1.0 };
        if self.is_map() {
            // Zoom about the cursor.
            let (x, y) = self.ndc((e.offset_x() as f64, e.offset_y() as f64));
            self.map.zoom_at(px * 0.001, x, y, self.camera.aspect);
        } else {
            self.camera.zoom(px * 0.001);
        }
    }
}

fn set_text(el: &Element, text: &str) {
    if el.text_content().as_deref() != Some(text) {
        el.set_text_content(Some(text));
    }
}

/// A new seed small enough to read and type in the URL.
fn random_seed() -> u64 {
    (js_sys::Math::random() * u32::MAX as f64) as u64
}

/// The `seed` parameter of a query string like `?seed=42`.
fn seed_from_query(query: &str) -> Option<u64> {
    query
        .trim_start_matches('?')
        .split('&')
        .find_map(|pair| pair.strip_prefix("seed="))
        .and_then(|v| v.parse().ok())
}

fn write_seed_to_url(seed: u64) {
    // replaceState, so regenerating doesn't flood the back button.
    let url = format!("?seed={seed}");
    if let Some(history) = web_sys::window().and_then(|w| w.history().ok()) {
        let _ = history.replace_state_with_url(&JsValue::NULL, "", Some(&url));
    }
}

/// Kilometers to one decimal with thousands separators, e.g. `1,234.5`.
fn format_km(km: f64) -> String {
    let tenths = (km.abs() * 10.0).round() as u64;
    format!("{}.{}", group_thousands(tenths / 10), tenths % 10)
}

/// Signed elevation like `+1,234 m` or `−4,812 m`.
fn format_meters(m: f64) -> String {
    let sign = if m < 0.0 { '\u{2212}' } else { '+' };
    format!("{sign}{} m", group_thousands(m.abs().round() as u64))
}

/// Puts `text` on the clipboard and briefly swaps `button`'s icon to a
/// check mark (see `.copied` in index.html).
/// Milliseconds from the page's high-resolution clock.
fn now_ms() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map_or(0.0, |p| p.now())
}

fn copy_to_clipboard(text: &str, button: &Element) {
    let Some(window) = web_sys::window() else {
        return;
    };
    // The promise rejects only without permission; there is nothing more
    // useful to do then than skip the check mark.
    let _ = window.navigator().clipboard().write_text(text);
    button.set_class_name("copied");
    let button = button.clone();
    let reset = Closure::once_into_js(move || button.set_class_name(""));
    let _ = window
        .set_timeout_with_callback_and_timeout_and_arguments_0(reset.unchecked_ref(), COPIED_MS);
}

async fn run() -> Result<(), String> {
    let window = web_sys::window().ok_or("no window")?;
    let document = window.document().ok_or("no document")?;
    let element = |id: &str| {
        document
            .get_element_by_id(id)
            .ok_or_else(|| format!("missing #{id}"))
    };
    let canvas: HtmlCanvasElement = element("world")?
        .dyn_into()
        .map_err(|_| "#world is not a canvas")?;

    if !window.is_secure_context() {
        return Err(
            "WebGPU needs a secure context: open this page over https or on localhost.".into(),
        );
    }
    let has_webgpu = js_sys::Reflect::has(&window.navigator(), &"gpu".into()).unwrap_or(false);
    if !has_webgpu {
        return Err(
            "This browser has WebGPU disabled or unsupported (navigator.gpu is missing).".into(),
        );
    }

    let seed = window
        .location()
        .search()
        .ok()
        .and_then(|q| seed_from_query(&q))
        .unwrap_or_else(random_seed);
    write_seed_to_url(seed);
    let grid = Grid::new(PLATE_RESOLUTION);
    let edges = edges(&grid);
    let start = now_ms();
    let world = World::generate(&grid, &edges, seed);
    let generation_ms = now_ms() - start;

    let renderer = Renderer::new(canvas.clone(), &grid, &world).await?;
    let aspect = canvas.client_width().max(1) as f64 / canvas.client_height().max(1) as f64;
    let app = Rc::new(RefCell::new(App {
        canvas: canvas.clone(),
        renderer,
        camera: OrbitCamera::over(EARTH_RADIUS_M, START_VIEW, START_ALTITUDE_RADII, aspect),
        map: MapView::new(MapProjection::new(Projection::Globe, START_VIEW), aspect),
        measure: Measure::default(),
        measure_end: None,
        measure_box: MeasureBox {
            card: element("measure")?,
            source: element("measure-source")?,
            bearing: element("measure-bearing")?,
            distance: element("measure-distance")?,
            status: element("measure-status")?,
            copy: element("measure-copy")?,
        },
        grid,
        edges,
        world,
        arrow_sites: h3_centers(ARROW_RESOLUTION),
        seed,
        layers: 0,
        color: COLOR_ELEVATION,
        cursor: None,
        press: None,
        hud_cursor: element("hud-cursor")?,
        hud_cell: element("hud-cell")?,
        hud_crust: element("hud-crust")?,
        hud_deep: element("hud-deep")?,
        hud_rock: element("hud-rock")?,
        hud_boundary: element("hud-boundary")?,
        cell_boundary: HashMap::new(),
        section: Section {
            card: element("section")?,
            title: element("section-title")?,
            body: element("section-body")?,
            brush: element("section-brush")?,
            reset: element("section-reset")?,
            visible: false,
            shown: None,
            dirty: false,
            zoom: cross_section::Zoom::default(),
            frame: None,
            drag: None,
            samples: Vec::new(),
            guide: element("section-guide")?,
            readout: element("section-readout")?,
            probe_km: None,
        },
        inspector: Inspector {
            card: element("inspector")?,
            body: element("inspector-body")?,
            visible: false,
            shown: None,
        },
        world_seed: element("world-seed")?,
        legend: element("legend")?,
        perf: Perf {
            stats: FrameStats::default(),
            card: element("perf")?,
            body: element("perf-body")?,
            visible: false,
            shown_at: 0.0,
            generation_ms,
        },
        hud_dirty: true,
    }));
    app.borrow_mut().update_overlays();
    app.borrow_mut().index_boundaries();
    log::info!("renderer ready");

    listen(&canvas, "pointerdown", &app, |app, e: PointerEvent| {
        app.pointer_down(&e)
    });
    listen(&canvas, "pointermove", &app, |app, e: PointerEvent| {
        app.pointer_move(&e)
    });
    listen(&canvas, "pointerup", &app, |app, e: PointerEvent| {
        app.pointer_up(&e)
    });
    listen(&canvas, "pointerleave", &app, |app, _: PointerEvent| {
        app.cursor = None
    });
    listen(&canvas, "wheel", &app, |app, e: WheelEvent| app.wheel(&e));
    listen(&window, "keydown", &app, |app, e: KeyboardEvent| {
        app.key_down(&e)
    });

    // The checkboxes' `checked` attributes in index.html set the defaults.
    for (id, bit) in LAYER_TOGGLES {
        let input: HtmlInputElement = element(id)?
            .dyn_into()
            .map_err(|_| format!("#{id} is not an input"))?;
        app.borrow_mut().set_layer(bit, input.checked());
        listen(
            &input.clone(),
            "change",
            &app,
            move |app, _: web_sys::Event| app.set_layer(bit, input.checked()),
        );
    }
    let color_select: HtmlSelectElement = element("color-mode")?
        .dyn_into()
        .map_err(|_| "#color-mode is not a select")?;
    let mode_of = |value: String| {
        COLOR_CHOICES
            .iter()
            .find(|(v, _)| *v == value)
            .map_or(COLOR_ELEVATION, |&(_, mode)| mode)
    };
    app.borrow_mut().set_color(mode_of(color_select.value()));
    listen(
        &color_select.clone(),
        "change",
        &app,
        move |app, _: web_sys::Event| app.set_color(mode_of(color_select.value())),
    );
    let chart = element("section-body")?;
    listen(&chart, "pointerdown", &app, |app, e: PointerEvent| {
        app.section_down(&e)
    });
    listen(&chart, "pointermove", &app, |app, e: PointerEvent| {
        app.section_move(&e)
    });
    listen(&chart, "pointerup", &app, |app, e: PointerEvent| {
        app.section_up(&e)
    });
    listen(&chart, "pointerleave", &app, |app, _: PointerEvent| {
        app.section_leave()
    });
    listen(&chart, "dblclick", &app, |app, _: web_sys::MouseEvent| {
        app.reset_section_zoom()
    });
    let reset = element("section-reset")?;
    listen(&reset, "click", &app, |app, _: web_sys::Event| {
        app.reset_section_zoom()
    });
    let section_toggle: HtmlInputElement = element("inspect-section")?
        .dyn_into()
        .map_err(|_| "#inspect-section is not an input")?;
    app.borrow_mut()
        .set_section_visible(section_toggle.checked());
    listen(
        &section_toggle.clone(),
        "change",
        &app,
        move |app, _: web_sys::Event| app.set_section_visible(section_toggle.checked()),
    );
    let inspector_toggle: HtmlInputElement = element("inspect-column")?
        .dyn_into()
        .map_err(|_| "#inspect-column is not an input")?;
    app.borrow_mut()
        .set_inspector_visible(inspector_toggle.checked());
    listen(
        &inspector_toggle.clone(),
        "change",
        &app,
        move |app, _: web_sys::Event| app.set_inspector_visible(inspector_toggle.checked()),
    );
    for (id, kind) in PROJECTION_CHOICES {
        let input: HtmlInputElement = element(id)?
            .dyn_into()
            .map_err(|_| format!("#{id} is not an input"))?;
        if input.checked() {
            app.borrow_mut().set_projection(kind);
        }
        listen(
            &input.clone(),
            "change",
            &app,
            move |app, _: web_sys::Event| {
                if input.checked() {
                    app.set_projection(kind);
                }
            },
        );
    }
    let perf_toggle: HtmlInputElement = element("perf-stats")?
        .dyn_into()
        .map_err(|_| "#perf-stats is not an input")?;
    app.borrow_mut().set_perf_visible(perf_toggle.checked());
    listen(
        &perf_toggle.clone(),
        "change",
        &app,
        move |app, _: web_sys::Event| app.set_perf_visible(perf_toggle.checked()),
    );
    let copy = element("measure-copy")?;
    listen(&copy, "click", &app, |app, _: web_sys::Event| {
        app.copy_measurement()
    });
    let new_world = element("new-world")?;
    listen(&new_world, "click", &app, |app, _: web_sys::Event| {
        app.set_seed(random_seed())
    });

    animate(app);
    Ok(())
}

/// Register `handler` for `event` on `target` for the life of the page.
fn listen<E: JsCast + 'static>(
    target: &web_sys::EventTarget,
    event: &str,
    app: &Rc<RefCell<App>>,
    handler: impl Fn(&mut App, E) + 'static,
) {
    let app = app.clone();
    let closure = Closure::<dyn FnMut(web_sys::Event)>::new(move |e: web_sys::Event| {
        handler(&mut app.borrow_mut(), e.unchecked_into());
    });
    target
        .add_event_listener_with_callback(event, closure.as_ref().unchecked_ref())
        .expect("addEventListener");
    closure.forget();
}

/// Drive `frame` from requestAnimationFrame. The closure re-schedules itself.
fn animate(app: Rc<RefCell<App>>) {
    let callback: FrameCallback = Rc::new(RefCell::new(None));
    let next = callback.clone();
    *callback.borrow_mut() = Some(Closure::new(move |time_ms: f64| {
        app.borrow_mut().frame(time_ms);
        request_frame(next.borrow().as_ref().unwrap());
    }));
    request_frame(callback.borrow().as_ref().unwrap());
}

fn request_frame(f: &Closure<dyn FnMut(f64)>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .expect("requestAnimationFrame");
}

fn show_error(message: &str) {
    if let Some(el) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("error"))
    {
        el.set_text_content(Some(message));
        let _ = el.remove_attribute("hidden");
    }
}
