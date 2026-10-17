use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{
    Element, HtmlCanvasElement, HtmlInputElement, KeyboardEvent, PointerEvent, WheelEvent,
};

use crate::camera::{OrbitCamera, ray_sphere};
use crate::draw::Drawing;
use crate::geo::{EARTH_RADIUS_M, LonLat};
use crate::grid::Grid;
use crate::plates::Plates;
use crate::projection::{MapProjection, MapView, Projection};
use crate::renderer::{Globals, LAYER_BOUNDARIES, LAYER_GRATICULE, LAYER_PLATES, Renderer};
use crate::scene::Scene;
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
/// Layer tray checkboxes and the layer bit each controls.
const LAYER_TOGGLES: [(&str, u32); 3] = [
    ("layer-plates", LAYER_PLATES),
    ("layer-boundaries", LAYER_BOUNDARIES),
    ("layer-graticule", LAYER_GRATICULE),
];
/// Projection tray radio buttons.
const PROJECTION_CHOICES: [(&str, Projection); 5] = [
    ("proj-globe", Projection::Globe),
    ("proj-equirectangular", Projection::Equirectangular),
    ("proj-mercator", Projection::Mercator),
    ("proj-albers", Projection::Albers),
    ("proj-azimuthal", Projection::AzimuthalEquidistant),
];
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
    drawing: Drawing,
    grid: Grid,
    plates: Plates,
    /// World seed, mirrored in the URL as `?seed=` so worlds can be shared.
    seed: u64,
    /// Visible `LAYER_*` bits, mirroring the tray checkboxes.
    layers: u32,
    /// Cursor in CSS pixels relative to the canvas, if over it.
    cursor: Option<(f64, f64)>,
    press: Option<Press>,
    hud_cursor: Element,
    hud_cell: Element,
    world_seed: Element,
    hud_polygons: Element,
    hud_dirty: bool,
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

    /// Regenerate the plates from `seed` and record it in the URL.
    fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.plates = Plates::generate(&self.grid, seed);
        let boundaries = Scene::plate_boundaries(&self.grid, &self.plates, self.camera.radius);
        self.renderer.set_plates(&self.plates, &boundaries);
        write_seed_to_url(seed);
        self.hud_dirty = true;
    }

    fn set_layer(&mut self, bit: u32, on: bool) {
        if on {
            self.layers |= bit;
        } else {
            self.layers &= !bit;
        }
    }

    fn frame(&mut self) {
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
            flags: [self.layers, self.map.projection.kind.id(), 0, 0],
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
        let scene = Scene::build(&self.drawing, hover, self.camera.radius, marker_scale);
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
        let cell = hover.map_or_else(String::new, |h| {
            let id = self.grid.cell_at(h);
            let plate = self.plates.plate_at(id);
            format!(
                "Cell {} · plate {}, {}, {:.1} cm/yr",
                self.grid.cells[id as usize],
                self.plates.cell_plate[id as usize] + 1,
                plate.kind.name(),
                plate.speed_cm_per_year(h.to_unit(), self.camera.radius)
            )
        });
        set_text(&self.hud_cell, &cell);
        if !self.hud_dirty {
            return;
        }
        self.hud_dirty = false;
        set_text(&self.world_seed, &format!("Seed {}", self.seed));
        let mut html: String = self
            .drawing
            .polygons
            .iter()
            .enumerate()
            .map(|(i, p)| {
                format!(
                    "<li>Polygon {}: {} vertices, {} km²</li>",
                    i + 1,
                    p.exterior.len(),
                    format_km2(p.area_m2(self.camera.radius))
                )
            })
            .collect();
        if !self.drawing.current.is_empty() {
            html.push_str(&format!(
                "<li class=\"drawing\">Drawing: {} vertices</li>",
                self.drawing.current.len()
            ));
        }
        self.hud_polygons.set_inner_html(&html);
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
            self.drawing.add_vertex(p);
            self.hud_dirty = true;
        }
    }

    fn key_down(&mut self, e: &KeyboardEvent) {
        // Leave browser shortcuts alone, and keys aimed at tray controls
        // (Enter on a focused button must not also close a polygon).
        let on_control = e
            .target()
            .and_then(|t| t.dyn_into::<Element>().ok())
            .is_some_and(|el| matches!(el.tag_name().as_str(), "INPUT" | "BUTTON"));
        if on_control || e.ctrl_key() || e.meta_key() || e.alt_key() {
            return;
        }
        match e.key().as_str() {
            "Enter" => {
                self.drawing.close();
            }
            "Backspace" => self.drawing.undo_vertex(),
            "Escape" => self.drawing.cancel(),
            _ => return,
        }
        e.prevent_default();
        self.hud_dirty = true;
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

fn format_km2(m2: f64) -> String {
    let km2 = (m2 / 1e6).round() as u64;
    // Thousands separators, e.g. 1,234,567.
    let digits = km2.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
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
    let plates = Plates::generate(&grid, seed);
    let boundaries = Scene::plate_boundaries(&grid, &plates, EARTH_RADIUS_M);

    let renderer = Renderer::new(canvas.clone(), &grid, &plates, &boundaries).await?;
    let aspect = canvas.client_width().max(1) as f64 / canvas.client_height().max(1) as f64;
    let app = Rc::new(RefCell::new(App {
        canvas: canvas.clone(),
        renderer,
        camera: OrbitCamera::over(EARTH_RADIUS_M, START_VIEW, START_ALTITUDE_RADII, aspect),
        map: MapView::new(MapProjection::new(Projection::Globe, START_VIEW), aspect),
        drawing: Drawing::default(),
        grid,
        plates,
        seed,
        layers: 0,
        cursor: None,
        press: None,
        hud_cursor: element("hud-cursor")?,
        hud_cell: element("hud-cell")?,
        world_seed: element("world-seed")?,
        hud_polygons: element("hud-polygons")?,
        hud_dirty: true,
    }));
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
    listen(&canvas, "dblclick", &app, |app, _: web_sys::MouseEvent| {
        app.drawing.close();
        app.hud_dirty = true;
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
    *callback.borrow_mut() = Some(Closure::new(move |_time_ms: f64| {
        app.borrow_mut().frame();
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
