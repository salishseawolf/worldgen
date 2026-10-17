use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::{Element, HtmlCanvasElement, KeyboardEvent, PointerEvent, WheelEvent};

use crate::camera::{OrbitCamera, ray_sphere};
use crate::draw::Drawing;
use crate::geo::{EARTH_RADIUS_M, LonLat};
use crate::renderer::{Globals, Renderer};
use crate::scene::Scene;
use glam::DVec3;

/// Start over the Salish Sea.
const START_VIEW: LonLat = LonLat::new(-123.0, 48.5);
const START_ALTITUDE_RADII: f64 = 2.0;
/// A press that moves less than this (CSS px) is a click, not a drag.
const CLICK_SLOP_PX: f64 = 4.0;
const LINE_WIDTH_CSS_PX: f64 = 2.0;

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
    /// keeps it under the cursor. None if the press started off the globe.
    grabbed: Option<DVec3>,
}

struct App {
    canvas: HtmlCanvasElement,
    renderer: Renderer,
    camera: OrbitCamera,
    drawing: Drawing,
    /// Cursor in CSS pixels relative to the canvas, if over it.
    cursor: Option<(f64, f64)>,
    press: Option<Press>,
    hud_cursor: Element,
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

    /// Unit vector of the surface point under a CSS-pixel position.
    fn surface_at(&self, pos: (f64, f64)) -> Option<DVec3> {
        let (x, y) = self.ndc(pos);
        ray_sphere(&self.camera.ray(x, y), self.camera.radius).map(DVec3::normalize)
    }

    fn frame(&mut self) {
        let (width, height) = self.renderer.sync_size();
        self.camera.aspect = width as f64 / height as f64;
        let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());

        let hover = self.hover();
        let globals = Globals {
            view_proj: self.camera.view_proj().as_mat4().to_cols_array_2d(),
            light_dir: self.camera.light_dir().as_vec3().extend(0.0).to_array(),
            params: [
                width as f32,
                height as f32,
                self.camera.radius as f32,
                (LINE_WIDTH_CSS_PX * dpr) as f32,
            ],
        };
        let scene = Scene::build(
            &self.drawing,
            hover,
            self.camera.radius,
            self.camera.altitude,
        );
        self.renderer.render(&globals, &scene);
        self.update_hud(hover);
    }

    fn update_hud(&mut self, hover: Option<LonLat>) {
        let text = hover.map_or_else(|| "Off the globe".to_string(), LonLat::display);
        if self.hud_cursor.text_content().as_deref() != Some(text.as_str()) {
            self.hud_cursor.set_text_content(Some(&text));
        }
        if !self.hud_dirty {
            return;
        }
        self.hud_dirty = false;
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
            if press.dragging {
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
        self.camera.zoom(px * 0.001);
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

    let renderer = Renderer::new(canvas.clone()).await?;
    let aspect = canvas.client_width().max(1) as f64 / canvas.client_height().max(1) as f64;
    let app = Rc::new(RefCell::new(App {
        canvas: canvas.clone(),
        renderer,
        camera: OrbitCamera::over(EARTH_RADIUS_M, START_VIEW, START_ALTITUDE_RADII, aspect),
        drawing: Drawing::default(),
        cursor: None,
        press: None,
        hud_cursor: element("hud-cursor")?,
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
