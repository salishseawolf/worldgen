use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

use crate::renderer::Renderer;

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

async fn run() -> Result<(), String> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or("no document")?;
    let canvas: HtmlCanvasElement = document
        .get_element_by_id("world")
        .ok_or("missing #world canvas")?
        .dyn_into()
        .map_err(|_| "#world is not a canvas")?;

    let window = web_sys::window().ok_or("no window")?;
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

    let renderer = Renderer::new(canvas).await?;
    log::info!("renderer ready");
    animate(renderer);
    Ok(())
}

/// Drive `render` from requestAnimationFrame. The closure re-schedules itself.
fn animate(mut renderer: Renderer) {
    let callback: FrameCallback = Rc::new(RefCell::new(None));
    let next = callback.clone();
    *callback.borrow_mut() = Some(Closure::new(move |time_ms: f64| {
        renderer.render(time_ms);
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
