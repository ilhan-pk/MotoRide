mod game;
mod renderer;

use std::{sync::Arc, time::Instant};

use winit::{
    application::ApplicationHandler,
    event::{TouchPhase, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// Touch layout (landscape):
///   left half            -> drag left/right to steer
///   right half, upper    -> throttle
///   right half, bottom   -> brake
#[derive(Default)]
struct Touches {
    steer_id: Option<u64>,
    steer_x0: f64,
    throttle_id: Option<u64>,
    brake_id: Option<u64>,
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<renderer::Renderer>,
    game: game::Game,
    input: game::Input,
    touches: Touches,
    last: Instant,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            game: game::Game::new(),
            input: game::Input::default(),
            touches: Touches::default(),
            last: Instant::now(),
        }
    }

    fn touch(&mut self, id: u64, phase: TouchPhase, x: f64, y: f64) {
        let Some(win) = &self.window else { return };
        let size = win.inner_size();
        let (w, h) = (size.width as f64, size.height as f64);
        let t = &mut self.touches;
        match phase {
            TouchPhase::Started => {
                if x < w * 0.5 {
                    t.steer_id = Some(id);
                    t.steer_x0 = x;
                } else if y > h * 0.65 {
                    t.brake_id = Some(id);
                } else {
                    t.throttle_id = Some(id);
                }
            }
            TouchPhase::Moved => {
                if t.steer_id == Some(id) {
                    self.input.steer = (((x - t.steer_x0) / (w * 0.12)) as f32).clamp(-1.0, 1.0);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if t.steer_id == Some(id) {
                    t.steer_id = None;
                    self.input.steer = 0.0;
                }
                if t.throttle_id == Some(id) {
                    t.throttle_id = None;
                }
                if t.brake_id == Some(id) {
                    t.brake_id = None;
                }
            }
        }
        self.input.throttle = t.throttle_id.is_some();
        self.input.brake = t.brake_id.is_some();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes().with_title("MotoRide"))
                .expect("window"),
        );
        let size = window.inner_size();
        let r = pollster::block_on(renderer::Renderer::new(
            window.clone(),
            size.width.max(1),
            size.height.max(1),
        ));
        self.renderer = Some(r);
        self.window = Some(window);
        self.last = Instant::now();
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        // Android destroys the surface here; recreate in resumed().
        self.renderer = None;
        self.window = None;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(s) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(s.width, s.height);
                }
            }
            WindowEvent::Touch(t) => self.touch(t.id, t.phase, t.location.x, t.location.y),
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last).as_secs_f32().min(0.05);
                self.last = now;
                self.game.update(&self.input, dt);
                if let Some(r) = &mut self.renderer {
                    r.render(&self.game);
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

pub fn run(event_loop: EventLoop<()>) {
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("event loop");
}

#[cfg(target_os = "android")]
#[no_mangle]
fn android_main(app: winit::platform::android::activity::AndroidApp) {
    use winit::platform::android::EventLoopBuilderExtAndroid;
    android_logger::init_once(android_logger::Config::default().with_max_level(log::LevelFilter::Info));
    let event_loop = EventLoop::builder().with_android_app(app).build().expect("event loop");
    run(event_loop);
}
