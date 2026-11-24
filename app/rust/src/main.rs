// Desktop test runner (not used on Android).
fn main() {
    let event_loop = winit::event_loop::EventLoop::new().unwrap();
    motoride::run(event_loop);
}
