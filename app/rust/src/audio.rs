//! Procedural single-cylinder "thumper" engine sound (Royal Enfield Bullet style), generated live from the
//! bike's speed and throttle. A 4-stroke single fires once every two crank turns (rpm / 120 Hz), which gives
//! the slow, heavy "dug-dug-dug" at idle that speeds up into a deep drone as the revs rise.

use std::sync::atomic::{AtomicU32, Ordering};

static SPEED: AtomicU32 = AtomicU32::new(0);
static THROTTLE: AtomicU32 = AtomicU32::new(0);

/// Called every frame by the game loop; read by the audio thread.
pub fn set_state(speed: f32, throttle: bool) {
    SPEED.store(speed.max(0.0).to_bits(), Ordering::Relaxed);
    THROTTLE.store((if throttle { 1.0f32 } else { 0.0 }).to_bits(), Ordering::Relaxed);
}

fn state() -> (f32, f32) {
    (f32::from_bits(SPEED.load(Ordering::Relaxed)), f32::from_bits(THROTTLE.load(Ordering::Relaxed)))
}

/// Top speed (m/s) of each of the five gears.
const GEARS: [f32; 5] = [9.0, 16.0, 24.0, 34.0, 50.0];
const IDLE_RPM: f32 = 945.0;
const MAX_RPM: f32 = 4200.0;

/// Seamless loops cut from a real Royal Enfield Bullet 350 recording, each tagged with the rpm it was recorded at.
const LOOP_RPM: [f32; 3] = [945.0, 2105.0, 2727.0];
static LOOP_DATA: [&[u8]; 3] = [include_bytes!("eng0.bin"), include_bytes!("eng1.bin"), include_bytes!("eng2.bin")];

pub struct Engine {
    sr: f32,
    loops: Vec<Vec<f32>>,
    pos: [f32; 3],
    rpm: f32,
    thr: f32,
    nl: f32,
    wind: f32,
    rng: u32,
}

impl Engine {
    pub fn new(sr: f32) -> Self {
        let loops = LOOP_DATA
            .iter()
            .map(|d| d.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect())
            .collect();
        Self { sr, loops, pos: [0.0, 123.0, 457.0], rpm: IDLE_RPM, thr: 0.0, nl: 0.0, wind: 0.0, rng: 0x1234_5678 }
    }

    fn noise(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / 8_388_608.0 - 1.0
    }

    /// Fill `out` (mono, -1..1) for the given bike state.
    pub fn render(&mut self, speed: f32, throttle: f32, out: &mut [f32]) {
        let dt = out.len() as f32 / self.sr;
        // simulated gearbox: pick the gear, derive rpm from road speed (+ throttle blip, clutch revving at a standstill)
        let mut g = 0;
        while g < 4 && speed > GEARS[g] {
            g += 1;
        }
        let mut target = IDLE_RPM + (MAX_RPM - 1100.0) * (speed / GEARS[g]).min(1.0) * 0.95;
        if speed < 1.5 && throttle > 0.5 {
            target = 2400.0; // revving on the spot
        } else {
            target += throttle * 400.0;
        }
        let k = 1.0 - (-dt * if target > self.rpm { 5.0 } else { 3.0 }).exp();
        self.rpm += (target - self.rpm) * k;
        self.rpm = self.rpm.clamp(IDLE_RPM, MAX_RPM);
        self.thr += (throttle - self.thr) * (1.0 - (-dt * 6.0).exp());

        // crossfade (equal power) between the two loops whose rpm brackets the engine rpm
        let r = self.rpm;
        let (i0, i1, f) = if r <= LOOP_RPM[0] {
            (0, 0, 0.0)
        } else if r >= LOOP_RPM[2] {
            (2, 2, 0.0)
        } else if r < LOOP_RPM[1] {
            (0, 1, (r - LOOP_RPM[0]) / (LOOP_RPM[1] - LOOP_RPM[0]))
        } else {
            (1, 2, (r - LOOP_RPM[1]) / (LOOP_RPM[2] - LOOP_RPM[1]))
        };
        let (w0, w1) = ((f * std::f32::consts::FRAC_PI_2).cos(), (f * std::f32::consts::FRAC_PI_2).sin());
        let (r0, r1) = (r / LOOP_RPM[i0], r / LOOP_RPM[i1]);
        // louder on the throttle, a bit softer on the overrun; a touch louder as the revs climb
        let level = (0.62 + 0.30 * self.thr) * (0.85 + 0.15 * (r - IDLE_RPM) / (MAX_RPM - IDLE_RPM));
        let wind_amp = (speed / 40.0).min(1.0);

        let (l0, l1) = (self.loops[i0].len(), self.loops[i1].len());
        let (mut p0, mut p1) = (self.pos[i0], self.pos[i1]);
        for o in out.iter_mut() {
            let mut s = 0.0;
            if w0 > 0.001 {
                s += w0 * sample(&self.loops[i0], p0);
            }
            if i1 != i0 && w1 > 0.001 {
                s += w1 * sample(&self.loops[i1], p1);
            }
            p0 += r0;
            if p0 >= l0 as f32 {
                p0 -= l0 as f32;
            }
            p1 += r1;
            if p1 >= l1 as f32 {
                p1 -= l1 as f32;
            }
            let n = self.noise();
            self.wind += (n - self.wind) * 0.04;
            let y = s * level + self.wind * wind_amp * 0.30;
            *o = (y * 1.1).tanh() * 0.9;
        }
        self.pos[i0] = p0;
        self.pos[i1] = p1;
        let _ = self.nl;
    }
}

#[inline]
fn sample(buf: &[f32], p: f32) -> f32 {
    let i = p as usize;
    let f = p - i as f32;
    let a = buf[i];
    let b = buf[if i + 1 >= buf.len() { 0 } else { i + 1 }];
    a + (b - a) * f
}

// ---------------------------------------------------------------------------------------------
// Android: AAudio output stream (low latency, float mono) driven by `Engine`.
// ---------------------------------------------------------------------------------------------

#[cfg(target_os = "android")]
pub struct AudioOut {
    _stream: Option<ndk::audio::AudioStream>,
}

#[cfg(target_os = "android")]
impl AudioOut {
    pub fn start() -> Self {
        use ndk::audio::{AudioCallbackResult, AudioDirection, AudioFormat, AudioPerformanceMode, AudioStreamBuilder};
        let mut engine = Engine::new(44100.0);
        let built = AudioStreamBuilder::new().and_then(|b| {
            b.direction(AudioDirection::Output)
                .format(AudioFormat::PCM_Float)
                .channel_count(1)
                .sample_rate(44100)
                .performance_mode(AudioPerformanceMode::LowLatency)
                .data_callback(Box::new(move |_stream, data, frames| {
                    let out = unsafe { std::slice::from_raw_parts_mut(data as *mut f32, frames.max(0) as usize) };
                    let (speed, thr) = state();
                    engine.render(speed, thr, out);
                    AudioCallbackResult::Continue
                }))
                .open_stream()
        });
        match built {
            Ok(stream) => {
                if let Err(e) = stream.request_start() {
                    log::warn!("audio start failed: {e:?}");
                }
                Self { _stream: Some(stream) }
            }
            Err(e) => {
                log::warn!("audio open failed: {e:?}");
                Self { _stream: None }
            }
        }
    }
}

#[cfg(not(target_os = "android"))]
pub struct AudioOut;

#[cfg(not(target_os = "android"))]
impl AudioOut {
    pub fn start() -> Self {
        let _ = state();
        Self
    }
}
