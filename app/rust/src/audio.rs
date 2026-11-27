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
const IDLE_RPM: f32 = 850.0; // lazy thumper idle: dug . . dug . . dug
const MAX_RPM: f32 = 4300.0;

/// Physical-model Bullet: each firing is a short pressure pulse that rings in a long, damped exhaust pipe
/// (the deep "dug"), plus a sharp combustion crack ("tuk"), valve ticks and a block thump. Because the pipe
/// resonates at a fixed pitch, revving only changes the *rate* of the beats, never the tone (no tape-speed
/// squeak), exactly like the real engine.
pub struct Engine {
    sr: f32,
    rpm: f32,
    thr: f32,
    fire: f32,
    e1: f32,
    e2: f32,
    pipe1: Vec<f32>,
    pipe2: Vec<f32>,
    i1: usize,
    i2: usize,
    lp1: f32,
    lp2: f32,
    crack: f32,
    nlp: f32,
    nprev: f32,
    tick: f32,
    thump: f32,
    tph: f32,
    body_lp: f32,
    wind: f32,
    out_lp: f32,
    amp: f32,
    modes: [[f32; 2]; 4], // resonator states (body / muffler shell modes)
    hunt: f32,
    nlp2: f32,
    road: f32,
    rev: [Vec<f32>; 3],
    ri: [usize; 3],
    rlp: [f32; 3],
    rng: u32,
}

impl Engine {
    pub fn new(sr: f32) -> Self {
        let d1 = (2.0 * 0.95 / 343.0 * sr) as usize; // main pipe, quarter-wave ~90 Hz
        let d2 = (2.0 * 0.36 / 343.0 * sr) as usize; // header, ~240 Hz
        Self {
            sr,
            rpm: IDLE_RPM,
            thr: 0.0,
            fire: 0.9,
            e1: 0.0,
            e2: 0.0,
            pipe1: vec![0.0; d1],
            pipe2: vec![0.0; d2],
            i1: 0,
            i2: 0,
            lp1: 0.0,
            lp2: 0.0,
            crack: 0.0,
            nlp: 0.0,
            nprev: 0.0,
            tick: 0.0,
            thump: 0.0,
            tph: 0.0,
            body_lp: 0.0,
            wind: 0.0,
            out_lp: 0.0,
            amp: 0.6,
            modes: [[0.0; 2]; 4],
            hunt: 0.0,
            nlp2: 0.0,
            road: 0.0,
            rev: [vec![0.0; (sr * 0.0297) as usize], vec![0.0; (sr * 0.0437) as usize], vec![0.0; (sr * 0.0713) as usize]],
            ri: [0; 3],
            rlp: [0.0; 3],
            rng: 0x1234_5678
        }
    }

    fn rand(&mut self) -> u32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        self.rng
    }

    fn noise(&mut self) -> f32 {
        (self.rand() >> 8) as f32 / 8_388_608.0 - 1.0
    }

    /// Fill `out` (mono, -1..1) for the given bike state.
    pub fn render(&mut self, speed: f32, throttle: f32, out: &mut [f32]) {
        let dt = out.len() as f32 / self.sr;
        // simulated gearbox: pick the gear, derive rpm from road speed (+ throttle blip, clutch revving at a standstill)
        let mut g = 0;
        while g < 4 && speed > GEARS[g] {
            g += 1;
        }
        let mut target = IDLE_RPM + (MAX_RPM - 1000.0) * (speed / GEARS[g]).min(1.0) * 0.95;
        if speed < 1.5 && throttle > 0.5 {
            target = 2800.0; // revving on the spot
        } else {
            target += throttle * 300.0;
        }
        let k = 1.0 - (-dt * if target > self.rpm { 4.5 } else { 2.5 }).exp();
        self.rpm += (target - self.rpm) * k;
        self.rpm = self.rpm.clamp(IDLE_RPM, MAX_RPM);
        self.thr += (throttle - self.thr) * (1.0 - (-dt * 7.0).exp());

        let rn = (self.rpm - IDLE_RPM) / (MAX_RPM - IDLE_RPM);
        // the heavy flywheel makes the idle "hunt" slowly: revs wander a few percent
        self.hunt += dt * 5.6;
        let wander = (self.hunt.sin() * 0.022 + (self.hunt * 0.37).sin() * 0.015) * (1.0 - rn);
        let f_fire = self.rpm * (1.0 + wander) / 120.0; // one firing per two crank turns
        let sr = self.sr;
        let d_pulse_slow = (-1.0 / (sr * 0.0035)).exp();
        let d_pulse_fast = (-1.0 / (sr * 0.0008)).exp();
        let d_crack = (-1.0 / (sr * 0.0045)).exp();
        let d_tick = (-1.0 / (sr * 0.0015)).exp();
        let d_thump = (-1.0 / (sr * 0.028)).exp();
        let wind_amp = (speed / 40.0).min(1.0);
        let g1 = 0.80 - 0.10 * rn; // exhaust rings less at high revs
        let g2 = 0.55;

        for o in out.iter_mut() {
            let prev_fire = self.fire;
            self.fire += f_fire / sr;
            let mut fired = false;
            if self.fire >= 1.0 {
                self.fire -= 1.0;
                fired = true;
            }
            // valve gear ticks at the firing point and half a cycle later
            if (prev_fire < 0.5 && self.fire >= 0.5) || fired {
                self.tick = 1.0;
            }
            if fired {
                // lumpy single-cylinder combustion: every firing a little different, a weak one now and then at idle
                let r = (self.rand() >> 8) as f32 / 16_777_216.0;
                let mut a = (0.50 + 0.50 * self.thr) * (0.80 + 0.40 * r);
                if rn < 0.2 && (self.rand() % 100) < 5 {
                    a *= 0.35;
                }
                // overrun pop: closed throttle at higher revs now and then bangs in the exhaust
                if self.thr < 0.15 && rn > 0.25 && (self.rand() % 100) < 30 {
                    a = 0.9 + 0.6 * r;
                    self.crack = 1.4;
                } else {
                    self.crack = a * 1.1;
                }
                self.e1 += a;
                self.e2 += a;
                self.thump = a;
                // slight timing jitter from the heavy flywheel
                self.fire -= (self.noise() * 0.5 + 0.5) * 0.04;
            }
            // exhaust pressure pulse: fast attack, slower decay
            self.e1 *= d_pulse_slow;
            self.e2 *= d_pulse_fast;
            let pulse = self.e1 - self.e2;

            // main pipe: quarter-wave resonator (open end inverts), damped by the muffler
            let b1 = self.pipe1[self.i1];
            self.lp1 += (b1 - self.lp1) * 0.30;
            let p1 = pulse - g1 * self.lp1;
            self.pipe1[self.i1] = p1;
            self.i1 += 1;
            if self.i1 >= self.pipe1.len() {
                self.i1 = 0;
            }
            // header pipe
            let b2 = self.pipe2[self.i2];
            self.lp2 += (b2 - self.lp2) * 0.45;
            let p2 = pulse - g2 * self.lp2;
            self.pipe2[self.i2] = p2;
            self.i2 += 1;
            if self.i2 >= self.pipe2.len() {
                self.i2 = 0;
            }
            let exhaust = 0.55 * p1 + 0.30 * p2;

            let n = self.noise();
            self.nlp += (n - self.nlp) * 0.18;
            let hp = self.nlp - self.nprev * 0.0; // band-limited noise for the crack
            let crack = (n - self.nlp) * 0.6 + hp * 0.4;
            let ck = crack * self.crack * 0.55;
            self.crack *= d_crack;
            let dn = n - self.nprev;
            self.nprev = n;
            let tick = dn * self.tick * 0.05;
            self.tick *= d_tick;

            self.tph += (46.0 + 12.0 * rn) / sr;
            if self.tph >= 1.0 {
                self.tph -= 1.0;
            }
            let thump = (self.tph * std::f32::consts::TAU).sin() * self.thump * 0.55;
            self.thump *= d_thump;

            self.wind += (n - self.wind) * 0.04;
            let wind = self.wind * wind_amp * 0.35;

            // body / muffler shell modes ring after every pulse: the "dwoom" body of the thump
            const MODE: [(f32, f32, f32); 4] = [(66.0, 0.085, 0.55), (112.0, 0.060, 0.40), (186.0, 0.040, 0.28), (305.0, 0.025, 0.16)];
            let mut modal = 0.0;
            for (m, &(fq, tau, gain)) in self.modes.iter_mut().zip(MODE.iter()) {
                let r = (-1.0 / (sr * tau)).exp();
                let c = 2.0 * r * (std::f32::consts::TAU * fq / sr).cos();
                let y0 = pulse * (1.0 - r) * 0.30 + c * m[0] - r * r * m[1];
                m[1] = m[0];
                m[0] = y0;
                modal += y0 * gain;
            }
            // intake chuff: low rumbling noise swelling with each pulse, stronger on the throttle
            self.nlp2 += (n - self.nlp2) * 0.07;
            let intake = self.nlp2 * pulse.abs() * (0.25 + 0.9 * self.thr) * 1.6;
            // tyre / road rumble with speed
            self.road += (n - self.road) * 0.015;
            let road = self.road * wind_amp * 0.5;

            let mix = exhaust * 1.0 + modal * 0.9 + ck + tick + thump + intake;
            let dry = (mix * 1.25).tanh() * 0.78;
            // small room / garage reverb for depth
            let mut wet = 0.0;
            for i in 0..3 {
                let o = self.rev[i][self.ri[i]];
                self.rlp[i] += (o - self.rlp[i]) * 0.35;
                self.rev[i][self.ri[i]] = dry + 0.42 * self.rlp[i];
                self.ri[i] += 1;
                if self.ri[i] >= self.rev[i].len() {
                    self.ri[i] = 0;
                }
                wet += o;
            }
            let y = dry + wet * 0.10 + wind + road;
            self.out_lp += (y - self.out_lp) * 0.75;
            // quiet idle would vanish on a phone speaker: lift it, and leave headroom when revving
            *o = (self.out_lp * (0.80 + 0.9 * (1.0 - rn) * (1.0 - rn))).clamp(-0.98, 0.98);
        }
        let _ = (self.body_lp, self.amp);
    }
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
