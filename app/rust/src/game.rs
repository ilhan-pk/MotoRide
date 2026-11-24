use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    /// 0 = procedural checker, 1 = flat vertex colour
    pub kind: f32,
}

pub const MAX_SPEED: f32 = 48.0; // m/s (~173 km/h)

const G: f32 = 12.0;
const CELL: f32 = 8.0; // terrain grid cell (m)
const N: i32 = 80; // terrain cells per side -> +-320 m view
const WATER_Y: f32 = -7.5;
const ROAD_HALF: f32 = 3.0;
const WALL_OFF: f32 = 4.6;
const SEG: f32 = 4.0; // road ribbon step (m)
const BLOCK: f32 = 6.0; // stone wall block length (m)
const VIEW: f32 = 300.0; // road / wall draw distance each way (m)

#[derive(Default)]
pub struct Input {
    /// -1 (left) .. 1 (right)
    pub steer: f32,
    pub throttle: bool,
    pub brake: bool,
}

// ------------------------------------------------------------------
// Procedural, infinite world
// ------------------------------------------------------------------

fn hash(ix: i32, iz: i32) -> u32 {
    let mut h = (ix as u32).wrapping_mul(0x9E37_79B1) ^ (iz as u32).wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    h
}

fn h01(ix: i32, iz: i32) -> f32 {
    (hash(ix, iz) & 0xffff) as f32 / 65535.0
}

fn vnoise(x: f32, z: f32) -> f32 {
    let xi = x.floor();
    let zi = z.floor();
    let fx = x - xi;
    let fz = z - zi;
    let u = fx * fx * (3.0 - 2.0 * fx);
    let v = fz * fz * (3.0 - 2.0 * fz);
    let (ix, iz) = (xi as i32, zi as i32);
    let a = h01(ix, iz);
    let b = h01(ix + 1, iz);
    let c = h01(ix, iz + 1);
    let d = h01(ix + 1, iz + 1);
    let top = a + (b - a) * u;
    let bot = c + (d - c) * u;
    top + (bot - top) * v
}

/// Rolling hills.
fn natural(x: f32, z: f32) -> f32 {
    let n = vnoise(x / 300.0, z / 300.0) * 0.66
        + vnoise(x / 110.0 + 17.3, z / 110.0 - 9.1) * 0.27
        + vnoise(x / 60.0 - 3.7, z / 60.0 + 5.2) * 0.07;
    (n - 0.5) * 42.0
}

/// The rally road winds along the z axis forever.
fn road_x(z: f32) -> f32 {
    45.0 * (z / 200.0).sin() + 22.0 * (z / 85.0 + 1.3).sin() + 5.0 * (z / 37.0).sin()
}

/// d(road_x)/dz
fn road_dx(z: f32) -> f32 {
    45.0 / 200.0 * (z / 200.0).cos() + 22.0 / 85.0 * (z / 85.0 + 1.3).cos() + 5.0 / 37.0 * (z / 37.0).cos()
}

/// Height of the road surface at road-parameter z (never below the lakes).
fn road_target(z: f32) -> f32 {
    natural(road_x(z), z).max(WATER_Y + 1.3)
}

pub fn terrain_h(x: f32, z: f32) -> f32 {
    let nat = natural(x, z);
    let dx = (x - road_x(z)).abs();
    if dx >= 30.0 {
        return nat;
    }
    let w = if dx <= 14.0 {
        1.0
    } else {
        let t = (dx - 14.0) / 16.0;
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    nat + (road_target(z) - nat) * w
}

/// (height, is_water)
fn ground(x: f32, z: f32) -> (f32, bool) {
    let t = terrain_h(x, z);
    if t < WATER_Y {
        (WATER_Y, true)
    } else {
        (t, false)
    }
}

fn grad(x: f32, z: f32) -> (f32, f32) {
    let e = 1.5;
    (
        (ground(x + e, z).0 - ground(x - e, z).0) / (2.0 * e),
        (ground(x, z + e).0 - ground(x, z - e).0) / (2.0 * e),
    )
}

/// Approximate perpendicular distance to the road centre line.
fn road_dist(x: f32, z: f32) -> f32 {
    let rp = road_dx(z);
    (x - road_x(z)).abs() / (1.0 + rp * rp).sqrt()
}

fn wall_exists(side: i32, blk: i32) -> bool {
    hash(blk, side * 7 + 3) % 4 != 0
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn terrain_color(x: f32, z: f32, h: f32, slope: f32) -> [f32; 3] {
    let n1 = vnoise(x / 23.0 + 7.0, z / 23.0 - 3.0);
    let n2 = vnoise(x / 6.0, z / 6.0);
    let t = (n1 * 0.7 + n2 * 0.3).clamp(0.0, 1.0);
    let mut c = [0.27 + 0.16 * t, 0.50 + 0.10 * t, 0.14 + 0.06 * t];
    let dry = ((h - 3.0) / 8.0).clamp(0.0, 1.0);
    c = mix(c, [0.55, 0.57, 0.28], dry * 0.55);
    let beach = ((WATER_Y + 1.2 - h) / 1.2).clamp(0.0, 1.0);
    c = mix(c, [0.72, 0.68, 0.48], beach);
    let rock = ((slope - 0.22) / 0.2).clamp(0.0, 1.0);
    c = mix(c, [0.45, 0.44, 0.40], rock);
    let dx = (x - road_x(z)).abs();
    let dirt = (1.0 - (dx - 4.0) / 6.0).clamp(0.0, 1.0);
    mix(c, [0.40, 0.35, 0.25], dirt * 0.8)
}

// ------------------------------------------------------------------
// Bike
// ------------------------------------------------------------------

pub struct Game {
    /// Contact point with the ground (x, ground height, z).
    pub pos: Vec3,
    /// Horizontal velocity (y unused).
    pub vel: Vec3,
    pub vy: f32,
    pub yaw: f32,
    pub speed: f32,
    pub steer: f32,
    pub lean: f32,
    pub pitch: f32,
    pub grounded: bool,
}

impl Game {
    pub fn new() -> Self {
        let x = road_x(0.0);
        let y = ground(x, 0.0).0;
        Self {
            pos: Vec3::new(x, y, 0.0),
            vel: Vec3::ZERO,
            vy: 0.0,
            yaw: road_dx(0.0).atan(),
            speed: 0.0,
            steer: 0.0,
            lean: 0.0,
            pitch: 0.0,
            grounded: true,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    pub fn eye(&self) -> Vec3 {
        Vec3::new(self.pos.x, self.pos.y + 1.25, self.pos.z)
    }

    pub fn look_dir(&self) -> Vec3 {
        let f = self.forward();
        let c = self.pitch.cos();
        Vec3::new(f.x * c, self.pitch.sin(), f.z * c)
    }

    pub fn update(&mut self, input: &Input, dt: f32) {
        let f = self.forward();
        let r = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let mut vf = self.vel.dot(f);
        let mut vr = self.vel.dot(r);

        let (_, in_water) = ground(self.pos.x, self.pos.z);
        let on_road = road_dist(self.pos.x, self.pos.z) < ROAD_HALF + 0.5;

        self.steer += (input.steer - self.steer) * (dt * 7.0).min(1.0);

        if self.grounded {
            let accel = if input.throttle { 17.0 * (1.0 - vf / MAX_SPEED).max(0.0) } else { 0.0 };
            vf += accel * dt;
            if input.brake && vf > 0.0 {
                vf = (vf - 32.0 * dt).max(0.0);
            }
            if !input.throttle && vf > 0.0 {
                vf = (vf - 1.2 * dt).max(0.0); // engine braking
            }
            let surface_drag = if in_water {
                3.0
            } else if on_road {
                0.0
            } else {
                0.12
            };
            vf -= (0.035 + surface_drag) * vf * dt;
            if vf < 0.0 {
                vf -= 2.0 * vf * dt; // rolling backwards stops quickly
            }

            // Dirt grip: lower off-road, and the bike slides when turned hard at speed.
            let grip = if in_water {
                1.0
            } else if on_road {
                4.5
            } else {
                2.6
            };
            let slide = 1.0 / (1.0 + self.steer.abs() * vf.abs() / 22.0);
            vr *= (-grip * slide * dt).exp();

            self.vel = f * vf + r * vr;

            // Gravity along the slope: downhill speeds you up, uphill slows you down.
            let (gx, gz) = grad(self.pos.x, self.pos.z);
            self.vel.x -= gx * G * 0.7 * dt;
            self.vel.z -= gz * G * 0.7 * dt;
        } else {
            self.vel = f * vf + r * vr;
        }

        let vmax = MAX_SPEED * 1.3;
        if self.vel.length() > vmax {
            self.vel = self.vel.normalize() * vmax;
        }

        // Steering: curvature drops with speed, so high speed feels stable.
        let curv = self.steer * 0.085 / (1.0 + vf * vf / 1100.0);
        let ctrl = if self.grounded { 1.0 } else { 0.15 };
        self.yaw -= curv * vf * ctrl * dt;

        self.pos.x += self.vel.x * dt;
        self.pos.z += self.vel.z * dt;

        self.wall_collision();

        // Vertical motion: follow the ground, or fly off crests.
        let (g1, _) = ground(self.pos.x, self.pos.z);
        self.vy -= G * dt;
        let y_new = self.pos.y + self.vy * dt;
        if y_new <= g1 {
            if self.grounded {
                self.vy = ((g1 - self.pos.y) / dt).clamp(-25.0, 25.0);
            } else {
                if -self.vy > 8.0 {
                    self.vel = self.vel * 0.88; // hard landing
                }
                self.vy = 0.0;
            }
            self.pos.y = g1;
            self.grounded = true;
        } else {
            self.pos.y = y_new;
            self.grounded = false;
        }

        self.speed = self.vel.length();

        let target_lean = self.steer * (self.speed / 22.0).min(1.0);
        self.lean += (target_lean - self.lean) * (dt * 5.0).min(1.0);

        let f2 = self.forward();
        let (gx, gz) = grad(self.pos.x, self.pos.z);
        let target_pitch = if self.grounded {
            (gx * f2.x + gz * f2.z).atan()
        } else {
            self.vy.atan2(vf.abs().max(3.0)) * 0.6
        };
        self.pitch += (target_pitch.clamp(-0.6, 0.6) - self.pitch) * (dt * 5.0).min(1.0);
    }

    /// Solid stone walls along the road (they have gaps, so you can leave the track).
    fn wall_collision(&mut self) {
        let z = self.pos.z;
        let rp = road_dx(z);
        let s = (1.0 + rp * rp).sqrt();
        let d = (self.pos.x - road_x(z)) / s;
        let ad = d.abs();
        if ad < 3.9 || ad > 5.4 {
            return;
        }
        let (gh, _) = ground(self.pos.x, self.pos.z);
        if self.pos.y - gh > 0.9 {
            return; // flying over it
        }
        let side: i32 = if d > 0.0 { 1 } else { -1 };
        let sf = side as f32;
        let zr = z + d * rp / s;
        let blk = (zr / BLOCK).floor() as i32;
        if !wall_exists(side, blk) {
            return;
        }
        let from_road = ad < WALL_OFF;
        let target = if from_road { 3.85 } else { 5.5 };
        self.pos.x = road_x(z) + sf * target * s;

        let n = Vec3::new(1.0, 0.0, -rp) / s * sf;
        let vn = self.vel.dot(n);
        let toward = if from_road { vn > 0.0 } else { vn < 0.0 };
        if toward {
            self.vel -= n * (vn * 1.3);
            self.vel = self.vel * 0.75;
        }
    }

    /// Terrain, water, road and stone walls, in world space.
    pub fn build_world(&self, out: &mut Vec<Vertex>) {
        // ---- terrain on a fixed world grid (no vertex swimming) ----
        let half = (N / 2) as f32;
        let x0 = (self.pos.x / CELL).floor() * CELL - half * CELL;
        let z0 = (self.pos.z / CELL).floor() * CELL - half * CELL;
        let n1 = (N + 1) as usize;
        let mut hs = vec![0f32; n1 * n1];
        for j in 0..n1 {
            for i in 0..n1 {
                hs[j * n1 + i] = terrain_h(x0 + i as f32 * CELL, z0 + j as f32 * CELL);
            }
        }
        let at = |i: usize, j: usize| hs[j.min(n1 - 1) * n1 + i.min(n1 - 1)];
        let sun = Vec3::new(0.45, 0.80, 0.40).normalize();
        let mut cols = vec![[0f32; 3]; n1 * n1];
        for j in 0..n1 {
            for i in 0..n1 {
                let hl = at(i.saturating_sub(1), j);
                let hr = at(i + 1, j);
                let hd = at(i, j.saturating_sub(1));
                let hu = at(i, j + 1);
                let nrm = Vec3::new(hl - hr, 2.0 * CELL, hd - hu).normalize();
                let slope = ((hl - hr) * (hl - hr) + (hd - hu) * (hd - hu)).sqrt() / (2.0 * CELL);
                let x = x0 + i as f32 * CELL;
                let z = z0 + j as f32 * CELL;
                let base = terrain_color(x, z, hs[j * n1 + i], slope);
                let shade = (0.50 + 0.60 * nrm.dot(sun).max(0.0)).min(1.1);
                cols[j * n1 + i] = [base[0] * shade, base[1] * shade, base[2] * shade];
            }
        }
        let vert = |i: usize, j: usize| Vertex {
            pos: [x0 + i as f32 * CELL, hs[j * n1 + i], z0 + j as f32 * CELL],
            color: cols[j * n1 + i],
            kind: 1.0,
        };
        for j in 0..N as usize {
            for i in 0..N as usize {
                let a = vert(i, j);
                let b = vert(i + 1, j);
                let c = vert(i, j + 1);
                let d = vert(i + 1, j + 1);
                out.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }

        // ---- lakes ----
        let wc = [0.12, 0.30, 0.45];
        let (wx, wz, ws) = (self.pos.x, self.pos.z, 900.0);
        let wq = [
            [wx - ws, WATER_Y, wz - ws],
            [wx + ws, WATER_Y, wz - ws],
            [wx + ws, WATER_Y, wz + ws],
            [wx - ws, WATER_Y, wz + ws],
        ];
        for &i in &[0usize, 1, 2, 0, 2, 3] {
            out.push(Vertex { pos: wq[i], color: wc, kind: 1.0 });
        }

        // ---- road ribbon ----
        let zr0 = (self.pos.z / SEG).floor() * SEG - VIEW;
        let nseg = (2.0 * VIEW / SEG) as usize;
        struct Row {
            rx: f32,
            rp: f32,
            s: f32,
            y: f32,
            z: f32,
        }
        let mut rows: Vec<Row> = Vec::with_capacity(nseg + 1);
        for k in 0..=nseg {
            let z = zr0 + k as f32 * SEG;
            let rp = road_dx(z);
            rows.push(Row { rx: road_x(z), rp, s: (1.0 + rp * rp).sqrt(), y: road_target(z), z });
        }
        let pt = |row: &Row, off: f32, lift: f32| -> [f32; 3] {
            [row.rx + off / row.s, row.y + lift, row.z - off * row.rp / row.s]
        };
        let strips: [(f32, f32, f32, [f32; 3]); 5] = [
            (-3.8, 3.8, 0.10, [0.40, 0.35, 0.26]),  // dirt shoulder
            (-3.0, 3.0, 0.14, [0.58, 0.55, 0.50]),  // gravel road
            (-1.5, -0.7, 0.16, [0.46, 0.43, 0.39]), // wheel tracks
            (0.7, 1.5, 0.16, [0.46, 0.43, 0.39]),
            (-0.15, 0.15, 0.16, [0.52, 0.49, 0.44]), // centre crown
        ];
        for k in 0..nseg {
            let (a, b) = (&rows[k], &rows[k + 1]);
            let tint = 0.94 + 0.12 * h01((a.z / SEG).floor() as i32, 91);
            for &(o0, o1, lift, col) in strips.iter() {
                let c = [col[0] * tint, col[1] * tint, col[2] * tint];
                let p = [pt(a, o0, lift), pt(a, o1, lift), pt(b, o1, lift), pt(b, o0, lift)];
                for &i in &[0usize, 1, 2, 0, 2, 3] {
                    out.push(Vertex { pos: p[i], color: c, kind: 1.0 });
                }
            }
        }

        // ---- stone walls ----
        let blk0 = ((self.pos.z - VIEW) / BLOCK).floor() as i32;
        let blk1 = ((self.pos.z + VIEW) / BLOCK).floor() as i32;
        for side in [-1i32, 1] {
            let sf = side as f32;
            for blk in blk0..=blk1 {
                if !wall_exists(side, blk) {
                    continue;
                }
                let zr = (blk as f32 + 0.5) * BLOCK;
                let rp = road_dx(zr);
                let s = (1.0 + rp * rp).sqrt();
                let cx = road_x(zr) + sf * WALL_OFF / s;
                let cz = zr - sf * WALL_OFF * rp / s;
                let gy = road_target(zr);
                let m = Mat4::from_translation(Vec3::new(cx, gy + 0.40, cz)) * Mat4::from_rotation_y(rp.atan());
                let v = h01(blk, side + 50);
                let col = [0.48 + 0.10 * v, 0.47 + 0.10 * v, 0.43 + 0.09 * v];
                push_box(out, m, Vec3::ZERO, Vec3::new(0.28, 0.55, BLOCK * 0.5 * 0.97), col);
            }
        }
    }

    /// Handlebar, stem, red tank and speed bar, in camera space (view = identity).
    pub fn build_cockpit(&self, out: &mut Vec<Vertex>) {
        let dark = [0.09, 0.10, 0.11];
        let red = [0.72, 0.23, 0.20];
        let base = Mat4::from_translation(Vec3::new(0.0, -0.40, -1.1))
            * Mat4::from_rotation_y(-self.steer * 0.35)
            * Mat4::from_rotation_x(0.25);
        push_box(out, base, Vec3::ZERO, Vec3::new(0.55, 0.03, 0.035), dark); // bar
        push_box(out, base, Vec3::new(0.0, -0.13, 0.0), Vec3::new(0.04, 0.13, 0.04), dark); // stem

        let tank = Mat4::from_translation(Vec3::new(0.0, -0.64, -1.15)) * Mat4::from_rotation_x(0.12);
        push_box(out, tank, Vec3::ZERO, Vec3::new(0.38, 0.05, 0.55), red);

        let w = 0.01 + 0.36 * (self.speed / MAX_SPEED).min(1.0);
        push_box(
            out,
            tank,
            Vec3::new(-0.36 + w, 0.06, 0.25),
            Vec3::new(w, 0.008, 0.012),
            [0.2, 0.9, 0.3],
        );
    }
}

pub fn push_box(out: &mut Vec<Vertex>, m: Mat4, c: Vec3, h: Vec3, color: [f32; 3]) {
    const FACES: [(f32, [[f32; 3]; 4]); 6] = [
        (0.8, [[1., -1., -1.], [1., 1., -1.], [1., 1., 1.], [1., -1., 1.]]),
        (0.6, [[-1., -1., 1.], [-1., 1., 1.], [-1., 1., -1.], [-1., -1., -1.]]),
        (1.0, [[-1., 1., -1.], [-1., 1., 1.], [1., 1., 1.], [1., 1., -1.]]),
        (0.4, [[-1., -1., 1.], [-1., -1., -1.], [1., -1., -1.], [1., -1., 1.]]),
        (0.9, [[-1., -1., 1.], [1., -1., 1.], [1., 1., 1.], [-1., 1., 1.]]),
        (0.7, [[1., -1., -1.], [-1., -1., -1.], [-1., 1., -1.], [1., 1., -1.]]),
    ];
    for (shade, corners) in FACES.iter() {
        let col = [color[0] * shade, color[1] * shade, color[2] * shade];
        for &i in &[0usize, 1, 2, 0, 2, 3] {
            let k = corners[i];
            let p = m.transform_point3(c + Vec3::new(k[0] * h.x, k[1] * h.y, k[2] * h.z));
            out.push(Vertex { pos: p.to_array(), color: col, kind: 1.0 });
        }
    }
}
