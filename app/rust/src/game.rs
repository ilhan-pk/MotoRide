use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    /// 0 = procedural ground, 1 = flat colour
    pub kind: f32,
}

pub const CELL: f32 = 30.0;
pub const RADIUS: i32 = 12;
pub const MAX_SPEED: f32 = 45.0; // m/s (~162 km/h)

#[derive(Default)]
pub struct Input {
    /// -1 (left) .. 1 (right)
    pub steer: f32,
    pub throttle: bool,
    pub brake: bool,
}

pub struct Game {
    pub pos: Vec3,
    pub yaw: f32,
    pub speed: f32,
    pub steer: f32,
    pub lean: f32,
}

fn hash(ix: i32, iz: i32) -> u32 {
    let mut h = (ix as u32).wrapping_mul(0x9E37_79B1) ^ (iz as u32).wrapping_mul(0x85EB_CA6B);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h
}

/// Deterministic roadside post for a grid cell: (x, z, is_orange)
fn obstacle(ix: i32, iz: i32) -> Option<(f32, f32, bool)> {
    if ix.abs() <= 1 && iz.abs() <= 1 {
        return None; // clear start area
    }
    let h = hash(ix, iz);
    if h % 3 != 0 {
        return None;
    }
    let ox = ((h >> 8) & 0xff) as f32 / 255.0;
    let oz = ((h >> 16) & 0xff) as f32 / 255.0;
    Some((
        (ix as f32 + 0.15 + ox * 0.7) * CELL,
        (iz as f32 + 0.15 + oz * 0.7) * CELL,
        (h >> 24) & 1 == 1,
    ))
}

impl Game {
    pub fn new() -> Self {
        Self { pos: Vec3::new(0.0, 1.1, 0.0), yaw: 0.0, speed: 0.0, steer: 0.0, lean: 0.0 }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    pub fn update(&mut self, input: &Input, dt: f32) {
        let accel = if input.throttle { 18.0 * (1.0 - self.speed / MAX_SPEED).max(0.0) } else { 0.0 };
        let brake = if input.brake { 35.0 } else { 0.0 };
        let drag = 0.05 * self.speed + if input.throttle { 0.0 } else { 1.5 };
        self.speed = (self.speed + (accel - brake - drag) * dt).clamp(0.0, MAX_SPEED);

        self.steer += (input.steer - self.steer) * (dt * 6.0).min(1.0);
        let grip = self.speed / (self.speed + 12.0);
        self.yaw -= self.steer * 1.2 * grip * dt;
        let target_lean = self.steer * (self.speed / 25.0).min(1.0);
        self.lean += (target_lean - self.lean) * (dt * 5.0).min(1.0);

        let fwd = self.forward();
        self.pos += fwd * self.speed * dt;

        // collisions with posts
        let cx = (self.pos.x / CELL).floor() as i32;
        let cz = (self.pos.z / CELL).floor() as i32;
        for ix in cx - 1..=cx + 1 {
            for iz in cz - 1..=cz + 1 {
                if let Some((ox, oz, _)) = obstacle(ix, iz) {
                    let dx = self.pos.x - ox;
                    let dz = self.pos.z - oz;
                    if dx * dx + dz * dz < 1.0 {
                        self.speed *= 0.25;
                        self.pos -= fwd * 1.2;
                    }
                }
            }
        }
    }

    /// Ground + roadside posts, in world space.
    pub fn build_world(&self, out: &mut Vec<Vertex>) {
        let s = 2500.0;
        let (cx, cz) = (self.pos.x, self.pos.z);
        let q = [
            [cx - s, 0.0, cz - s],
            [cx + s, 0.0, cz - s],
            [cx + s, 0.0, cz + s],
            [cx - s, 0.0, cz + s],
        ];
        for &i in &[0usize, 1, 2, 0, 2, 3] {
            out.push(Vertex { pos: q[i], color: [0.0; 3], kind: 0.0 });
        }

        let ix0 = (self.pos.x / CELL).floor() as i32;
        let iz0 = (self.pos.z / CELL).floor() as i32;
        for ix in ix0 - RADIUS..=ix0 + RADIUS {
            for iz in iz0 - RADIUS..=iz0 + RADIUS {
                if let Some((x, z, orange)) = obstacle(ix, iz) {
                    let col = if orange { [0.95, 0.45, 0.10] } else { [0.92, 0.92, 0.92] };
                    push_box(out, Mat4::IDENTITY, Vec3::new(x, 1.25, z), Vec3::new(0.3, 1.25, 0.3), col);
                }
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

        let w = 0.01 + 0.36 * (self.speed / MAX_SPEED);
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
