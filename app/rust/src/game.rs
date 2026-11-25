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
/// World is built and cached in square chunks of this size (m).
pub const CHUNK: f32 = 80.0;

const G: f32 = 12.0;
const CELL: f32 = 10.0; // terrain grid cell (m)
const CPC: i32 = 8; // cells per chunk side (CHUNK / CELL)
const TREE_CELL: f32 = 16.0;
const TPC: i32 = 5; // tree cells per chunk side (CHUNK / TREE_CELL)
const ROAD_HALF: f32 = 5.5;
const RAIL_OFF: f32 = 12.0;
const SEG: f32 = 3.0; // road ribbon step (m)
const BLOCK: f32 = 6.0; // guard-rail block length (m)
const LAMP: f32 = 48.0;
const SIGN: f32 = 24.0;
/// Set to false to remove all trees.
const TREES: bool = true;
/// 1.0 = full forest. Lower (e.g. 0.5) if the FPS counter shows a slow phone.
const TREE_DENSITY: f32 = 1.0;

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

/// Big hills only (the road follows these, so the track stays smooth).
fn base_h(x: f32, z: f32) -> f32 {
    (vnoise(x / 240.0, z / 240.0) - 0.5) * 72.0
}

/// Steep rolling hills: big hills + medium ridges + small bumps.
fn natural(x: f32, z: f32) -> f32 {
    base_h(x, z)
        + (vnoise(x / 95.0 + 17.3, z / 95.0 - 9.1) - 0.5) * 22.0
        + (vnoise(x / 40.0 - 3.7, z / 40.0 + 5.2) - 0.5) * 4.5
}

/// The race track winds along the z axis forever (S-bends).
pub fn road_x(z: f32) -> f32 {
    40.0 * (z / 150.0).sin() + 16.0 * (z / 62.0 + 1.3).sin() + 4.0 * (z / 31.0).sin()
}

pub fn road_dx(z: f32) -> f32 {
    40.0 / 150.0 * (z / 150.0).cos() + 16.0 / 62.0 * (z / 62.0 + 1.3).cos() + 4.0 / 31.0 * (z / 31.0).cos()
}

fn road_ddx(z: f32) -> f32 {
    -40.0 / (150.0 * 150.0) * (z / 150.0).sin()
        - 16.0 / (62.0 * 62.0) * (z / 62.0 + 1.3).sin()
        - 4.0 / (31.0 * 31.0) * (z / 31.0).sin()
}

fn road_target(z: f32) -> f32 {
    base_h(road_x(z), z)
}

/// Road parameter (z along the centre line) of the centre-line point nearest to (x, z).
fn road_param(x: f32, z: f32) -> f32 {
    let mut zr = z;
    for _ in 0..3 {
        zr = z + (x - road_x(zr)) * road_dx(zr);
    }
    zr
}

pub fn terrain_h(x: f32, z: f32) -> f32 {
    let nat = natural(x, z);
    // cheap reject for points far from the road
    if (x - road_x(z)).abs() >= 80.0 {
        return nat;
    }
    let zr = road_param(x, z);
    let dxr = x - road_x(zr);
    let dzr = z - zr;
    let dist = (dxr * dxr + dzr * dzr).sqrt();
    if dist >= 60.0 {
        return nat;
    }
    // Flat across the road (height depends only on the nearest road point), blending into the hills.
    let w = if dist <= 30.0 {
        1.0
    } else {
        let t = (dist - 30.0) / 30.0;
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    nat + (road_target(zr) - nat) * w
}

/// Perpendicular distance to the road centre line (accurate, via the nearest road point).
fn road_off(x: f32, z: f32) -> f32 {
    if (x - road_x(z)).abs() >= 90.0 {
        return 99.0;
    }
    let zr = road_param(x, z);
    let dx = x - road_x(zr);
    let dz = z - zr;
    (dx * dx + dz * dz).sqrt()
}

fn smooth(e0: f32, e1: f32, v: f32) -> f32 {
    let t = ((v - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Visual terrain height: like `terrain_h`, but sunk under the road bed so the coarse 10 m grass mesh
/// can never poke through the asphalt / sand (the physics ground stays `terrain_h`).
fn mesh_h(x: f32, z: f32) -> f32 {
    let d = road_off(x, z);
    terrain_h(x, z) - 0.45 * (1.0 - smooth(8.0, 17.0, d))
}

fn grad(x: f32, z: f32) -> (f32, f32) {
    let e = 1.5;
    (
        (terrain_h(x + e, z) - terrain_h(x - e, z)) / (2.0 * e),
        (terrain_h(x, z + e) - terrain_h(x, z - e)) / (2.0 * e),
    )
}

/// Approximate perpendicular distance to the road centre line.
fn road_dist(x: f32, z: f32) -> f32 {
    let rp = road_dx(z);
    (x - road_x(z)).abs() / (1.0 + rp * rp).sqrt()
}

fn rail_exists(side: i32, blk: i32) -> bool {
    hash(blk, side * 7 + 3) % 4 != 0
}

/// Deterministic tree for a world tree-cell: (x, z, scale).
fn tree_at(ix: i32, iz: i32) -> Option<(f32, f32, f32)> {
    if !TREES {
        return None;
    }
    let h = hash(ix * 3 + 11, iz * 5 + 7);
    let jx = ((h >> 8) & 0xff) as f32 / 255.0;
    let jz = ((h >> 16) & 0xff) as f32 / 255.0;
    let x = (ix as f32 + 0.1 + 0.8 * jx) * TREE_CELL;
    let z = (iz as f32 + 0.1 + 0.8 * jz) * TREE_CELL;
    let dd = road_dist(x, z);
    if dd < RAIL_OFF + 2.5 {
        return None;
    }
    if near_building(x, z, 3.0) {
        return None;
    }
    let p = (if dd < 70.0 { 0.50 } else { 0.16 }) * TREE_DENSITY;
    if ((h & 0xff) as f32 / 255.0) > p {
        return None;
    }
    let s = 0.8 + 0.5 * (((h >> 24) & 0xff) as f32 / 255.0);
    Some((x, z, s))
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// Muddy patches away from the road (0..1).
fn mud_amount(x: f32, z: f32) -> f32 {
    if road_off(x, z) < 14.0 {
        return 0.0;
    }
    smooth(0.66, 0.78, vnoise(x / 26.0 + 40.0, z / 26.0 - 12.0))
}

fn terrain_color(x: f32, z: f32, slope: f32) -> [f32; 3] {
    let patch = vnoise(x / 45.0 + 3.0, z / 45.0 - 8.0);
    let t = ((patch - 0.35) / 0.3).clamp(0.0, 1.0);
    let mut c = mix([0.50, 0.78, 0.26], [0.62, 0.87, 0.34], t);
    let fine = vnoise(x / 9.0, z / 9.0);
    c = mix(c, [0.56, 0.82, 0.30], fine * 0.25);
    let mud = mud_amount(x, z);
    c = mix(c, [0.45, 0.32, 0.20], mud * 0.92);
    let rock = ((slope - 0.45) / 0.3).clamp(0.0, 1.0);
    mix(c, [0.50, 0.50, 0.44], rock * 0.55)
}

fn sand_w(z: f32, side: i32) -> f32 {
    let sf = side as f32;
    let t = vnoise(z / 38.0 + 13.7 * sf, 4.3 * sf);
    ((t - 0.42) / 0.58).clamp(0.0, 1.0) * 5.0
}

struct Row {
    rx: f32,
    rp: f32,
    s: f32,
    y: f32,
    z: f32,
    wl: f32,
    wr: f32,
    kerb: i32,
}

fn make_row(z: f32) -> Row {
    let rp = road_dx(z);
    let ddx = road_ddx(z);
    let s = (1.0 + rp * rp).sqrt();
    let kappa = ddx / (s * s * s);
    let kerb = if kappa.abs() > 0.0011 {
        if ddx > 0.0 {
            1
        } else {
            -1
        }
    } else {
        0
    };
    Row { rx: road_x(z), rp, s, y: road_target(z), z, wl: sand_w(z, -1), wr: sand_w(z, 1), kerb }
}

fn in_chunk(x: f32, z: f32, x0: f32, z0: f32) -> bool {
    x >= x0 && x < x0 + CHUNK && z >= z0 && z < z0 + CHUNK
}

// ------------------------------------------------------------------
// Chunk mesh building (pure function of the world; cached by the renderer)
// ------------------------------------------------------------------

pub fn build_chunk(cx: i32, cz: i32, out: &mut Vec<Vertex>) {
    let x0 = cx as f32 * CHUNK;
    let z0 = cz as f32 * CHUNK;
    build_terrain(x0, z0, out);
    build_ribbon(x0, z0, out);
    build_props(x0, z0, out);
    build_flowers(cx, cz, out);
    if TREES {
        build_trees(cx, cz, out);
    }
}

fn build_terrain(x0: f32, z0: f32, out: &mut Vec<Vertex>) {
    let n = CPC as usize;
    let m = n + 3; // heights from -1 ..= n+1 so normals match across chunk borders
    let mut hs = vec![0f32; m * m];
    for j in 0..m {
        for i in 0..m {
            hs[j * m + i] = mesh_h(x0 + (i as f32 - 1.0) * CELL, z0 + (j as f32 - 1.0) * CELL);
        }
    }
    let h = |i: i32, j: i32| hs[((j + 1) as usize) * m + (i + 1) as usize];
    let sun = Vec3::new(0.45, 0.80, 0.40).normalize();
    let stride = n + 1;
    let mut verts: Vec<Vertex> = Vec::with_capacity(stride * stride);
    for j in 0..=CPC {
        for i in 0..=CPC {
            let hl = h(i - 1, j);
            let hr = h(i + 1, j);
            let hd = h(i, j - 1);
            let hu = h(i, j + 1);
            let nrm = Vec3::new(hl - hr, 2.0 * CELL, hd - hu).normalize();
            let slope = ((hl - hr) * (hl - hr) + (hd - hu) * (hd - hu)).sqrt() / (2.0 * CELL);
            let x = x0 + i as f32 * CELL;
            let z = z0 + j as f32 * CELL;
            let mut base = terrain_color(x, z, slope);
            // dusty verge under / next to the road, so no bright green ever shows on the asphalt
            let dust = 0.75 * (1.0 - smooth(4.5, 11.0, road_off(x, z)));
            base = mix(base, [0.62, 0.62, 0.52], dust);
            let shade = (0.66 + 0.42 * nrm.dot(sun).max(0.0)).min(1.1);
            verts.push(Vertex {
                pos: [x, h(i, j), z],
                color: [base[0] * shade, base[1] * shade, base[2] * shade],
                kind: 1.0,
            });
        }
    }
    for j in 0..n {
        for i in 0..n {
            let a = verts[j * stride + i];
            let b = verts[j * stride + i + 1];
            let c = verts[(j + 1) * stride + i];
            let d = verts[(j + 1) * stride + i + 1];
            out.extend_from_slice(&[a, b, d, a, d, c]);
        }
    }
}

fn quad(out: &mut Vec<Vertex>, p: [[f32; 3]; 4], c: [f32; 3]) {
    for &i in &[0usize, 1, 2, 0, 2, 3] {
        out.push(Vertex { pos: p[i], color: c, kind: 1.0 });
    }
}

fn scale(c: [f32; 3], k: f32) -> [f32; 3] {
    [c[0] * k, c[1] * k, c[2] * k]
}

/// Asphalt, white edge lines, kerbs and sand run-off.
fn build_ribbon(x0: f32, z0: f32, out: &mut Vec<Vertex>) {
    let k0 = ((z0 - 20.0) / SEG).floor() as i32;
    let k1 = ((z0 + CHUNK + 20.0) / SEG).ceil() as i32;
    let rows: Vec<Row> = (k0..=k1).map(|k| make_row(k as f32 * SEG)).collect();
    let pt = |row: &Row, off: f32, lift: f32| -> [f32; 3] {
        [row.rx + off / row.s, row.y + lift, row.z - off * row.rp / row.s]
    };
    for w in rows.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        let mx = (a.rx + b.rx) * 0.5;
        let mz = (a.z + b.z) * 0.5;
        if !in_chunk(mx, mz, x0, z0) {
            continue;
        }
        let idx = (a.z / SEG).round() as i32;
        let tint = 0.97 + 0.06 * h01(idx, 91);

        // asphalt
        quad(
            out,
            [pt(a, -ROAD_HALF, 0.20), pt(a, ROAD_HALF, 0.20), pt(b, ROAD_HALF, 0.20), pt(b, -ROAD_HALF, 0.20)],
            scale([0.56, 0.61, 0.66], tint),
        );
        // white edge lines
        for &sd in &[-1.0f32, 1.0] {
            let o0 = sd * (ROAD_HALF - 0.38);
            let o1 = sd * (ROAD_HALF - 0.15);
            quad(out, [pt(a, o0, 0.24), pt(a, o1, 0.24), pt(b, o1, 0.24), pt(b, o0, 0.24)], [0.96, 0.96, 0.96]);
        }
        // sand run-off (varying width)
        let sand = scale([0.93, 0.82, 0.58], tint);
        if a.wl > 0.05 || b.wl > 0.05 {
            let i0 = -(ROAD_HALF + 0.7);
            quad(
                out,
                [pt(a, i0, 0.16), pt(a, i0 - a.wl, 0.16), pt(b, i0 - b.wl, 0.16), pt(b, i0, 0.16)],
                sand,
            );
        }
        if a.wr > 0.05 || b.wr > 0.05 {
            let i0 = ROAD_HALF + 0.7;
            quad(
                out,
                [pt(a, i0, 0.16), pt(a, i0 + a.wr, 0.16), pt(b, i0 + b.wr, 0.16), pt(b, i0, 0.16)],
                sand,
            );
        }
        // black / white kerbs on the inside of bends
        if a.kerb != 0 {
            let sd = a.kerb as f32;
            let o0 = sd * ROAD_HALF;
            let o1 = sd * (ROAD_HALF + 1.3);
            let c = if idx & 1 == 0 { [0.12, 0.12, 0.14] } else { [0.96, 0.96, 0.96] };
            quad(out, [pt(a, o0, 0.24), pt(a, o1, 0.24), pt(b, o1, 0.24), pt(b, o0, 0.24)], c);
        }
    }
}

/// Guard rails, street lamps and chevron signs.
fn build_props(x0: f32, z0: f32, out: &mut Vec<Vertex>) {
    let z_lo = z0 - 14.0;
    let z_hi = z0 + CHUNK + 14.0;

    // ---- guard rails ----
    for side in [-1i32, 1] {
        let sf = side as f32;
        let b0 = (z_lo / BLOCK).floor() as i32;
        let b1 = (z_hi / BLOCK).floor() as i32;
        for blk in b0..=b1 {
            if !rail_exists(side, blk) {
                continue;
            }
            let zr = (blk as f32 + 0.5) * BLOCK;
            let rp = road_dx(zr);
            let s = (1.0 + rp * rp).sqrt();
            let cx = road_x(zr) + sf * RAIL_OFF / s;
            let cz = zr - sf * RAIL_OFF * rp / s;
            if !in_chunk(cx, cz, x0, z0) {
                continue;
            }
            // follow the road's slope (pitch) and sit on the sunk verge
            let gy = road_target(zr) - 0.22;
            let slope_z = (road_target(zr + 2.0) - road_target(zr - 2.0)) / 4.0;
            let pitch = -(slope_z / s).atan();
            let m = Mat4::from_translation(Vec3::new(cx, gy, cz))
                * Mat4::from_rotation_y(rp.atan())
                * Mat4::from_rotation_x(pitch);
            push_box(out, m, Vec3::new(0.0, 0.66, 0.0), Vec3::new(0.05, 0.17, BLOCK * 0.5), [0.66, 0.71, 0.76]);
            push_box(out, m, Vec3::new(0.0, 0.36, 0.0), Vec3::new(0.04, 0.08, BLOCK * 0.5), [0.52, 0.57, 0.62]);
            for &pz in &[-BLOCK * 0.42, BLOCK * 0.42] {
                push_box(out, m, Vec3::new(0.0, 0.28, pz), Vec3::new(0.06, 0.58, 0.06), [0.40, 0.44, 0.48]);
            }
        }
    }

    // ---- street lamps ----
    let l0 = (z_lo / LAMP).floor() as i32;
    let l1 = (z_hi / LAMP).floor() as i32;
    for lm in l0..=l1 {
        if hash(lm, 555) % 10 >= 8 {
            continue;
        }
        let side: i32 = if lm & 1 == 0 { 1 } else { -1 };
        let sf = side as f32;
        let zr = (lm as f32 + 0.5) * LAMP;
        let rp = road_dx(zr);
        let s = (1.0 + rp * rp).sqrt();
        let off = ROAD_HALF + 3.4;
        let cx = road_x(zr) + sf * off / s;
        let cz = zr - sf * off * rp / s;
        if !in_chunk(cx, cz, x0, z0) {
            continue;
        }
        let gy = road_target(zr);
        let m = Mat4::from_translation(Vec3::new(cx, gy, cz)) * Mat4::from_rotation_y(rp.atan());
        let pole = [0.42, 0.50, 0.57];
        push_box(out, m, Vec3::new(0.0, 3.8, 0.0), Vec3::new(0.11, 3.8, 0.11), pole);
        push_box(out, m, Vec3::new(-sf * 1.3, 7.55, 0.0), Vec3::new(1.4, 0.07, 0.07), pole);
        push_box(out, m, Vec3::new(-sf * 2.6, 7.42, 0.0), Vec3::new(0.45, 0.08, 0.18), [0.95, 0.95, 0.80]);
    }

    // ---- red chevron signs on the outside of sharp bends ----
    let s0 = (z_lo / SIGN).floor() as i32;
    let s1 = (z_hi / SIGN).floor() as i32;
    for sg in s0..=s1 {
        let zr = (sg as f32 + 0.5) * SIGN;
        let ddx = road_ddx(zr);
        let rp = road_dx(zr);
        let s = (1.0 + rp * rp).sqrt();
        let kappa = ddx / (s * s * s);
        if kappa.abs() < 0.0016 {
            continue;
        }
        let side: i32 = if ddx > 0.0 { -1 } else { 1 }; // outside of the bend
        let sf = side as f32;
        let off = RAIL_OFF - 1.6;
        let cx = road_x(zr) + sf * off / s;
        let cz = zr - sf * off * rp / s;
        if !in_chunk(cx, cz, x0, z0) {
            continue;
        }
        let gy = road_target(zr);
        let m = Mat4::from_translation(Vec3::new(cx, gy, cz)) * Mat4::from_rotation_y(rp.atan());
        push_box(out, m, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.05, 1.0, 0.05), [0.40, 0.44, 0.48]);
        push_box(out, m, Vec3::new(0.0, 2.1, 0.0), Vec3::new(0.55, 0.55, 0.04), [0.96, 0.96, 0.96]);
        push_box(out, m, Vec3::new(0.0, 2.1, 0.05), Vec3::new(0.38, 0.38, 0.03), [0.85, 0.10, 0.10]);
    }
}


// ------------------------------------------------------------------
// Roadside buildings (low-poly models from buildings.bin, shared GPU buffers, one draw per building)
// ------------------------------------------------------------------

static BUILDINGS_BIN: &[u8] = include_bytes!("buildings.bin");

pub struct Model {
    pub verts: Vec<Vertex>,
    pub hx: f32,
    pub hz: f32,
    pub h: f32,
}

pub fn models() -> &'static Vec<Model> {
    static M: std::sync::OnceLock<Vec<Model>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let d = BUILDINGS_BIN;
        let rf = |o: usize| f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
        let mut o = 0usize;
        let mut v = Vec::new();
        while o + 20 <= d.len() {
            let n = u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize;
            let (hx, hz, h) = (rf(o + 4), rf(o + 8), rf(o + 12));
            o += 20;
            let mut verts = Vec::with_capacity(n);
            for _ in 0..n {
                verts.push(Vertex {
                    pos: [rf(o), rf(o + 4), rf(o + 8)],
                    color: [rf(o + 12), rf(o + 16), rf(o + 20)],
                    kind: 1.0,
                });
                o += 24;
            }
            v.push(Model { verts, hx, hz, h });
        }
        v
    })
}

#[derive(Clone, Copy)]
pub struct Bld {
    pub model: usize,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub hx: f32,
    pub hz: f32,
}

const BLD_CELL: f32 = 200.0;
/// Which local axis is the facade of each model (true = +x, false = +z).
const FRONT_X: [bool; 3] = [true, true, false];

fn bld_compute(iz: i32) -> Vec<Bld> {
    let ms = models();
    let mut res = Vec::new();
    if ms.len() < 3 {
        return res;
    }
    let h = hash(iz * 7 + 1, 4242);
    let r = h % 100;
    // some stretches have no building, some one, two or three
    let count = if r < 28 { 0 } else if r < 60 { 1 } else if r < 82 { 2 } else { 3 };
    let side: f32 = if (h >> 8) & 1 == 0 { 1.0 } else { -1.0 };
    let mut zr = iz as f32 * BLD_CELL + 25.0 + 50.0 * h01(iz, 31);
    for k in 0..count {
        let hk = hash(iz * 11 + k, 777);
        let mi = (hk % 3) as usize;
        let m = &ms[mi];
        let rad = m.hx.max(m.hz);
        let rp = road_dx(zr);
        let s = (1.0 + rp * rp).sqrt();
        let mut best: Option<(f32, f32, f32, f32)> = None;
        for &off0 in &[24.0f32, 30.0, 38.0, 46.0] {
            let off = off0 + rad * 0.5;
            let bx = road_x(zr) + side * off / s;
            let bz = zr - side * off * rp / s;
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for &(ox, oz) in &[(0.0, 0.0), (rad, rad), (-rad, rad), (rad, -rad), (-rad, -rad)] {
                let t = terrain_h(bx + ox, bz + oz);
                lo = lo.min(t);
                hi = hi.max(t);
            }
            if hi - lo < 3.2 {
                best = Some((bx, bz, lo, rp));
                break;
            }
        }
        if let Some((bx, bz, lo, rp)) = best {
            // face the road
            let (dx, dz) = (-side / s, side * rp / s);
            let jit = (h01(iz * 5 + k, 99) - 0.5) * 0.16;
            let yaw = if FRONT_X[mi] { (-dz).atan2(dx) } else { dx.atan2(dz) } + jit;
            res.push(Bld { model: mi, x: bx, y: lo - 0.3, z: bz, yaw, hx: m.hx, hz: m.hz });
        }
        zr += 2.0 * rad + 14.0 + 18.0 * h01(iz * 3 + k, 55);
    }
    res
}

fn bld_cell(iz: i32) -> Vec<Bld> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<i32, Vec<Bld>>> = RefCell::new(HashMap::new());
    }
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 600 {
            c.clear();
        }
        c.entry(iz).or_insert_with(|| bld_compute(iz)).clone()
    })
}

/// All buildings within a few hundred metres (in z) of `z`.
pub fn buildings_around(z: f32) -> Vec<Bld> {
    let iz0 = (z / BLD_CELL).floor() as i32;
    let mut v = Vec::new();
    for iz in iz0 - 2..=iz0 + 2 {
        v.extend(bld_cell(iz));
    }
    v
}

fn near_building(x: f32, z: f32, margin: f32) -> bool {
    let iz0 = (z / BLD_CELL).floor() as i32;
    for iz in iz0 - 1..=iz0 + 1 {
        for b in bld_cell(iz) {
            let r = (b.hx * b.hx + b.hz * b.hz).sqrt() + margin;
            let (dx, dz) = (x - b.x, z - b.z);
            if dx * dx + dz * dz < r * r {
                return true;
            }
        }
    }
    false
}

const FLOWER_COLORS: [[f32; 3]; 7] = [
    [0.95, 0.15, 0.20], // red
    [1.00, 0.85, 0.12], // yellow
    [0.98, 0.98, 0.96], // white
    [1.00, 0.45, 0.70], // pink
    [0.62, 0.30, 0.90], // purple
    [1.00, 0.55, 0.10], // orange
    [0.30, 0.50, 1.00], // blue
];
const FL_CELL: f32 = 8.0;
const FL_PER_CLUSTER: u32 = 6;

/// Wild-flower patches on both sides of the road: tilted little blooms facing the rider.
fn build_flowers(cx: i32, cz: i32, out: &mut Vec<Vertex>) {
    let n = (CHUNK / FL_CELL) as i32;
    for ix in cx * n..cx * n + n {
        for iz in cz * n..cz * n + n {
            let h = hash(ix * 13 + 5, iz * 17 + 9);
            let px = (ix as f32 + 0.15 + 0.7 * ((h >> 8) & 0xff) as f32 / 255.0) * FL_CELL;
            let pz = (iz as f32 + 0.15 + 0.7 * ((h >> 16) & 0xff) as f32 / 255.0) * FL_CELL;
            let d = road_off(px, pz);
            if d < 9.0 || d > 70.0 {
                continue;
            }
            // patchy meadows: more flowers near the road, in noise-driven patches
            let patch = vnoise(px / 55.0 + 5.1, pz / 55.0 - 2.7);
            let p = if d < 26.0 { 0.62 } else { 0.28 } * (0.4 + patch);
            if ((h & 0xff) as f32 / 255.0) > p {
                continue;
            }
            if near_building(px, pz, 3.5) || mud_amount(px, pz) > 0.25 {
                continue;
            }
            let col = FLOWER_COLORS[((h >> 24) % 7) as usize];
            let col2 = FLOWER_COLORS[((h >> 27) % 7) as usize];
            // direction towards the road (horizontal)
            let rx = road_x(road_param(px, pz));
            let to = Vec3::new(if rx > px { 1.0 } else { -1.0 }, 0.0, 0.0);
            for k in 0..FL_PER_CLUSTER {
                let hk = hash(ix * 31 + k as i32 * 7, iz * 37 + 3);
                let a = ((hk & 0xff) as f32 / 255.0) * std::f32::consts::TAU;
                let rr = 0.3 + 1.9 * (((hk >> 8) & 0xff) as f32 / 255.0);
                let x = px + a.cos() * rr;
                let z = pz + a.sin() * rr;
                let gy = terrain_h(x, z);
                let hh = 0.32 + 0.30 * (((hk >> 16) & 0xff) as f32 / 255.0);
                let r = 0.10 + 0.07 * (((hk >> 24) & 0xff) as f32 / 255.0);
                let petal = if hk & 0x100 != 0 { col } else { col2 };
                let top = Vec3::new(x, gy + hh, z);
                // stem
                let sg = [0.20, 0.52, 0.16];
                for &(p, c) in &[
                    ([x - 0.015, gy, z], sg),
                    ([x + 0.015, gy, z], sg),
                    ([top.x, top.y, top.z], [0.30, 0.62, 0.20]),
                ] {
                    out.push(Vertex { pos: p, color: c, kind: 1.0 });
                }
                // bloom: tilted fan, yellow centre -> coloured petals
                let nrm = (to * 0.8 + Vec3::Y * 0.6).normalize();
                let t1 = Vec3::Y.cross(nrm).normalize();
                let t2 = nrm.cross(t1);
                let ring = |i: u32| -> [f32; 3] {
                    let an = std::f32::consts::TAU * i as f32 / 5.0;
                    let q = top + (t1 * an.cos() + t2 * an.sin()) * r;
                    [q.x, q.y, q.z]
                };
                for i in 0..5u32 {
                    out.push(Vertex { pos: [top.x, top.y, top.z], color: [1.0, 0.86, 0.20], kind: 1.0 });
                    out.push(Vertex { pos: ring(i), color: petal, kind: 1.0 });
                    out.push(Vertex { pos: ring(i + 1), color: petal, kind: 1.0 });
                }
            }
        }
    }
}

/// Realistic trees: tapered trunks, branches and soft smooth-shaded leaf clusters.
fn build_trees(cx: i32, cz: i32, out: &mut Vec<Vertex>) {
    for ix in cx * TPC..cx * TPC + TPC {
        for iz in cz * TPC..cz * TPC + TPC {
            let Some((x, z, s)) = tree_at(ix, iz) else { continue };
            let gy = terrain_h(x, z);
            push_tree(out, x, gy, z, s, hash(ix, iz));
        }
    }
}

const SUN: [f32; 3] = [0.45, 0.80, 0.40];

fn sun_dir() -> Vec3 {
    Vec3::new(SUN[0], SUN[1], SUN[2]).normalize()
}

/// Tapered, smooth-shaded branch / trunk segment.
fn push_limb(out: &mut Vec<Vertex>, a: Vec3, b: Vec3, ra: f32, rb: f32, ca: [f32; 3], cb: [f32; 3], sides: usize) {
    use std::f32::consts::TAU;
    let axis = b - a;
    let len = axis.length();
    if len < 1e-4 {
        return;
    }
    let dir = axis / len;
    let helper = if dir.y.abs() < 0.95 { Vec3::Y } else { Vec3::new(1.0, 0.0, 0.0) };
    let u = dir.cross(helper).normalize();
    let v = dir.cross(u);
    let sun = sun_dir();
    let mut ring_a: Vec<(Vec3, [f32; 3])> = Vec::with_capacity(sides);
    let mut ring_b: Vec<(Vec3, [f32; 3])> = Vec::with_capacity(sides);
    for i in 0..sides {
        let th = TAU * i as f32 / sides as f32;
        let off = u * th.cos() + v * th.sin();
        let lit = 0.50 + 0.65 * off.dot(sun).max(0.0);
        ring_a.push((a + off * ra, scale(ca, lit)));
        ring_b.push((b + off * rb, scale(cb, lit)));
    }
    for i in 0..sides {
        let j = (i + 1) % sides;
        let (a0, a1, b0, b1) = (ring_a[i], ring_a[j], ring_b[i], ring_b[j]);
        for &(p, c) in &[a0, a1, b1, a0, b1, b0] {
            out.push(Vertex { pos: p.to_array(), color: c, kind: 1.0 });
        }
    }
}

/// Soft, lumpy, smooth-shaded leaf cluster (ellipsoid with noise, light on top, dark underneath).
fn push_canopy(
    out: &mut Vec<Vertex>,
    c: Vec3,
    r: Vec3,
    top: [f32; 3],
    bot: [f32; 3],
    seed: i32,
    segs: usize,
    rings: usize,
    jit: f32,
) {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    let sun = sun_dir();
    let mut pts: Vec<Vec3> = Vec::with_capacity((rings + 1) * segs);
    let mut cols: Vec<[f32; 3]> = Vec::with_capacity((rings + 1) * segs);
    for k in 0..=rings {
        let phi = -FRAC_PI_2 + PI * k as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for i in 0..segs {
            let th = TAU * i as f32 / segs as f32;
            let pole = k == 0 || k == rings;
            let j = if pole { 1.0 } else { 1.0 + jit * (h01(seed + k as i32 * 13, i as i32 * 7 + seed) - 0.5) * 2.0 };
            let p = Vec3::new(
                c.x + r.x * cp * th.cos() * j,
                c.y + r.y * sp * j,
                c.z + r.z * cp * th.sin() * j,
            );
            let d = p - c;
            let n = Vec3::new(d.x / (r.x * r.x), d.y / (r.y * r.y), d.z / (r.z * r.z));
            let nl = n.length();
            let n = if nl > 1e-6 { n / nl } else { Vec3::Y };
            let up = ((d.y / r.y) * 0.5 + 0.5).clamp(0.0, 1.0);
            let lit = (0.66 + 0.52 * n.dot(sun).max(0.0)).min(1.2);
            let mott = 0.95 + 0.10 * h01(seed * 31 + (k * 17 + i) as i32, 5);
            let base = mix(bot, top, up * up * 0.5 + up * 0.5);
            cols.push(scale(base, lit * mott));
            pts.push(p);
        }
    }
    let idx = |k: usize, i: usize| k * segs + (i % segs);
    let mut tri = |out: &mut Vec<Vertex>, a: usize, b: usize, d: usize| {
        for &q in &[a, b, d] {
            out.push(Vertex { pos: pts[q].to_array(), color: cols[q], kind: 1.0 });
        }
    };
    for k in 0..rings {
        for i in 0..segs {
            let a = idx(k, i);
            let b = idx(k, i + 1);
            let cc = idx(k + 1, i + 1);
            let d = idx(k + 1, i);
            if k == 0 {
                tri(out, a, cc, d);
            } else if k == rings - 1 {
                tri(out, a, b, d);
            } else {
                tri(out, a, b, cc);
                tri(out, a, cc, d);
            }
        }
    }
}

/// Three kinds of tree: broadleaf (oak-like), umbrella pine and tall cypress.
fn push_tree(out: &mut Vec<Vertex>, x: f32, gy: f32, z: f32, s: f32, h: u32) {
    use std::f32::consts::TAU;
    let hs = (h & 0xffff) as i32;
    let rnd = |k: i32| (hash(hs, k) & 0xff) as f32 / 255.0;
    let kind = hash(hs, 99) % 20;
    let ang = rnd(1) * TAU;
    let g = rnd(2);
    let bark_lo = [0.20, 0.13, 0.10];
    let bark_hi = [0.36, 0.26, 0.19];
    let base = Vec3::new(x, gy - 0.6, z);
    let seed = hs;

    if kind < 11 {
        // ---- broadleaf ----
        let top = [0.42 + 0.16 * g, 0.74 + 0.10 * g, 0.20 + 0.07 * g];
        let bot = [0.16, 0.40, 0.12];
        let hh = 9.5 * s;
        let lean = (ang.cos() * 0.7 * s, ang.sin() * 0.7 * s);
        let flare = base + Vec3::new(0.0, 1.5, 0.0);
        push_limb(out, base, flare, 0.72 * s, 0.46 * s, bark_lo, bark_lo, 7);
        let fork = Vec3::new(x + lean.0, gy + hh * 0.42, z + lean.1);
        push_limb(out, flare, fork, 0.46 * s, 0.32 * s, bark_lo, bark_hi, 7);
        let nb = if rnd(3) > 0.5 { 4 } else { 3 };
        for k in 0..nb {
            let a = ang + k as f32 * TAU / nb as f32 + rnd(10 + k) * 0.7;
            let len = (2.6 + 1.4 * rnd(20 + k)) * s;
            let end = fork + Vec3::new(a.cos() * len, hh * (0.26 + 0.1 * rnd(30 + k)), a.sin() * len);
            push_limb(out, fork, end, 0.27 * s, 0.09 * s, bark_hi, bark_hi, 6);
            let rr = (2.9 + 0.8 * rnd(40 + k)) * s;
            push_canopy(out, end + Vec3::new(0.0, 1.1 * s, 0.0), Vec3::new(rr, 2.2 * s, rr), top, bot, seed + k * 3, 10, 5, 0.18);
        }
        let crown = Vec3::new(fork.x, gy + hh * 0.86, fork.z);
        push_canopy(out, crown, Vec3::new(3.7 * s, 2.7 * s, 3.7 * s), top, bot, seed + 50, 10, 5, 0.16);
        push_canopy(
            out,
            crown + Vec3::new(0.9 * s, 1.9 * s, -0.7 * s),
            Vec3::new(2.4 * s, 1.8 * s, 2.4 * s),
            top,
            bot,
            seed + 60,
            10,
            5,
            0.14,
        );
    } else if kind < 17 {
        // ---- umbrella pine (tall bare trunk, flat wide crown) ----
        let top = [0.30 + 0.10 * g, 0.62 + 0.08 * g, 0.24 + 0.05 * g];
        let bot = [0.12, 0.34, 0.14];
        let hh = 12.5 * s;
        let l1 = (ang.cos() * 0.9 * s, ang.sin() * 0.9 * s);
        let l2 = (ang.cos() * 1.9 * s, ang.sin() * 1.9 * s);
        let mid = Vec3::new(x + l1.0, gy + hh * 0.55, z + l1.1);
        let tip = Vec3::new(x + l2.0, gy + hh * 0.84, z + l2.1);
        push_limb(out, base, mid, 0.46 * s, 0.30 * s, bark_lo, bark_hi, 7);
        push_limb(out, mid, tip, 0.30 * s, 0.17 * s, bark_hi, bark_hi, 6);
        for k in 0..2 {
            let a = ang + 1.6 + k as f32 * 3.1 + rnd(11 + k) * 0.5;
            let end = Vec3::new(tip.x + a.cos() * 3.4 * s, tip.y - 0.5 * s, tip.z + a.sin() * 3.4 * s);
            push_limb(out, Vec3::new(tip.x, tip.y - 1.6 * s, tip.z), end, 0.16 * s, 0.07 * s, bark_hi, bark_hi, 5);
            push_canopy(out, end + Vec3::new(0.0, 0.8 * s, 0.0), Vec3::new(3.6 * s, 1.2 * s, 3.6 * s), top, bot, seed + k * 7, 10, 5, 0.14);
        }
        push_canopy(out, tip + Vec3::new(0.0, 1.0 * s, 0.0), Vec3::new(5.0 * s, 1.6 * s, 5.0 * s), top, bot, seed + 70, 10, 5, 0.14);
        push_canopy(out, tip + Vec3::new(-1.2 * s, 2.0 * s, 0.8 * s), Vec3::new(3.0 * s, 1.4 * s, 3.0 * s), top, bot, seed + 80, 10, 5, 0.14);
    } else {
        // ---- cypress (tall narrow flame) ----
        let top = [0.18 + 0.06 * g, 0.52 + 0.06 * g, 0.24];
        let bot = [0.07, 0.24, 0.11];
        let tops: [(f32, f32); 4] = [(1.9, 3.0), (1.7, 2.9), (1.4, 2.7), (1.0, 2.4)];
        push_limb(out, base, base + Vec3::new(0.0, 3.2 * s, 0.0), 0.34 * s, 0.22 * s, bark_lo, bark_lo, 6);
        for (i, &(rx, ry)) in tops.iter().enumerate() {
            let c = Vec3::new(x, gy + (2.8 + i as f32 * 2.7) * s, z);
            push_canopy(out, c, Vec3::new(rx * s, ry * s, rx * s), top, bot, seed + i as i32 * 11, 10, 5, 0.12);
        }
    }
}

/// Faceted, flat-shaded blob (tree canopy / balloon).
fn push_blob(
    out: &mut Vec<Vertex>,
    c: Vec3,
    r: Vec3,
    top: [f32; 3],
    bot: [f32; 3],
    seed: i32,
    segs: usize,
    rings: usize,
    jit: f32,
) {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    let sun = Vec3::new(0.45, 0.80, 0.40).normalize();
    let mut pts: Vec<Vec3> = Vec::with_capacity((rings + 1) * segs);
    for k in 0..=rings {
        let phi = -FRAC_PI_2 + PI * k as f32 / rings as f32;
        let (sp, cp) = phi.sin_cos();
        for i in 0..segs {
            let th = TAU * i as f32 / segs as f32;
            let j = if k == 0 || k == rings {
                1.0
            } else {
                1.0 + jit * (h01(seed + k as i32 * 13, i as i32 * 7 + seed) - 0.5) * 2.0
            };
            pts.push(Vec3::new(
                c.x + r.x * cp * th.cos() * j,
                c.y + r.y * sp * j,
                c.z + r.z * cp * th.sin() * j,
            ));
        }
    }
    let at = |k: usize, i: usize| pts[k * segs + (i % segs)];
    let emit = |out: &mut Vec<Vertex>, p0: Vec3, p1: Vec3, p2: Vec3| {
        let n = (p1 - p0).cross(p2 - p0);
        let len = n.length();
        if len < 1e-6 {
            return;
        }
        let mut n = n / len;
        let cen = (p0 + p1 + p2) / 3.0;
        if n.dot(cen - c) < 0.0 {
            n = n * -1.0;
        }
        let shade = (0.60 + 0.55 * n.dot(sun).max(0.0)).min(1.15);
        let t = (((cen.y - c.y) / r.y) * 0.5 + 0.5).clamp(0.0, 1.0);
        let col = scale(mix(bot, top, t), shade);
        for p in [p0, p1, p2] {
            out.push(Vertex { pos: p.to_array(), color: col, kind: 1.0 });
        }
    };
    for k in 0..rings {
        for i in 0..segs {
            let a = at(k, i);
            let b = at(k, i + 1);
            let cc = at(k + 1, i + 1);
            let d = at(k + 1, i);
            if k == 0 {
                emit(out, a, cc, d);
            } else if k == rings - 1 {
                emit(out, a, b, d);
            } else {
                emit(out, a, b, cc);
                emit(out, a, cc, d);
            }
        }
    }
}

// ------------------------------------------------------------------
// Bike
// ------------------------------------------------------------------

/// Ground surface under the wheels.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Surf {
    Asphalt,
    Sand,
    Grass,
    Mud,
    Rock,
}

struct SurfP {
    /// tyre grip coefficient
    mu: f32,
    /// rolling resistance
    drag: f32,
    /// bump amplitude (m) fed into the suspension
    rough: f32,
}

fn surf_params(s: Surf) -> SurfP {
    match s {
        Surf::Asphalt => SurfP { mu: 1.10, drag: 0.00, rough: 0.004 },
        Surf::Sand => SurfP { mu: 0.62, drag: 0.16, rough: 0.020 },
        Surf::Grass => SurfP { mu: 0.80, drag: 0.09, rough: 0.035 },
        Surf::Mud => SurfP { mu: 0.50, drag: 0.26, rough: 0.055 },
        Surf::Rock => SurfP { mu: 0.85, drag: 0.07, rough: 0.100 },
    }
}

fn surface_at(x: f32, z: f32, slope: f32) -> Surf {
    let zr = road_param(x, z);
    let dxr = x - road_x(zr);
    let dzr = z - zr;
    let off = (dxr * dxr + dzr * dzr).sqrt();
    if off < ROAD_HALF + 0.7 {
        return Surf::Asphalt;
    }
    let side = if dxr >= 0.0 { 1 } else { -1 };
    if off < ROAD_HALF + 0.7 + sand_w(zr, side) {
        return Surf::Sand;
    }
    if slope > 0.62 {
        return Surf::Rock;
    }
    if mud_amount(x, z) > 0.5 {
        Surf::Mud
    } else {
        Surf::Grass
    }
}

/// Small deterministic ground bumps (suspension input).
fn bump(x: f32, z: f32, amp: f32) -> f32 {
    let a = vnoise(x * 0.16 + 7.1, z * 0.16 - 3.3) - 0.5;
    let b = vnoise(x * 0.55 - 1.9, z * 0.55 + 8.4) - 0.5;
    (a * 1.4 + b * 0.6) * amp * 2.0
}

pub struct Game {
    /// Contact point with the ground (x, ground height, z).
    pub pos: Vec3,
    /// Horizontal velocity (y unused).
    pub vel: Vec3,
    pub vy: f32,
    pub yaw: f32,
    pub speed: f32,
    pub steer: f32,
    /// Bike lean angle (rad, + = right).
    pub lean: f32,
    /// Camera roll (rad) derived from the lean.
    pub roll: f32,
    /// Camera pitch (rad): slope / flight attitude + weight transfer.
    pub pitch: f32,
    pub grounded: bool,
    pub surf: Surf,
    // suspension (body height offset, m)
    susp: f32,
    susp_v: f32,
    bump_prev: f32,
    // body attitude and filtered longitudinal acceleration
    body_pitch: f32,
    ax: f32,
    // rider body offset relative to the bike (right, up, forward) in metres
    pub rider: [f32; 3],
    spin: f32,
}

impl Game {
    pub fn new() -> Self {
        let x = road_x(0.0);
        let y = terrain_h(x, 0.0);
        Self {
            pos: Vec3::new(x, y, 0.0),
            vel: Vec3::ZERO,
            vy: 0.0,
            yaw: road_dx(0.0).atan(),
            speed: 0.0,
            steer: 0.0,
            lean: 0.0,
            roll: 0.0,
            pitch: 0.0,
            grounded: true,
            surf: Surf::Asphalt,
            susp: 0.0,
            susp_v: 0.0,
            bump_prev: 0.0,
            body_pitch: 0.0,
            ax: 0.0,
            rider: [0.0; 3],
            spin: 0.0,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    /// The rider's head: bike position + rider body movement + suspension travel.
    pub fn eye(&self) -> Vec3 {
        let f = self.forward();
        let r = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let up = 1.25 + self.susp + self.rider[1];
        Vec3::new(
            self.pos.x + r.x * self.rider[0] + f.x * self.rider[2],
            self.pos.y + up,
            self.pos.z + r.z * self.rider[0] + f.z * self.rider[2],
        )
    }

    pub fn look_dir(&self) -> Vec3 {
        let f = self.forward();
        let c = self.pitch.cos();
        Vec3::new(f.x * c, self.pitch.sin(), f.z * c)
    }

    pub fn update(&mut self, input: &Input, dt: f32) {
        let f = self.forward();
        let r = Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin());
        let (gx, gz) = grad(self.pos.x, self.pos.z);
        let slope = (gx * gx + gz * gz).sqrt();
        self.surf = surface_at(self.pos.x, self.pos.z, slope);
        let sp = surf_params(self.surf);

        self.steer += (input.steer - self.steer) * (dt * 7.0).min(1.0);

        let mut vf = self.vel.dot(f);
        let vf_before = vf;
        let slope_f = gx * f.x + gz * f.z; // rise per metre along the heading
        let cos_s = 1.0 / (1.0 + slope_f * slope_f).sqrt();

        // ---------------- longitudinal: engine, brakes, traction, weight transfer ----------------
        if self.grounded {
            let engine = if input.throttle { 17.0 * (1.0 - vf / MAX_SPEED).max(0.0) } else { 0.0 };
            // rear tyre grip limits drive; braking also loads the front tyre (more stopping grip)
            let drive_lim = sp.mu * G * 1.15 * cos_s;
            self.spin += (if engine > drive_lim { 1.0 } else { 0.0 } - self.spin) * (dt * 6.0).min(1.0);
            vf += engine.min(drive_lim) * dt;
            if input.brake && vf > 0.0 {
                vf = (vf - sp.mu * G * 1.5 * cos_s * dt).max(0.0);
            }
            if !input.throttle && vf > 0.0 {
                vf = (vf - 1.2 * dt).max(0.0); // engine braking
            }
            vf -= (0.035 + sp.drag) * vf * dt;
            if vf < 0.0 {
                vf -= 2.0 * vf * dt; // rolling backwards stops quickly
            }
            // hills: gravity along the slope
            vf -= slope_f * cos_s * G * 0.7 * dt;
        } else {
            self.spin = 0.0;
        }

        // ---------------- steering: lean-based, grip-limited (understeer instead of skidding) ----------------
        let curv = self.steer * 0.085 / (1.0 + vf * vf / 1100.0);
        let ctrl = if self.grounded { 1.0 } else { 0.15 };
        let mut yawrate = curv * vf * ctrl;
        if self.grounded {
            let lat = yawrate * vf;
            let lat_max = sp.mu * G * 1.6;
            if lat.abs() > lat_max {
                let k = lat_max / lat.abs();
                yawrate *= k;
                vf -= vf * (1.0 - k) * 0.25 * dt; // tyres scrub speed, but never slide sideways
            }
        }
        self.yaw -= yawrate * dt;

        if self.grounded {
            // no sideways sliding: velocity always follows the wheels
            self.vel = f * vf;
        } else {
            // in the air momentum is kept
            let vr = self.vel.dot(r);
            self.vel = f * vf + r * vr;
        }

        let vmax = MAX_SPEED * 1.3;
        if self.vel.length() > vmax {
            self.vel = self.vel.normalize() * vmax;
        }

        self.pos.x += self.vel.x * dt;
        self.pos.z += self.vel.z * dt;

        self.rail_collision();
        self.tree_collision();
        self.building_collision();

        // ---------------- vertical: follow the ground or fly (jumps) ----------------
        let g1 = terrain_h(self.pos.x, self.pos.z);
        self.vy -= G * dt;
        let y_new = self.pos.y + self.vy * dt;
        if y_new <= g1 {
            if self.grounded {
                self.vy = ((g1 - self.pos.y) / dt).clamp(-25.0, 25.0);
            } else {
                self.land(g1);
            }
            self.pos.y = g1;
            self.grounded = true;
        } else {
            self.pos.y = y_new;
            self.grounded = false;
        }

        self.speed = self.vel.length();
        let f2 = self.forward();
        let vf_now = self.vel.dot(f2);

        // ---------------- weight transfer (filtered longitudinal acceleration) ----------------
        let ax_raw = ((vf_now - vf_before) / dt.max(1e-3)).clamp(-40.0, 40.0);
        self.ax += (ax_raw - self.ax) * (dt * 7.0).min(1.0);
        let wt = (self.ax / (G * 1.5)).clamp(-1.0, 1.0); // + accelerating, - braking

        // ---------------- lean ----------------
        let lat_acc = yawrate * vf_now;
        let lean_t = ((lat_acc / G).atan() + self.steer * 0.2 * (vf_now / 8.0).clamp(0.0, 1.0))
            .clamp(-0.75, 0.75)
            * if self.grounded { 1.0 } else { 0.3 };
        self.lean += (lean_t - self.lean) * (dt * 6.0).min(1.0);
        self.roll = self.lean * 0.42;

        // ---------------- suspension: spring + damper driven by bumps ----------------
        let w_in = if self.grounded { bump(self.pos.x, self.pos.z, sp.rough) } else { 0.0 };
        let w_v = (w_in - self.bump_prev) / dt.max(1e-3);
        self.bump_prev = w_in;
        const K: f32 = 150.0;
        const C: f32 = 17.0;
        let acc = -K * (self.susp - w_in) - C * (self.susp_v - w_v.clamp(-3.0, 3.0));
        self.susp_v += acc * dt;
        self.susp += self.susp_v * dt;
        if self.susp < -0.30 {
            self.susp = -0.30; // bottomed out
            self.susp_v = self.susp_v.max(0.0);
        }
        if self.susp > 0.18 {
            self.susp = 0.18;
            self.susp_v = self.susp_v.min(0.0);
        }

        // ---------------- body attitude: slope on the ground, flight path + rider control in the air ----------------
        let slope_pitch = (gx * f2.x + gz * f2.z).atan();
        if self.grounded {
            self.body_pitch += (slope_pitch.clamp(-0.7, 0.7) - self.body_pitch) * (dt * 7.0).min(1.0);
        } else {
            let flight = self.vy.atan2(vf_now.abs().max(3.0));
            let mut target = flight * 0.85;
            if input.throttle {
                target += 0.35; // gas in the air: nose up
            }
            if input.brake {
                target -= 0.35; // rear brake: nose down
            }
            self.body_pitch += (target.clamp(-0.9, 0.9) - self.body_pitch) * (dt * 1.6).min(1.0);
        }
        // squat when accelerating, dive when braking
        self.pitch = self.body_pitch + wt * 0.06 + (self.susp_v * 0.01).clamp(-0.03, 0.03);

        // ---------------- rider body ----------------
        let tuck = (vf_now / MAX_SPEED).clamp(0.0, 1.0);
        let tx = self.lean * 0.34;
        let ty = -self.lean.abs() * 0.10 - tuck * 0.07 - wt.min(0.0).abs() * 0.04 + self.susp * -0.35;
        let tz = if wt < 0.0 { -wt * 0.16 } else { -wt * 0.10 } - tuck * 0.05;
        let k = (dt * 5.0).min(1.0);
        self.rider[0] += (tx - self.rider[0]) * k;
        self.rider[1] += (ty - self.rider[1]) * k;
        self.rider[2] += (tz - self.rider[2]) * k;
    }

    /// Touchdown after a jump: the suspension absorbs the hit; a bad attitude or a huge drop costs speed.
    fn land(&mut self, g1: f32) {
        let (gx, gz) = grad(self.pos.x, self.pos.z);
        let n = Vec3::new(-gx, 1.0, -gz).normalize();
        let v = Vec3::new(self.vel.x, self.vy, self.vel.z);
        let impact = (-v.dot(n)).max(0.0);
        let f = self.forward();
        let slope_pitch = (gx * f.x + gz * f.z).atan();
        let mism = (self.body_pitch - slope_pitch).abs();
        let sev = ((impact - 7.0) / 14.0).clamp(0.0, 1.0) * 0.45 + ((mism - 0.35) / 0.6).clamp(0.0, 1.0) * 0.45;
        self.vel = self.vel * (1.0 - sev);
        self.susp_v -= impact * 0.32;
        self.rider[1] -= (impact * 0.012).min(0.12);
        self.vy = 0.0;
    }

    /// Solid guard rails along the track (they have gaps, so you can leave it).
    fn rail_collision(&mut self) {
        let z = self.pos.z;
        let rp = road_dx(z);
        let s = (1.0 + rp * rp).sqrt();
        let d = (self.pos.x - road_x(z)) / s;
        let ad = d.abs();
        if ad < RAIL_OFF - 1.2 || ad > RAIL_OFF + 1.2 {
            return;
        }
        if self.pos.y - terrain_h(self.pos.x, self.pos.z) > 0.7 {
            return; // flying over it
        }
        let side: i32 = if d > 0.0 { 1 } else { -1 };
        let sf = side as f32;
        let zr = z + d * rp / s;
        let blk = (zr / BLOCK).floor() as i32;
        if !rail_exists(side, blk) {
            return;
        }
        let from_road = ad < RAIL_OFF;
        let target = if from_road { RAIL_OFF - 1.05 } else { RAIL_OFF + 1.05 };
        self.pos.x = road_x(z) + sf * target * s;

        let n = Vec3::new(1.0, 0.0, -rp) / s * sf;
        let vn = self.vel.dot(n);
        let toward = if from_road { vn > 0.0 } else { vn < 0.0 };
        if toward {
            self.vel -= n * (vn * 1.3);
            self.vel = self.vel * 0.75;
        }
    }

    /// Tree trunks are solid.
    fn tree_collision(&mut self) {
        let ix0 = (self.pos.x / TREE_CELL).floor() as i32;
        let iz0 = (self.pos.z / TREE_CELL).floor() as i32;
        for ix in ix0 - 1..=ix0 + 1 {
            for iz in iz0 - 1..=iz0 + 1 {
                let Some((tx, tz, s)) = tree_at(ix, iz) else { continue };
                let dx = self.pos.x - tx;
                let dz = self.pos.z - tz;
                let rr = 0.45 * s + 0.35;
                let d2 = dx * dx + dz * dz;
                if d2 < rr * rr {
                    let d = d2.sqrt().max(0.01);
                    let (nx, nz) = (dx / d, dz / d);
                    self.pos.x = tx + nx * rr;
                    self.pos.z = tz + nz * rr;
                    let vn = self.vel.x * nx + self.vel.z * nz;
                    if vn < 0.0 {
                        self.vel.x -= vn * nx * 1.3;
                        self.vel.z -= vn * nz * 1.3;
                        self.vel = self.vel * 0.6;
                    }
                }
            }
        }
    }

    fn building_collision(&mut self) {
        let iz0 = (self.pos.z / BLD_CELL).floor() as i32;
        for iz in iz0 - 1..=iz0 + 1 {
            for b in bld_cell(iz) {
                let (dx, dz) = (self.pos.x - b.x, self.pos.z - b.z);
                if dx * dx + dz * dz > 400.0 {
                    continue;
                }
                // into the building's local frame (rotation_y(yaw): local x -> (cos, -sin), local z -> (sin, cos))
                let (s, c) = b.yaw.sin_cos();
                let lx = dx * c - dz * s;
                let lz = dx * s + dz * c;
                let (ex, ez) = (b.hx + 0.5, b.hz + 0.5);
                if lx.abs() < ex && lz.abs() < ez {
                    let (px, pz) = (ex - lx.abs(), ez - lz.abs());
                    let (nlx, nlz, push) = if px < pz { (lx.signum(), 0.0, px) } else { (0.0, lz.signum(), pz) };
                    // local normal back to world
                    let nx = nlx * c + nlz * s;
                    let nz = -nlx * s + nlz * c;
                    self.pos.x += nx * push;
                    self.pos.z += nz * push;
                    let vn = self.vel.x * nx + self.vel.z * nz;
                    if vn < 0.0 {
                        self.vel.x -= vn * nx * 1.2;
                        self.vel.z -= vn * nz * 1.2;
                        self.vel = self.vel * 0.5;
                    }
                }
            }
        }
    }

    /// Hot-air balloons drifting in the sky (a few hundred vertices, rebuilt each frame).
    pub fn build_balloons(&self, out: &mut Vec<Vertex>) {
        const SPACING: f32 = 260.0;
        let t0 = ((self.pos.z - 450.0) / SPACING).floor() as i32;
        let t1 = ((self.pos.z + 450.0) / SPACING).floor() as i32;
        for t in t0..=t1 {
            let hh = hash(t, 777);
            let bz = t as f32 * SPACING + (hh % 200) as f32;
            let sgn = if hh & 0x100 != 0 { 1.0 } else { -1.0 };
            let bx = road_x(bz) + sgn * (55.0 + ((hh >> 12) & 31) as f32);
            let by = 105.0 + ((hh >> 20) & 15) as f32;
            push_blob(
                out,
                Vec3::new(bx, by, bz),
                Vec3::new(9.0, 11.0, 9.0),
                [0.98, 0.45, 0.12],
                [0.60, 0.26, 0.16],
                t,
                10,
                6,
                0.04,
            );
            push_box(
                out,
                Mat4::from_translation(Vec3::new(bx, by - 13.5, bz)),
                Vec3::ZERO,
                Vec3::new(0.9, 0.6, 0.9),
                [0.35, 0.22, 0.12],
            );
        }
    }

    /// Handlebar, stem, red tank and speed bar, in camera space (view = identity).
    pub fn build_cockpit(&self, out: &mut Vec<Vertex>) {
        let dark = [0.09, 0.10, 0.11];
        let red = [0.72, 0.23, 0.20];
        // the bars move against the rider's head: lean, weight shift and suspension travel
        let sx = -self.rider[0] * 0.6;
        let sy = -(self.rider[1] + self.susp * 0.5) * 0.6;
        let sz = self.rider[2] * 0.6;
        let base = Mat4::from_translation(Vec3::new(sx, -0.40 + sy, -1.1 + sz))
            * Mat4::from_rotation_z(-self.rider[0] * 0.25)
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
