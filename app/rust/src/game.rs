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

/// Gentle rolling ground.
fn natural(x: f32, z: f32) -> f32 {
    let n = vnoise(x / 300.0, z / 300.0) * 0.66
        + vnoise(x / 110.0 + 17.3, z / 110.0 - 9.1) * 0.27
        + vnoise(x / 60.0 - 3.7, z / 60.0 + 5.2) * 0.07;
    (n - 0.5) * 20.0
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
    natural(road_x(z), z)
}

pub fn terrain_h(x: f32, z: f32) -> f32 {
    let nat = natural(x, z);
    let dx = (x - road_x(z)).abs();
    if dx >= 44.0 {
        return nat;
    }
    let w = if dx <= 22.0 {
        1.0
    } else {
        let t = (dx - 22.0) / 22.0;
        1.0 - t * t * (3.0 - 2.0 * t)
    };
    nat + (road_target(z) - nat) * w
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
    let p = if dd < 70.0 { 0.50 } else { 0.16 };
    if ((h & 0xff) as f32 / 255.0) > p {
        return None;
    }
    let s = 0.8 + 0.5 * (((h >> 24) & 0xff) as f32 / 255.0);
    Some((x, z, s))
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn terrain_color(x: f32, z: f32, slope: f32) -> [f32; 3] {
    let patch = vnoise(x / 45.0 + 3.0, z / 45.0 - 8.0);
    let t = ((patch - 0.35) / 0.3).clamp(0.0, 1.0);
    let mut c = mix([0.50, 0.78, 0.26], [0.62, 0.87, 0.34], t);
    let fine = vnoise(x / 9.0, z / 9.0);
    c = mix(c, [0.56, 0.82, 0.30], fine * 0.25);
    let rock = ((slope - 0.28) / 0.2).clamp(0.0, 1.0);
    mix(c, [0.52, 0.55, 0.40], rock * 0.6)
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
            hs[j * m + i] = terrain_h(x0 + (i as f32 - 1.0) * CELL, z0 + (j as f32 - 1.0) * CELL);
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
            let base = terrain_color(x, z, slope);
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
            [pt(a, -ROAD_HALF, 0.16), pt(a, ROAD_HALF, 0.16), pt(b, ROAD_HALF, 0.16), pt(b, -ROAD_HALF, 0.16)],
            scale([0.56, 0.61, 0.66], tint),
        );
        // white edge lines
        for &sd in &[-1.0f32, 1.0] {
            let o0 = sd * (ROAD_HALF - 0.38);
            let o1 = sd * (ROAD_HALF - 0.15);
            quad(out, [pt(a, o0, 0.2), pt(a, o1, 0.2), pt(b, o1, 0.2), pt(b, o0, 0.2)], [0.96, 0.96, 0.96]);
        }
        // sand run-off (varying width)
        let sand = scale([0.93, 0.82, 0.58], tint);
        if a.wl > 0.05 || b.wl > 0.05 {
            let i0 = -(ROAD_HALF + 0.7);
            quad(
                out,
                [pt(a, i0, 0.10), pt(a, i0 - a.wl, 0.10), pt(b, i0 - b.wl, 0.10), pt(b, i0, 0.10)],
                sand,
            );
        }
        if a.wr > 0.05 || b.wr > 0.05 {
            let i0 = ROAD_HALF + 0.7;
            quad(
                out,
                [pt(a, i0, 0.10), pt(a, i0 + a.wr, 0.10), pt(b, i0 + b.wr, 0.10), pt(b, i0, 0.10)],
                sand,
            );
        }
        // black / white kerbs on the inside of bends
        if a.kerb != 0 {
            let sd = a.kerb as f32;
            let o0 = sd * ROAD_HALF;
            let o1 = sd * (ROAD_HALF + 1.3);
            let c = if idx & 1 == 0 { [0.12, 0.12, 0.14] } else { [0.96, 0.96, 0.96] };
            quad(out, [pt(a, o0, 0.2), pt(a, o1, 0.2), pt(b, o1, 0.2), pt(b, o0, 0.2)], c);
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
            let gy = road_target(zr);
            let m = Mat4::from_translation(Vec3::new(cx, gy, cz)) * Mat4::from_rotation_y(rp.atan());
            push_box(out, m, Vec3::new(0.0, 0.62, 0.0), Vec3::new(0.05, 0.17, BLOCK * 0.48), [0.60, 0.65, 0.70]);
            push_box(out, m, Vec3::new(0.0, 0.34, 0.0), Vec3::new(0.04, 0.08, BLOCK * 0.48), [0.52, 0.57, 0.62]);
            push_box(out, m, Vec3::new(0.0, 0.40, 0.0), Vec3::new(0.06, 0.40, 0.06), [0.40, 0.44, 0.48]);
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

/// Low-poly trees: tall dark trunk and three faceted canopy blobs.
fn build_trees(cx: i32, cz: i32, out: &mut Vec<Vertex>) {
    for ix in cx * TPC..cx * TPC + TPC {
        for iz in cz * TPC..cz * TPC + TPC {
            let Some((x, z, s)) = tree_at(ix, iz) else { continue };
            let h = hash(ix, iz);
            let gy = terrain_h(x, z);
            let th = 6.5 * s;
            let yaw = (h & 0xff) as f32 / 255.0 * 6.283;
            let m = Mat4::from_translation(Vec3::new(x, gy + th * 0.5 - 0.2, z)) * Mat4::from_rotation_y(yaw);
            push_box(out, m, Vec3::ZERO, Vec3::new(0.28 * s, th * 0.5 + 0.2, 0.28 * s), [0.24, 0.14, 0.11]);

            let v = ((h >> 4) & 0xff) as f32 / 255.0;
            let top = [0.28 + 0.12 * v, 0.62 + 0.10 * v, 0.14 + 0.06 * v];
            let bot = [0.10, 0.30, 0.08];
            let cy = gy + th + 0.6 * s;
            let seed = (h & 0x7fff) as i32;
            push_blob(out, Vec3::new(x, cy, z), Vec3::new(3.4 * s, 1.7 * s, 3.4 * s), top, bot, seed, 6, 3, 0.22);
            push_blob(out, Vec3::new(x + 1.9 * s, cy + 1.0 * s, z - 1.0 * s), Vec3::new(2.5 * s, 1.3 * s, 2.5 * s), top, bot, seed + 5, 6, 3, 0.22);
            push_blob(out, Vec3::new(x - 1.7 * s, cy + 1.2 * s, z + 1.3 * s), Vec3::new(2.3 * s, 1.2 * s, 2.3 * s), top, bot, seed + 9, 6, 3, 0.22);
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
        let y = terrain_h(x, 0.0);
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

        let dperp = road_dist(self.pos.x, self.pos.z);
        let on_road = dperp < ROAD_HALF + 0.3;
        let on_sand = !on_road && dperp < ROAD_HALF + 4.0;

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
            let surface_drag = if on_road {
                0.0
            } else if on_sand {
                0.07
            } else {
                0.12
            };
            vf -= (0.035 + surface_drag) * vf * dt;
            if vf < 0.0 {
                vf -= 2.0 * vf * dt; // rolling backwards stops quickly
            }

            // Grip: best on asphalt; the bike slides when turned hard at speed.
            let grip = if on_road {
                4.5
            } else if on_sand {
                3.0
            } else {
                2.4
            };
            let slide = 1.0 / (1.0 + self.steer.abs() * vf.abs() / 22.0);
            vr *= (-grip * slide * dt).exp();

            self.vel = f * vf + r * vr;

            // Gravity along the slope.
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

        self.rail_collision();
        self.tree_collision();

        // Vertical motion: follow the ground, or fly off crests.
        let g1 = terrain_h(self.pos.x, self.pos.z);
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

    /// Hot-air balloons drifting in the sky (a few hundred vertices, rebuilt each frame).
    pub fn build_balloons(&self, out: &mut Vec<Vertex>) {
        const SPACING: f32 = 380.0;
        let t0 = ((self.pos.z - 450.0) / SPACING).floor() as i32;
        let t1 = ((self.pos.z + 450.0) / SPACING).floor() as i32;
        for t in t0..=t1 {
            let hh = hash(t, 777);
            let bz = t as f32 * SPACING + (hh % 300) as f32;
            let sgn = if hh & 0x100 != 0 { 1.0 } else { -1.0 };
            let bx = road_x(bz) + sgn * (70.0 + ((hh >> 12) & 63) as f32);
            let by = 62.0 + ((hh >> 20) & 15) as f32;
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
