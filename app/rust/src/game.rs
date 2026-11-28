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
    trail_adjust(x, z, terrain_road(x, z))
}

fn terrain_road(x: f32, z: f32) -> f32 {
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
    terrain_h(x, z) - 0.45 * (1.0 - smooth(8.0, 17.0, d)) - trail_mesh_sink(x, z)
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
    // open the roadside next to a detour's fork so you can turn off the road
    let zc = blk as f32 * BLOCK + 3.0;
    if let Some(zn) = zone_at(zc) {
        if side as f32 == zn.side && (zc < zn.a + 160.0 || zc > zn.b - 160.0) {
            return false;
        }
    }
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
    if canyon_tint(x, z) > 0.12 {
        return None;
    }
    if near_building(x, z, 3.0) || near_bull(x, z, 2.5) {
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
    let ct = canyon_tint(x, z);
    c = mix(c, [0.80, 0.70, 0.64], ct * 0.92);
    let rock = ((slope - 0.45) / 0.3).clamp(0.0, 1.0);
    mix(c, if ct > 0.3 { [0.62, 0.55, 0.52] } else { [0.50, 0.50, 0.44] }, rock * 0.55)
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
// Canyon detour: a rough dirt trail forks off the main road, winds between giant low-poly rocks, crosses
// chasms on plank bridges and rejoins the road further on. No traffic here.
// ------------------------------------------------------------------

const ZONE_CELL: f32 = 1500.0;
const TRAIL_HALF: f32 = 3.6; // half width of the dirt path
const TRAIL_OFF: f32 = 66.0; // lateral distance from the main road at the far point of the detour
const TRAIL_START_OFF: f32 = 10.5; // where the trail forks off the road
const PIT_DEPTH: f32 = 26.0;
const ROCK_CELL: f32 = 100.0;

#[derive(Clone, Copy)]
struct Zone {
    a: f32,
    b: f32,
    side: f32,
}

fn zone_cell(iz: i32) -> Option<Zone> {
    if iz == -1 {
        // the first detour is always near the start so it is easy to find
        return Some(Zone { a: -1100.0, b: -400.0, side: 1.0 });
    }
    let h = hash(iz * 5 + 2, 9001);
    if h % 100 >= 62 {
        return None;
    }
    let a = iz as f32 * ZONE_CELL + 150.0 + ((h >> 8) % 300) as f32;
    let len = 640.0 + ((h >> 16) % 5) as f32 * 60.0;
    let side = if (h >> 20) & 1 == 0 { 1.0 } else { -1.0 };
    Some(Zone { a, b: a + len, side })
}

fn zone_at(z: f32) -> Option<Zone> {
    let zn = zone_cell((z / ZONE_CELL).floor() as i32)?;
    if z >= zn.a - 2.0 && z <= zn.b + 2.0 {
        Some(zn)
    } else {
        None
    }
}

/// 0 at both ends of a detour, 1 in the middle.
fn zone_gate(z: f32, zn: &Zone, edge: f32) -> f32 {
    smooth(zn.a, zn.a + edge, z) * (1.0 - smooth(zn.b - edge, zn.b, z))
}

fn trail_off(z: f32, zn: &Zone) -> f32 {
    let g = zone_gate(z, zn, 230.0);
    TRAIL_START_OFF + (TRAIL_OFF - TRAIL_START_OFF) * g + 8.0 * (z / 55.0).sin() * zone_gate(z, zn, 250.0)
}

fn trail_x(z: f32, zn: &Zone) -> f32 {
    road_x(z) + zn.side * trail_off(z, zn)
}

fn trail_dx(z: f32, zn: &Zone) -> f32 {
    (trail_x(z + 0.5, zn) - trail_x(z - 0.5, zn))
}

/// Trail surface height: equals the road height at the fork, follows the hills further in.
fn trail_h(z: f32, zn: &Zone) -> f32 {
    let r = road_target(z);
    r + (base_h(trail_x(z, zn), z) - r) * 0.55 * zone_gate(z, zn, 230.0)
}

/// Chasm amount (0..1) along the trail: plank bridges span these.
fn pit_amt(z: f32, zn: &Zone) -> f32 {
    let mut best = 0.0f32;
    let mut c = zn.a + 300.0;
    while c < zn.b - 220.0 {
        best = best.max(1.0 - smooth(11.0, 19.0, (z - c).abs()));
        c += 280.0;
    }
    best
}

/// Signed lateral distance from the trail centre line (perpendicular approximation).
fn trail_lat(x: f32, z: f32, zn: &Zone) -> f32 {
    let tdx = trail_dx(z, zn);
    (x - trail_x(z, zn)) / (1.0 + tdx * tdx).sqrt()
}

/// 0..1: how much of a rocky canyon this spot is (no trees / flowers / buildings / bulls in there).
fn canyon_tint(x: f32, z: f32) -> f32 {
    let Some(zn) = zone_at(z) else { return 0.0 };
    let al = (x - trail_x(z, &zn)).abs();
    if al > 50.0 {
        return 0.0;
    }
    let wg = smooth(zn.a, zn.a + 60.0, z) * (1.0 - smooth(zn.b - 60.0, zn.b, z));
    (1.0 - smooth(18.0, 46.0, al)) * wg
}

/// Ground height change caused by the detour (flat trail bed + chasms), applied on top of the road terrain.
fn trail_adjust(x: f32, z: f32, h: f32) -> f32 {
    let Some(zn) = zone_at(z) else { return h };
    let tx = trail_x(z, &zn);
    if (x - tx).abs() > 60.0 {
        return h;
    }
    let al = trail_lat(x, z, &zn).abs();
    if al >= 28.0 {
        return h;
    }
    let w = 1.0 - smooth(9.0, 28.0, al);
    let wg = smooth(zn.a, zn.a + 25.0, z) * (1.0 - smooth(zn.b - 25.0, zn.b, z));
    let mut r = h + (trail_h(z, &zn) - h) * w * wg;
    let p = pit_amt(z, &zn);
    if p > 0.0 {
        r -= PIT_DEPTH * p * (1.0 - smooth(6.0, 24.0, al)) * smooth(3.8, 6.0, al);
    }
    r
}

/// Extra visual sinking of the coarse terrain mesh under the trail / bridge (physics ground unchanged).
fn trail_mesh_sink(x: f32, z: f32) -> f32 {
    let Some(zn) = zone_at(z) else { return 0.0 };
    if (x - trail_x(z, &zn)).abs() > 40.0 {
        return 0.0;
    }
    let al = trail_lat(x, z, &zn).abs();
    let wg = smooth(zn.a, zn.a + 25.0, z) * (1.0 - smooth(zn.b - 25.0, zn.b, z));
    let mut s = 0.45 * (1.0 - smooth(5.0, 12.0, al)) * wg;
    let p = pit_amt(z, &zn);
    if p > 0.0 {
        s += PIT_DEPTH * p * (1.0 - smooth(6.0, 24.0, al)) * (1.0 - smooth(3.8, 6.0, al));
    }
    s
}

#[derive(Clone, Copy)]
pub struct Rock {
    pub x: f32,
    pub z: f32,
    pub r: f32,
    pub h: f32,
    pub seed: u32,
}

fn rocks_compute(iz: i32) -> Vec<Rock> {
    let mut v = Vec::new();
    let z0 = iz as f32 * ROCK_CELL;
    if zone_cell((z0 / ZONE_CELL).floor() as i32).is_none() {
        return v;
    }
    const N: i32 = 7;
    for row in 0..2i32 {
        for j in 0..N {
            for sgi in 0..2i32 {
                let sg = if sgi == 0 { -1.0f32 } else { 1.0 };
                let hs = hash(iz * 97 + j * 13 + row * 5 + sgi * 3, 4711 + row);
                let u = |k: u32| ((hs >> (k * 5)) & 31) as f32 / 31.0;
                if u(5) < 0.12 {
                    continue; // a gap now and then
                }
                let z = z0 + (j as f32 + 0.5 + (u(0) - 0.5) * 0.9) * (ROCK_CELL / N as f32);
                let Some(zn) = zone_at(z) else { continue };
                if zone_gate(z, &zn, 230.0) < 0.55 || pit_amt(z, &zn) > 0.02 {
                    continue;
                }
                let (r, lat, h) = if row == 0 {
                    let r = 3.4 + 4.6 * u(1);
                    (r, TRAIL_HALF + 1.6 + r * 0.85 + 1.5 * u(2), r * (1.5 + 1.8 * u(3)))
                } else {
                    let r = 7.0 + 8.0 * u(1);
                    (r, TRAIL_HALF + 13.0 + r + 8.0 * u(2), r * (2.0 + 2.2 * u(3)))
                };
                let tdx = trail_dx(z, &zn);
                let sd = (1.0 + tdx * tdx).sqrt();
                let x = trail_x(z, &zn) + sg * lat / sd;
                let zz = z - sg * lat * tdx / sd;
                if road_off(x, zz) < 21.0 + r {
                    continue;
                }
                v.push(Rock { x, z: zz, r, h, seed: hs });
            }
        }
    }
    v
}

fn rock_cell(iz: i32) -> Vec<Rock> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<i32, Vec<Rock>>> = RefCell::new(HashMap::new());
    }
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 800 {
            c.clear();
        }
        c.entry(iz).or_insert_with(|| rocks_compute(iz)).clone()
    })
}

/// Faceted low-poly boulder / pillar (pale rock, flat shaded like the reference picture).
fn push_rock(out: &mut Vec<Vertex>, rk: &Rock) {
    let gy = terrain_h(rk.x, rk.z);
    let mut s = rk.seed ^ 0x5bd1_e995;
    let mut rnd = move || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((s >> 8) & 0xffff) as f32 / 65535.0
    };
    const N: usize = 6;
    let tone = 0.88 + 0.12 * rnd();
    let warm = rnd();
    let base = [(0.84 + 0.05 * warm) * tone, (0.77 - 0.02 * warm) * tone, (0.74 - 0.05 * warm) * tone];
    // rings: (height, radius factor, sideways shift factor)
    let rings: [(f32, f32, f32); 3] = [(-2.5, 1.05, 0.0), (0.34 * rk.h, 0.95 + 0.25 * rnd(), 0.08), (0.70 * rk.h, 0.58 + 0.12 * rnd(), 0.20)];
    let lean = (rnd() * 6.283, rnd());
    let mut pts: Vec<[Vec3; N]> = Vec::new();
    for (k, &(yy, rf, sh)) in rings.iter().enumerate() {
        let mut ring = [Vec3::ZERO; N];
        let ox = lean.0.cos() * sh * rk.r * lean.1;
        let oz = lean.0.sin() * sh * rk.r * lean.1;
        for i in 0..N {
            let a = (i as f32 + 0.35 * (rnd() - 0.5)) * 6.283 / N as f32;
            let rr = rk.r * rf * (0.82 + 0.36 * rnd());
            let y = if k == 0 { gy + yy } else { gy + yy + 0.8 * (rnd() - 0.5) };
            ring[i] = Vec3::new(rk.x + ox + a.cos() * rr, y, rk.z + oz + a.sin() * rr);
        }
        pts.push(ring);
    }
    let apex = Vec3::new(rk.x + lean.0.cos() * 0.3 * rk.r * lean.1 + (rnd() - 0.5) * 0.3 * rk.r, gy + rk.h, rk.z + lean.0.sin() * 0.3 * rk.r * lean.1 + (rnd() - 0.5) * 0.3 * rk.r);
    let centre = Vec3::new(rk.x, gy + rk.h * 0.4, rk.z);
    let sun = Vec3::new(0.45, 0.80, 0.40).normalize();
    let mut tri = |a: Vec3, b: Vec3, c: Vec3| {
        let mut n = (b - a).cross(c - a);
        if n.length() < 1e-6 {
            return;
        }
        n = n.normalize();
        if n.dot((a + b + c) / 3.0 - centre) < 0.0 {
            n = Vec3::new(-n.x, -n.y, -n.z);
        }
        let shade = (0.46 + 0.62 * n.dot(sun).max(0.0) + 0.14 * n.y).clamp(0.40, 1.10);
        let var = 0.94 + 0.10 * ((a.x * 7.3 + a.z * 3.1).sin() * 0.5 + 0.5);
        let col = [(base[0] * shade * var).min(1.0), (base[1] * shade * var).min(1.0), (base[2] * shade * var).min(1.0)];
        for p in [a, b, c] {
            out.push(Vertex { pos: p.to_array(), color: col, kind: 1.0 });
        }
    };
    for k in 0..2 {
        for i in 0..N {
            let j = (i + 1) % N;
            let (a, b, c, d) = (pts[k][i], pts[k][j], pts[k + 1][j], pts[k + 1][i]);
            tri(a, b, c);
            tri(a, c, d);
        }
    }
    for i in 0..N {
        let j = (i + 1) % N;
        tri(pts[2][i], pts[2][j], apex);
    }
}

fn build_rocks(x0: f32, z0: f32, out: &mut Vec<Vertex>) {
    let i0 = (z0 / ROCK_CELL).floor() as i32;
    let i1 = ((z0 + CHUNK) / ROCK_CELL).floor() as i32;
    for iz in i0..=i1 {
        for rk in rock_cell(iz) {
            if in_chunk(rk.x, rk.z, x0, z0) {
                push_rock(out, &rk);
            }
        }
    }
}

/// Dirt trail ribbon and plank bridges.
fn build_trail(x0: f32, z0: f32, out: &mut Vec<Vertex>) {
    const STEP: f32 = 3.0;
    let c0 = ((z0 - 10.0) / ZONE_CELL).floor() as i32;
    let c1 = ((z0 + CHUNK + 10.0) / ZONE_CELL).floor() as i32;
    for ci in c0..=c1 {
        let Some(zn) = zone_cell(ci) else { continue };
        if zn.b < z0 - 10.0 || zn.a > z0 + CHUNK + 10.0 {
            continue;
        }
        let tp = |z: f32, off: f32, lift: f32| -> [f32; 3] {
            let tx = trail_x(z, &zn);
            let tdx = trail_dx(z, &zn);
            let sd = (1.0 + tdx * tdx).sqrt();
            [tx + off / sd, trail_h(z, &zn) + lift, z - off * tdx / sd]
        };
        let half = |z: f32| TRAIL_HALF * smooth(zn.a, zn.a + 30.0, z) * (1.0 - smooth(zn.b - 30.0, zn.b, z));
        let k0 = ((z0.max(zn.a) - STEP) / STEP).floor() as i32;
        let k1 = ((((z0 + CHUNK).min(zn.b)) + STEP) / STEP).ceil() as i32;
        for k in k0..k1 {
            let (za, zb) = (k as f32 * STEP, (k + 1) as f32 * STEP);
            let zm = 0.5 * (za + zb);
            if zm < zn.a || zm > zn.b {
                continue;
            }
            let (wa, wb) = (half(za), half(zb));
            if wa < 0.15 && wb < 0.15 {
                continue;
            }
            let mx = trail_x(zm, &zn);
            if !in_chunk(mx, zm, x0, z0) {
                continue;
            }
            let idx = k;
            let tint = 0.93 + 0.12 * h01(idx, 313);
            let pit = pit_amt(zm, &zn);
            if pit < 0.06 {
                // packed dirt with two darker wheel ruts
                quad(out, [tp(za, -wa, 0.21), tp(za, wa, 0.21), tp(zb, wb, 0.21), tp(zb, -wb, 0.21)], scale([0.66, 0.50, 0.36], tint));
                for &sd in &[-1.0f32, 1.0] {
                    let (o0, o1) = (sd * 0.7, sd * 1.5);
                    quad(out, [tp(za, o0 * wa / TRAIL_HALF, 0.23), tp(za, o1 * wa / TRAIL_HALF, 0.23), tp(zb, o1 * wb / TRAIL_HALF, 0.23), tp(zb, o0 * wb / TRAIL_HALF, 0.23)], scale([0.52, 0.38, 0.27], tint));
                }
            } else {
                // wooden plank bridge: planks, stringers, posts and rope rails
                for q in 0..3 {
                    let pz = za + q as f32;
                    let c = if (idx * 3 + q) & 1 == 0 { [0.80, 0.47, 0.20] } else { [0.66, 0.38, 0.17] };
                    quad(out, [tp(pz, -2.55, 0.27), tp(pz, 2.55, 0.27), tp(pz + 0.84, 2.55, 0.27), tp(pz + 0.84, -2.55, 0.27)], scale(c, tint));
                }
                for &sd in &[-2.2f32, 2.2] {
                    quad(out, [tp(za, sd - 0.14, 0.20), tp(za, sd + 0.14, 0.20), tp(zb, sd + 0.14, 0.20), tp(zb, sd - 0.14, 0.20)], [0.30, 0.20, 0.12]);
                }
                for &sd in &[-2.8f32, 2.8] {
                    let p = tp(za, sd, 0.0);
                    push_box(out, Mat4::IDENTITY, Vec3::new(p[0], p[1] + 0.68, p[2]), Vec3::new(0.09, 0.68, 0.09), [0.36, 0.24, 0.14]);
                    for &hgt in &[0.60f32, 1.15] {
                        quad(out, [tp(za, sd, hgt), tp(zb, sd, hgt), tp(zb, sd, hgt + 0.07), tp(za, sd, hgt + 0.07)], [0.86, 0.76, 0.54]);
                    }
                }
            }
        }
    }
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
    build_trail(x0, z0, out);
    build_rocks(x0, z0, out);
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
// Fox: baked gallop animation (16 frames), crosses the road now and then. Harmless: no collision.
// ------------------------------------------------------------------

static FOX_BIN: &[u8] = include_bytes!("fox.bin");
static BULL_BIN: &[u8] = include_bytes!("bull.bin");

pub struct FoxModel {
    /// all frames, one after another (`per_frame` vertices each)
    pub verts: Vec<Vertex>,
    pub per_frame: usize,
    pub frames: usize,
}

pub fn fox_model() -> &'static FoxModel {
    static M: std::sync::OnceLock<FoxModel> = std::sync::OnceLock::new();
    M.get_or_init(|| parse_anim(FOX_BIN))
}

pub fn bull_model() -> &'static FoxModel {
    static M: std::sync::OnceLock<FoxModel> = std::sync::OnceLock::new();
    M.get_or_init(|| parse_anim(BULL_BIN))
}

/// Baked skeletal animation: all frames as plain flat-shaded triangles.
fn parse_anim(d: &'static [u8]) -> FoxModel {
    let u32at = |o: usize| u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize;
    let f32at = |o: usize| f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
    let nf = u32at(0);
    let nv = u32at(4);
    let ni = u32at(8);
    let mut o = 24usize;
    let col_o = o;
    o += nv * 3;
    o = (o + 3) & !3;
    let idx_o = o;
    o += ni * 2;
    o = (o + 3) & !3;
    let pos_o = o;
    let mut verts = Vec::with_capacity(nf * ni);
    for f in 0..nf {
        let base = pos_o + f * nv * 12;
        let p = |i: usize| -> [f32; 3] {
            let q = base + i * 12;
            [f32at(q), f32at(q + 4), f32at(q + 8)]
        };
        for t in 0..ni / 3 {
            let ix = |k: usize| {
                let q = idx_o + (t * 3 + k) * 2;
                u16::from_le_bytes([d[q], d[q + 1]]) as usize
            };
            let (a, b, c) = (ix(0), ix(1), ix(2));
            let (pa, pb, pc) = (p(a), p(b), p(c));
            let e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            let e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            let n = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-9);
            let shade = 0.80 + 0.20 * (n[1] / len).abs();
            for &vi in &[a, b, c] {
                let q = col_o + vi * 3;
                verts.push(Vertex {
                    pos: p(vi),
                    color: [
                        d[q] as f32 / 255.0 * shade,
                        d[q + 1] as f32 / 255.0 * shade,
                        d[q + 2] as f32 / 255.0 * shade,
                    ],
                    kind: 1.0,
                });
            }
        }
    }
    FoxModel { verts, per_frame: ni, frames: nf }
}

pub const FOX_SCALE: f32 = 1.3;
const FOX_ANIM_FPS: f32 = 29.5;

#[derive(Clone, Copy)]
pub struct Fox {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub frame: usize,
    zr: f32,
    side: f32, // direction of travel across the road (+1 = towards +x side)
    off: f32,  // lateral offset from the road centre (m)
    t: f32,
    speed: f32,
}


// ------------------------------------------------------------------
// Grazing bulls beside the road: one alone, or a family (adult + calves), eating animation
// ------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Bull {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub scale: f32,
    /// animation phase 0..1 so the herd does not move in sync
    pub phase: f32,
}

pub const BULL_ANIM_FPS: f32 = 8.0;

fn bull_compute(iz: i32) -> Vec<Bull> {
    let mut res = Vec::new();
    if bull_model().frames == 0 {
        return res;
    }
    let h = hash(iz * 17 + 3, 9091);
    let r = h % 100;
    // some stretches: nothing, a single bull, or a family of 3-5
    let count = if r < 48 { 0 } else if r < 72 { 1 } else { 3 + (h >> 12) as usize % 3 };
    if count == 0 {
        return res;
    }
    let side: f32 = if (h >> 8) & 1 == 0 { 1.0 } else { -1.0 };
    let zr = iz as f32 * BLD_CELL + 20.0 + 160.0 * h01(iz, 71);
    let rp = road_dx(zr);
    let s = (1.0 + rp * rp).sqrt();
    let mut off = 18.0 + 10.0 * h01(iz, 72);
    let base_yaw = 6.283 * h01(iz, 73);
    // try a few cluster positions until the ground is gentle and free
    let mut centre = None;
    for _ in 0..4 {
        let cx = road_x(zr) + side * off / s;
        let cz = zr - side * off * rp / s;
        let (gx, gz) = grad(cx, cz);
        if (gx * gx + gz * gz).sqrt() < 0.30 && canyon_tint(cx, cz) < 0.1 && !near_building(cx, cz, 7.0) && mud_amount(cx, cz) < 0.5 {
            centre = Some((cx, cz));
            break;
        }
        off += 6.0;
    }
    let Some((cx, cz)) = centre else { return res };
    for k in 0..count {
        let a = 6.283 * h01(iz * 7 + k as i32, 74);
        let rr = if k == 0 { 0.0 } else { 2.2 + 2.2 * h01(iz * 7 + k as i32, 75) };
        let x = cx + a.cos() * rr;
        let z = cz + a.sin() * rr;
        if near_building(x, z, 5.0) {
            continue;
        }
        let scale = if k == 0 { 1.0 } else { 0.62 + 0.18 * h01(iz * 7 + k as i32, 76) };
        let yaw = base_yaw + (h01(iz * 7 + k as i32, 77) - 0.5) * 1.4;
        res.push(Bull { x, y: mesh_h(x, z), z, yaw, scale, phase: h01(iz * 7 + k as i32, 78) });
    }
    res
}

fn bull_cell(iz: i32) -> Vec<Bull> {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        static CACHE: RefCell<HashMap<i32, Vec<Bull>>> = RefCell::new(HashMap::new());
    }
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        if c.len() > 600 {
            c.clear();
        }
        c.entry(iz).or_insert_with(|| bull_compute(iz)).clone()
    })
}

/// All bulls within a few hundred metres (in z) of `z`.
pub fn bulls_around(z: f32) -> Vec<Bull> {
    let iz0 = (z / BLD_CELL).floor() as i32;
    let mut v = Vec::new();
    for iz in iz0 - 2..=iz0 + 2 {
        v.extend(bull_cell(iz));
    }
    v
}

fn near_bull(x: f32, z: f32, margin: f32) -> bool {
    let iz0 = (z / BLD_CELL).floor() as i32;
    for iz in iz0 - 1..=iz0 + 1 {
        for b in bull_cell(iz) {
            let r = 1.6 * b.scale + margin;
            let (dx, dz) = (x - b.x, z - b.z);
            if dx * dx + dz * dz < r * r {
                return true;
            }
        }
    }
    false
}

// ------------------------------------------------------------------
// One-way traffic (4 low-poly car models, shared GPU buffers, one draw per car)
// ------------------------------------------------------------------

static CARS_BIN: &[u8] = include_bytes!("cars.bin");

pub struct CarModel {
    pub verts: Vec<Vertex>,
    pub hw: f32,
    pub hl: f32,
    pub h: f32,
}

pub fn car_models() -> &'static Vec<CarModel> {
    static M: std::sync::OnceLock<Vec<CarModel>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let d = CARS_BIN;
        let rf = |o: usize| f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]);
        let mut o = 0usize;
        let mut v = Vec::new();
        while o + 20 <= d.len() {
            let n = u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as usize;
            let (hw, hl, h) = (rf(o + 4), rf(o + 8), rf(o + 12));
            o += 20;
            let mut verts = Vec::with_capacity(n);
            for _ in 0..n {
                verts.push(Vertex {
                    pos: [rf(o), rf(o + 4), rf(o + 8)],
                    color: [d[o + 12] as f32 / 255.0, d[o + 13] as f32 / 255.0, d[o + 14] as f32 / 255.0],
                    kind: 1.0,
                });
                o += 16;
            }
            v.push(CarModel { verts, hw, hl, h });
        }
        v
    })
}

/// A traffic car driving the same way as the player (towards -z) in one of two lanes.
#[derive(Clone, Copy)]
pub struct Car {
    pub model: usize,
    /// road parameter (z of the centre line)
    pub zr: f32,
    /// lateral lane offset (m, + = right)
    pub lane: f32,
    pub cruise: f32,
    pub v: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub hw: f32,
    pub hl: f32,
    pub h: f32,
}

const LANES: [f32; 2] = [-2.9, 2.9];
const N_CARS: usize = 9;

fn rnd(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    (*state & 0xffff) as f32 / 65535.0
}

impl Car {
    fn place(&mut self) {
        let rp = road_dx(self.zr);
        let s = (1.0 + rp * rp).sqrt();
        self.x = road_x(self.zr) + self.lane / s;
        self.z = self.zr - self.lane * rp / s;
        self.y = road_target(self.zr) + 0.20;
        self.yaw = rp.atan();
    }
}

fn new_car(rng: &mut u32, zr: f32, lane_idx: usize) -> Car {
    let ms = car_models();
    let model = ((rnd(rng) * ms.len() as f32) as usize).min(ms.len().saturating_sub(1));
    let m = &ms[model];
    // relaxed traffic: 45..80 km/h, so the bike (up to ~170 km/h) overtakes everything easily
    let cruise = 12.5 + 10.0 * rnd(rng);
    let mut c = Car {
        model,
        zr,
        lane: LANES[lane_idx % 2],
        cruise,
        v: cruise,
        x: 0.0,
        y: 0.0,
        z: 0.0,
        yaw: 0.0,
        pitch: 0.0,
        hw: m.hw,
        hl: m.hl,
        h: m.h,
    };
    c.place();
    c
}

fn init_traffic(rng: &mut u32, from_z: f32) -> Vec<Car> {
    if car_models().is_empty() {
        return Vec::new();
    }
    let mut v: Vec<Car> = Vec::new();
    for i in 0..N_CARS {
        let zr = from_z - 110.0 - i as f32 * 62.0 - rnd(rng) * 30.0;
        let lane = if rnd(rng) < 0.5 { 0 } else { 1 };
        v.push(new_car(rng, zr, lane));
    }
    v
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
const FRONT_X: [bool; 9] = [true, true, false, false, false, false, false, false, false];

fn bld_compute(iz: i32) -> Vec<Bld> {
    let ms = models();
    let mut res = Vec::new();
    if ms.is_empty() {
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
        let mi = (hk as usize) % ms.len();
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
            if hi - lo < 3.2 && canyon_tint(bx, bz) < 0.1 {
                best = Some((bx, bz, lo, rp));
                break;
            }
        }
        if let Some((bx, bz, lo, rp)) = best {
            // face the road
            let (dx, dz) = (-side / s, side * rp / s);
            let jit = (h01(iz * 5 + k, 99) - 0.5) * 0.16;
            let yaw = if FRONT_X.get(mi).copied().unwrap_or(false) { (-dz).atan2(dx) } else { dx.atan2(dz) } + jit;
            res.push(Bld { model: mi, x: bx, y: lo - 0.3, z: bz, yaw, hx: m.hx, hz: m.hz });
        }
        zr += 2.0 * rad + 14.0 + 18.0 * h01(iz * 3 + k, 55);
        if zr > (iz + 1) as f32 * BLD_CELL - 12.0 {
            break; // keep each cell's buildings inside the cell
        }
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
            if canyon_tint(px, pz) > 0.1 || near_building(px, pz, 3.5) || near_bull(px, pz, 2.5) || mud_amount(px, pz) > 0.25 {
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
    Dirt,
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
        Surf::Dirt => SurfP { mu: 0.92, drag: 0.05, rough: 0.060 },
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
    if let Some(zn) = zone_at(z) {
        if (x - trail_x(z, &zn)).abs() < TRAIL_HALF + 1.0 && trail_lat(x, z, &zn).abs() < TRAIL_HALF + 0.3 && zone_gate(z, &zn, 40.0) > 0.05 {
            return if pit_amt(z, &zn) > 0.05 { Surf::Rock } else { Surf::Dirt };
        }
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
    pub cars: Vec<Car>,
    rng: u32,
    pub fox: Option<Fox>,
    fox_timer: f32,
    pub time: f32,
}

impl Game {
    pub fn new() -> Self {
        let x = road_x(0.0);
        let y = terrain_h(x, 0.0);
        let mut rng = 0x1234_5678u32;
        let cars = init_traffic(&mut rng, 0.0);
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
            cars,
            rng,
            fox: None,
            fox_timer: 8.0,
            time: 0.0,
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

        self.time += dt;
        self.update_cars(dt);
        self.update_fox(dt);
        self.rail_collision();
        self.tree_collision();
        self.building_collision();
        self.rock_collision();
        self.bridge_collision();
        self.bull_collision();
        self.car_collision();

        // ---------------- vertical: no jumping - the bike always stays glued to the ground ----------------
        let g1 = terrain_h(self.pos.x, self.pos.z);
        self.vy = ((g1 - self.pos.y) / dt.max(1e-3)).clamp(-25.0, 25.0);
        self.pos.y = g1;
        self.grounded = true;

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

    #[allow(dead_code)]
    /// Touchdown after a jump (unused while jumping is disabled): the suspension absorbs the hit; a bad attitude or a huge drop costs speed.
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

    /// A fox now and then gallops across the road ahead. It is only scenery: the bike cannot hit it.
    fn update_fox(&mut self, dt: f32) {
        if fox_model().frames == 0 {
            return;
        }
        let f = self.forward();
        let bz = self.pos.z;
        if let Some(fx) = &mut self.fox {
            fx.t += dt;
            fx.off += fx.side * fx.speed * dt;
            fx.frame = ((fx.t * FOX_ANIM_FPS) as usize) % fox_model().frames;
            let rp = road_dx(fx.zr);
            let s = (1.0 + rp * rp).sqrt();
            fx.x = road_x(fx.zr) + fx.off / s;
            fx.z = fx.zr - fx.off * rp / s;
            let ao = fx.off.abs();
            let target = if ao < ROAD_HALF + 0.7 {
                road_target(fx.zr) + 0.20
            } else if ao < 11.5 {
                road_target(fx.zr) + 0.16
            } else {
                mesh_h(fx.x, fx.z)
            };
            fx.y += (target - fx.y) * (dt * 14.0).min(1.0);
            let (hx, hz) = (fx.side / s, -fx.side * rp / s);
            fx.yaw = (-hx).atan2(-hz);
            // done: crossed the road, or the bike left it far behind
            if ao > 19.0 || fx.z - bz > 80.0 {
                self.fox = None;
                self.fox_timer = 14.0 + 26.0 * rnd(&mut self.rng);
            }
            return;
        }
        self.fox_timer -= dt;
        if self.fox_timer <= 0.0 {
            if self.speed < 6.0 || f.z > -0.3 {
                self.fox_timer = 2.0; // wait until the bike is rolling along the road
                return;
            }
            // appear far ahead (visible from a distance), timed so it crosses as the bike arrives
            let d = (self.speed * (1.9 + 2.6 * rnd(&mut self.rng))).clamp(80.0, 260.0);
            let zr = bz - d;
            let side = if rnd(&mut self.rng) < 0.5 { 1.0 } else { -1.0 };
            let speed = 9.0 + 3.5 * rnd(&mut self.rng);
            let mut fx = Fox {
                x: 0.0,
                y: road_target(zr),
                z: zr,
                yaw: 0.0,
                frame: 0,
                zr,
                side,
                off: -side * 18.0,
                t: 0.0,
                speed,
            };
            fx.x = road_x(zr);
            self.fox = Some(fx);
        }
    }

    /// Move the traffic: slower on climbs, faster downhill, queueing behind slower cars in the same lane.
    fn update_cars(&mut self, dt: f32) {
        let bz = self.pos.z;
        let bike_v = self.speed;
        let snapshot: Vec<(f32, f32, f32)> = self.cars.iter().map(|c| (c.zr, c.lane, c.v)).collect();
        for i in 0..self.cars.len() {
            let (zr, lane) = (self.cars[i].zr, self.cars[i].lane);
            let rp = road_dx(zr);
            let sh = (1.0 + rp * rp).sqrt();
            let rd = (road_target(zr + 1.5) - road_target(zr - 1.5)) / 3.0; // dh/dz
            let climb = -rd / sh; // rise per metre while driving towards -z
            let mut vt = self.cars[i].cruise * (1.0 - 1.7 * climb).clamp(0.6, 1.3);
            // keep distance to the car in front (same lane)
            for (j, &(zj, lj, vj)) in snapshot.iter().enumerate() {
                if j == i || (lj - lane).abs() > 0.5 {
                    continue;
                }
                let gap = zr - zj; // > 0 : the other car is ahead
                if gap > 0.0 && gap < 30.0 {
                    vt = vt.min(vj * if gap < 15.0 { 0.8 } else { 1.0 });
                }
            }
            // never run the rider down from behind
            let behind = zr - bz; // > 0 : behind the bike
            if behind > 0.0 && behind < 25.0 && bike_v < vt {
                vt = vt.min(bike_v.max(0.0) * 0.9);
            }
            let c = &mut self.cars[i];
            let dv = (vt - c.v).clamp(-6.0 * dt, 3.0 * dt);
            c.v += dv;
            let s3 = (1.0 + rp * rp + rd * rd).sqrt();
            c.zr -= c.v * dt / s3;
            c.place();
            // body pitch follows the slope, nose dips when braking and squats when accelerating
            let target = climb.atan() - dv / dt.max(1e-3) * 0.006;
            c.pitch += (target - c.pitch) * (dt * 4.0).min(1.0);
        }
        // recycle cars that were left far behind (the bike is much faster) or are too far ahead
        for i in 0..self.cars.len() {
            let d = self.cars[i].zr - bz; // + behind, - ahead
            if d > 70.0 || d < -700.0 {
                let mut ok = None;
                for _ in 0..6 {
                    let zr = bz - 340.0 - rnd(&mut self.rng) * 240.0;
                    let lane = if rnd(&mut self.rng) < 0.5 { 0 } else { 1 };
                    let lane_off = LANES[lane];
                    let clear = self
                        .cars
                        .iter()
                        .enumerate()
                        .all(|(j, o)| j == i || (o.lane - lane_off).abs() > 0.5 || (o.zr - zr).abs() > 45.0);
                    if clear {
                        ok = Some((zr, lane));
                        break;
                    }
                }
                if let Some((zr, lane)) = ok {
                    let mut c = new_car(&mut self.rng, zr, lane);
                    c.pitch = 0.0;
                    self.cars[i] = c;
                }
            }
        }
    }

    /// Bulls are solid (like a tree trunk): the bike stops.
    fn bull_collision(&mut self) {
        let iz0 = (self.pos.z / BLD_CELL).floor() as i32;
        for iz in iz0 - 1..=iz0 + 1 {
            for b in bull_cell(iz) {
                let rr = 1.0 * b.scale + 0.45;
                let (dx, dz) = (self.pos.x - b.x, self.pos.z - b.z);
                let d2 = dx * dx + dz * dz;
                if d2 < rr * rr && self.pos.y < b.y + 1.6 * b.scale {
                    let d = d2.sqrt().max(0.01);
                    let (nx, nz) = (dx / d, dz / d);
                    self.pos.x = b.x + nx * rr;
                    self.pos.z = b.z + nz * rr;
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

    /// Cars are solid: hitting one stops the bike like a guard rail.
    fn car_collision(&mut self) {
        for k in 0..self.cars.len() {
            let c = self.cars[k];
            let (dx, dz) = (self.pos.x - c.x, self.pos.z - c.z);
            if dx * dx + dz * dz > 36.0 {
                continue;
            }
            if self.pos.y > c.y + c.h + 0.2 {
                continue; // flying over it
            }
            let (s, co) = c.yaw.sin_cos();
            let lx = dx * co - dz * s;
            let lz = dx * s + dz * co;
            let (ex, ez) = (c.hw + 0.45, c.hl + 0.45);
            if lx.abs() < ex && lz.abs() < ez {
                let (px, pz) = (ex - lx.abs(), ez - lz.abs());
                let (nlx, nlz, push) = if px < pz { (lx.signum(), 0.0, px) } else { (0.0, lz.signum(), pz) };
                let nx = nlx * co + nlz * s;
                let nz = -nlx * s + nlz * co;
                self.pos.x += nx * push;
                self.pos.z += nz * push;
                // relative to the (moving) car
                let cv = Vec3::new(-c.yaw.sin() * c.v, 0.0, -c.yaw.cos() * c.v);
                let rel = self.vel - cv;
                let vn = rel.x * nx + rel.z * nz;
                if vn < 0.0 {
                    self.vel.x -= nx * vn * 1.3;
                    self.vel.z -= nz * vn * 1.3;
                    self.vel = self.vel * 0.75;
                }
            }
        }
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

    /// Giant canyon rocks are solid.
    fn rock_collision(&mut self) {
        if zone_cell((self.pos.z / ZONE_CELL).floor() as i32).is_none() {
            return;
        }
        let iz0 = (self.pos.z / ROCK_CELL).floor() as i32;
        for iz in iz0 - 1..=iz0 + 1 {
            for rk in rock_cell(iz) {
                let (dx, dz) = (self.pos.x - rk.x, self.pos.z - rk.z);
                let rr = rk.r * 0.88 + 0.5;
                let d2 = dx * dx + dz * dz;
                if d2 < rr * rr {
                    let d = d2.sqrt().max(0.01);
                    let (nx, nz) = (dx / d, dz / d);
                    self.pos.x = rk.x + nx * rr;
                    self.pos.z = rk.z + nz * rr;
                    let vn = self.vel.x * nx + self.vel.z * nz;
                    if vn < 0.0 {
                        self.vel.x -= vn * nx * 1.2;
                        self.vel.z -= vn * nz * 1.2;
                        self.vel = self.vel * 0.55;
                    }
                }
            }
        }
    }

    /// Rope rails of the plank bridges keep the bike on the deck.
    fn bridge_collision(&mut self) {
        let z = self.pos.z;
        let Some(zn) = zone_at(z) else { return };
        if pit_amt(z, &zn) < 0.005 {
            return;
        }
        let tdx = trail_dx(z, &zn);
        let sd = (1.0 + tdx * tdx).sqrt();
        let lat = (self.pos.x - trail_x(z, &zn)) / sd;
        const LIM: f32 = 2.85;
        // only while actually on / right beside the deck (never drag a bike riding the main road)
        if lat.abs() > LIM && lat.abs() < 4.6 {
            let sgn = lat.signum();
            self.pos.x = trail_x(z, &zn) + sgn * LIM * sd;
            let n = Vec3::new(1.0, 0.0, -tdx) / sd * sgn;
            let vn = self.vel.dot(n);
            if vn > 0.0 {
                self.vel -= n * (vn * 1.15);
                self.vel = self.vel * 0.985;
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

    /// First-person cockpit (view = identity): sport-bike bars, mirrors, forks, instrument cluster with a live
    /// speed arc and a glossy two-tone tank. The static mesh is built once and only transformed per frame.
    pub fn build_cockpit(&self, out: &mut Vec<Vertex>) {
        let v3 = Vec3::new;
        thread_local! {
            static STATIC: (Vec<Vertex>, Vec<Vertex>) = {
                let mut bars = Vec::with_capacity(3000);
                cockpit_bars(&mut bars);
                let mut tank = Vec::with_capacity(3000);
                cockpit_tank(&mut tank);
                (bars, tank)
            };
        }
        // the bars move against the rider's head: lean, weight shift and suspension travel
        let sx = -self.rider[0] * 0.6;
        let sy = -(self.rider[1] + self.susp * 0.5) * 0.6;
        let sz = self.rider[2] * 0.6;
        let base = Mat4::from_translation(v3(sx, -0.40 + sy, -1.1 + sz))
            * Mat4::from_rotation_z(-self.rider[0] * 0.25)
            * Mat4::from_rotation_y(-self.steer * 0.35)
            * Mat4::from_rotation_x(0.25);
        STATIC.with(|(bars, tank)| {
            for q in bars.iter() {
                out.push(Vertex { pos: base.transform_point3(v3(q.pos[0], q.pos[1], q.pos[2])).to_array(), color: q.color, kind: 1.0 });
            }
            out.extend_from_slice(tank);
        });

        // live speed arc + needle (the only per-frame geometry)
        let frac = (self.speed / MAX_SPEED).clamp(0.0, 1.0);
        let c = v3(0.0, 0.058, -0.092);
        let pol = |a: f32, r: f32, z: f32| v3(c.x + a.cos() * r, c.y + a.sin() * r, c.z + z);
        let a0 = 225f32.to_radians();
        let sweep = 270f32.to_radians();
        let n_seg = 22;
        for i in 0..n_seg {
            let t0 = a0 - sweep * i as f32 / n_seg as f32;
            let t1 = a0 - sweep * (i as f32 + 0.8) / n_seg as f32;
            let on = (i as f32) < frac * n_seg as f32;
            let k = i as f32 / n_seg as f32;
            let col = if !on {
                [0.08, 0.10, 0.15]
            } else if k < 0.55 {
                [0.20, 0.95, 0.45]
            } else if k < 0.8 {
                [1.0, 0.82, 0.15]
            } else {
                [1.0, 0.25, 0.15]
            };
            push_quad_m(out, base, [pol(t0, 0.060, 0.002), pol(t1, 0.060, 0.002), pol(t1, 0.078, 0.002), pol(t0, 0.078, 0.002)], col);
        }
        let na = a0 - sweep * frac;
        let (nx, ny) = (na.cos(), na.sin());
        let (px, py) = (-ny * 0.004, nx * 0.004);
        push_quad_m(
            out,
            base,
            [v3(c.x + px, c.y + py, c.z + 0.004), v3(c.x - px, c.y - py, c.z + 0.004), v3(c.x + nx * 0.060 - px * 0.3, c.y + ny * 0.060 - py * 0.3, c.z + 0.004), v3(c.x + nx * 0.060 + px * 0.3, c.y + ny * 0.060 + py * 0.3, c.z + 0.004)],
            [1.0, 0.18, 0.12],
        );
    }
}

/// Everything attached to the handlebar, in bar-local space (no hands).
fn cockpit_bars(out: &mut Vec<Vertex>) {
    let metal = [0.20, 0.21, 0.24];
    let chrome = [0.78, 0.80, 0.84];
    let rubber = [0.05, 0.05, 0.055];
    let red = [0.70, 0.06, 0.09];
    let gold = [0.90, 0.68, 0.16];
    let v3 = Vec3::new;
    let m = Mat4::IDENTITY;

    let bar = [v3(-0.57, -0.02, 0.11), v3(-0.40, 0.0, 0.03), v3(-0.20, 0.0, 0.0), v3(0.20, 0.0, 0.0), v3(0.40, 0.0, 0.03), v3(0.57, -0.02, 0.11)];
    for w in bar.windows(2) {
        push_tube(out, m, w[0], w[1], 0.016, 0.016, metal, 0.7, 8);
    }
    for s in [-1.0f32, 1.0] {
        // grip with chrome bar-end weight
        push_tube(out, m, v3(s * 0.36, 0.0, 0.035), v3(s * 0.57, -0.02, 0.11), 0.029, 0.029, rubber, 0.15, 8);
        push_ell(out, m, v3(s * 0.58, -0.02, 0.115), v3(0.026, 0.026, 0.026), chrome, 1.0, 8, 5);
        // clamp + switch block
        push_ell(out, m, v3(s * 0.31, 0.0, 0.0), v3(0.045, 0.032, 0.04), chrome, 0.9, 8, 5);
        push_ell(out, m, v3(s * 0.27, 0.025, 0.03), v3(0.05, 0.022, 0.03), [0.06, 0.06, 0.07], 0.5, 8, 4);
        // lever
        push_tube(out, m, v3(s * 0.40, -0.005, 0.0), v3(s * 0.33, -0.03, -0.24), 0.010, 0.006, chrome, 0.9, 6);
        // mirror
        push_tube(out, m, v3(s * 0.31, 0.01, -0.03), v3(s * 0.45, 0.17, -0.07), 0.007, 0.006, rubber, 0.3, 6);
        push_ell(out, m, v3(s * 0.46, 0.18, -0.08), v3(0.07, 0.046, 0.024), red, 1.0, 10, 6);
        push_ell(out, m, v3(s * 0.46, 0.18, -0.059), v3(0.058, 0.036, 0.005), [0.55, 0.76, 0.92], 1.0, 9, 4);
        // fork leg: chrome upper tube, gold lower slider
        push_tube(out, m, v3(s * 0.12, -0.05, -0.03), v3(s * 0.13, -0.30, -0.12), 0.024, 0.024, chrome, 1.0, 8);
        push_tube(out, m, v3(s * 0.13, -0.30, -0.12), v3(s * 0.145, -0.62, -0.25), 0.03, 0.03, gold, 0.9, 8);
        // fairing side panel
        push_ell(out, m, v3(s * 0.25, -0.02, -0.25), v3(0.07, 0.075, 0.22), red, 1.0, 10, 6);
    }
    // brake fluid reservoir (right) and triple clamp
    push_ell(out, m, v3(0.26, 0.055, 0.0), v3(0.02, 0.025, 0.02), [0.85, 0.85, 0.9], 1.0, 8, 4);
    push_ell(out, m, v3(0.0, -0.045, -0.02), v3(0.17, 0.026, 0.06), chrome, 1.0, 10, 5);

    // instrument cluster housing, dial face, ring and tick marks
    push_ell(out, m, v3(0.0, 0.055, -0.15), v3(0.125, 0.075, 0.055), [0.05, 0.05, 0.06], 0.6, 12, 7);
    let c = v3(0.0, 0.058, -0.092);
    let pol = |a: f32, r: f32, z: f32| v3(c.x + a.cos() * r, c.y + a.sin() * r, c.z + z);
    let n_face = 24;
    for i in 0..n_face {
        let t0 = std::f32::consts::TAU * i as f32 / n_face as f32;
        let t1 = std::f32::consts::TAU * (i + 1) as f32 / n_face as f32;
        push_tri_m(out, m, [c, pol(t0, 0.088, 0.0), pol(t1, 0.088, 0.0)], [0.02, 0.04, 0.09]);
        push_quad_m(out, m, [pol(t0, 0.088, 0.001), pol(t1, 0.088, 0.001), pol(t1, 0.098, 0.001), pol(t0, 0.098, 0.001)], [0.70, 0.86, 1.0]);
    }
    let a0 = 225f32.to_radians();
    let sweep = 270f32.to_radians();
    for i in 0..=10 {
        let t = a0 - sweep * i as f32 / 10.0;
        let w = 0.004;
        push_quad_m(out, m, [pol(t - w, 0.080, 0.002), pol(t + w, 0.080, 0.002), pol(t + w, 0.087, 0.002), pol(t - w, 0.087, 0.002)], [0.85, 0.92, 1.0]);
    }
    push_ell(out, m, v3(c.x, c.y, c.z + 0.004), v3(0.012, 0.012, 0.006), chrome, 1.0, 8, 4);
}

/// Tank in camera space: deep metallic red, white racing stripe, chrome filler cap, knee pads.
fn cockpit_tank(out: &mut Vec<Vertex>) {
    let v3 = Vec3::new;
    let tank = Mat4::from_translation(v3(0.0, -0.56, -1.0)) * Mat4::from_rotation_x(0.12);
    push_tank(out, tank, v3(0.26, 0.14, 0.55), [0.70, 0.06, 0.09], [0.95, 0.95, 0.97]);
    push_ell(out, tank, v3(0.0, 0.135, 0.03), v3(0.055, 0.016, 0.055), [0.78, 0.80, 0.84], 1.0, 12, 3);
    push_ell(out, tank, v3(0.0, 0.147, 0.03), v3(0.025, 0.008, 0.025), [0.1, 0.1, 0.1], 0.6, 8, 3);
    for s in [-1.0f32, 1.0] {
        push_ell(out, tank, v3(s * 0.235, -0.01, 0.12), v3(0.03, 0.09, 0.2), [0.05, 0.05, 0.055], 0.1, 8, 4);
    }
}

fn shade_col(n: Vec3, base: [f32; 3], gloss: f32) -> [f32; 3] {
    let l = Vec3::new(-0.35, 0.8, 0.5).normalize();
    let h = (l + Vec3::new(0.0, 0.0, 1.0)).normalize();
    let d = n.dot(l).max(0.0);
    let sp = n.dot(h).max(0.0).powf(26.0) * gloss * 0.9;
    let s = 0.36 + 0.64 * d;
    [(base[0] * s + sp).min(1.0), (base[1] * s + sp).min(1.0), (base[2] * s + sp).min(1.0)]
}

fn xf_dir(m: Mat4, d: Vec3) -> Vec3 {
    let o = m.transform_point3(Vec3::ZERO);
    (m.transform_point3(d) - o).normalize()
}

/// Smooth-shaded ellipsoid.
fn push_ell(out: &mut Vec<Vertex>, m: Mat4, c: Vec3, r: Vec3, color: [f32; 3], gloss: f32, seg: usize, rings: usize) {
    let pt = |i: usize, j: usize| -> Vertex {
        let th = std::f32::consts::PI * i as f32 / rings as f32;
        let ph = std::f32::consts::TAU * j as f32 / seg as f32;
        let d = Vec3::new(ph.cos() * th.sin(), th.cos(), ph.sin() * th.sin());
        let p = m.transform_point3(c + Vec3::new(d.x * r.x, d.y * r.y, d.z * r.z));
        let n = xf_dir(m, Vec3::new(d.x / r.x, d.y / r.y, d.z / r.z));
        Vertex { pos: p.to_array(), color: shade_col(n, color, gloss), kind: 1.0 }
    };
    for i in 0..rings {
        for j in 0..seg {
            let (a, b, cc, d) = (pt(i, j), pt(i, j + 1), pt(i + 1, j + 1), pt(i + 1, j));
            out.extend_from_slice(&[a, b, cc, a, cc, d]);
        }
    }
}

/// Glossy tank ellipsoid with a painted racing stripe (per-vertex colours, no overlapping geometry).
fn push_tank(out: &mut Vec<Vertex>, m: Mat4, r: Vec3, body: [f32; 3], stripe: [f32; 3]) {
    let (seg, rings) = (32usize, 12usize);
    let pt = |i: usize, j: usize| -> Vertex {
        let th = std::f32::consts::PI * i as f32 / rings as f32;
        let ph = std::f32::consts::TAU * j as f32 / seg as f32;
        let d = Vec3::new(ph.cos() * th.sin(), th.cos(), ph.sin() * th.sin());
        let p = m.transform_point3(Vec3::new(d.x * r.x, d.y * r.y, d.z * r.z));
        let n = xf_dir(m, Vec3::new(d.x / r.x, d.y / r.y, d.z / r.z));
        let on = d.y > 0.45 && (d.x * r.x).abs() < 0.032;
        Vertex { pos: p.to_array(), color: shade_col(n, if on { stripe } else { body }, 1.0), kind: 1.0 }
    };
    for i in 0..rings {
        for j in 0..seg {
            let (a, b, cc, d) = (pt(i, j), pt(i, j + 1), pt(i + 1, j + 1), pt(i + 1, j));
            out.extend_from_slice(&[a, b, cc, a, cc, d]);
        }
    }
}

/// Smooth-shaded tapered tube from a to b (in m space).
fn push_tube(out: &mut Vec<Vertex>, m: Mat4, a: Vec3, b: Vec3, r0: f32, r1: f32, color: [f32; 3], gloss: f32, seg: usize) {
    let ax = (b - a).normalize();
    let helper = if ax.y.abs() < 0.9 { Vec3::Y } else { Vec3::new(1.0, 0.0, 0.0) };
    let u = ax.cross(helper).normalize();
    let w = ax.cross(u);
    let ring = |t: f32, j: usize| -> Vertex {
        let ph = std::f32::consts::TAU * j as f32 / seg as f32;
        let d = u * ph.cos() + w * ph.sin();
        let (p, r) = if t < 0.5 { (a, r0) } else { (b, r1) };
        let pos = m.transform_point3(p + d * r);
        Vertex { pos: pos.to_array(), color: shade_col(xf_dir(m, d), color, gloss), kind: 1.0 }
    };
    for j in 0..seg {
        let (p0, p1, q0, q1) = (ring(0.0, j), ring(0.0, j + 1), ring(1.0, j), ring(1.0, j + 1));
        out.extend_from_slice(&[p0, p1, q1, p0, q1, q0]);
    }
}

fn push_tri_m(out: &mut Vec<Vertex>, m: Mat4, p: [Vec3; 3], color: [f32; 3]) {
    for q in p.iter() {
        out.push(Vertex { pos: m.transform_point3(*q).to_array(), color, kind: 1.0 });
    }
}

fn push_quad_m(out: &mut Vec<Vertex>, m: Mat4, p: [Vec3; 4], color: [f32; 3]) {
    push_tri_m(out, m, [p[0], p[1], p[2]], color);
    push_tri_m(out, m, [p[0], p[2], p[3]], color);
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
