struct U {
    vp: mat4x4<f32>,
    fog: vec4<f32>,     // rgb = fog / horizon colour
    params: vec4<f32>,  // x = fog density (0 = UI), y = hidden/grow amount (1 = just spawned), z = time
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) kind: f32,
    @location(3) dist: f32,
};

@vertex
fn vs_main(@location(0) p: vec3<f32>, @location(1) c: vec3<f32>, @location(2) k: f32) -> VOut {
    var o: VOut;
    var q = p;
    if (k >= 2.0) {
        // grass: blades sway (tip moves, base stays) and grow in
        let s = clamp(k - 2.0, 0.0, 1.0);
        let t = u.params.z;
        let sw = sin(t * 2.3 + p.x * 0.55 + p.z * 0.42) * 0.13 + sin(t * 3.7 + p.z * 0.9 + p.x * 0.3) * 0.05;
        q.x = p.x + sw * s;
        q.z = p.z + sw * 0.6 * s;
        q.y = p.y - u.params.y * s * 1.2;
    } else {
        q.y = p.y - u.params.y * 28.0;
    }
    o.pos = u.vp * vec4<f32>(q, 1.0);
    o.color = c;
    o.world = p;
    o.kind = k;
    o.dist = o.pos.w;
    return o;
}

// Anime / cel look: luminance snapped to soft bands (hue kept), richer colours, warm lights and cool shadows.
fn toon(c: vec3<f32>) -> vec3<f32> {
    let l = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    let bands = 5.0;
    let x = l * bands;
    let fl = floor(x);
    let fr = x - fl;
    let st = smoothstep(0.40, 0.60, fr);
    let lq = (fl + st) / bands;
    var o = c * (lq / max(l, 0.001));
    let g = dot(o, vec3<f32>(0.299, 0.587, 0.114));
    o = mix(vec3<f32>(g, g, g), o, 1.20);
    o = o + vec3<f32>(0.035, 0.020, -0.010) * lq + vec3<f32>(-0.020, 0.000, 0.030) * (1.0 - lq);
    return clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    var col = i.color;
    if (i.kind < 0.5) {
        // procedural checkered ground, like the reference video
        let cx = i32(floor(i.world.x / 4.0));
        let cz = i32(floor(i.world.z / 4.0));
        let chk = f32((cx + cz) & 1);
        col = mix(vec3<f32>(0.42, 0.44, 0.48), vec3<f32>(0.50, 0.52, 0.56), chk);
    }
    if (u.params.x > 0.0) {
        col = toon(col);
    }
    let t = i.dist * u.params.x;
    let f = 1.0 - exp(-t * t);
    var g = max(f, u.params.y);
    if (i.kind >= 2.0) {
        g = f; // grass fades by growing, not by fogging
    }
    return vec4<f32>(mix(col, u.fog.rgb, g), 1.0);
}

// ---- sky: 2k equirectangular image ----
struct SkyU {
    inv_vp: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> sk: SkyU;
@group(1) @binding(1) var sky_tex: texture_2d<f32>;
@group(1) @binding(2) var sky_smp: sampler;

struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_sky(@builtin(vertex_index) idx: u32) -> SkyOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var o: SkyOut;
    o.pos = vec4<f32>(p[idx], 0.0, 1.0);
    o.ndc = p[idx];
    return o;
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    let a = sk.inv_vp * vec4<f32>(i.ndc, 1.0, 1.0);
    let b = sk.inv_vp * vec4<f32>(i.ndc, 0.0, 1.0);
    let dir = normalize(a.xyz / a.w - b.xyz / b.w);
    // sun in the image lines up with the game's light direction
    let az = atan2(dir.x, dir.z) / 6.2831853 + 0.056;
    let el = asin(clamp(dir.y, -1.0, 1.0));
    let uv = vec2<f32>(fract(az), clamp(0.49 - el / 3.14159265, 0.0, 1.0));
    let c = textureSampleLevel(sky_tex, sky_smp, uv, 0.0).rgb;
    return vec4<f32>(c, 1.0);
}
