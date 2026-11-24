struct U {
    vp: mat4x4<f32>,
    fog: vec4<f32>,     // rgb = fog / horizon colour
    params: vec4<f32>,  // x = fog density
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
    o.pos = u.vp * vec4<f32>(p, 1.0);
    o.color = c;
    o.world = p;
    o.kind = k;
    o.dist = o.pos.w;
    return o;
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
    let t = i.dist * u.params.x;
    let f = 1.0 - exp(-t * t);
    return vec4<f32>(mix(col, u.fog.rgb, f), 1.0);
}

// ---- sky: fullscreen gradient ----
struct SkyOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
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
    o.uv = p[idx] * 0.5 + vec2<f32>(0.5, 0.5);
    return o;
}

@fragment
fn fs_sky(i: SkyOut) -> @location(0) vec4<f32> {
    let top = vec3<f32>(0.36, 0.50, 0.74);
    let k = smoothstep(0.5, 1.0, clamp(i.uv.y, 0.0, 1.0));
    return vec4<f32>(mix(u.fog.rgb, top, k), 1.0);
}
