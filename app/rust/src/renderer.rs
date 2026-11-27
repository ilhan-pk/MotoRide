use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{Mat4, Vec3};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use winit::window::Window;

use crate::game::{build_chunk, bull_model, bulls_around, buildings_around, car_models, fox_model, models, Game, Vertex, BULL_ANIM_FPS, CHUNK, FOX_SCALE};
use crate::hud;

const MAX_BALLOON_VERTS: usize = 3_000;
const MAX_COCKPIT_VERTS: usize = 9000;
const MAX_HUD_VERTS: usize = 600;
const MAX_DYN_VERTS: usize = MAX_BALLOON_VERTS + MAX_COCKPIT_VERTS + MAX_HUD_VERTS;
/// Chunks kept around the player (each side). 4 -> 9 x 9 chunks = 720 m square.
const R_CHUNKS: i32 = 4;
// matches the horizon colour of the sky image
const FOG: [f32; 4] = [0.42, 0.69, 0.75, 1.0];

static SKY_PNG: &[u8] = include_bytes!("sky.png");
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SkyUniforms {
    inv_vp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    vp: [[f32; 4]; 4],
    fog: [f32; 4],
    params: [f32; 4],
}

/// A static piece of world, built once and kept on the GPU.
struct Chunk {
    vb: wgpu::Buffer,
    count: u32,
    born: Instant,
    /// Own uniforms while the spawn animation runs (None once settled).
    anim: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

const REVEAL_SECS: f32 = 1.4;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    sky_ub: wgpu::Buffer,
    sky_bg: wgpu::BindGroup,
    world_ub: wgpu::Buffer,
    cockpit_ub: wgpu::Buffer,
    hud_ub: wgpu::Buffer,
    world_bg: wgpu::BindGroup,
    cockpit_bg: wgpu::BindGroup,
    hud_bg: wgpu::BindGroup,
    /// Small per-frame buffer: [balloons | cockpit | hud].
    dyn_vb: wgpu::Buffer,
    chunks: HashMap<(i32, i32), Chunk>,
    depth_view: wgpu::TextureView,
    batch: Vec<Vertex>,
    part: Vec<Vertex>,
    hud_cache: Vec<Vertex>,
    hud_key: (u32, u32),
    bgl: wgpu::BindGroupLayout,
    /// Shared building models (vertex buffer, vertex count) and per-instance uniform pool.
    bmodels: Vec<(wgpu::Buffer, u32)>,
    bpool: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    cmodels: Vec<(wgpu::Buffer, u32)>,
    cpool: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
    fox_vb: wgpu::Buffer,
    fox_ub: wgpu::Buffer,
    fox_bg: wgpu::BindGroup,
    bull_vb: wgpu::Buffer,
    bull_pool: Vec<(wgpu::Buffer, wgpu::BindGroup)>,
}

fn make_depth(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Prefer Vulkan; fall back to OpenGL ES if Vulkan is unavailable on this device.
async fn pick_backend(
    window: Arc<Window>,
) -> (wgpu::Instance, wgpu::Surface<'static>, wgpu::Adapter) {
    for backends in [wgpu::Backends::VULKAN, wgpu::Backends::GL] {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let Ok(surface) = instance.create_surface(window.clone()) else { continue };
        // `into_iter().next()` works whether request_adapter returns Option or Result.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .into_iter()
            .next();
        if let Some(adapter) = adapter {
            let info = adapter.get_info();
            log::info!("MotoRide backend: {:?} on {}", info.backend, info.name);
            return (instance, surface, adapter);
        }
        log::warn!("MotoRide: backend {:?} not available, trying next", backends);
    }
    panic!("no Vulkan or OpenGL ES adapter found");
}

impl Renderer {
    pub async fn new(window: Arc<Window>, w: u32, h: u32) -> Self {
        let (_instance, surface, adapter) = pick_backend(window).await;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("device");

        let caps = surface.get_capabilities(&adapter);
        // Prefer a non-sRGB format so shader colours appear exactly as written.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        // ---- sky: 2k equirectangular texture ----
        let sky_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sky_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sky_layout"),
            bind_group_layouts: &[&bgl, &sky_bgl],
            push_constant_ranges: &[],
        });
        let (sky_w, sky_h, sky_rgba) = load_sky();
        let sky_tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("sky"),
                size: wgpu::Extent3d { width: sky_w, height: sky_h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &sky_rgba,
        );
        let sky_view = sky_tex.create_view(&Default::default());
        let sky_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sky_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let sky_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sky_ub"),
            size: std::mem::size_of::<SkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sky_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_bg"),
            layout: &sky_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sky_ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sky_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sky_sampler) },
            ],
        });

        let make_ub = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let world_ub = make_ub("world_ub");
        let cockpit_ub = make_ub("cockpit_ub");
        let hud_ub = make_ub("hud_ub");
        let make_bg = |ub: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }],
            })
        };
        let world_bg = make_bg(&world_ub);
        let cockpit_bg = make_bg(&cockpit_ub);
        let hud_bg = make_bg(&hud_ub);

        let vsize = std::mem::size_of::<Vertex>() as u64;
        let dyn_vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dyn_vb"),
            size: vsize * MAX_DYN_VERTS as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let attrs = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32];
        let vlayout = wgpu::VertexBufferLayout {
            array_stride: vsize,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &attrs,
        };
        let target = [Some(wgpu::ColorTargetState {
            format,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[vlayout],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &target,
            }),
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sky"),
            layout: Some(&sky_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_sky"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &target,
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let depth_view = make_depth(&device, w, h);

        let fox_vb = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("fox"),
            contents: bytemuck::cast_slice(&fox_model().verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let bull_vb = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("bull"),
            contents: bytemuck::cast_slice(&bull_model().verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let fox_ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fox_ub"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fox_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: fox_ub.as_entire_binding() }],
        });
        let cmodels: Vec<(wgpu::Buffer, u32)> = car_models()
            .iter()
            .map(|m| {
                let vb = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("car"),
                    contents: bytemuck::cast_slice(&m.verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                (vb, m.verts.len() as u32)
            })
            .collect();
        let bmodels: Vec<(wgpu::Buffer, u32)> = models()
            .iter()
            .map(|m| {
                let vb = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("building"),
                    contents: bytemuck::cast_slice(&m.verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                (vb, m.verts.len() as u32)
            })
            .collect();

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            sky_pipeline,
            sky_ub,
            sky_bg,
            world_ub,
            cockpit_ub,
            hud_ub,
            world_bg,
            cockpit_bg,
            hud_bg,
            dyn_vb,
            chunks: HashMap::new(),
            depth_view,
            batch: Vec::with_capacity(MAX_DYN_VERTS),
            part: Vec::with_capacity(8_000),
            hud_cache: Vec::new(),
            hud_key: (u32::MAX, 0),
            bgl,
            bmodels,
            bpool: Vec::new(),
            cmodels,
            cpool: Vec::new(),
            fox_vb,
            fox_ub,
            fox_bg,
            bull_vb,
            bull_pool: Vec::new(),
        }
    }

    pub fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth(&self.device, w, h);
    }

    /// Build missing chunks around the player (a few ms per frame at most) and drop far ones.
    fn stream_chunks(&mut self, g: &Game) {
        let pcx = (g.pos.x / CHUNK).floor() as i32;
        let pcz = (g.pos.z / CHUNK).floor() as i32;
        self.chunks
            .retain(|k, _| (k.0 - pcx).abs() <= R_CHUNKS + 1 && (k.1 - pcz).abs() <= R_CHUNKS + 1);

        let mut missing: Vec<(i32, (i32, i32))> = Vec::new();
        for dz in -R_CHUNKS..=R_CHUNKS {
            for dx in -R_CHUNKS..=R_CHUNKS {
                let key = (pcx + dx, pcz + dz);
                if !self.chunks.contains_key(&key) {
                    missing.push((dx * dx + dz * dz, key));
                }
            }
        }
        if missing.is_empty() {
            return;
        }
        missing.sort_by_key(|m| m.0); // nearest first

        // First frame: build everything at once. Afterwards: a small time budget per frame.
        let budget = if self.chunks.is_empty() {
            Duration::from_millis(1500)
        } else {
            Duration::from_millis(3)
        };
        let first = self.chunks.is_empty();
        let t0 = Instant::now();
        let mut built = 0;
        for (_, key) in missing {
            if built > 0 && t0.elapsed() > budget {
                break;
            }
            self.part.clear();
            build_chunk(key.0, key.1, &mut self.part);
            built += 1;
            if self.part.is_empty() {
                continue;
            }
            let vb = self.device.create_buffer_init(&BufferInitDescriptor {
                label: Some("chunk"),
                contents: bytemuck::cast_slice(&self.part),
                usage: wgpu::BufferUsages::VERTEX,
            });
            let anim = if first {
                None // initial load appears instantly
            } else {
                let ub = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("chunk_ub"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }],
                });
                Some((ub, bg))
            };
            self.chunks.insert(key, Chunk { vb, count: self.part.len() as u32, born: Instant::now(), anim });
        }
    }

    pub fn render(&mut self, g: &Game, fps: u32) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Mat4::perspective_rh(70f32.to_radians(), aspect, 0.3, 2000.0);
        let view = Mat4::from_rotation_z(g.roll) * Mat4::look_to_rh(g.eye(), g.look_dir(), Vec3::Y);

        self.queue.write_buffer(
            &self.world_ub,
            0,
            bytemuck::bytes_of(&Uniforms {
                vp: (proj * view).to_cols_array_2d(),
                fog: FOG,
                params: [1.0 / 240.0, 0.0, g.time % 628.0, 0.0],
            }),
        );
        self.queue.write_buffer(
            &self.cockpit_ub,
            0,
            bytemuck::bytes_of(&Uniforms { vp: proj.to_cols_array_2d(), fog: FOG, params: [0.0; 4] }),
        );
        self.queue.write_buffer(
            &self.hud_ub,
            0,
            bytemuck::bytes_of(&Uniforms {
                vp: Mat4::IDENTITY.to_cols_array_2d(),
                fog: FOG,
                params: [0.0; 4],
            }),
        );

        // ---- static world: built once per chunk, cached on the GPU ----
        self.stream_chunks(g);

        // spawn animation: new chunks rise out of the fog with an ease-out curve
        let world_u = Uniforms {
            vp: (proj * view).to_cols_array_2d(),
            fog: FOG,
            params: [1.0 / 240.0, 0.0, g.time % 628.0, 0.0],
        };
        for c in self.chunks.values_mut() {
            if c.anim.is_none() {
                continue;
            }
            let t = (c.born.elapsed().as_secs_f32() / REVEAL_SECS).min(1.0);
            if t >= 1.0 {
                c.anim = None;
                continue;
            }
            let e = 1.0 - (1.0 - t) * (1.0 - t) * (1.0 - t);
            let mut u = world_u;
            u.params[1] = 1.0 - e;
            if let Some((ub, _)) = &c.anim {
                self.queue.write_buffer(ub, 0, bytemuck::bytes_of(&u));
            }
        }

        // ---- small dynamic batch: balloons + cockpit + HUD (one upload) ----
        self.batch.clear();

        self.part.clear();
        g.build_balloons(&mut self.part);
        self.part.truncate(MAX_BALLOON_VERTS - MAX_BALLOON_VERTS % 3);
        let n_balloon = self.part.len() as u32;
        self.batch.extend_from_slice(&self.part);

        self.part.clear();
        g.build_cockpit(&mut self.part);
        self.part.truncate(MAX_COCKPIT_VERTS);
        let n_cockpit = self.part.len() as u32;
        self.batch.extend_from_slice(&self.part);

        // The FPS text only changes a couple of times per second: rebuild it only then.
        let aspect_key = (aspect * 1000.0) as u32;
        if self.hud_key != (fps, aspect_key) {
            self.hud_cache.clear();
            hud::build_fps(&mut self.hud_cache, fps, aspect);
            self.hud_cache.truncate(MAX_HUD_VERTS);
            self.hud_key = (fps, aspect_key);
        }
        let n_hud = self.hud_cache.len() as u32;
        self.batch.extend_from_slice(&self.hud_cache);

        self.queue.write_buffer(&self.dyn_vb, 0, bytemuck::cast_slice(&self.batch));

        let cockpit_start = n_balloon;
        let hud_start = n_balloon + n_cockpit;

        // ---- cull chunks that are behind or far to the side ----
        let eye = g.eye();
        let f = g.forward();
        let tan_h = 35f32.to_radians().tan() * aspect;
        self.queue.write_buffer(
            &self.sky_ub,
            0,
            bytemuck::bytes_of(&SkyUniforms { inv_vp: (proj * view).inverse().to_cols_array_2d() }),
        );

        // ---- roadside buildings: one draw each, own uniforms (view-projection * model) ----
        let mut bdraw: Vec<(usize, usize)> = Vec::with_capacity(16);
        for b in buildings_around(eye.z) {
            let dx = b.x - eye.x;
            let dz = b.z - eye.z;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist > 340.0 {
                continue;
            }
            let along = dx * f.x + dz * f.z;
            let perp = (dx * f.z - dz * f.x).abs();
            if dist > 40.0 && (along < -20.0 || perp > along.max(0.0) * tan_h * 1.2 + 40.0) {
                continue;
            }
            let i = bdraw.len();
            if self.bpool.len() <= i {
                let ub = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("building_ub"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }],
                });
                self.bpool.push((ub, bg));
            }
            let model = Mat4::from_translation(Vec3::new(b.x, b.y, b.z)) * Mat4::from_rotation_y(b.yaw);
            // buildings rise out of the fog as they enter view distance
            let t = ((dist - 265.0) / 70.0).clamp(0.0, 1.0);
            let hidden = t * t * (3.0 - 2.0 * t);
            self.queue.write_buffer(
                &self.bpool[i].0,
                0,
                bytemuck::bytes_of(&Uniforms {
                    vp: (proj * view * model).to_cols_array_2d(),
                    fog: FOG,
                    params: [1.0 / 240.0, hidden, 0.0, 0.0],
                }),
            );
            bdraw.push((b.model, i));
        }

        // ---- traffic cars: same trick, own uniform pool ----
        let mut cdraw: Vec<(usize, usize)> = Vec::with_capacity(12);
        for c in g.cars.iter() {
            let dx = c.x - eye.x;
            let dz = c.z - eye.z;
            let dist = (dx * dx + dz * dz).sqrt();
            if dist > 340.0 {
                continue;
            }
            let along = dx * f.x + dz * f.z;
            let perp = (dx * f.z - dz * f.x).abs();
            if dist > 25.0 && (along < -10.0 || perp > along.max(0.0) * tan_h * 1.2 + 25.0) {
                continue;
            }
            let i = cdraw.len();
            if self.cpool.len() <= i {
                let ub = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("car_ub"),
                    size: std::mem::size_of::<Uniforms>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bgl,
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }],
                });
                self.cpool.push((ub, bg));
            }
            let model = Mat4::from_translation(Vec3::new(c.x, c.y, c.z))
                * Mat4::from_rotation_y(c.yaw)
                * Mat4::from_rotation_x(c.pitch);
            let t = ((dist - 265.0) / 70.0).clamp(0.0, 1.0);
            let hidden = t * t * (3.0 - 2.0 * t);
            self.queue.write_buffer(
                &self.cpool[i].0,
                0,
                bytemuck::bytes_of(&Uniforms {
                    vp: (proj * view * model).to_cols_array_2d(),
                    fog: FOG,
                    params: [1.0 / 240.0, hidden, 0.0, 0.0],
                }),
            );
            cdraw.push((c.model, i));
        }

        // ---- fox (one draw, current animation frame) ----
        let mut fox_draw: Option<(u32, u32)> = None;
        if let Some(fx) = &g.fox {
            let m = fox_model();
            let dx = fx.x - eye.x;
            let dz = fx.z - eye.z;
            let dist = (dx * dx + dz * dz).sqrt();
            let along = dx * f.x + dz * f.z;
            if dist < 340.0 && along > -15.0 {
                let model = Mat4::from_translation(Vec3::new(fx.x, fx.y, fx.z))
                    * Mat4::from_rotation_y(fx.yaw)
                    * Mat4::from_scale(Vec3::splat(FOX_SCALE));
                let t = ((dist - 265.0) / 70.0).clamp(0.0, 1.0);
                let hidden = t * t * (3.0 - 2.0 * t);
                self.queue.write_buffer(
                    &self.fox_ub,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        vp: (proj * view * model).to_cols_array_2d(),
                        fog: FOG,
                        params: [1.0 / 240.0, hidden, 0.0, 0.0],
                    }),
                );
                let a = (fx.frame * m.per_frame) as u32;
                fox_draw = Some((a, a + m.per_frame as u32));
            }
        }

        // ---- grazing bulls: one draw each (current animation frame), nearby ones only ----
        let mut bull_draw: Vec<(usize, u32, u32)> = Vec::with_capacity(16);
        {
            let bm = bull_model();
            for b in bulls_around(eye.z) {
                if bull_draw.len() >= 16 {
                    break;
                }
                let dx = b.x - eye.x;
                let dz = b.z - eye.z;
                let dist = (dx * dx + dz * dz).sqrt();
                if dist > 250.0 {
                    continue;
                }
                let along = dx * f.x + dz * f.z;
                let perp = (dx * f.z - dz * f.x).abs();
                if dist > 30.0 && (along < -10.0 || perp > along.max(0.0) * tan_h * 1.2 + 30.0) {
                    continue;
                }
                let i = bull_draw.len();
                if self.bull_pool.len() <= i {
                    let ub = self.device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("bull_ub"),
                        size: std::mem::size_of::<Uniforms>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bgl,
                        entries: &[wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() }],
                    });
                    self.bull_pool.push((ub, bg));
                }
                let model = Mat4::from_translation(Vec3::new(b.x, b.y, b.z))
                    * Mat4::from_rotation_y(b.yaw)
                    * Mat4::from_scale(Vec3::splat(b.scale));
                let t = ((dist - 200.0) / 50.0).clamp(0.0, 1.0);
                let hidden = t * t * (3.0 - 2.0 * t);
                self.queue.write_buffer(
                    &self.bull_pool[i].0,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        vp: (proj * view * model).to_cols_array_2d(),
                        fog: FOG,
                        params: [1.0 / 240.0, hidden, 0.0, 0.0],
                    }),
                );
                let fr = ((g.time * BULL_ANIM_FPS + b.phase * bm.frames as f32) as usize) % bm.frames;
                let a = (fr * bm.per_frame) as u32;
                bull_draw.push((i, a, a + bm.per_frame as u32));
            }
        }

        let mut visible: Vec<&Chunk> = Vec::with_capacity(48);
        for (key, chunk) in self.chunks.iter() {
            let cx = (key.0 as f32 + 0.5) * CHUNK - eye.x;
            let cz = (key.1 as f32 + 0.5) * CHUNK - eye.z;
            let dist = (cx * cx + cz * cz).sqrt();
            let along = cx * f.x + cz * f.z;
            let perp = (cx * f.z - cz * f.x).abs();
            let near = dist < CHUNK * 1.3;
            if near || (along > -CHUNK * 0.9 && perp < along.max(0.0) * tan_h * 1.15 + CHUNK * 1.2) {
                visible.push(chunk);
            }
        }

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
        };
        let view_tex = frame.texture.create_view(&Default::default());
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.72, g: 0.96, b: 0.99, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &self.world_bg, &[]);
            pass.set_bind_group(1, &self.sky_bg, &[]);
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.world_bg, &[]);
            for c in visible.iter() {
                match &c.anim {
                    Some((_, bg)) => pass.set_bind_group(0, bg, &[]),
                    None => pass.set_bind_group(0, &self.world_bg, &[]),
                }
                pass.set_vertex_buffer(0, c.vb.slice(..));
                pass.draw(0..c.count, 0..1);
            }

            for &(m, i) in bdraw.iter() {
                if let Some((vb, count)) = self.bmodels.get(m) {
                    pass.set_bind_group(0, &self.bpool[i].1, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.draw(0..*count, 0..1);
                }
            }

            for &(m, i) in cdraw.iter() {
                if let Some((vb, count)) = self.cmodels.get(m) {
                    pass.set_bind_group(0, &self.cpool[i].1, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.draw(0..*count, 0..1);
                }
            }

            for &(i, a, b) in bull_draw.iter() {
                pass.set_bind_group(0, &self.bull_pool[i].1, &[]);
                pass.set_vertex_buffer(0, self.bull_vb.slice(..));
                pass.draw(a..b, 0..1);
            }

            if let Some((a, b)) = fox_draw {
                pass.set_bind_group(0, &self.fox_bg, &[]);
                pass.set_vertex_buffer(0, self.fox_vb.slice(..));
                pass.draw(a..b, 0..1);
            }

            pass.set_bind_group(0, &self.world_bg, &[]);
            pass.set_vertex_buffer(0, self.dyn_vb.slice(..));
            if n_balloon > 0 {
                pass.draw(0..n_balloon, 0..1);
            }

            pass.set_bind_group(0, &self.cockpit_bg, &[]);
            pass.draw(cockpit_start..cockpit_start + n_cockpit, 0..1);

            if n_hud > 0 {
                pass.set_bind_group(0, &self.hud_bg, &[]);
                pass.draw(hud_start..hud_start + n_hud, 0..1);
            }
        }
        self.queue.submit(Some(enc.finish()));
        frame.present();
    }
}


/// Decode the embedded 2k equirectangular sky (PNG) to RGBA8.
fn load_sky() -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::Cursor::new(SKY_PNG));
    let mut reader = decoder.read_info().expect("sky png header");
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).expect("sky png data");
    let (w, h) = (info.width, info.height);
    let ch = match info.color_type {
        png::ColorType::Rgba => 4,
        png::ColorType::Rgb => 3,
        png::ColorType::GrayscaleAlpha => 2,
        _ => 1,
    };
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for i in 0..(w * h) as usize {
        let p = &buf[i * ch..i * ch + ch];
        match ch {
            4 => rgba.extend_from_slice(&[p[0], p[1], p[2], 255]),
            3 => rgba.extend_from_slice(&[p[0], p[1], p[2], 255]),
            2 => rgba.extend_from_slice(&[p[0], p[0], p[0], 255]),
            _ => rgba.extend_from_slice(&[p[0], p[0], p[0], 255]),
        }
    }
    (w, h, rgba)
}
