use std::sync::Arc;

use glam::{Mat4, Vec3};
use winit::window::Window;

use crate::game::{Game, Vertex};
use crate::hud;

const MAX_WORLD_VERTS: usize = 60_000; // multiple of 3
const MAX_COCKPIT_VERTS: usize = 600;
const MAX_HUD_VERTS: usize = 600;
const MAX_VERTS: usize = MAX_WORLD_VERTS + MAX_COCKPIT_VERTS + MAX_HUD_VERTS;
const FOG: [f32; 4] = [0.66, 0.72, 0.82, 1.0];
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    vp: [[f32; 4]; 4],
    fog: [f32; 4],
    params: [f32; 4],
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    world_ub: wgpu::Buffer,
    cockpit_ub: wgpu::Buffer,
    hud_ub: wgpu::Buffer,
    world_bg: wgpu::BindGroup,
    cockpit_bg: wgpu::BindGroup,
    hud_bg: wgpu::BindGroup,
    /// One shared vertex buffer: [world | cockpit | hud], uploaded once per frame.
    vb: wgpu::Buffer,
    depth_view: wgpu::TextureView,
    batch: Vec<Vertex>,
    part: Vec<Vertex>,
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
        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch_vb"),
            size: vsize * MAX_VERTS as u64,
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
            layout: Some(&layout),
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

        Self {
            surface,
            device,
            queue,
            config,
            pipeline,
            sky_pipeline,
            world_ub,
            cockpit_ub,
            hud_ub,
            world_bg,
            cockpit_bg,
            hud_bg,
            vb,
            depth_view,
            batch: Vec::with_capacity(MAX_VERTS),
            part: Vec::with_capacity(MAX_WORLD_VERTS),
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

    pub fn render(&mut self, g: &Game, fps: u32) {
        let aspect = self.config.width as f32 / self.config.height as f32;
        let proj = Mat4::perspective_rh(70f32.to_radians(), aspect, 0.3, 2000.0);
        let view = Mat4::from_rotation_z(g.lean * 0.28) * Mat4::look_to_rh(g.eye(), g.look_dir(), Vec3::Y);

        self.queue.write_buffer(
            &self.world_ub,
            0,
            bytemuck::bytes_of(&Uniforms {
                vp: (proj * view).to_cols_array_2d(),
                fog: FOG,
                params: [1.0 / 210.0, 0.0, 0.0, 0.0],
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

        // ---- batch all geometry into one buffer, one upload ----
        self.batch.clear();

        self.part.clear();
        g.build_world(&mut self.part);
        self.part.truncate(MAX_WORLD_VERTS);
        let n_world = self.part.len() as u32;
        self.batch.extend_from_slice(&self.part);

        self.part.clear();
        g.build_cockpit(&mut self.part);
        self.part.truncate(MAX_COCKPIT_VERTS);
        let n_cockpit = self.part.len() as u32;
        self.batch.extend_from_slice(&self.part);

        self.part.clear();
        hud::build_fps(&mut self.part, fps, aspect);
        self.part.truncate(MAX_HUD_VERTS);
        let n_hud = self.part.len() as u32;
        self.batch.extend_from_slice(&self.part);

        self.queue.write_buffer(&self.vb, 0, bytemuck::cast_slice(&self.batch));

        let cockpit_start = n_world;
        let hud_start = n_world + n_cockpit;

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
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.66, g: 0.72, b: 0.82, a: 1.0 }),
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
            pass.draw(0..3, 0..1);

            pass.set_pipeline(&self.pipeline);
            pass.set_vertex_buffer(0, self.vb.slice(..));

            pass.set_bind_group(0, &self.world_bg, &[]);
            pass.draw(0..n_world, 0..1);

            pass.set_bind_group(0, &self.cockpit_bg, &[]);
            pass.draw(cockpit_start..cockpit_start + n_cockpit, 0..1);

            pass.set_bind_group(0, &self.hud_bg, &[]);
            pass.draw(hud_start..hud_start + n_hud, 0..1);
        }
        self.queue.submit(Some(enc.finish()));
        frame.present();
    }
}
