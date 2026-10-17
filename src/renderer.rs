use web_sys::HtmlCanvasElement;
use wgpu::util::DeviceExt;

use crate::grid::Grid;
use crate::mesh::{CellVertex, Vertex, cell_mesh, cylinder};
use crate::palette;
use crate::projection::Projection;
use crate::scene::{Column, Scene, Segment};
use crate::view::{LAYER_BOUNDARIES, LAYER_MANTLE, LAYER_MOTION};
use crate::world::World;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const COLUMN_SIDES: u32 = 16;
/// Cell-id cube map face size. Cells at resolution 3 span about 70 texels.
const CELL_MAP_SIZE: u32 = 1024;
const CELL_MAP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Line overlays, each drawn when its layer bit is set.
pub const OVERLAYS: [u32; 3] = [LAYER_BOUNDARIES, LAYER_MOTION, LAYER_MANTLE];

/// Per-cell surface data; matches `CellData` in surface.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CellData {
    plate: u32,
    elevation_m: f32,
    /// Seafloor age (Myr), or -1 for continental crust.
    age_myr: f32,
    thickness_km: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Globals {
    pub view_proj: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
    /// Viewport width and height (physical px), planet radius (m), line width (px).
    pub params: [f32; 4],
    /// Bitmask of visible `view::LAYER_*` flags, projection id,
    /// `view::COLOR_*` mode, then unused.
    pub flags: [u32; 4],
    /// Flat maps: `MapProjection::params`.
    pub projection: [f32; 4],
    /// Flat maps: map point at the screen center, map units to the top
    /// edge, aspect.
    pub map: [f32; 4],
}

/// Builds a shader from common.wgsl, any shared snippets, then the file.
macro_rules! shader {
    ($device:expr, $($file:literal),+) => {
        $device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(concat!($($file, " "),+)),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("shaders/common.wgsl"),
                    $(include_str!(concat!("shaders/", $file))),+
                )
                .into(),
            ),
        })
    };
}

struct GpuMesh {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
}

impl GpuMesh {
    fn new<V: bytemuck::Pod>(
        device: &wgpu::Device,
        label: &str,
        vertices: &[V],
        indices: &[u32],
    ) -> Self {
        Self {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: indices.len() as u32,
        }
    }
}

/// Per-frame instance data in a vertex buffer that grows as needed.
struct InstanceBuffer {
    label: &'static str,
    buffer: wgpu::Buffer,
    count: u32,
}

impl InstanceBuffer {
    fn new(device: &wgpu::Device, label: &'static str) -> Self {
        Self {
            label,
            buffer: Self::allocate(device, label, 4096),
            count: 0,
        }
    }

    fn allocate(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn write<T: bytemuck::Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.len() as u64 > self.buffer.size() {
            let size = (bytes.len() as u64).next_power_of_two();
            self.buffer = Self::allocate(device, self.label, size);
        }
        queue.write_buffer(&self.buffer, 0, bytes);
        self.count = data.len() as u32;
    }
}

pub struct Renderer {
    canvas: HtmlCanvasElement,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    /// sRGB view format that pipelines render to (see `new`).
    view_format: wgpu::TextureFormat,
    depth_view: wgpu::TextureView,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    globe_pipeline: wgpu::RenderPipeline,
    globe: GpuMesh,
    map_pipeline: wgpu::RenderPipeline,
    cell_map: wgpu::TextureView,
    surface_layout: wgpu::BindGroupLayout,
    surface_bind_group: wgpu::BindGroup,
    /// Line overlays in the order of `OVERLAYS`.
    overlays: Vec<InstanceBuffer>,
    column_pipeline: wgpu::RenderPipeline,
    column_mesh: GpuMesh,
    columns: InstanceBuffer,
    line_pipeline: wgpu::RenderPipeline,
    segments: InstanceBuffer,
}

impl Renderer {
    pub async fn new(
        canvas: HtmlCanvasElement,
        grid: &Grid,
        world: &World,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas.clone()))
            .map_err(|e| format!("create surface: {e}"))?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("request device: {e}"))?;

        let (width, height) = canvas_size(&canvas);
        let caps = surface.get_capabilities(&adapter);
        // Shaders output linear color. WebGPU canvases are never sRGB
        // (bgra8unorm and friends), so render through an sRGB view of the
        // canvas, which encodes linear to sRGB on write. Without it every
        // color comes out too dark.
        let canvas_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let format = canvas_format.add_srgb_suffix();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: canvas_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: if format == canvas_format {
                vec![]
            } else {
                vec![format]
            },
            desired_maximum_frame_latency: 2,
            color_space: Default::default(),
        };
        surface.configure(&device, &config);

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("globals"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let storage_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let surface_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface"),
            entries: &[
                storage_entry(0),
                storage_entry(1),
                storage_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let globe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("globe"),
            bind_group_layouts: &[Some(&bind_group_layout), Some(&surface_layout)],
            immediate_size: 0,
        });

        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        };

        let cell_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<CellVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Uint32],
        };
        let globe_shader = shader!(device, "surface.wgsl", "globe.wgsl");
        let globe_pipeline = create_pipeline(
            &device,
            &globe_layout,
            "globe",
            &globe_shader,
            &[Some(cell_vertex_layout.clone())],
            format,
            Some(wgpu::Face::Back),
            true,
            None,
        );

        let map_shader = shader!(device, "surface.wgsl", "map.wgsl");
        let map_pipeline = create_pipeline(
            &device,
            &globe_layout,
            "map",
            &map_shader,
            &[],
            format,
            None,
            false,
            None,
        );

        let column_shader = shader!(device, "columns.wgsl");
        let column_pipeline = create_pipeline(
            &device,
            &layout,
            "columns",
            &column_shader,
            &[
                Some(mesh_layout),
                Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Column>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        2 => Float32x3, 3 => Float32, 4 => Float32x3, 5 => Float32, 6 => Float32x4
                    ],
                }),
            ],
            format,
            None,
            true,
            None,
        );

        let line_shader = shader!(device, "lines.wgsl");
        let line_pipeline = create_pipeline(
            &device,
            &layout,
            "lines",
            &line_shader,
            &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Segment>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4],
            })],
            format,
            None,
            false,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        let (globe_vertices, globe_indices) = cell_mesh(grid);
        let globe = GpuMesh::new(&device, "globe", &globe_vertices, &globe_indices);
        let cell_map = bake_cell_map(&device, &queue, &globe, cell_vertex_layout);
        let column = cylinder(COLUMN_SIDES);
        Ok(Self {
            depth_view: create_depth_view(&device, width, height),
            globe,
            surface_bind_group: create_surface_bind_group(
                &device,
                &surface_layout,
                world,
                &cell_map,
            ),
            cell_map,
            map_pipeline,
            surface_layout,
            overlays: OVERLAYS
                .iter()
                .map(|_| InstanceBuffer::new(&device, "overlay"))
                .collect(),
            column_mesh: GpuMesh::new(&device, "column", &column.vertices, &column.indices),
            columns: InstanceBuffer::new(&device, "column instances"),
            segments: InstanceBuffer::new(&device, "segment instances"),
            canvas,
            surface,
            device,
            queue,
            config,
            view_format: format,
            globals_buffer,
            globals_bind_group,
            globe_pipeline,
            column_pipeline,
            line_pipeline,
        })
    }

    /// Resize the surface to the canvas's displayed size if it changed, and
    /// return the size in physical pixels.
    pub fn sync_size(&mut self) -> (u32, u32) {
        let (width, height) = canvas_size(&self.canvas);
        if width != self.config.width || height != self.config.height {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.depth_view = create_depth_view(&self.device, width, height);
        }
        (width, height)
    }

    /// Replace the per-cell surface data, e.g. after regenerating.
    pub fn set_world(&mut self, world: &World) {
        self.surface_bind_group =
            create_surface_bind_group(&self.device, &self.surface_layout, world, &self.cell_map);
    }

    /// Replace the lines of the overlay drawn for `layer` (one of
    /// `OVERLAYS`).
    pub fn set_overlay(&mut self, layer: u32, segments: &[Segment]) {
        if let Some(i) = OVERLAYS.iter().position(|&l| l == layer) {
            self.overlays[i].write(&self.device, &self.queue, segments);
        }
    }

    pub fn render(&mut self, globals: &Globals, scene: &Scene) {
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(globals));
        self.columns
            .write(&self.device, &self.queue, &scene.columns);
        self.segments
            .write(&self.device, &self.queue, &scene.segments);

        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(&self.device, &self.config);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.view_format),
            ..Default::default()
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.004,
                            g: 0.005,
                            b: 0.008,
                            a: 1.0,
                        }),
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
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.globals_bind_group, &[]);

            pass.set_bind_group(1, &self.surface_bind_group, &[]);
            if globals.flags[1] == Projection::Globe.id() {
                pass.set_pipeline(&self.globe_pipeline);
                pass.set_vertex_buffer(0, self.globe.vertices.slice(..));
                pass.set_index_buffer(self.globe.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..self.globe.index_count, 0, 0..1);
            } else {
                pass.set_pipeline(&self.map_pipeline);
                pass.draw(0..3, 0..1);
            }

            if self.columns.count > 0 {
                pass.set_pipeline(&self.column_pipeline);
                pass.set_vertex_buffer(0, self.column_mesh.vertices.slice(..));
                pass.set_vertex_buffer(1, self.columns.buffer.slice(..));
                pass.set_index_buffer(
                    self.column_mesh.indices.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                pass.draw_indexed(0..self.column_mesh.index_count, 0, 0..self.columns.count);
            }

            // Lines last: they test depth but don't write it, and blend.
            pass.set_pipeline(&self.line_pipeline);
            for (layer, lines) in OVERLAYS.iter().zip(&self.overlays) {
                if globals.flags[0] & layer != 0 && lines.count > 0 {
                    pass.set_vertex_buffer(0, lines.buffer.slice(..));
                    pass.draw(0..6, 0..lines.count);
                }
            }
            if self.segments.count > 0 {
                pass.set_vertex_buffer(0, self.segments.buffer.slice(..));
                pass.draw(0..6, 0..self.segments.count);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.queue.present(frame);
    }
}

#[allow(clippy::too_many_arguments)]
fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    label: &str,
    shader: &wgpu::ShaderModule,
    buffers: &[Option<wgpu::VertexBufferLayout>],
    format: wgpu::TextureFormat,
    cull_mode: Option<wgpu::Face>,
    depth_write: bool,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers,
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_surface_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    world: &World,
    cell_map: &wgpu::TextureView,
) -> wgpu::BindGroup {
    let crust = &world.crust;
    let cells: Vec<CellData> = (0..crust.kind.len())
        .map(|c| CellData {
            plate: world.plates.cell_plate[c],
            elevation_m: crust.elevation_m[c] as f32,
            age_myr: if crust.kind[c].is_continental() {
                -1.0
            } else {
                crust.age_myr[c] as f32
            },
            thickness_km: crust.thickness_km[c] as f32,
        })
        .collect();
    let colors: Vec<[f32; 4]> = world
        .plates
        .plates
        .iter()
        .map(|p| [p.color[0], p.color[1], p.color[2], 1.0])
        .collect();
    let storage = |label, contents: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let cell_data = storage("cells", bytemuck::cast_slice(&cells));
    let plate_color = storage("plate color", bytemuck::cast_slice(&colors));
    let palettes = storage("palettes", bytemuck::cast_slice(&palette::gpu_table()));
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("surface"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: cell_data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: plate_color.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(cell_map),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: palettes.as_entire_binding(),
            },
        ],
    })
}

/// Renders the globe mesh from the planet's center into the six faces of a
/// cube map, storing the grid cell id seen in each direction, so flat maps
/// can find the cell under any pixel. Cells never change, so this runs once.
fn bake_cell_map(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    globe: &GpuMesh,
    vertex_layout: wgpu::VertexBufferLayout,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("cell map"),
        size: wgpu::Extent3d {
            width: CELL_MAP_SIZE,
            height: CELL_MAP_SIZE,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: CELL_MAP_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shader = shader!(device, "cells.wgsl");
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("cell map"),
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cell map"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[Some(vertex_layout)],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(CELL_MAP_FORMAT.into())],
        }),
        // Seen from inside, so the globe's front faces wind backwards.
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    for face in 0..6 {
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: face,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cell map"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    // NO_CELL in surface.wgsl.
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: u32::MAX as f64,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_vertex_buffer(0, globe.vertices.slice(..));
        pass.set_index_buffer(globe.indices.slice(..), wgpu::IndexFormat::Uint32);
        // The instance index tells the shader which face to render.
        pass.draw_indexed(0..globe.index_count, 0, face..face + 1);
    }
    queue.submit([encoder.finish()]);
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

/// Canvas size in physical pixels, never zero.
fn canvas_size(canvas: &HtmlCanvasElement) -> (u32, u32) {
    let dpr = web_sys::window().map_or(1.0, |w| w.device_pixel_ratio());
    let width = ((canvas.client_width() as f64 * dpr) as u32).max(1);
    let height = ((canvas.client_height() as f64 * dpr) as u32).max(1);
    // Assigning the size, even unchanged, can reset the canvas, so only on change.
    if canvas.width() != width || canvas.height() != height {
        canvas.set_width(width);
        canvas.set_height(height);
    }
    (width, height)
}

fn create_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}
