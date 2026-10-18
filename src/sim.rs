//! Runs the time step on the GPU: owns the plates' crust samples and the
//! working buffers, dispatches the passes in timestep.wgsl, and reads world
//! cells back for the HUD, overlays and cross-section.
//!
//! The rules live in `timestep.rs` (the tested reference) and the shader;
//! this is the plumbing.

use std::cell::Cell;
use std::rc::Rc;

use bytemuck::{Pod, Zeroable};
use glam::{DQuat, DVec3};
use wgpu::util::DeviceExt;

use crate::grid::Grid;
use crate::renderer::Renderer;
use crate::timestep::{ColumnOut, Frames, MAX_PLATES, STEP_MYR, wgsl_source};
use crate::world::World;

const WORKGROUP: u32 = 64;
/// The passes, in order.
const PASSES: [&str; 6] = ["resolve", "diverge", "apply", "age", "spill", "view"];
/// Passes that run once per plate sample rather than per world cell.
const PER_SAMPLE: [&str; 2] = ["age", "spill"];

/// The uniform block; must match Params in timestep.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    cells: u32,
    plates: u32,
    dt_myr: f32,
    _pad: f32,
    rotation: [[f32; 4]; MAX_PLATES],
}

/// Fixed per world cell; must match WorldCell in timestep.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WorldCell {
    center: [f32; 3],
    dynamic_m: f32,
    swell_m: f32,
    _pad: [f32; 3],
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Readback {
    Idle,
    Pending,
    Ready,
}

pub struct GpuSim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    columns: wgpu::Buffer,
    readback: wgpu::Buffer,
    readback_state: Rc<Cell<Readback>>,
    /// The time of the readback on its way.
    readback_myr: f64,
    cells: u32,
    plates: u32,
    rotation: Vec<DQuat>,
    omega: Vec<DVec3>,
    pub time_myr: f64,
}

impl GpuSim {
    /// Starts at time 0 from `world`, writing into the renderer's cell
    /// buffer (so call again after `Renderer::set_world`).
    pub fn new(renderer: &Renderer, grid: &Grid, world: &World) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let frames = Frames::from_world(world, grid);
        let (cells, plates) = (frames.cells as u32, frames.plates as u32);

        let storage = |label: &str, contents: &[u8], extra: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE | extra,
            })
        };
        let none = wgpu::BufferUsages::empty();
        let world_cells: Vec<WorldCell> = (0..grid.len())
            .map(|c| WorldCell {
                center: grid.centers[c].as_vec3().to_array(),
                dynamic_m: world.dynamic_m[c] as f32,
                swell_m: world.swell_m[c] as f32,
                _pad: [0.0; 3],
            })
            .collect();
        let world_buffer = storage("sim world", bytemuck::cast_slice(&world_cells), none);
        let neighbors = storage("sim neighbors", bytemuck::cast_slice(&grid.neighbors), none);
        let samples = storage("sim samples", bytemuck::cast_slice(&frames.samples), none);
        // Alive flags, then merged-crust tallies and spill shares (start at
        // zero).
        let mut counters: Vec<u32> = frames.alive.iter().map(|&a| u32::from(a)).collect();
        counters.resize(counters.len() * 4, 0);
        let counters = storage("sim counters", bytemuck::cast_slice(&counters), none);
        let scratch = storage(
            "sim scratch",
            &vec![0u8; grid.len() * 16],
            wgpu::BufferUsages::empty(),
        );
        let columns = storage(
            "sim columns",
            &vec![0u8; grid.len() * std::mem::size_of::<ColumnOut>()],
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim readback"),
            size: columns.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sim params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sim"),
            entries: &[
                buffer_entry(0, wgpu::BufferBindingType::Uniform),
                buffer_entry(1, read),
                buffer_entry(2, read),
                buffer_entry(3, write),
                buffer_entry(4, write),
                buffer_entry(5, write),
                buffer_entry(6, write),
                buffer_entry(7, write),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let buffers = [
            &params,
            &world_buffer,
            &neighbors,
            &samples,
            &counters,
            &scratch,
            renderer.cell_buffer(),
            &columns,
        ];
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(renderer.cell_map()),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim"),
            layout: &layout,
            entries: &entries,
        });

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("timestep.wgsl"),
            source: wgpu::ShaderSource::Wgsl(wgsl_source().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sim"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = PASSES
            .iter()
            .map(|&entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();

        Self {
            device,
            queue,
            pipelines,
            bind_group,
            params,
            columns,
            readback,
            readback_state: Rc::new(Cell::new(Readback::Idle)),
            readback_myr: 0.0,
            cells,
            plates,
            rotation: frames.rotation,
            omega: world.plates.plates.iter().map(|p| p.omega).collect(),
            time_myr: 0.0,
        }
    }

    /// Plates turn at `omega` (rad/Myr) from the next step on.
    pub fn set_omega(&mut self, omega: &[DVec3]) {
        self.omega = omega.to_vec();
    }

    /// Advances one step.
    pub fn step(&mut self) {
        for (rot, w) in self.rotation.iter_mut().zip(&self.omega) {
            if w.length() > 0.0 {
                *rot = (DQuat::from_axis_angle(w.normalize(), w.length() * STEP_MYR) * *rot)
                    .normalize();
            }
        }
        let mut params = Params {
            cells: self.cells,
            plates: self.plates,
            dt_myr: STEP_MYR as f32,
            _pad: 0.0,
            rotation: [[0.0; 4]; MAX_PLATES],
        };
        for (slot, q) in params.rotation.iter_mut().zip(&self.rotation) {
            *slot = q.as_quat().to_array();
        }
        self.queue
            .write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let per_cell = self.cells.div_ceil(WORKGROUP);
        let per_sample = (self.cells * self.plates).div_ceil(WORKGROUP);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("time step"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            for (name, pipeline) in PASSES.iter().zip(&self.pipelines) {
                pass.set_pipeline(pipeline);
                let groups = if PER_SAMPLE.contains(name) {
                    per_sample
                } else {
                    per_cell
                };
                pass.dispatch_workgroups(groups, 1, 1);
            }
        }
        self.queue.submit([encoder.finish()]);
        self.time_myr += STEP_MYR;
    }

    /// Whether a readback is on its way.
    pub fn reading_back(&self) -> bool {
        self.readback_state.get() != Readback::Idle
    }

    /// Starts copying the world cells back to the CPU, unless a copy is
    /// already on its way. Collect it with `take_readback`.
    pub fn request_readback(&mut self) {
        if self.reading_back() {
            return;
        }
        self.readback_myr = self.time_myr;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.columns, 0, &self.readback, 0, self.columns.size());
        self.queue.submit([encoder.finish()]);
        self.readback_state.set(Readback::Pending);
        let state = self.readback_state.clone();
        self.readback
            .map_async(wgpu::MapMode::Read, .., move |result| {
                state.set(if result.is_ok() {
                    Readback::Ready
                } else {
                    Readback::Idle
                });
            });
    }

    /// The world cells from the last finished readback and the time they
    /// show, if one is ready.
    pub fn take_readback(&mut self) -> Option<(f64, Vec<ColumnOut>)> {
        if self.readback_state.get() != Readback::Ready {
            return None;
        }
        let cells = self
            .readback
            .get_mapped_range(..)
            .ok()
            .map(|view| bytemuck::cast_slice::<u8, ColumnOut>(&view).to_vec());
        self.readback.unmap();
        self.readback_state.set(Readback::Idle);
        cells.map(|c| (self.readback_myr, c))
    }
}
