use bytemuck::{Pod, Zeroable};
use glam::camera::rh;
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
    pub model: [[f32; 4]; 4],
    pub light_dir: [f32; 4],
}

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y: f32,
    pub aspect: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            eye: Vec3::new(0.0, 0.6, 3.0),
            target: Vec3::ZERO,
            fov_y: 45f32.to_radians(),
            aspect,
        }
    }

    pub fn uniform(&self, model: Mat4) -> CameraUniform {
        let view = rh::view::look_at_mat4(self.eye, self.target, Vec3::Y);
        // WebGPU clip space is Y-up with depth in 0..1, the DirectX convention.
        let proj = rh::proj::directx::perspective(self.fov_y, self.aspect, 0.1, 100.0);
        CameraUniform {
            view_proj: (proj * view).to_cols_array_2d(),
            model: model.to_cols_array_2d(),
            light_dir: [0.6, 0.5, 0.8, 0.0],
        }
    }
}
