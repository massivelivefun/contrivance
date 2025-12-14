use iced::widget::shader;
use iced::wgpu;
use iced::Rectangle;
use super::skybox_pipeline::SkyboxPipeline;
use crate::application::cube_primitive::CubeUniform;

// I need to think about this more, adding more fields to this struct
// just doesn't seem like good practice or good code smells
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyboxPrimitive {
    pub cube_transform: [[f32; 4]; 4],
    pub skybox_transform: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyboxUniform {
    pub transform: [[f32; 4]; 4],
}

impl shader::Primitive for SkyboxPrimitive {
    type Pipeline = SkyboxPipeline;

    fn prepare(
        &self,
        pipeline: &mut Self::Pipeline,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _bounds: &Rectangle,
        _viewport: &shader::Viewport,
    ) {
        // println!("PREPARE: Writing to buffer.");
        let cube_uniform = CubeUniform {
            transform: self.cube_transform,
        };
        let skybox_uniform = SkyboxUniform {
            transform: self.skybox_transform,
        };
        queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::cast_slice(&[cube_uniform]));
        queue.write_buffer(&pipeline.skybox_uniform_buffer, 0, bytemuck::cast_slice(&[skybox_uniform]));
    }

    // This needs to return false for render to run
    // idk why that's the case, beyond dumb
    fn draw(
        &self,
        pipeline: &Self::Pipeline,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        // println!("DRAW: Writing to buffer.");

        // Shared Vertex Buffer (same cube mesh for both)
        render_pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));

        // Draw the skybox
        render_pass.set_pipeline(&pipeline.skybox_pipeline);
        render_pass.set_bind_group(0, &pipeline.skybox_bind_group, &[]);
        // Draw 36 vertices (6 faces * 2 triangles * 3 vertices)
        render_pass.draw(0..36, 0..1);

        // Draw the cube
        render_pass.set_pipeline(&pipeline.pipeline);
        render_pass.set_bind_group(0, &pipeline.bind_group, &[]);
        // Draw 36 vertices (6 faces * 2 triangles * 3 vertices)
        render_pass.draw(0..36, 0..1);
        false
    }

    fn render(
        &self,
        pipeline: &Self::Pipeline,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        clip_bounds: &iced::Rectangle<u32>,
    ) {
        // println!("RENDER: Writing to buffer.");
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Skybox Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                // I don't think this should be None, docs say...
                // The depth slice index of a 3D view.
                // It must not be provided if the view is not 3D.
                depth_slice: None,
                ops: wgpu::Operations {
                    // load: wgpu::LoadOp::Clear(wgpu::Color::BLUE),
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // Shared Vertex Buffer (same cube mesh for both)
        pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));

        // Draw the skybox
        pass.set_pipeline(&pipeline.skybox_pipeline);
        pass.set_bind_group(0, &pipeline.skybox_bind_group, &[]);
        // Draw 36 vertices (6 faces * 2 triangles * 3 vertices)
        pass.draw(0..36, 0..1);

        // Draw the cube
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, &pipeline.bind_group, &[]);

        pass.set_scissor_rect(clip_bounds.x, clip_bounds.y, clip_bounds.width, clip_bounds.height);
        // Draw 36 vertices (6 faces * 2 triangles * 3 vertices)
        pass.draw(0..36, 0..1);
    }
}
