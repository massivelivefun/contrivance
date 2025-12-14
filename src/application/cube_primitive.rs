use iced::widget::shader;
use iced::wgpu;
use iced::Rectangle;
use super::cube_pipeline::CubePipeline;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CubePrimitive {
    pub transform: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CubeUniform {
    pub transform: [[f32; 4]; 4],
}

impl shader::Primitive for CubePrimitive {
    type Pipeline = CubePipeline;

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
            transform: self.transform,
        };
        queue.write_buffer(&pipeline.uniform_buffer, 0, bytemuck::cast_slice(&[cube_uniform]));
    }

    // This needs to return false for render to run
    // idk why that's the case, beyond dumb
    fn draw(
        &self,
        pipeline: &Self::Pipeline,
        render_pass: &mut wgpu::RenderPass<'_>,
    ) -> bool {
        // println!("DRAW: Writing to buffer.");
        render_pass.set_pipeline(&pipeline.pipeline);
        // render_pass.set_scissor_rect(clip_bounds.x, clip_bounds.y, clip_bounds.width, clip_bounds.height);
        render_pass.set_bind_group(0, &pipeline.bind_group, &[]);
        render_pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));
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
            label: Some("Cube Pass"),
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

        // Render the cube
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_scissor_rect(clip_bounds.x, clip_bounds.y, clip_bounds.width, clip_bounds.height);
        pass.set_bind_group(0, &pipeline.bind_group, &[]);
        pass.set_vertex_buffer(0, pipeline.vertex_buffer.slice(..));
        // Draw 36 vertices (6 faces * 2 triangles * 3 vertices)
        pass.draw(0..36, 0..1);
    }
}
