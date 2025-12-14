use iced::widget::shader;
use iced::wgpu;
use crate::application::cube_primitive::CubePrimitive;
use iced::wgpu::TextureFormat;
use crate::application::Vertex;
// Required to use buffer_init convenience function
use wgpu::util::DeviceExt;
use crate::application::CUBE_VERTICES;

pub const CUBE_WGSL_SHADER_SOURCE: &str = r#"
    // Uniform holding the transformation matrix
    struct Uniforms {
        transform: mat4x4<f32>,
    };
    @group(0) @binding(0)
    var<uniform> uniforms: Uniforms;
        
    struct VertexInput {
        @location(0) position: vec3<f32>,
        @location(1) color: vec3<f32>,
    };

    struct VertexOutput {
        @builtin(position) clip_position: vec4<f32>,
        @location(0) color: vec3<f32>,
    };
        
    @vertex
    fn vs_main(model: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        // Apply the transformation matrix to the vertex position
        out.clip_position = uniforms.transform * vec4<f32>(model.position, 1.0);
        // Pass the color through to the fragment shader
        out.color = model.color;
        return out;
    }
            
    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        // Return the interpolated color with full opacity
        return vec4<f32>(in.color, 1.0);
    }
"#;

#[derive(Clone, Debug)]
pub struct CubePipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl shader::Pipeline for CubePipeline {
    fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        format: TextureFormat,
    ) -> Self {
        log::info!("Creating WGPU resources for Cube Shader");

        // Load shader code
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CUBE_WGSL_SHADER_SOURCE)),
        });

        // Define vertex buffer layout
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        };

        // Cube primitive for now
        // Create uniform buffer & bind group layout
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<CubePrimitive>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create render pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some("Cube Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader_module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // Create vertex buffer loaded with the Cube data
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Cube Vertex Buffer"),
            contents: bytemuck::cast_slice(CUBE_VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Self {
            pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,
        }
    }

    #[allow(unused)]
    fn trim(&mut self) {
        // Maybe clear caches that are acculuating in the future
        // The cube's data is const and static so it doesn't need to be trimmed
    }
}
