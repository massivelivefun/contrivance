use iced::widget::shader;
use iced::wgpu;
use iced::wgpu::TextureFormat;
use crate::application::Vertex;
// Required to use buffer_init convenience function
use wgpu::util::DeviceExt;
use crate::application::CUBE_VERTICES;
use crate::application::cube_pipeline::CUBE_WGSL_SHADER_SOURCE;
use crate::application::cube_primitive::CubeUniform;
use crate::application::skybox_primitive::SkyboxUniform;
use image::GenericImageView;

pub const SKYBOX_WGSL_SHADER_SOURCE: &str = r#"
    struct Uniforms {
        transform: mat4x4<f32>,
    };

    @group(0) @binding(0) var<uniform> uniforms: Uniforms;
    @group(0) @binding(1) var t_cube: texture_cube<f32>;
    @group(0) @binding(2) var s_cube: sampler;

    struct VertexInput {
        @location(0) position: vec3<f32>,
        // We don't use the color from the buffer, but it's there in the layout
        @location(1) ignore_color: vec3<f32>, 
    };

    struct VertexOutput {
        @builtin(position) clip_position: vec4<f32>,
        @location(0) tex_coords: vec3<f32>,
    };

    @vertex
    fn vs_main(model: VertexInput) -> VertexOutput {
        var out: VertexOutput;
        // Standard projection
        out.clip_position = uniforms.transform * vec4<f32>(model.position, 1.0);
        // For a skybox, the 3D position of the cube vertex IS the texture coordinate direction
        out.tex_coords = model.position;
        return out;
    }

    @fragment
    fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
        // Sample the cubemap using the direction vector
        return textureSample(t_cube, s_cube, in.tex_coords);
    }
"#;

#[derive(Clone, Debug)]
pub struct SkyboxPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub vertex_buffer: wgpu::Buffer,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,

    pub skybox_pipeline: wgpu::RenderPipeline,
    pub skybox_bind_group: wgpu::BindGroup,
    pub skybox_uniform_buffer: wgpu::Buffer,
}

impl shader::Pipeline for SkyboxPipeline {
    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: TextureFormat,
    ) -> Self {
        log::info!("Creating WGPU resources for Skybox Shader");

        // Load shader code
        let cube_shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
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
            label: Some("Cube Uniform Buffer"),
            size: std::mem::size_of::<CubeUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cube_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cube Bind Group Layout"),
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
            label: Some("Cube Bind Group"),
            layout: &cube_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Create render pipeline layout
        let cube_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cube Pipeline Layout"),
            bind_group_layouts: &[&cube_bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some("Cube Pipeline"),
            layout: Some(&cube_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &cube_shader_module,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &cube_shader_module,
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

        // Cube ^^^^^

        // Skybox vvvvv

        let image_paths: &[&[u8]] = &[
            include_bytes!("../../assets/skyboxes/right.png") as &[u8],  // 0: +X
            include_bytes!("../../assets/skyboxes/left.png") as &[u8],   // 1: -X
            include_bytes!("../../assets/skyboxes/top.png") as &[u8],    // 2: +Y
            include_bytes!("../../assets/skyboxes/bottom.png") as &[u8], // 3: -Y
            include_bytes!("../../assets/skyboxes/front.png") as &[u8],  // 4: +Z
            include_bytes!("../../assets/skyboxes/back.png") as &[u8],   // 5: -Z
        ];

        let first_image = image::load_from_memory(image_paths[0]).expect("Failed to load image");
        let dimensions = first_image.dimensions();
        let width = dimensions.0;
        let height = dimensions.1;

        let texture_size = wgpu::Extent3d { width: width, height: height, depth_or_array_layers: 6 };
        let cubemap_texture = device.create_texture(&wgpu::TextureDescriptor{
            label: Some("Cubemap Texture"),
            size: texture_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // let colors: [[u8; 4]; 6] = [
        //     [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
        //     [255, 255, 0, 255], [0, 255, 255, 255], [255, 0, 255, 255],
        // ];
        for (i, image_bytes) in image_paths.iter().enumerate() {
            // Decode the image
            let img = image::load_from_memory(image_bytes).expect("Failed to load image");
            let rgba = img.to_rgba8();

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &cubemap_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: i as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    // 4 bytes per pixel (R, G, B, A)
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d { width: width, height: height, depth_or_array_layers: 1 },
            );
        }

        let cubemap_view = cubemap_texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SKYBOX_WGSL_SHADER_SOURCE)),
        });

        let skybox_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Skybox Uniform Buffer"),
            size: std::mem::size_of::<SkyboxUniform>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let skybox_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Texture (Cube)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float {
                            filterable: true,
                        },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                // Sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("Cubemap (Skybox) Bind Group Layout"),
        });

        let skybox_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &skybox_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: skybox_uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&cubemap_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
            label: Some("Cubemap (Skybox) Bind Group"),
        });

        let skybox_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&skybox_bind_group_layout],
            ..Default::default()
        });

        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            cache: None,
            label: Some("Cubemap (Skybox) Pipeline"),
            layout: Some(&skybox_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: Some("vs_main"),
                // We reuse the existing vertex buffer layout because it has positions!
                // We just ignore the color attribute in the shader.
                buffers: &[vertex_buffer_layout.clone()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // Important: We are inside the cube, so we see back faces.
                // Standard approach: Disable culling for skybox or cull Front.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None, // No depth testing for the background (its' always behind everything)
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,

            skybox_pipeline,
            skybox_bind_group,
            skybox_uniform_buffer,
        }
    }

    #[allow(unused)]
    fn trim(&mut self) {
        // Maybe clear caches that are acculuating in the future
        // The cube's data is const and static so it doesn't need to be trimmed
    }
}
