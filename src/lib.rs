//! [egui](https://docs.rs/egui) rendering backend for [Vulkano](https://docs.rs/vulkano).
#![warn(missing_docs)]

use std::collections::{HashMap, HashSet};
use std::default::Default;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use egui::epaint::{
    textures::TexturesDelta, ClippedPrimitive, ClippedShape, ImageData, ImageDelta, Primitive,
};
use egui::{Color32, Context, Rect, TextureId};
use smallvec::smallvec;
use thiserror::Error;
use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, BufferImageCopy, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
    SubpassBeginInfo, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::{
    StandardDescriptorSetAllocator, StandardDescriptorSetAllocatorCreateInfo,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::device::Device;
use vulkano::format::Format;
use vulkano::image::sampler::{Filter, Sampler, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::image::view::ImageView;
use vulkano::image::{
    AllocateImageError, Image, ImageAspects, ImageCreateInfo, ImageSubresourceLayers, ImageUsage,
    SampleCount,
};
use vulkano::memory::allocator::{
    AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::vertex_input::VertexDefinition;
use vulkano::pipeline::graphics::viewport::{Scissor, Viewport, ViewportState};
use vulkano::pipeline::graphics::{GraphicsPipeline, GraphicsPipelineCreateInfo};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{DynamicState, PipelineBindPoint};
use vulkano::pipeline::{Pipeline, PipelineLayout, PipelineShaderStageCreateInfo};
use vulkano::render_pass::Subpass;
use vulkano::{Validated, ValidationError, VulkanError};

mod shaders;

#[derive(Vertex, Default, Debug, Clone, Copy, Zeroable, Pod)]
#[repr(C)]
struct EguiVertex {
    #[format(R32G32_SFLOAT)]
    pub pos: [f32; 2],
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
    #[format(R32G32B32A32_SFLOAT)]
    pub color: [f32; 4],
}

impl From<&egui::epaint::Vertex> for EguiVertex {
    fn from(v: &egui::epaint::Vertex) -> Self {
        let convert = {
            |c: Color32| {
                [
                    c.r() as f32 / 255.0,
                    c.g() as f32 / 255.0,
                    c.b() as f32 / 255.0,
                    c.a() as f32 / 255.0,
                ]
            }
        };

        Self {
            pos: [v.pos.x, v.pos.y],
            uv: [v.uv.x, v.uv.y],
            color: convert(v.color),
        }
    }
}

#[derive(Error, Debug)]
pub enum PainterCreationError {
    #[error(transparent)]
    CreatePipelineFailed(Validated<VulkanError>),
    #[error(transparent)]
    CreateSamplerFailed(Validated<VulkanError>),
}

#[derive(Error, Debug)]
pub enum UpdateTexturesError {
    #[error(transparent)]
    CreateDescriptorSet(Validated<VulkanError>),
    #[error(transparent)]
    CreateImage(#[from] Validated<AllocateImageError>),
    #[error(transparent)]
    CreateImageView(Validated<VulkanError>),
    #[error(transparent)]
    Alloc(#[from] Validated<AllocateBufferError>),
}

#[derive(Error, Debug)]
pub enum DrawError {
    #[error(transparent)]
    CreateBuffersFailed(#[from] Validated<AllocateBufferError>),
    #[error(transparent)]
    CommandBuilderValidation(#[from] Box<ValidationError>),
}

#[must_use = "You must use this to avoid attempting to modify a texture that's still in use"]
#[derive(PartialEq)]
/// You must use this to avoid attempting to modify a texture that's still in use.
pub enum UpdateTexturesResult {
    /// No texture will be modified in this frame.
    Unchanged,
    /// A texture will be modified in this frame,
    /// and you must wait for the last frame to finish before submitting the next command buffer.
    Changed,
}

/// Contains everything needed to render the gui.
pub struct Painter {
    allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    /// Graphics pipeline used to render the gui.
    pub pipeline: Arc<GraphicsPipeline>,
    /// Texture sampler used to render the gui.
    pub sampler: Arc<Sampler>,
    images: HashMap<TextureId, Arc<Image>>,
    texture_sets: HashMap<TextureId, Arc<DescriptorSet>>,
    texture_free_queue: Vec<TextureId>,
}

impl Painter {
    /// Pass in the vulkano [`Device`], [`Queue`] and [`Subpass`]
    /// that you want to use to render the gui.
    pub fn new(device: Arc<Device>, subpass: Subpass) -> Result<Self, PainterCreationError> {
        let allocator = StandardMemoryAllocator::new_default(device.clone());
        let descriptor_set_allocator = StandardDescriptorSetAllocator::new(
            device.clone(),
            StandardDescriptorSetAllocatorCreateInfo {
                ..Default::default()
            },
        );
        Self::new_with_allocators(
            device,
            subpass,
            allocator.into(),
            descriptor_set_allocator.into(),
        )
    }

    /// Pass in the vulkano [`Device`], [`Queue`], [`Subpass`] and ['StandardMemoryAllocator']
    /// that you want to use to render the gui.
    pub fn new_with_allocators(
        device: Arc<Device>,
        subpass: Subpass,
        allocator: Arc<StandardMemoryAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> Result<Self, PainterCreationError> {
        let pipeline = create_pipeline(device.clone(), subpass)
            .map_err(PainterCreationError::CreatePipelineFailed)?;
        let sampler =
            create_sampler(device.clone()).map_err(PainterCreationError::CreateSamplerFailed)?;
        Ok(Self {
            allocator,
            descriptor_set_allocator,
            pipeline,
            sampler,
            images: Default::default(),
            texture_sets: Default::default(),
            texture_free_queue: Vec::new(),
        })
    }

    fn write_image_delta(
        &mut self,
        image: Arc<Image>,
        delta: &ImageDelta,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<(), Validated<AllocateBufferError>> {
        let image_data = match &delta.image {
            ImageData::Color(image) => image
                .pixels
                .iter()
                .flat_map(|c| c.to_array())
                .collect::<Vec<_>>(),
            ImageData::Font(image) => image
                .srgba_pixels(Some(1.0))
                .flat_map(|c| c.to_array())
                .collect::<Vec<_>>(),
        };

        let img_buffer = Buffer::from_iter(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            image_data,
        )?;

        let size = [delta.image.width() as u32, delta.image.height() as u32, 1];
        let offset = match delta.pos {
            None => [0, 0, 0],
            Some(pos) => [pos[0] as u32, pos[1] as u32, 0],
        };

        let info = CopyBufferToImageInfo {
            regions: smallvec![BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: ImageSubresourceLayers {
                    aspects: ImageAspects::COLOR,
                    mip_level: 0,
                    array_layers: 0..1,
                },
                image_offset: offset,
                image_extent: size,
                ..Default::default()
            }],
            ..CopyBufferToImageInfo::buffer_image(img_buffer, image)
        };
        builder.copy_buffer_to_image(info)?;
        Ok(())
    }

    /// Uploads all newly created and modified textures to the GPU.
    /// Has to be called before entering the first render pass.  
    /// If the return value is [`UpdateTexturesResult::Changed`],
    /// a texture will be changed in this frame and you need to wait for the last frame to finish
    /// before submitting the command buffer for this frame.
    pub fn update_textures(
        &mut self,
        textures_delta: TexturesDelta,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> Result<UpdateTexturesResult, UpdateTexturesError> {
        for texture_id in textures_delta.free {
            self.texture_free_queue.push(texture_id);
        }

        let mut result = UpdateTexturesResult::Unchanged;

        for (texture_id, delta) in &textures_delta.set {
            let image = if delta.is_whole() {
                let image = create_image(self.allocator.clone(), &delta.image)?;
                let layout = &self.pipeline.layout().set_layouts()[0];

                let set = DescriptorSet::new(
                    self.descriptor_set_allocator.clone(),
                    layout.clone(),
                    [WriteDescriptorSet::image_view_sampler(
                        0,
                        ImageView::new_default(image.clone())
                            .map_err(UpdateTexturesError::CreateImageView)?,
                        self.sampler.clone(),
                    )],
                    [],
                )
                .map_err(UpdateTexturesError::CreateDescriptorSet)?;

                self.texture_sets.insert(*texture_id, set);
                self.images.insert(*texture_id, image.clone());
                image
            } else {
                result = UpdateTexturesResult::Changed; //modifying an existing image that might be in use
                self.images[texture_id].clone()
            };
            self.write_image_delta(image, delta, builder)?;
        }

        Ok(result)
    }

    /// Free textures freed by egui, *after* drawing
    fn free_textures(&mut self) {
        for texture_id in &self.texture_free_queue {
            self.texture_sets.remove(texture_id);
            self.images.remove(texture_id);
        }

        self.texture_free_queue.clear();
    }

    /// Advances to the next rendering subpass and uses the [`ClippedShape`]s from [`egui::FullOutput`] to draw the gui.
    pub fn draw(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        window_size_points: [f32; 2],
        egui_ctx: &Context,
        clipped_shapes: Vec<ClippedShape>,
    ) -> Result<(), DrawError> {
        builder
            .next_subpass(SubpassEndInfo::default(), SubpassBeginInfo::default())?
            .bind_pipeline_graphics(self.pipeline.clone())?
            .set_viewport(
                0,
                smallvec![Viewport {
                    offset: [0.0, 0.0],
                    extent: window_size_points,
                    depth_range: 0.0..=1.0,
                }],
            )?;

        let clipped_primitives: Vec<ClippedPrimitive> = egui_ctx.tessellate(clipped_shapes, 1.0);
        let num_meshes = clipped_primitives.len();

        let mut verts = Vec::<EguiVertex>::with_capacity(num_meshes * 4);
        let mut indices = Vec::<u32>::with_capacity(num_meshes * 6);
        let mut clips = Vec::<Rect>::with_capacity(num_meshes);
        let mut texture_ids = Vec::<TextureId>::with_capacity(num_meshes);
        let mut offsets = Vec::<(usize, usize)>::with_capacity(num_meshes);

        for cm in clipped_primitives.iter() {
            let clip = cm.clip_rect;
            let mesh = match &cm.primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => {
                    continue; // callbacks not supported at the moment
                }
            };

            // Skip empty meshes
            if mesh.vertices.len() == 0 || mesh.indices.len() == 0 {
                continue;
            }

            offsets.push((verts.len(), indices.len()));
            texture_ids.push(mesh.texture_id);

            for v in mesh.vertices.iter() {
                verts.push(v.into());
            }

            for i in mesh.indices.iter() {
                indices.push(*i);
            }

            clips.push(clip);
        }
        offsets.push((verts.len(), indices.len()));

        // Return if there's nothing to render
        if clips.len() == 0 {
            return Ok(());
        }

        let (vertex_buf, index_buf) = self.create_buffers((verts, indices))?;
        for (idx, clip) in clips.iter().enumerate() {
            let mut scissors = Vec::with_capacity(1);
            let o = clip.min;
            let (w, h) = (clip.width() as u32, clip.height() as u32);
            scissors.push(Scissor {
                offset: [(o.x as u32), (o.y as u32)],
                extent: [w, h],
            });
            builder.set_scissor(0, scissors.into())?;

            let offset = offsets[idx];
            let end = offsets[idx + 1];

            let vb_slice = vertex_buf.clone().slice(offset.0 as u64..end.0 as u64);
            let ib_slice = index_buf.clone().slice(offset.1 as u64..end.1 as u64);

            let texture_set = self.texture_sets.get(&texture_ids[idx]);
            if texture_set.is_none() {
                continue; //skip if we don't have a texture
            }

            builder
                .bind_vertex_buffers(0, vb_slice.clone())?
                .bind_index_buffer(ib_slice.clone())?
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    texture_set.unwrap().clone(),
                )?
                .push_constants(self.pipeline.layout().clone(), 0, window_size_points)?
                .draw_indexed(ib_slice.len() as u32, 1, 0, 0, 0)?;
        }
        self.free_textures();
        Ok(())
    }

    /// Create vulkano CpuAccessibleBuffer objects for the vertices and indices
    fn create_buffers(
        &self,
        triangles: (Vec<EguiVertex>, Vec<u32>),
    ) -> Result<(Subbuffer<[EguiVertex]>, Subbuffer<[u32]>), Validated<AllocateBufferError>> {
        let vertex_buffer = Buffer::from_iter(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            triangles.0.iter().cloned(),
        )?;

        let index_buffer = Buffer::from_iter(
            self.allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            triangles.1.iter().cloned(),
        )?;

        Ok((vertex_buffer.into(), index_buffer.into()))
    }
}

/// Create a graphics pipeline with the shaders and settings necessary to render egui output
fn create_pipeline(
    device: Arc<Device>,
    subpass: Subpass,
) -> Result<Arc<GraphicsPipeline>, Validated<VulkanError>> {
    let vs = shaders::vs::load(device.clone())?
        .entry_point("main")
        .unwrap();
    let fs = shaders::fs::load(device.clone())?
        .entry_point("main")
        .unwrap();

    let pipeline = {
        let mut blend = AttachmentBlend::alpha();
        blend.src_color_blend_factor = BlendFactor::One;
        blend.src_alpha_blend_factor = BlendFactor::OneMinusDstAlpha;
        blend.dst_alpha_blend_factor = BlendFactor::One;
        let blend_state = ColorBlendState {
            attachments: vec![ColorBlendAttachmentState {
                blend: Some(blend),
                ..Default::default()
            }],
            ..Default::default()
        };

        let vertex_input_state =
            [EguiVertex::per_vertex()].definition(&vs.info().input_interface)?;
        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(device.clone())
                .unwrap(),
        )?;

        GraphicsPipeline::new(
            device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                stages: stages.into_iter().collect(),
                rasterization_state: Some(RasterizationState::default()),
                multisample_state: Some(MultisampleState {
                    rasterization_samples: subpass.num_samples().unwrap_or(SampleCount::Sample1),
                    ..Default::default()
                }),
                color_blend_state: Some(blend_state),
                vertex_input_state: Some(vertex_input_state),
                dynamic_state: HashSet::from_iter([DynamicState::Viewport, DynamicState::Scissor]),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )?
    };

    Ok(pipeline)
}

/// Create a texture sampler for the textures used by egui
fn create_sampler(device: Arc<Device>) -> Result<Arc<Sampler>, Validated<VulkanError>> {
    Sampler::new(
        device.clone(),
        SamplerCreateInfo {
            mag_filter: Filter::Linear,
            min_filter: Filter::Linear,
            mipmap_mode: SamplerMipmapMode::Linear,
            ..Default::default()
        },
    )
}

/// Create a Vulkano image for the given egui texture
fn create_image(
    allocator: Arc<dyn MemoryAllocator>,
    texture: &ImageData,
) -> Result<Arc<Image>, Validated<AllocateImageError>> {
    Image::new(
        allocator,
        ImageCreateInfo {
            extent: [texture.width() as u32, texture.height() as u32, 1],
            format: Format::R8G8B8A8_UNORM,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::STORAGE,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
}
