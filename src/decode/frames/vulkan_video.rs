//! Vulkan Video frame adapter with zero-copy direct sampling (default) and
//! GPU-copy staging (fallback).
//!
//! The default path samples FFmpeg's Vulkan hardware frames (AVVkFrame) in
//! place: the decoder pool images are wrapped as external wgpu textures (no
//! CPU readback, no GPU copy) and the decode stays on Vulkan hardware. The
//! AVVkFrame consumer contract is followed: wait on the per-plane timeline
//! semaphore at `sem_value`, transition to SHADER_READ_ONLY_OPTIMAL, write
//! layout/access/queue_family back into the AVVkFrame, and — once our
//! sampling submission has completed (the next import's device poll) — signal
//! `sem_value + 1` so the decoder's next decode of the same pool image waits
//! until our reads are done. Set FFGPU_VULKAN_VIDEO_ZERO_COPY=0 to disable
//! direct sampling.
//!
//! The fallback path copies FFmpeg's Vulkan hardware frames (AVVkFrame) into
//! wgpu-owned presentation textures. The decode stays on Vulkan hardware,
//! and the copy is GPU-side (zero CPU readback), but it is not true zero
//! GPU-copy. This avoids the synchronization issues that arise from sampling
//! FFmpeg's decoder pool images directly, which wgpu cannot safely track
//! because `texture_from_raw` does not record the external image's layout,
//! access mask, or queue family ownership.
//!
//! The destination textures are created via `create_texture_from_hal`
//! (backed by raw Vulkan images with dedicated memory) so that wgpu's
//! init tracker marks them as already initialized. This mirrors the
//! D3D11VA path and avoids the discarding clear that wgpu would
//! otherwise issue on first use, which would destroy the copied data.
//!
//! Note: wgpu-core starts every `create_texture_from_hal` texture in
//! `UNINITIALIZED` tracker state, so the first wgpu sampling used to emit a
//! barrier with `oldLayout=UNDEFINED`, which discards the image contents
//! (black frames — zero-copy creates a fresh texture every frame, so this hit
//! every frame). Fixed in the vendored wgpu-core patch
//! (patches/wgpu-core/src/device/resource.rs): sampled hal-imported textures
//! now start in `RESOURCE` tracker state (in the ordered uses mask), so no
//! barrier is emitted on first use. The raw Vulkan transition we record
//! before handing the frame to wgpu leaves the image in
//! SHADER_READ_ONLY_OPTIMAL, matching what wgpu-hal derives for
//! `TextureUses::RESOURCE`.

use super::FrameAdapter;
use super::GlInteropTicket;
use crate::{
    context::{layout, pipeline_cache::PipelineCache},
    decode::{
        frames::FrameAdapterBuilder,
        vulkan_hwcontext::{AVVkFrame, AVVulkanFramesContext, AV_NUM_DATA_POINTERS},
    },
    error::{Error, Result},
};
use ffmpeg_next::sys as ff;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

/// Map a wgpu texture format to the corresponding Vulkan format.
///
/// Only the formats used by Vulkan Video presentation textures are
/// supported (NV12, P010, and their per-plane R/Rg 8/16-bit variants).
fn wgpu_to_vk_format(format: wgpu::TextureFormat) -> Result<ash::vk::Format> {
    use ash::vk::Format as F;
    use wgpu::TextureFormat as Tf;
    let vk = match format {
        Tf::NV12 => F::G8_B8R8_2PLANE_420_UNORM,
        Tf::P010 => F::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16,
        Tf::R8Unorm => F::R8_UNORM,
        Tf::Rg8Unorm => F::R8G8_UNORM,
        Tf::R16Unorm => F::R16_UNORM,
        Tf::Rg16Unorm => F::R16G16_UNORM,
        _ => {
            log::error!("[VulkanVideo] Unsupported presentation texture format: {:?}", format);
            return Err(Error::UnsupportedBackend);
        }
    };
    Ok(vk)
}

/// Create a wgpu texture backed by a raw Vulkan image with dedicated memory.
///
/// This uses `create_texture_from_hal` which marks the texture as already
/// initialized (init: false), avoiding wgpu's discarding clear on first use.
/// This is the same approach used by the D3D11VA path for its shared
/// presentation texture.
///
/// # Safety
///
/// The `hal_device` must be the Vulkan HAL device backing `device`.
unsafe fn create_raw_vulkan_texture(
    device: &wgpu::Device,
    hal_device: &wgpu::hal::vulkan::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    label: &str,
) -> Result<wgpu::Texture> {
    let vk_format = wgpu_to_vk_format(format)?;
    let raw_device = hal_device.raw_device();
    let raw_instance = hal_device.shared_instance().raw_instance();
    let physical_device = hal_device.raw_physical_device();

    // Multi-plane formats (NV12, P010) require MUTABLE_FORMAT and
    // EXTENDED_USAGE to create per-plane image views (e.g. R8Unorm for
    // plane 0, Rg8Unorm for plane 1). Without these flags, vkCreateImageView
    // with per-plane aspects reads garbage data, producing a green screen.
    // This mirrors what wgpu-hal does internally for multi-plane formats.
    let is_multi_plane = format.is_multi_planar_format();
    let mut image_flags = ash::vk::ImageCreateFlags::empty();
    if is_multi_plane {
        image_flags |=
            ash::vk::ImageCreateFlags::MUTABLE_FORMAT | ash::vk::ImageCreateFlags::EXTENDED_USAGE;
    }

    let mut image_info = ash::vk::ImageCreateInfo::default()
        .flags(image_flags)
        .image_type(ash::vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(ash::vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(ash::vk::SampleCountFlags::TYPE_1)
        .tiling(ash::vk::ImageTiling::OPTIMAL)
        .usage(ash::vk::ImageUsageFlags::TRANSFER_DST | ash::vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
        .initial_layout(ash::vk::ImageLayout::UNDEFINED);

    // For mutable-format images, provide a format list with the per-plane
    // formats so the driver knows which view formats will be used. The
    // view_formats Vec must outlive the ImageCreateInfo, so it's declared
    // outside the if block.
    let view_formats: Vec<ash::vk::Format> = if is_multi_plane {
        match vk_format {
            ash::vk::Format::G8_B8R8_2PLANE_420_UNORM => vec![
                ash::vk::Format::R8_UNORM,
                ash::vk::Format::R8G8_UNORM,
                vk_format,
            ],
            ash::vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => vec![
                ash::vk::Format::R16_UNORM,
                ash::vk::Format::R16G16_UNORM,
                vk_format,
            ],
            _ => vec![vk_format],
        }
    } else {
        Vec::new()
    };
    let mut format_list_info = ash::vk::ImageFormatListCreateInfo::default();
    if is_multi_plane {
        format_list_info = format_list_info.view_formats(&view_formats);
        image_info = image_info.push_next(&mut format_list_info);
    }

    let vk_image = unsafe {
        raw_device.create_image(&image_info, None).map_err(|e| {
            log::error!("[VulkanVideo] vkCreateImage failed: {:?}", e);
            Error::UnsupportedBackend
        })?
    };

    let mem_requirements =
        unsafe { raw_device.get_image_memory_requirements(vk_image) };

    let mem_properties =
        unsafe { raw_instance.get_physical_device_memory_properties(physical_device) };

    let memory_type_index = mem_properties
        .memory_types_as_slice()
        .iter()
        .enumerate()
        .find(|(i, mt)| {
            (mem_requirements.memory_type_bits & (1 << i)) != 0
                && mt
                    .property_flags
                    .contains(ash::vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .map(|(i, _)| i as u32)
        .ok_or_else(|| {
            log::error!("[VulkanVideo] No suitable DEVICE_LOCAL memory type found");
            unsafe { raw_device.destroy_image(vk_image, None) };
            Error::UnsupportedBackend
        })?;

    let alloc_info = ash::vk::MemoryAllocateInfo::default()
        .allocation_size(mem_requirements.size)
        .memory_type_index(memory_type_index);

    let device_memory = unsafe {
        raw_device.allocate_memory(&alloc_info, None).map_err(|e| {
            log::error!("[VulkanVideo] vkAllocateMemory failed: {:?}", e);
            raw_device.destroy_image(vk_image, None);
            Error::UnsupportedBackend
        })?
    };

    unsafe {
        raw_device
            .bind_image_memory(vk_image, device_memory, 0)
            .map_err(|e| {
                log::error!("[VulkanVideo] vkBindImageMemory failed: {:?}", e);
                raw_device.free_memory(device_memory, None);
                raw_device.destroy_image(vk_image, None);
                Error::UnsupportedBackend
            })?;
    }

    let hal_texture = unsafe {
        hal_device.texture_from_raw(
            vk_image,
            &wgpu::hal::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUses::COPY_DST | wgpu::TextureUses::RESOURCE,
                memory_flags: wgpu::hal::MemoryFlags::empty(),
                view_formats: vec![],
            },
            None,
            wgpu::hal::vulkan::TextureMemory::Dedicated(device_memory),
        )
    };

    let texture = unsafe {
        device.create_texture_from_hal::<wgpu::hal::vulkan::Api>(
            hal_texture,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
        )
    };

    Ok(texture)
}

/// Persistent presentation state for the GPU-copy staging path.
///
/// The destination texture(s) are wgpu-owned (via `create_texture_from_hal`)
/// and reused across frames. Each frame, the decoded VkImage is copied into
/// the destination via raw Vulkan commands on a dedicated command buffer.
/// The textures are marked as already initialized (init: false) so wgpu
/// does not issue a discarding clear on first use.
struct PresentationTexture {
    /// The wgpu-owned destination texture(s).
    /// For multi-plane (NV12/P010): a single texture.
    /// For separate-plane: one texture per plane.
    textures: Vec<wgpu::Texture>,
    /// The layout descriptor for shader binding.
    #[allow(dead_code)]
    texture_format: layout::FrameDescriptor<wgpu::TextureFormat>,
    /// Whether the source uses a single multi-plane VkImage.
    #[allow(dead_code)]
    is_multiplane: bool,
    /// The wgpu multi-plane format (if multi-plane).
    #[allow(dead_code)]
    multiplane_format: Option<wgpu::TextureFormat>,
    /// Frame dimensions.
    width: u32,
    height: u32,
    /// Bind group for the presentation texture views.
    bg0: wgpu::BindGroup,
    /// Identity layout.
    identity: layout::FrameDescriptor<()>,
}

/// A retained AVFrame reference for zero-copy sampling.
struct HeldAvFrame {
    frame: NonNull<ff::AVFrame>,
}

impl HeldAvFrame {
    unsafe fn new(src: NonNull<ff::AVFrame>) -> Result<Self> {
        let frame = unsafe { ff::av_frame_alloc() };
        let Some(frame) = NonNull::new(frame) else {
            return Err(Error::Unknown);
        };

        let ret = unsafe { ff::av_frame_ref(frame.as_ptr(), src.as_ptr()) };
        if ret < 0 {
            let mut frame_ptr = frame.as_ptr();
            unsafe { ff::av_frame_free(&mut frame_ptr) };
            return Err(Error::Unknown);
        }

        Ok(HeldAvFrame { frame })
    }
}

impl Drop for HeldAvFrame {
    fn drop(&mut self) {
        let mut frame = self.frame.as_ptr();
        unsafe { ff::av_frame_free(&mut frame) };
    }
}

/// Per-frame direct import state for zero-copy sampling.
struct ZeroCopyImportedFrame {
    _held_frame: HeldAvFrame,
    _textures: Vec<wgpu::Texture>,
    bg0: wgpu::BindGroup,
    identity: layout::FrameDescriptor<()>,
}

impl ZeroCopyImportedFrame {
    /// Per-plane texture views for the engine's YUV shader. Multi-plane images
    /// (NV12/P010) are a single texture; expose the per-plane aspect views so
    /// the engine can sample Y (Plane0) and UV (Plane1) separately.
    fn plane_views(&self) -> Vec<wgpu::TextureView> {
        let mut views = Vec::with_capacity(self._textures.len());
        match &self.identity.planes {
            layout::PlaneLayout::PackedYUV420(_) => {
                let tex = &self._textures[0];
                views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                    aspect: wgpu::TextureAspect::Plane0,
                    ..Default::default()
                }));
                views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                    aspect: wgpu::TextureAspect::Plane1,
                    ..Default::default()
                }));
            }
            _ => {
                for tex in &self._textures {
                    views.push(tex.create_view(&wgpu::TextureViewDescriptor::default()));
                }
            }
        }
        views
    }
}

/// Raw Vulkan copy context, created lazily on first frame.
///
/// This owns a dedicated Vulkan command pool and command buffer used to
/// record the GPU-side image copy. We use a separate command buffer
/// (not wgpu's command encoder) because wgpu forbids mixing its encoding
/// API with the raw encoding API (`as_hal_mut`) on the same encoder,
/// and because the copy needs explicit source/destination layout
/// transitions that wgpu's tracker cannot manage for external images.
struct VulkanCopyContext {
    command_pool: ash::vk::CommandPool,
    raw_device: ash::Device,
    queue: ash::vk::Queue,
    #[allow(dead_code)]
    queue_family_index: u32,
    slots: Vec<VulkanCopySlot>,
}

struct VulkanCopySlot {
    command_buffer: ash::vk::CommandBuffer,
    fence: ash::vk::Fence,
}

const MAX_COPY_SLOTS: usize = 4;

impl VulkanCopyContext {
    /// Create a new copy context from the wgpu hal device and queue.
    ///
    /// # Safety
    ///
    /// The device and queue must remain valid for the lifetime of this
    /// context.
    unsafe fn new(
        raw_device: ash::Device,
        queue: ash::vk::Queue,
        queue_family_index: u32,
    ) -> std::result::Result<Self, ash::vk::Result> {
        let command_pool = unsafe {
            raw_device.create_command_pool(
                &ash::vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_index)
                    .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        Ok(VulkanCopyContext {
            command_pool,
            raw_device,
            queue,
            queue_family_index,
            slots: Vec::new(),
        })
    }

    /// Record and submit a GPU-side image copy.
    ///
    /// The `record` closure receives the raw `vk::CommandBuffer` and should
    /// record all commands (barriers, copies, etc.) onto it. The command
    /// buffer is begun, the closure is called, the buffer is ended, submitted
    /// to the queue. Later work on the same queue is ordered after it.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all resources used in the recorded commands
    /// are valid and not being used by other queue submissions concurrently.
    unsafe fn submit_copy<F>(&mut self, record: F) -> std::result::Result<(), ash::vk::Result>
    where
        F: FnOnce(ash::vk::CommandBuffer),
    {
        unsafe {
            let slot_index = self
                .slots
                .iter()
                .position(|slot| self.raw_device.get_fence_status(slot.fence).unwrap_or(false));
            let slot_index = match slot_index {
                Some(index) => index,
                None if self.slots.len() < MAX_COPY_SLOTS => {
                    let command_buffer = self.raw_device.allocate_command_buffers(
                        &ash::vk::CommandBufferAllocateInfo::default()
                            .command_pool(self.command_pool)
                            .level(ash::vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?[0];
                    let fence = self.raw_device.create_fence(
                        &ash::vk::FenceCreateInfo::default()
                            .flags(ash::vk::FenceCreateFlags::SIGNALED),
                        None,
                    )?;
                    self.slots.push(VulkanCopySlot {
                        command_buffer,
                        fence,
                    });
                    self.slots.len() - 1
                }
                None => {
                    let fence = self.slots[0].fence;
                    self.raw_device.wait_for_fences(&[fence], true, u64::MAX)?;
                    0
                }
            };
            let slot = &self.slots[slot_index];
            self.raw_device.reset_fences(&[slot.fence])?;
            self.raw_device.reset_command_buffer(
                slot.command_buffer,
                ash::vk::CommandBufferResetFlags::empty(),
            )?;

            self.raw_device.begin_command_buffer(
                slot.command_buffer,
                &ash::vk::CommandBufferBeginInfo::default()
                    .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            record(slot.command_buffer);

            self.raw_device.end_command_buffer(slot.command_buffer)?;

            let submit_info = ash::vk::SubmitInfo::default()
                .command_buffers(std::slice::from_ref(&slot.command_buffer));

            self.raw_device.queue_submit(
                self.queue,
                std::slice::from_ref(&submit_info),
                slot.fence,
            )?;

        }

        Ok(())
    }
}

impl Drop for VulkanCopyContext {
    fn drop(&mut self) {
        unsafe {
            let _ = self.raw_device.device_wait_idle();
            for slot in &self.slots {
                self.raw_device.destroy_fence(slot.fence, None);
            }
            let command_buffers: Vec<_> = self
                .slots
                .iter()
                .map(|slot| slot.command_buffer)
                .collect();
            self.raw_device
                .free_command_buffers(self.command_pool, &command_buffers);
            self.raw_device
                .destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct VulkanVideoFrameAdapter {
    /// Persistent presentation texture, created lazily on first frame.
    presentation: Option<PresentationTexture>,
    /// Current direct-import frame for zero-copy sampling.
    zero_copy_current: Option<ZeroCopyImportedFrame>,
    /// Whether direct zero-copy sampling is enabled (default on; FFGPU_VULKAN_VIDEO_ZERO_COPY=0 disables).
    zero_copy_enabled: bool,
    /// Raw Vulkan copy/transition context, created lazily on first frame.
    copy_ctx: Option<VulkanCopyContext>,
    /// Imported frames kept alive until the queue submission that sampled them completes.
    retired_zero_copy: Vec<ZeroCopyImportedFrame>,
}

impl FrameAdapterBuilder for VulkanVideoFrameAdapter {
    unsafe fn new(_decoder: NonNull<ff::AVCodecContext>) -> Result<Self> {
        // Zero-copy direct sampling is the default. Set
        // FFGPU_VULKAN_VIDEO_ZERO_COPY=0|false|off to force the GPU-copy path.
        let zero_copy_enabled = !std::env::var("FFGPU_VULKAN_VIDEO_ZERO_COPY")
            .map(|v| matches!(v.as_str(), "0" | "false" | "FALSE" | "no" | "NO" | "off" | "OFF"))
            .unwrap_or(false);

        Ok(VulkanVideoFrameAdapter {
            presentation: None,
            zero_copy_current: None,
            zero_copy_enabled,
            copy_ctx: None,
            retired_zero_copy: Vec::new(),
        })
    }

    fn supports_format(format: ff::AVPixelFormat) -> bool {
        format == ff::AVPixelFormat::AV_PIX_FMT_VULKAN
    }
}

impl VulkanVideoFrameAdapter {
    /// Create the persistent presentation texture and bind group.
    ///
    /// This is called once (on the first frame, or when dimensions change).
    /// The destination textures are created via `create_texture_from_hal`
    /// (backed by raw Vulkan images with dedicated memory) so that wgpu's
    /// init tracker marks them as already initialized. This avoids the
    /// discarding clear that wgpu would otherwise issue on first use,
    /// which would destroy the data we copy via raw Vulkan commands.
    #[allow(clippy::too_many_arguments)]
    fn create_presentation(
        &mut self,
        device: &wgpu::Device,
        pipeline_cache: &mut PipelineCache,
        width: u32,
        height: u32,
        texture_format: &layout::FrameDescriptor<wgpu::TextureFormat>,
        is_multiplane: bool,
        multiplane_format: Option<wgpu::TextureFormat>,
        color_space: ffmpeg_next::color::Space,
    ) -> Result<&PresentationTexture> {
        let hal_device = unsafe {
            device
                .as_hal::<wgpu::hal::vulkan::Api>()
                .ok_or_else(|| {
                    log::error!(
                        "[VulkanVideo] device.as_hal::<vulkan::Api>() returned None"
                    );
                    Error::UnsupportedBackend
                })?
        };

        let textures = if is_multiplane {
            // Single multi-plane destination texture (NV12 or P010).
            let mp_fmt = multiplane_format.unwrap();
            let tex = unsafe {
                create_raw_vulkan_texture(
                    device,
                    &*hal_device,
                    mp_fmt,
                    width,
                    height,
                    "ffgpu Vulkan Video presentation (multi-plane)",
                )?
            };
            vec![tex]
        } else {
            // Separate-plane destination textures, one per plane.
            let make_tex = |fmt: &wgpu::TextureFormat, wd: u32, hd: u32| {
                unsafe {
                    create_raw_vulkan_texture(
                        device,
                        &*hal_device,
                        *fmt,
                        width / wd,
                        height / hd,
                        "ffgpu Vulkan Video presentation (plane)",
                    )
                }
            };
            match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420([y, uv]) => {
                    vec![make_tex(y, 1, 1)?, make_tex(uv, 2, 2)?]
                }
                layout::PlaneLayout::YUV420([y, u, v]) => {
                    vec![make_tex(y, 1, 1)?, make_tex(u, 2, 2)?, make_tex(v, 2, 2)?]
                }
                layout::PlaneLayout::YUV444([y, u, v]) => {
                    vec![make_tex(y, 1, 1)?, make_tex(u, 1, 1)?, make_tex(v, 1, 1)?]
                }
                layout::PlaneLayout::RGB(fmt) => {
                    vec![make_tex(fmt, 1, 1)?]
                }
            }
        };

        // Create texture views for the bind group.
        let plane_views = if is_multiplane {
            let texture = &textures[0];
            match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420(_) => {
                    let y_view = texture.create_view(&wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::Plane0,
                        ..Default::default()
                    });
                    let uv_view = texture.create_view(&wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::Plane1,
                        ..Default::default()
                    });
                    layout::PlaneLayout::PackedYUV420([y_view, uv_view])
                }
                _ => {
                    log::error!(
                        "[VulkanVideo] Multi-plane image with unexpected layout {:?}",
                        texture_format.planes
                    );
                    return Err(Error::UnsupportedBackend);
                }
            }
        } else {
            let views: Vec<wgpu::TextureView> = textures
                .iter()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                .collect();

            let mut iter = views.into_iter();
            match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420(_) => {
                    let y = iter.next().ok_or(Error::InvalidFrame)?;
                    let uv = iter.next().ok_or(Error::InvalidFrame)?;
                    layout::PlaneLayout::PackedYUV420([y, uv])
                }
                layout::PlaneLayout::YUV420(_) => {
                    let y = iter.next().ok_or(Error::InvalidFrame)?;
                    let u = iter.next().ok_or(Error::InvalidFrame)?;
                    let v = iter.next().ok_or(Error::InvalidFrame)?;
                    layout::PlaneLayout::YUV420([y, u, v])
                }
                layout::PlaneLayout::YUV444(_) => {
                    let y = iter.next().ok_or(Error::InvalidFrame)?;
                    let u = iter.next().ok_or(Error::InvalidFrame)?;
                    let v = iter.next().ok_or(Error::InvalidFrame)?;
                    layout::PlaneLayout::YUV444([y, u, v])
                }
                layout::PlaneLayout::RGB(_) => {
                    let view = iter.next().ok_or(Error::InvalidFrame)?;
                    layout::PlaneLayout::RGB(view)
                }
            }
        };

        let frame_descriptor = layout::FrameDescriptor {
            planes: plane_views,
            depth: texture_format.depth,
        };

        let bg0 = pipeline_cache.bind_frame_textures(&frame_descriptor, color_space);

        self.presentation = Some(PresentationTexture {
            textures,
            texture_format: *texture_format,
            is_multiplane,
            multiplane_format,
            width,
            height,
            bg0,
            identity: texture_format.as_identity(),
        });

        // Safe because we just assigned to self.presentation.
        Ok(self.presentation.as_ref().unwrap())
    }
}

impl VulkanVideoFrameAdapter {
    /// Signal an imported zero-copy frame back to FFmpeg once our sampling has
    /// completed (the caller must have waited for all wgpu work first).
    ///
    /// Per the AVVkFrame consumer contract (libavutil/hwcontext_vulkan.h):
    /// at every submission touching the frame's images the consumer must wait
    /// on `sem[i]` at `sem_value[i]`, then signal `sem_value[i] + 1` so the
    /// decoder's next decode of the same pool image GPU-waits until our work
    /// is done. The layout/access/queue_family fields were already written
    /// back at import time ("updated after every barrier").
    unsafe fn release_zero_copy_frame(frame: &HeldAvFrame, raw_device: &ash::Device) {
        let av_frame = unsafe { frame.frame.as_ref() };
        if av_frame.data[0].is_null() {
            return;
        }
        let vk_frame = unsafe { &mut *(av_frame.data[0] as *mut AVVkFrame) };

        let mut plane_count = 0;
        while plane_count < AV_NUM_DATA_POINTERS
            && vk_frame.img[plane_count] != ash::vk::Image::null()
        {
            plane_count += 1;
        }

        let mut deduped: Vec<(ash::vk::Semaphore, u64)> = Vec::with_capacity(plane_count);
        for i in 0..plane_count {
            let sem = vk_frame.sem[i];
            if sem == ash::vk::Semaphore::null() {
                continue;
            }
            let value = vk_frame.sem_value[i];
            if let Some((_, v)) = deduped.iter_mut().find(|(s, _)| *s == sem) {
                *v = (*v).max(value);
            } else {
                deduped.push((sem, value));
            }
        }

        for (sem, value) in deduped {
            // Monotonic bump: the pool image is still owned by our AVFrame ref,
            // so no other submission can be signalling this semaphore. The
            // decoder's next use reads the updated value and waits on it.
            let signal_info = ash::vk::SemaphoreSignalInfo::default()
                .semaphore(sem)
                .value(value + 1);
            if let Err(e) = unsafe { raw_device.signal_semaphore(&signal_info) } {
                log::error!(
                    "[VulkanVideo] release: vkSignalSemaphore({:?}, {}) failed: {:?}",
                    sem,
                    value + 1,
                    e
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn import_zero_copy_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<()> {
        let av_frame = unsafe { frame.as_ref() };
        if av_frame.format != ff::AVPixelFormat::AV_PIX_FMT_VULKAN as i32 {
            return Err(Error::UnsupportedBackend);
        }
        if av_frame.hw_frames_ctx.is_null() || av_frame.data[0].is_null() {
            return Err(Error::UnsupportedBackend);
        }

        let vk_frame = unsafe { &mut *(av_frame.data[0] as *mut AVVkFrame) };
        let frames_ctx = unsafe {
            let buf_ref = &*(av_frame.hw_frames_ctx as *mut ff::AVBufferRef);
            &*(buf_ref.data as *mut ff::AVHWFramesContext)
        };
        let vulkan_frames_ctx = frames_ctx.hwctx as *mut AVVulkanFramesContext;
        if vulkan_frames_ctx.is_null() {
            return Err(Error::UnsupportedBackend);
        }
        let vulkan_frames_ctx = unsafe { &*vulkan_frames_ctx };

        let width = av_frame.width as u32;
        let height = av_frame.height as u32;
        let vk_format = vulkan_frames_ctx.format[0];
        let sw_format = frames_ctx.sw_format;
        let texture_format = layout::vk_format_texture_format(
            vk_format,
            ff::AVPixelFormat::try_from(sw_format)
                .unwrap_or(ff::AVPixelFormat::AV_PIX_FMT_NONE),
        )
        .ok_or(Error::UnsupportedBackend)?;

        let expected_planes = match &texture_format.planes {
            layout::PlaneLayout::PackedYUV420(_) => 2,
            layout::PlaneLayout::YUV420(_) => 3,
            layout::PlaneLayout::YUV444(_) => 3,
            layout::PlaneLayout::RGB(_) => 1,
        };

        let mut plane_count = 0;
        while plane_count < AV_NUM_DATA_POINTERS
            && vk_frame.img[plane_count] != ash::vk::Image::null()
        {
            plane_count += 1;
        }
        if plane_count == 0 {
            return Err(Error::UnsupportedBackend);
        }

        let is_multiplane = plane_count == 1 && expected_planes > 1;

        // One-shot diagnostics: dump the AVVkFrame contract state (decode
        // semaphore, layout, access, queue family) so a runtime mismatch with
        // the linked FFmpeg's struct layout is immediately visible.
        {
            static DIAG: AtomicBool = AtomicBool::new(false);
            if !DIAG.swap(true, Ordering::Relaxed) {
                let mut sem_brief = String::new();
                for i in 0..plane_count {
                    sem_brief.push_str(&format!(
                        "  plane {}: img={:?} sem={:?} sem_value={} layout={:?} access={:?} qf={}\n",
                        i,
                        vk_frame.img[i],
                        vk_frame.sem[i],
                        vk_frame.sem_value[i],
                        vk_frame.layout[i],
                        vk_frame.access[i],
                        vk_frame.queue_family[i],
                    ));
                }
                eprintln!(
                    "[VulkanVideo] zero-copy import: vk_format={:?} sw_format={:?} multiplane={} expected_planes={} frame={}x{}\n{}",
                    vulkan_frames_ctx.format[0],
                    sw_format,
                    is_multiplane,
                    expected_planes,
                    width,
                    height,
                    sem_brief,
                );
            }
        }

        let multiplane_format = if is_multiplane {
            match vk_format {
                ash::vk::Format::G8_B8R8_2PLANE_420_UNORM => wgpu::TextureFormat::NV12,
                ash::vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => {
                    wgpu::TextureFormat::P010
                }
                _ => return Err(Error::UnsupportedBackend),
            }
        } else {
            wgpu::TextureFormat::R8Unorm
        };

        let hal_device = unsafe {
            device
                .as_hal::<wgpu::hal::vulkan::Api>()
                .ok_or(Error::UnsupportedBackend)?
        };
        let raw_device = (*hal_device).raw_device();
        let graphics_queue_family = (*hal_device).queue_family_index();

        for i in 0..plane_count {
            let src_qf = vk_frame.queue_family[i];
            if src_qf != ash::vk::QUEUE_FAMILY_IGNORED && src_qf != graphics_queue_family {
                return Err(Error::UnsupportedBackend);
            }
        }

        // Keep the previous pool image and its external texture wrappers alive
        // until the queue submission that sampled it completes. The render
        // queue callback releases it without blocking the event-loop thread.
        if let Some(prev) = self.zero_copy_current.take() {
            self.retired_zero_copy.push(prev);
        }

        let mut semaphores: Vec<ash::vk::Semaphore> = Vec::with_capacity(plane_count);
        let mut values: Vec<u64> = Vec::with_capacity(plane_count);
        for i in 0..plane_count {
            let sem = vk_frame.sem[i];
            if sem != ash::vk::Semaphore::null() {
                let value = vk_frame.sem_value[i];
                if let Some(pos) = semaphores.iter().position(|s| *s == sem) {
                    values[pos] = values[pos].max(value);
                } else {
                    semaphores.push(sem);
                    values.push(value);
                }
            }
        }
        if !semaphores.is_empty() {
            let wait_info = ash::vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            unsafe { raw_device.wait_semaphores(&wait_info, u64::MAX) }
                .map_err(|_| Error::UnsupportedBackend)?;
        }

        if self.copy_ctx.is_none() {
            let hal_queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::vulkan::Api>()
                    .ok_or(Error::UnsupportedBackend)?
            };
            let copy_ctx = unsafe {
                VulkanCopyContext::new(
                    raw_device.clone(),
                    (*hal_queue).as_raw(),
                    graphics_queue_family,
                )
            }
            .map_err(|_| Error::UnsupportedBackend)?;
            self.copy_ctx = Some(copy_ctx);
        }
        let copy_ctx = self.copy_ctx.as_mut().unwrap();

        unsafe {
            copy_ctx
                .submit_copy(|cmd_buffer| {
                    for i in 0..plane_count {
                        let aspect = if is_multiplane {
                            ash::vk::ImageAspectFlags::PLANE_0 | ash::vk::ImageAspectFlags::PLANE_1
                        } else {
                            ash::vk::ImageAspectFlags::COLOR
                        };
                        let barrier = ash::vk::ImageMemoryBarrier::default()
                            .image(vk_frame.img[i])
                            .subresource_range(ash::vk::ImageSubresourceRange {
                                aspect_mask: aspect,
                                base_mip_level: 0,
                                level_count: ash::vk::REMAINING_MIP_LEVELS,
                                base_array_layer: 0,
                                layer_count: ash::vk::REMAINING_ARRAY_LAYERS,
                            })
                            .src_access_mask(vk_frame.access[i])
                            .dst_access_mask(ash::vk::AccessFlags::SHADER_READ)
                            .old_layout(vk_frame.layout[i])
                            .new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED);
                        raw_device.cmd_pipeline_barrier(
                            cmd_buffer,
                            ash::vk::PipelineStageFlags::ALL_COMMANDS,
                            ash::vk::PipelineStageFlags::FRAGMENT_SHADER,
                            ash::vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            std::slice::from_ref(&barrier),
                        );
                    }
                })
                .map_err(|_| Error::UnsupportedBackend)?;
        }
        for i in 0..plane_count {
            vk_frame.layout[i] = ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            vk_frame.access[i] = ash::vk::AccessFlags::SHADER_READ;
            vk_frame.queue_family[i] = ash::vk::QUEUE_FAMILY_IGNORED;
        }

        let make_external_texture = |image: ash::vk::Image,
                                     format: wgpu::TextureFormat,
                                     tex_width: u32,
                                     tex_height: u32|
         -> Result<wgpu::Texture> {
            let hal_texture = unsafe {
                (*hal_device).texture_from_raw(
                    image,
                    &wgpu::hal::TextureDescriptor {
                        label: Some("ffgpu Vulkan Video zero-copy external"),
                        size: wgpu::Extent3d {
                            width: tex_width,
                            height: tex_height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUses::RESOURCE,
                        memory_flags: wgpu::hal::MemoryFlags::empty(),
                        view_formats: vec![],
                    },
                    Some(Box::new(|| {})),
                    wgpu::hal::vulkan::TextureMemory::External,
                )
            };
            Ok(unsafe {
                device.create_texture_from_hal::<wgpu::hal::vulkan::Api>(
                    hal_texture,
                    &wgpu::TextureDescriptor {
                        label: Some("ffgpu Vulkan Video zero-copy external"),
                        size: wgpu::Extent3d {
                            width: tex_width,
                            height: tex_height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                )
            })
        };

        let textures = if is_multiplane {
            vec![make_external_texture(vk_frame.img[0], multiplane_format, width, height)?]
        } else {
            match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420([y, uv]) => vec![
                    make_external_texture(vk_frame.img[0], *y, width, height)?,
                    make_external_texture(vk_frame.img[1], *uv, width / 2, height / 2)?,
                ],
                layout::PlaneLayout::YUV420([y, u, v]) => vec![
                    make_external_texture(vk_frame.img[0], *y, width, height)?,
                    make_external_texture(vk_frame.img[1], *u, width / 2, height / 2)?,
                    make_external_texture(vk_frame.img[2], *v, width / 2, height / 2)?,
                ],
                layout::PlaneLayout::YUV444([y, u, v]) => vec![
                    make_external_texture(vk_frame.img[0], *y, width, height)?,
                    make_external_texture(vk_frame.img[1], *u, width, height)?,
                    make_external_texture(vk_frame.img[2], *v, width, height)?,
                ],
                layout::PlaneLayout::RGB(fmt) => {
                    vec![make_external_texture(vk_frame.img[0], *fmt, width, height)?]
                }
            }
        };

        let plane_views = if is_multiplane {
            let y_view = textures[0].create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::Plane0,
                ..Default::default()
            });
            let uv_view = textures[0].create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::Plane1,
                ..Default::default()
            });
            match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420(_) => layout::PlaneLayout::PackedYUV420([y_view, uv_view]),
                _ => return Err(Error::UnsupportedBackend),
            }
        } else {
            let views: Vec<wgpu::TextureView> = textures
                .iter()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                .collect();
            let mut iter = views.into_iter();
            match &texture_format.planes {
                layout::PlaneLayout::PackedYUV420(_) => {
                    layout::PlaneLayout::PackedYUV420([
                        iter.next().ok_or(Error::InvalidFrame)?,
                        iter.next().ok_or(Error::InvalidFrame)?,
                    ])
                }
                layout::PlaneLayout::YUV420(_) => {
                    layout::PlaneLayout::YUV420([
                        iter.next().ok_or(Error::InvalidFrame)?,
                        iter.next().ok_or(Error::InvalidFrame)?,
                        iter.next().ok_or(Error::InvalidFrame)?,
                    ])
                }
                layout::PlaneLayout::YUV444(_) => {
                    layout::PlaneLayout::YUV444([
                        iter.next().ok_or(Error::InvalidFrame)?,
                        iter.next().ok_or(Error::InvalidFrame)?,
                        iter.next().ok_or(Error::InvalidFrame)?,
                    ])
                }
                layout::PlaneLayout::RGB(_) => {
                    layout::PlaneLayout::RGB(iter.next().ok_or(Error::InvalidFrame)?)
                }
            }
        };

        let frame_descriptor = layout::FrameDescriptor {
            planes: plane_views,
            depth: texture_format.depth,
        };
        let bg0 = pipeline_cache.bind_frame_textures(&frame_descriptor, av_frame.colorspace.into());
        let held_frame = unsafe { HeldAvFrame::new(frame)? };

        self.zero_copy_current = Some(ZeroCopyImportedFrame {
            _held_frame: held_frame,
            _textures: textures,
            bg0,
            identity: texture_format.as_identity(),
        });

        Ok(())
    }
}

impl FrameAdapter for VulkanVideoFrameAdapter {
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        _instance: &wgpu::Instance,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        _pipeline_cache: &mut PipelineCache,
    ) -> Result<Option<GlInteropTicket>> {
        if self.zero_copy_enabled {
            match unsafe { self.import_zero_copy_frame(frame, device, queue, _pipeline_cache) } {
                Ok(()) => return Ok(None),
                Err(e) => {
                    log::warn!(
                        "[VulkanVideo] zero-copy import failed: {:?}; falling back to GPU-copy",
                        e
                    );
                }
            }
        }

        self.zero_copy_current = None;
        let frame = unsafe { frame.as_ref() };

        if frame.format != ff::AVPixelFormat::AV_PIX_FMT_VULKAN as i32 {
            log::error!(
                "[VulkanVideo] frame.format ({}) != AV_PIX_FMT_VULKAN",
                frame.format
            );
            return Err(Error::UnsupportedBackend);
        }

        if frame.hw_frames_ctx.is_null() {
            log::error!("[VulkanVideo] AVFrame has no hw_frames_ctx");
            return Err(Error::UnsupportedBackend);
        }

        // The AVVkFrame is stored in frame.data[0].
        let vk_frame = frame.data[0] as *mut AVVkFrame;
        if vk_frame.is_null() {
            log::error!("[VulkanVideo] AVFrame.data[0] is null");
            return Err(Error::UnsupportedBackend);
        }
        let vk_frame = unsafe { &*vk_frame };

        let frames_ctx = unsafe {
            let buf_ref = &*(frame.hw_frames_ctx as *mut ff::AVBufferRef);
            &*(buf_ref.data as *mut ff::AVHWFramesContext)
        };
        let vulkan_frames_ctx = frames_ctx.hwctx as *mut AVVulkanFramesContext;
        if vulkan_frames_ctx.is_null() {
            log::error!("[VulkanVideo] AVHWFramesContext.hwctx is null");
            return Err(Error::UnsupportedBackend);
        }
        let vulkan_frames_ctx = unsafe { &*vulkan_frames_ctx };

        let sw_format = frames_ctx.sw_format;
        let width = frame.width as u32;
        let height = frame.height as u32;
        let vk_format = vulkan_frames_ctx.format[0];

        // Count valid planes.
        let mut plane_count = 0;
        while plane_count < AV_NUM_DATA_POINTERS
            && vk_frame.img[plane_count] != ash::vk::Image::null()
        {
            plane_count += 1;
        }

        if plane_count == 0 {
            log::error!("[VulkanVideo] AVVkFrame has no images");
            return Err(Error::UnsupportedBackend);
        }

        let texture_format = layout::vk_format_texture_format(
            vk_format,
            ff::AVPixelFormat::try_from(sw_format)
                .unwrap_or(ff::AVPixelFormat::AV_PIX_FMT_NONE),
        )
        .ok_or_else(|| {
            log::error!(
                "[VulkanVideo] Unsupported Vulkan format {:?} / sw_format {:?}",
                vk_format,
                sw_format
            );
            Error::UnsupportedBackend
        })?;

        let expected_planes = match &texture_format.planes {
            layout::PlaneLayout::PackedYUV420(_) => 2,
            layout::PlaneLayout::YUV420(_) => 3,
            layout::PlaneLayout::YUV444(_) => 3,
            layout::PlaneLayout::RGB(_) => 1,
        };

        let is_multiplane = plane_count == 1 && expected_planes > 1;

        let multiplane_format = if is_multiplane {
            match vk_format {
                ash::vk::Format::G8_B8R8_2PLANE_420_UNORM => Some(wgpu::TextureFormat::NV12),
                ash::vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16 => {
                    Some(wgpu::TextureFormat::P010)
                }
                _ => {
                    log::error!(
                        "[VulkanVideo] Multi-plane VkImage with unsupported format {:?}",
                        vk_format
                    );
                    return Err(Error::UnsupportedBackend);
                }
            }
        } else {
            None
        };

        // Get the wgpu hal device for raw Vulkan access.
        let hal_device = device.as_hal::<wgpu::hal::vulkan::Api>().ok_or_else(|| {
            log::error!("[VulkanVideo] device.as_hal::<vulkan::Api>() returned None");
            Error::UnsupportedBackend
        })?;
        let raw_device = (*hal_device).raw_device();
        let graphics_queue_family = (*hal_device).queue_family_index();

        // Check queue family compatibility. If the FFmpeg image is on a
        // different queue family (not IGNORED), we would need a queue family
        // ownership transfer, which is complex. For now, return an error so
        // the fallback path takes over.
        for i in 0..plane_count {
            let src_qf = vk_frame.queue_family[i];
            if src_qf != ash::vk::QUEUE_FAMILY_IGNORED
                && src_qf != graphics_queue_family
            {
                log::error!(
                    "[VulkanVideo] Source image queue family {} != graphics queue family {} \
                     (plane {}); cross-queue copy not yet supported",
                    src_qf,
                    graphics_queue_family,
                    i
                );
                return Err(Error::UnsupportedBackend);
            }
        }

        // The copy is submitted to the same queue as the render work. Queue
        // order keeps the previous presentation read before this overwrite.

        // Wait for FFmpeg's decode work to finish. AVVkFrame provides timeline
        // semaphores per plane; we must wait on them before copying.
        {
            let mut semaphores: Vec<ash::vk::Semaphore> = Vec::with_capacity(plane_count);
            let mut values: Vec<u64> = Vec::with_capacity(plane_count);
            for i in 0..plane_count {
                let sem = vk_frame.sem[i];
                if sem != ash::vk::Semaphore::null() {
                    let value = vk_frame.sem_value[i];
                    if let Some(pos) = semaphores.iter().position(|s| *s == sem) {
                        values[pos] = values[pos].max(value);
                    } else {
                        semaphores.push(sem);
                        values.push(value);
                    }
                }
            }
            if !semaphores.is_empty() {
                let wait_info = ash::vk::SemaphoreWaitInfo::default()
                    .semaphores(&semaphores)
                    .values(&values);
                if let Err(e) = unsafe { raw_device.wait_semaphores(&wait_info, u64::MAX) } {
                    log::error!("[VulkanVideo] wait_semaphores failed: {:?}", e);
                    return Err(Error::UnsupportedBackend);
                }
            }
        }

        // Create or reuse the persistent presentation texture.
        let need_create = self.presentation.as_ref().map_or(true, |p| {
            p.width != width || p.height != height
        });

        if need_create {
            self.create_presentation(
                device,
                _pipeline_cache,
                width,
                height,
                &texture_format,
                is_multiplane,
                multiplane_format,
                frame.colorspace.into(),
            )?;
        }

        let presentation = self.presentation.as_ref().unwrap();
        let dst_textures = &presentation.textures;

        // Create the raw Vulkan copy context if it doesn't exist yet.
        // We use a dedicated command pool/buffer because wgpu forbids
        // mixing its encoding API (transition_resources) with the raw
        // encoding API (as_hal_mut) on the same command encoder.
        if self.copy_ctx.is_none() {
            let hal_queue = unsafe {
                queue
                    .as_hal::<wgpu::hal::vulkan::Api>()
                    .ok_or_else(|| {
                        log::error!(
                            "[VulkanVideo] queue.as_hal::<vulkan::Api>() returned None"
                        );
                        Error::UnsupportedBackend
                    })?
            };
            let copy_ctx = unsafe {
                VulkanCopyContext::new(
                    raw_device.clone(),
                    (*hal_queue).as_raw(),
                    graphics_queue_family,
                )
                .map_err(|e| {
                    log::error!("[VulkanVideo] failed to create copy context: {:?}", e);
                    Error::UnsupportedBackend
                })?
            };
            self.copy_ctx = Some(copy_ctx);
        }
        let copy_ctx = self.copy_ctx.as_mut().unwrap();

        // Record the GPU-side image copy on our dedicated command buffer.
        // This includes:
        // - Source barrier: from AVVkFrame.layout to TRANSFER_SRC_OPTIMAL
        // - Destination barrier: to TRANSFER_DST_OPTIMAL
        // - cmd_copy_image for each plane
        // - Source restore barrier: back to original layout for FFmpeg pool reuse
        // - Destination barrier: to SHADER_READ_ONLY_OPTIMAL for wgpu sampling
        let copy_result = unsafe {
            copy_ctx.submit_copy(|cmd_buffer| {
                let copy_plane_count = if is_multiplane {
                    expected_planes
                } else {
                    plane_count
                };

                for i in 0..copy_plane_count {
                    let src_index = if is_multiplane { 0 } else { i };
                    let src_image = vk_frame.img[src_index];
                    let src_layout = vk_frame.layout[src_index];
                    let src_access = vk_frame.access[src_index];

                    // Determine the destination VkImage and aspect for this plane.
                    let (dst_image, dst_aspect, src_aspect, copy_width, copy_height) =
                        if is_multiplane {
                            let dst_raw = dst_textures[0]
                                .as_hal::<wgpu::hal::vulkan::Api>()
                                .map(|t| t.raw_handle());
                            let dst_image = match dst_raw {
                                Some(img) => img,
                                None => {
                                    log::error!(
                                        "[VulkanVideo] dest texture as_hal returned None"
                                    );
                                    return;
                                }
                            };

                            match i {
                                0 => (
                                    dst_image,
                                    ash::vk::ImageAspectFlags::PLANE_0,
                                    ash::vk::ImageAspectFlags::PLANE_0,
                                    width,
                                    height,
                                ),
                                1 => (
                                    dst_image,
                                    ash::vk::ImageAspectFlags::PLANE_1,
                                    ash::vk::ImageAspectFlags::PLANE_1,
                                    width / 2,
                                    height / 2,
                                ),
                                _ => {
                                    log::error!(
                                        "[VulkanVideo] Unexpected plane index {} for multi-plane",
                                        i
                                    );
                                    return;
                                }
                            }
                        } else {
                            let dst_raw = dst_textures.get(i).and_then(|tex| {
                                tex.as_hal::<wgpu::hal::vulkan::Api>()
                                    .map(|t| t.raw_handle())
                            });
                            let dst_image = match dst_raw {
                                Some(img) => img,
                                None => {
                                    log::error!(
                                        "[VulkanVideo] dest texture {} as_hal returned None",
                                        i
                                    );
                                    return;
                                }
                            };

                            let (cw, ch) = match &texture_format.planes {
                                layout::PlaneLayout::PackedYUV420(_) => {
                                    if i == 0 {
                                        (width, height)
                                    } else {
                                        (width / 2, height / 2)
                                    }
                                }
                                layout::PlaneLayout::YUV420(_) => {
                                    if i == 0 {
                                        (width, height)
                                    } else {
                                        (width / 2, height / 2)
                                    }
                                }
                                layout::PlaneLayout::YUV444(_) => (width, height),
                                layout::PlaneLayout::RGB(_) => (width, height),
                            };

                            (
                                dst_image,
                                ash::vk::ImageAspectFlags::COLOR,
                                ash::vk::ImageAspectFlags::COLOR,
                                cw,
                                ch,
                            )
                        };

                    // Barrier: transition source to TRANSFER_SRC_OPTIMAL.
                    let src_barrier = ash::vk::ImageMemoryBarrier::default()
                        .image(src_image)
                        .subresource_range(ash::vk::ImageSubresourceRange {
                            aspect_mask: src_aspect,
                            base_mip_level: 0,
                            level_count: ash::vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: ash::vk::REMAINING_ARRAY_LAYERS,
                        })
                        .src_access_mask(src_access)
                        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
                        .old_layout(src_layout)
                        .new_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED);

                    // Barrier: transition destination to TRANSFER_DST_OPTIMAL.
                    // Using UNDEFINED as old_layout is always safe: on the first
                    // frame the texture is freshly created (UNDEFINED), and on
                    // subsequent frames it's in SHADER_READ_ONLY_OPTIMAL from
                    // the previous frame's final barrier. Vulkan allows
                    // transitioning from UNDEFINED to any layout (it discards
                    // old contents, which is fine since we're about to overwrite).
                    let dst_barrier = ash::vk::ImageMemoryBarrier::default()
                        .image(dst_image)
                        .subresource_range(ash::vk::ImageSubresourceRange {
                            aspect_mask: dst_aspect,
                            base_mip_level: 0,
                            level_count: ash::vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: ash::vk::REMAINING_ARRAY_LAYERS,
                        })
                        .src_access_mask(ash::vk::AccessFlags::empty())
                        .dst_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(ash::vk::ImageLayout::UNDEFINED)
                        .new_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED);

                    raw_device.cmd_pipeline_barrier(
                        cmd_buffer,
                        ash::vk::PipelineStageFlags::ALL_COMMANDS,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        std::slice::from_ref(&src_barrier),
                    );
                    raw_device.cmd_pipeline_barrier(
                        cmd_buffer,
                        ash::vk::PipelineStageFlags::ALL_COMMANDS,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        std::slice::from_ref(&dst_barrier),
                    );

                    // Copy the image.
                    let region = ash::vk::ImageCopy::default()
                        .src_subresource(ash::vk::ImageSubresourceLayers {
                            aspect_mask: src_aspect,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .src_offset(ash::vk::Offset3D { x: 0, y: 0, z: 0 })
                        .dst_subresource(ash::vk::ImageSubresourceLayers {
                            aspect_mask: dst_aspect,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .dst_offset(ash::vk::Offset3D { x: 0, y: 0, z: 0 })
                        .extent(ash::vk::Extent3D {
                            width: copy_width,
                            height: copy_height,
                            depth: 1,
                        });

                    raw_device.cmd_copy_image(
                        cmd_buffer,
                        src_image,
                        ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        dst_image,
                        ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(&region),
                    );

                    // Barrier: transition source back to its original layout
                    // so FFmpeg can safely reuse the image from its pool.
                    let restore_src = ash::vk::ImageMemoryBarrier::default()
                        .image(src_image)
                        .subresource_range(ash::vk::ImageSubresourceRange {
                            aspect_mask: src_aspect,
                            base_mip_level: 0,
                            level_count: ash::vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: ash::vk::REMAINING_ARRAY_LAYERS,
                        })
                        .src_access_mask(ash::vk::AccessFlags::TRANSFER_READ)
                        .dst_access_mask(src_access)
                        .old_layout(ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .new_layout(src_layout)
                        .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED);

                    // Barrier: transition destination to SHADER_READ_ONLY_OPTIMAL
                    // so wgpu can sample it.
                    let restore_dst = ash::vk::ImageMemoryBarrier::default()
                        .image(dst_image)
                        .subresource_range(ash::vk::ImageSubresourceRange {
                            aspect_mask: dst_aspect,
                            base_mip_level: 0,
                            level_count: ash::vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: ash::vk::REMAINING_ARRAY_LAYERS,
                        })
                        .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ)
                        .old_layout(ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED);

                    raw_device.cmd_pipeline_barrier(
                        cmd_buffer,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::PipelineStageFlags::ALL_COMMANDS,
                        ash::vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        std::slice::from_ref(&restore_src),
                    );
                    raw_device.cmd_pipeline_barrier(
                        cmd_buffer,
                        ash::vk::PipelineStageFlags::TRANSFER,
                        ash::vk::PipelineStageFlags::ALL_COMMANDS,
                        ash::vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        std::slice::from_ref(&restore_dst),
                    );
                }
            })
        };

        if let Err(e) = copy_result {
            log::error!("[VulkanVideo] raw Vulkan copy failed: {:?}", e);
            return Err(Error::UnsupportedBackend);
        }

        // We do NOT call transition_resources here. The destination textures
        // were created via create_texture_from_hal (init: false), so wgpu's
        // init tracker considers them already initialized — no discarding
        // clear will be issued. wgpu's usage tracker starts at UNINITIALIZED
        // and will transition to RESOURCE on first bind group use, issuing a
        // barrier with oldLayout=UNDEFINED. Since the actual Vulkan layout is
        // already SHADER_READ_ONLY_OPTIMAL (from our raw copy's final
        // barrier), this is the same pattern used by the D3D11VA path and
        // works correctly in practice. On subsequent frames, wgpu's tracker
        // will see RESOURCE→RESOURCE (no barrier), which is correct since
        // our raw copy always leaves the texture in SHADER_READ_ONLY_OPTIMAL.

        Ok(None)
    }

    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.zero_copy_current
            .as_ref()
            .map(|p| p.identity)
            .or_else(|| self.presentation.as_ref().map(|p| p.identity))
    }

    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.zero_copy_current
            .as_ref()
            .map(|p| &p.bg0)
            .or_else(|| self.presentation.as_ref().map(|p| &p.bg0))
    }

    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        // Zero-copy path: expose the current frame's external pool-image
        // textures (the engine samples them directly). The presentation
        // textures are only populated on the GPU-copy fallback path.
        if let Some(zc) = self.zero_copy_current.as_ref() {
            return Some(zc.plane_views());
        }

        let presentation = self.presentation.as_ref()?;
        // For a multi-plane Vulkan image (NV12/P010) the destination is a single
        // texture holding both planes; expose the per-plane aspect views so the
        // engine's universal YUV shader can sample Y (Plane0) and UV (Plane1)
        // separately. A single default view of the whole image would be len 1
        // and break the YUV bind group (which needs >= 2 plane views).
        if presentation.is_multiplane {
            let texture = &presentation.textures[0];
            match &presentation.identity.planes {
                layout::PlaneLayout::PackedYUV420(_) => {
                    let y_view = texture.create_view(&wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::Plane0,
                        ..Default::default()
                    });
                    let uv_view = texture.create_view(&wgpu::TextureViewDescriptor {
                        aspect: wgpu::TextureAspect::Plane1,
                        ..Default::default()
                    });
                    Some(vec![y_view, uv_view])
                }
                _ => {
                    log::error!(
                        "[VulkanVideo] plane_views: multi-plane image with unexpected layout {:?}",
                        presentation.identity.planes
                    );
                    None
                }
            }
        } else {
            let views: Vec<wgpu::TextureView> = presentation
                .textures
                .iter()
                .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
                .collect();
            Some(views)
        }
    }

    fn release_completed_frames(&mut self, device: &wgpu::Device) -> Result<()> {
        if self.retired_zero_copy.is_empty() {
            return Ok(());
        }
        let hal_device = unsafe {
            device
                .as_hal::<wgpu::hal::vulkan::Api>()
                .ok_or(Error::UnsupportedBackend)?
        };
        let raw_device = (*hal_device).raw_device();
        for frame in self.retired_zero_copy.drain(..) {
            unsafe { Self::release_zero_copy_frame(&frame._held_frame, &raw_device) };
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        if self.zero_copy_current.is_some() {
            "Vulkan Video (zero-copy)"
        } else {
            "Vulkan Video (GPU copy)"
        }
    }
}

impl Drop for VulkanVideoFrameAdapter {
    fn drop(&mut self) {
        // Explicit cleanup ordering: release frame-referencing state
        // (ZeroCopyImportedFrame holds a HeldAvFrame that frees an AVFrame,
        // plus wgpu textures created from raw Vulkan images) BEFORE the
        // presentation textures and the raw Vulkan copy context.
        // VulkanCopyContext's Drop destroys its command pool, so any state
        // recorded against it must be released first.
        self.retired_zero_copy.clear();
        self.zero_copy_current = None;
        self.presentation = None;
        self.copy_ctx = None;
    }
}
