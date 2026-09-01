mod software;

#[cfg(target_os = "windows")]
mod d3d11va;
#[cfg(target_os = "linux")]
mod vaapi;
#[cfg(target_os = "macos")]
mod video_toolbox;

mod vulkan_video;

// OpenGL zero-copy path (Linux VA-API/DRM PRIME/EGL, Windows D3D11VA).
// Compiled on every platform that can build the `gles` wgpu backend so the
// code path exists; per-format selection happens at runtime via the adapter.
mod opengl;

use crate::{
    Error, VideoMetadata,
    context::{layout, pipeline_cache::PipelineCache},
    error::Result,
};
use ffmpeg_next::{self as ffn, sys as ff};
use std::{
    ptr::NonNull,
    sync::{Arc, Mutex},
};

/// Opaque submission ticket identifying a GL-interop slot that has been
/// locked (ownership transferred to OpenGL) and is awaiting GPU consumption.
///
/// The engine collects these while recording the video draw pass and hands
/// them back via `Video::finish_gl_frames` immediately after `queue.submit`,
/// so the interop owner can insert a GL fence and defer the WGL unlock until
/// the GPU has actually finished sampling the textures.
///
/// `generation` is bumped whenever the interop device is (re)created, so a
/// stale ticket from a torn-down ring is never mistaken for a live slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlInteropTicket {
    pub generation: u32,
    pub slot_id: u8,
}



// needs to be separate from FrameAdapater to be dyn compatible
pub(crate) trait FrameAdapterBuilder: FrameAdapter + Sized {
    unsafe fn new(decoder: NonNull<ff::AVCodecContext>) -> Result<Self>;
    fn supports_format(format: ff::AVPixelFormat) -> bool;
}

pub(crate) trait FrameAdapter {
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<Option<GlInteropTicket>>;
    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>>;
    fn bind_group(&self) -> Option<&wgpu::BindGroup>;
    fn name(&self) -> &'static str;
    /// Per-plane `wgpu::TextureView`s (Y, then UV/Planes) for direct engine
    /// sampling. Each view is already formatted for its plane (R8/R16 for Y,
    /// Rg8/Rg16 for interleaved UV, R8 for planar). Returns `None` until the
    /// first frame has been imported.
    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>>;
    /// Insert a GL fence for every video slot sampled by the just-submitted
    /// command buffer and defer their WGL unlock until the fence signals.
    /// No-op for adapters that do not use a GL-interop ring (default).
    fn finish_gl_frames(&mut self, _tickets: &[GlInteropTicket]) -> Result<()> {
        Ok(())
    }
    /// Release frame references whose queue submission has completed.
    /// No-op for adapters that do not retain externally-owned frames.
    fn release_completed_frames(&mut self, _device: &wgpu::Device) -> Result<()> {
        Ok(())
    }
    /// Abandon a locked slot whose draw was never recorded (render error,
    /// early return, clipping). Unlocks the WGL object immediately so the slot
    /// returns to the Free state. No-op for non-ring adapters (default).
    fn cancel_gl_frame(&mut self, _ticket: GlInteropTicket) -> Result<()> {
        Ok(())
    }
}

pub(crate) struct FrameDecoder {
    pub(crate) adapter: Option<Box<dyn FrameAdapter>>,
    pipeline_cache: Arc<Mutex<PipelineCache>>,
    last_pixel_format: ff::AVPixelFormat,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    /// When `true` (default) each decoded frame is rendered through the
    /// `copy_to_rgb` pass into the RGBA8 `texture` (the legacy path the engine
    /// has always consumed via `Video::texture()`). When `false`, the
    /// `copy_to_rgb` pass is skipped and the engine samples the YUV planes
    /// directly (see `yuv_bind_group` / `layout_identity`), eliminating one
    /// full-frame RGBA8 copy + render pass.
    pub(crate) copy_to_rgb_enabled: bool,
    /// Color space detected from the stream metadata (BT.601 / BT.709 / etc.).
    /// Used by the engine's YUV shader to select the correct conversion matrix.
    color_space: ffn::color::Space,
    /// Color range detected from the stream metadata (MPEG/limited vs
    /// JPEG/full). Used by the engine's YUV shader to pick the right expansion.
    color_range: ffn::color::Range,
    /// GL-interop submission tickets produced by the Windows GL adapter's
    /// `import_frame` since the last drain. The engine pulls these during the
    /// draw pass and returns them via `finish_gl_frames` after `queue.submit`.
    pending_gl_tickets: Vec<GlInteropTicket>,
}

fn allow_cpu_fallback(adapter: &wgpu::Adapter, frame_format: i32) -> bool {
    let is_windows_gl_d3d11 = cfg!(target_os = "windows")
        && adapter.get_info().backend == wgpu::Backend::Gl
        && frame_format == ff::AVPixelFormat::AV_PIX_FMT_D3D11 as i32;
    if !is_windows_gl_d3d11 {
        return true;
    }

    let explicit_cpu_fallback = std::env::var("FFGPU_GL_ALLOW_CPU_FALLBACK")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let explicit_cpu_mode = std::env::var("FFGPU_GL_INTEROP")
        .map(|value| value.eq_ignore_ascii_case("cpu"))
        .unwrap_or(false);
    explicit_cpu_fallback || explicit_cpu_mode
}

impl FrameDecoder {
    pub fn new(
        device: &wgpu::Device,
        pipeline_cache: Arc<Mutex<PipelineCache>>,
        metadata: &VideoMetadata,
        supports_view_formats: bool,
    ) -> Result<Self> {
        Self::new_with_options(
            device,
            pipeline_cache,
            metadata,
            supports_view_formats,
            true,
        )
    }

    /// Like `new`, but lets the caller disable the `copy_to_rgb` RGBA8 pass.
    /// Disabling it keeps `texture` allocated (cheap, and still valid as a
    /// fallback) but skips the per-frame render pass when the engine samples
    /// YUV planes directly.
    pub fn new_with_options(
        device: &wgpu::Device,
        pipeline_cache: Arc<Mutex<PipelineCache>>,
        metadata: &VideoMetadata,
        supports_view_formats: bool,
        copy_to_rgb_enabled: bool,
    ) -> Result<Self> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: metadata.width,
                height: metadata.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            // Rgba8UnormSrgb as a view format requires DownlevelFlags::VIEW_FORMATS,
            // which backends like OpenGL/ANGLE lack. Only request it when supported.
            view_formats: if supports_view_formats {
                &[wgpu::TextureFormat::Rgba8UnormSrgb]
            } else {
                &[]
            },
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(FrameDecoder {
            adapter: None,
            pipeline_cache,
            last_pixel_format: ff::AVPixelFormat::AV_PIX_FMT_NONE,
            texture,
            texture_view,
            copy_to_rgb_enabled,
            color_space: metadata.color_space,
            color_range: metadata.color_range,
            pending_gl_tickets: Vec::new(),
        })
    }

    /// Enable/disable the `copy_to_rgb` RGBA8 pass. When disabled, the engine
    /// is expected to sample the YUV planes directly via `yuv_bind_group`.
    pub fn set_copy_to_rgb(&mut self, enabled: bool) {
        self.copy_to_rgb_enabled = enabled;
    }

    /// Returns the color space detected from the stream metadata.
    pub fn color_space(&self) -> ffn::color::Space {
        self.color_space
    }

    /// Returns the color range detected from the stream metadata.
    pub fn color_range(&self) -> ffn::color::Range {
        self.color_range
    }

    pub unsafe fn decode_frame(
        &mut self,
        instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        decoder: NonNull<ff::AVCodecContext>,
        frame: &ffn::Frame,
    ) -> Result<()> {
        let format =
            unsafe { std::mem::transmute::<_, ff::AVPixelFormat>((*frame.as_ptr()).format) };

        if format != self.last_pixel_format {
            self.last_pixel_format = format;
            self.adapter = None;
        }

        let frame_adapter = if let Some(frame_adapter) = self.adapter.as_mut() {
            frame_adapter
        } else {
            unsafe {
                // Dispatch by (wgpu backend, FFmpeg pixel format) so we never
                // steal frames meant for a different adapter (e.g. a D3D11VA
                // frame on a Dx12/Vulkan device must not be claimed by the GL
                // path, and vice versa).
                let backend = adapter.get_info().backend;
                let decoder = match (backend, format) {
                    #[cfg(target_os = "windows")]
                    (wgpu::Backend::Gl, ff::AVPixelFormat::AV_PIX_FMT_D3D11) => {
                        Box::new(opengl::OpenGlWindowsFrameAdapter::new(decoder)?) as _
                    }
                    #[cfg(target_os = "linux")]
                    (wgpu::Backend::Gl, ff::AVPixelFormat::AV_PIX_FMT_VAAPI) => {
                        Box::new(opengl::OpenGlLinuxFrameAdapter::new(decoder)?) as _
                    }
                    #[cfg(target_os = "windows")]
                    (_, ff::AVPixelFormat::AV_PIX_FMT_D3D11) => {
                        Box::new(d3d11va::D3D11VAFrameAdapter::new(decoder)?) as _
                    }
                    #[cfg(target_os = "linux")]
                    (_, ff::AVPixelFormat::AV_PIX_FMT_VAAPI) => {
                        return Err(Error::UnsupportedBackend);
                    }
                    (_, ff::AVPixelFormat::AV_PIX_FMT_VULKAN) => {
                        Box::new(vulkan_video::VulkanVideoFrameAdapter::new(decoder)?) as _
                    }
                    (_, format) if software::SoftwareFrameAdapter::supports_format(format) => {
                        log::warn!("using CPU frame copies");
                        Box::new(software::SoftwareFrameAdapter::new(decoder)?) as _
                    }
                    _ => return Err(Error::UnsupportedPixelFormat),
                };
                self.adapter.insert(decoder)
            }
        };

        let mut pipeline_cache = self.pipeline_cache.lock().unwrap();

        unsafe {
            let res = frame_adapter.import_frame(
                NonNull::new_unchecked(frame.as_ptr() as *mut _),
                instance,
                adapter,
                device,
                queue,
                encoder,
                &mut *pipeline_cache,
            );
            match res {
                Err(error @ Error::UnsupportedBackend)
                | Err(error @ Error::TextureShare)
                | Err(error @ Error::Probe(_)) => {
                    let frame_fmt = (*frame.as_ptr()).format;
                    if frame_fmt == ff::AVPixelFormat::AV_PIX_FMT_VULKAN as i32 {
                        eprintln!("[frames] Vulkan zero-copy failed and CPU transfer is not supported");
                        return Err(Error::UnsupportedBackend);
                    }
                    if !allow_cpu_fallback(adapter, frame_fmt) {
                        eprintln!(
                            "[frames] strict GL interop rejected CPU fallback: {:?}; set FFGPU_GL_ALLOW_CPU_FALLBACK=1 to opt in",
                            error
                        );
                        return Err(error);
                    }

                    eprintln!(
                        "[frames] zero-copy frame import failed: {:?}, falling back to CPU frame copies",
                        error
                    );
                    let mut sw_adapter = Box::new(software::SoftwareFrameAdapter::new(decoder)?);
                    sw_adapter.import_frame(
                        NonNull::new_unchecked(frame.as_ptr() as *mut _),
                        instance,
                        adapter,
                        device,
                        queue,
                        encoder,
                        &mut *pipeline_cache,
                    )?;
                    self.adapter = Some(sw_adapter);
                }
                Err(error) => return Err(error),
                Ok(maybe_ticket) => {
                    // A GL-interop ring adapter returns a ticket identifying
                    // the slot it locked; the engine finishes it after submit.
                    if let Some(ticket) = maybe_ticket {
                        self.pending_gl_tickets.push(ticket);
                    }
                }
            }
        };

        drop(pipeline_cache);

        // Only render into the RGBA8 texture when the engine consumes it. When
        // the engine samples YUV planes directly this pass is skipped, saving a
        // full-frame copy + render pass on every frame.
        if self.copy_to_rgb_enabled {
            self.copy_to_rgb(encoder, unsafe { (*frame.as_ptr()).colorspace.into() });
        }

        Ok(())
    }

    pub fn copy_to_rgb(&self, encoder: &mut wgpu::CommandEncoder, color_space: ffn::color::Space) {
        let Some(bg0) = self
            .adapter
            .as_ref()
            .and_then(|adapter| adapter.bind_group())
        else {
            return;
        };

        let Some(layout_identity) = self
            .adapter
            .as_ref()
            .and_then(|adapter| adapter.layout_identity())
        else {
            return;
        };

        let pipeline = self
            .pipeline_cache
            .lock()
            .unwrap()
            .get(layout_identity, color_space)
            .clone();

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, bg0, &[]);
        rpass.draw(0..3, 0..1);
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Whether at least one decoded frame has reached the active adapter.
    /// The RGBA fallback texture is uninitialized until that happens.
    pub fn has_frame(&self) -> bool {
        self.adapter.as_ref().and_then(|adapter| adapter.bind_group()).is_some()
    }

    /// YUV plane bind group (`bg0`) produced by the active frame adapter.
    /// Returns `None` until the first frame has been imported (or if the
    /// adapter failed to build one). When `Some`, the engine may sample the
    /// YUV planes directly instead of going through `copy_to_rgb` + `texture`.
    pub fn yuv_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.adapter.as_ref().and_then(|a| a.bind_group())
    }

    /// Layout identity describing the YUV plane formats/layout (NV12, planar
    /// I420, YUV444, bit depth, etc.). Use this to select the correct YUV→RGB
    /// conversion on the engine side.
    pub fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.adapter.as_ref().and_then(|a| a.layout_identity())
    }

    /// Per-plane texture views for direct engine sampling (Y, then UV/planes).
    /// Returns `None` until the first frame has been imported.
    pub fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.adapter.as_ref().and_then(|a| a.plane_views())
    }

    /// Release external frame references after the queue submission that sampled them.
    pub fn release_completed_frames(&mut self, device: &wgpu::Device) -> Result<()> {
        if let Some(adapter) = self.adapter.as_mut() {
            adapter.release_completed_frames(device)?;
        }
        Ok(())
    }

    /// Whether the engine should sample YUV planes directly (no `copy_to_rgb`).
    /// Only true for multi-plane (YUV) frames; a single-plane RGBA8 frame must
    /// keep `copy_to_rgb` enabled so the engine's RGBA8 texture is populated.
    pub fn direct_yuv(&self) -> bool {
        !self.copy_to_rgb_enabled
            && self.yuv_bind_group().is_some()
            && matches!(
                self.layout_identity().map(|l| l.planes),
                Some(layout::PlaneLayout::PackedYUV420(_))
                    | Some(layout::PlaneLayout::YUV420(_))
                    | Some(layout::PlaneLayout::YUV444(_))
            )
    }

    /// Drain any GL-interop submission tickets accumulated since the last call.
    /// The engine pulls these during the draw pass and returns them via
    /// `finish_gl_frames` after `queue.submit`.
    pub fn take_pending_gl_tickets(&mut self) -> Vec<GlInteropTicket> {
        std::mem::take(&mut self.pending_gl_tickets)
    }

    /// Insert a GL fence for every ticket's slot and defer the WGL unlock until
    /// the fence signals. Delegates to the active adapter (no-op for
    /// non-ring adapters).
    pub fn finish_gl_frames(&mut self, tickets: &[GlInteropTicket]) -> Result<()> {
        if let Some(a) = self.adapter.as_mut() {
            a.finish_gl_frames(tickets)?;
        }
        Ok(())
    }

    /// Abandon a locked slot whose draw was never recorded. Delegates to the
    /// active adapter (no-op for non-ring adapters).
    pub fn cancel_gl_frame(&mut self, ticket: GlInteropTicket) -> Result<()> {
        if let Some(a) = self.adapter.as_mut() {
            a.cancel_gl_frame(ticket)?;
        }
        Ok(())
    }
}
