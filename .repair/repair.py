from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return (ROOT / path).read_text()


def save(path, text):
    (ROOT / path).write_text(text)


def replace_once(path, old, new):
    text = load(path)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one literal match, found {count}")
    save(path, text.replace(old, new, 1))


def regex_once(path, pattern, replacement, flags=re.S):
    text = load(path)
    text2, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one regex match for {pattern!r}, found {count}")
    save(path, text2)


# ---------------------------------------------------------------------------
# 1. Vulkan device creation: never place FFmpeg and wgpu on one VkQueue.
# ---------------------------------------------------------------------------
replace_once(
    "src/vulkan_device.rs",
    '''    let video_queue_family_index = dedicated_video_qfi
        .or_else(|| {
            queue_families
                .iter()
                .enumerate()
                .find(|(_, props)| {
                    props
                        .queue_flags
                        .contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR)
                })
                .map(|(i, _)| i as u32)
        })
        .ok_or_else(|| {
            log::error!("[VulkanVideo] No queue family with VIDEO_DECODE_KHR found");
            Error::HardwareContext
        })?;

    log::info!(
        "[VulkanVideo] Selected video decode queue family: {}{}",
        video_queue_family_index,
        if dedicated_video_qfi.is_some() {
            " (dedicated)"
        } else {
            " (combined fallback)"
        }
    );
''',
    '''    let video_queue_family_index = dedicated_video_qfi.ok_or_else(|| {
        log::error!(
            "[VulkanVideo] Refusing combined graphics/video queue: no dedicated VIDEO_DECODE_KHR family"
        );
        Error::HardwareContext
    })?;

    log::info!(
        "[VulkanVideo] Selected dedicated video decode queue family: {}",
        video_queue_family_index,
    );
''',
)

# ---------------------------------------------------------------------------
# 2. Decode routing: a Vulkan AVFrame is valid only when FFmpeg shares wgpu's
#    VkDevice. D3D11VA->DX12/Vulkan is disabled until explicit cross-API fence
#    synchronization exists.
# ---------------------------------------------------------------------------
replace_once(
    "src/video.rs",
    '''#[cfg(target_os = "windows")]
fn preferred_device_type_for_backend(backend: wgpu::Backend) -> ff::AVHWDeviceType {
    match backend {
        wgpu::Backend::Vulkan => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        wgpu::Backend::Dx12 => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        _ => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
    }
}
''',
    '''#[cfg(target_os = "windows")]
fn preferred_device_type_for_backend(backend: wgpu::Backend) -> ff::AVHWDeviceType {
    match backend {
        wgpu::Backend::Vulkan => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        // Opening a D3D11 shared allocation from D3D12 is not execution
        // synchronization. Keep this path off until a shared D3D11/D3D12
        // fence is wired into the submission lifecycle.
        wgpu::Backend::Dx12 => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE,
        wgpu::Backend::Gl => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        _ => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE,
    }
}
''',
)
replace_once(
    "src/video.rs",
    '''        let device_type = if force_software {
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
        } else {
            preferred_device_type_for_backend(backend)
        };
''',
    '''        let device_type = if force_software {
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
        } else if backend == wgpu::Backend::Vulkan && hw_device_ctx.is_none() {
            // A separately-created FFmpeg Vulkan device produces device-scoped
            // VkImage/VkSemaphore handles that are invalid on wgpu's VkDevice.
            // Vulkan hardware decode is therefore allowed only with the shared
            // hwdevice created by Context::new_with_vulkan_queue_family.
            log::warn!(
                "[ffgpu] Vulkan backend has no shared FFmpeg hwdevice; using software decode"
            );
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
        } else {
            preferred_device_type_for_backend(backend)
        };
''',
)
replace_once(
    "src/decode/frames.rs",
    '''                    #[cfg(target_os = "windows")]
                    (_, ff::AVPixelFormat::AV_PIX_FMT_D3D11) => {
                        Box::new(d3d11va::D3D11VAFrameAdapter::new(decoder)?) as _
                    }
''',
    '''                    #[cfg(target_os = "windows")]
                    (_, ff::AVPixelFormat::AV_PIX_FMT_D3D11) => {
                        log::error!(
                            "[frames] D3D11VA cross-API import is disabled until explicit consumer synchronization is implemented"
                        );
                        return Err(Error::UnsupportedBackend);
                    }
''',
)

# ---------------------------------------------------------------------------
# 3. FFmpeg Vulkan hwcontext: real queue capabilities, dedicated decode queue,
#    FFmpeg's own queue mutexes, and failure cleanup.
# ---------------------------------------------------------------------------
replace_once(
    "src/decode/vulkan_hwcontext.rs",
    '''pub unsafe fn create_ffmpeg_vulkan_device_context(
    handles: &VulkanDeviceHandles,
    instance_extensions: &[&'static CStr],
    device_extensions: &[&'static CStr],
) -> Result<NonNull<ff::AVBufferRef>> {
    let ctx_ref: *mut ff::AVBufferRef =
''',
    '''pub unsafe fn create_ffmpeg_vulkan_device_context(
    handles: &VulkanDeviceHandles,
    instance_extensions: &[&'static CStr],
    device_extensions: &[&'static CStr],
) -> Result<NonNull<ff::AVBufferRef>> {
    if handles.video_queue_family_index == handles.queue_family_index {
        log::error!(
            "[VulkanHW] refusing shared graphics/video VkQueue family {}",
            handles.queue_family_index
        );
        return Err(Error::HardwareContext);
    }

    let queue_families = unsafe {
        handles
            .instance
            .get_physical_device_queue_family_properties(handles.physical_device)
    };
    let graphics_flags = queue_families
        .get(handles.queue_family_index as usize)
        .ok_or(Error::HardwareContext)?
        .queue_flags;
    let decode_flags = queue_families
        .get(handles.video_queue_family_index as usize)
        .ok_or(Error::HardwareContext)?
        .queue_flags;
    if !graphics_flags.contains(ash::vk::QueueFlags::GRAPHICS)
        || !decode_flags.contains(ash::vk::QueueFlags::VIDEO_DECODE_KHR)
    {
        log::error!(
            "[VulkanHW] queue family capability mismatch: graphics={:?}, decode={:?}",
            graphics_flags,
            decode_flags
        );
        return Err(Error::HardwareContext);
    }

    let ctx_ref: *mut ff::AVBufferRef =
''',
)
replace_once(
    "src/decode/vulkan_hwcontext.rs",
    '''    if vulkan_ctx.is_null() {
        log::error!("[VulkanHW] AVHWDeviceContext.hwctx is null");
        return Err(Error::HardwareContext);
    }
''',
    '''    if vulkan_ctx.is_null() {
        log::error!("[VulkanHW] AVHWDeviceContext.hwctx is null");
        let mut raw = ctx_ref.as_ptr();
        unsafe { ff::av_buffer_unref(&mut raw) };
        return Err(Error::HardwareContext);
    }
''',
)
regex_once(
    "src/decode/vulkan_hwcontext.rs",
    r'''    // Queue families: graphics queue is always required\..*?    // Set deprecated queue family fields for backwards compatibility\.''',
    '''    // Report the real Vulkan queue-family capabilities. FFmpeg uses these
    // flags to select compute/transfer/video execution queues; inventing flags
    // can make it submit unsupported work to a family.
    let graphics_qf = AVVulkanDeviceQueueFamily {
        idx: handles.queue_family_index as c_int,
        num: 1,
        flags: graphics_flags,
        video_caps: ash::vk::VideoCodecOperationFlagsKHR::empty(),
    };
    let decode_qf = AVVulkanDeviceQueueFamily {
        idx: handles.video_queue_family_index as c_int,
        num: 1,
        flags: decode_flags,
        video_caps: ash::vk::VideoCodecOperationFlagsKHR::empty(),
    };
    unsafe {
        (*vulkan_ctx).qf[0] = graphics_qf;
        (*vulkan_ctx).qf[1] = decode_qf;
        (*vulkan_ctx).nb_qf = 2;
    }

    // Set deprecated queue family fields for backwards compatibility.''',
)
regex_once(
    "src/decode/vulkan_hwcontext.rs",
    r'''    let gqf = handles\.queue_family_index as c_int;.*?    // Provide no-op lock_queue/unlock_queue callbacks\..*?    unsafe \{ \(\*vulkan_ctx\)\.unlock_queue = Some\(noop_lock_queue\) \};''',
    '''    let gqf = handles.queue_family_index as c_int;
    let dqf = handles.video_queue_family_index as c_int;
    let graphics_can_transfer = graphics_flags.intersects(
        ash::vk::QueueFlags::GRAPHICS
            | ash::vk::QueueFlags::COMPUTE
            | ash::vk::QueueFlags::TRANSFER,
    );
    unsafe {
        (*vulkan_ctx).queue_family_index = gqf;
        (*vulkan_ctx).nb_graphics_queues = 1;
        (*vulkan_ctx).queue_family_tx_index = if graphics_can_transfer { gqf } else { -1 };
        (*vulkan_ctx).nb_tx_queues = if graphics_can_transfer { 1 } else { 0 };
        (*vulkan_ctx).queue_family_comp_index =
            if graphics_flags.contains(ash::vk::QueueFlags::COMPUTE) {
                gqf
            } else {
                -1
            };
        (*vulkan_ctx).nb_comp_queues =
            if graphics_flags.contains(ash::vk::QueueFlags::COMPUTE) {
                1
            } else {
                0
            };
        (*vulkan_ctx).queue_family_encode_index = -1;
        (*vulkan_ctx).nb_encode_queues = 0;
        (*vulkan_ctx).queue_family_decode_index = dqf;
        (*vulkan_ctx).nb_decode_queues = 1;

        // Leave the callbacks NULL. FFmpeg installs its qf_mutex-backed default
        // implementation during av_hwdevice_ctx_init. This is required because
        // one ffgpu Context can feed multiple decoder threads concurrently.
        (*vulkan_ctx).lock_queue = None;
        (*vulkan_ctx).unlock_queue = None;
    }''',
)
replace_once(
    "src/decode/vulkan_hwcontext.rs",
    '''    if init_ret != 0 {
        log::error!("[VulkanHW] av_hwdevice_ctx_init failed: {}", init_ret);
        return Err(Error::HardwareContext);
    }
''',
    '''    if init_ret != 0 {
        log::error!("[VulkanHW] av_hwdevice_ctx_init failed: {}", init_ret);
        let mut raw = ctx_ref.as_ptr();
        unsafe { ff::av_buffer_unref(&mut raw) };
        return Err(Error::HardwareContext);
    }
''',
)

# ---------------------------------------------------------------------------
# 4. Audio PTS: keep timestamps in the stream time base across resampling.
# ---------------------------------------------------------------------------
replace_once(
    "src/decode/audio.rs",
    '''                        let resampled_pts = unsafe {
                            ff::swr_next_pts(
                                resampler.resampler.as_mut_ptr(),
                                (*frame.as_ptr()).pts * resampler.parameters.sample_rate as i64,
                            )
                        };

                        let resampled_pts =
                            resampled_pts / self.decoder.metadata.sample_rate as i64;
''',
    '''                        let input_pts = unsafe { (*frame.as_ptr()).pts };
                        let resampled_pts = if input_pts == ff::AV_NOPTS_VALUE {
                            None
                        } else {
                            let input_rate = self.decoder.metadata.sample_rate as i64;
                            let output_rate = resampler.parameters.sample_rate as i64;
                            let stream_tb: ff::AVRational = self.decoder.metadata.time_base.into();
                            let input_sample_tb = ff::AVRational {
                                num: 1,
                                den: input_rate as i32,
                            };
                            let output_sample_tb = ff::AVRational {
                                num: 1,
                                den: output_rate as i32,
                            };

                            // swr_next_pts uses units of 1/(in_rate*out_rate).
                            // Convert stream PTS -> input-sample units -> cross-rate
                            // units, then convert the compensated result back into
                            // the original stream time base. AudioSink therefore sees
                            // the same timestamp domain before and after resampling.
                            let input_sample_pts = unsafe {
                                ff::av_rescale_q(input_pts, stream_tb, input_sample_tb)
                            };
                            let swr_pts = input_sample_pts.saturating_mul(output_rate);
                            let compensated = unsafe {
                                ff::swr_next_pts(resampler.resampler.as_mut_ptr(), swr_pts)
                            };
                            let output_sample_pts = unsafe { ff::av_rescale(compensated, 1, input_rate) };
                            Some(unsafe {
                                ff::av_rescale_q(output_sample_pts, output_sample_tb, stream_tb)
                            })
                        };
''',
)
replace_once(
    "src/decode/audio.rs",
    '''                                audio_frame.set_pts(Some(resampled_pts));
''',
    '''                                audio_frame.set_pts(resampled_pts);
''',
)

# ---------------------------------------------------------------------------
# 5. Demux backpressure: len() is packet count, not bytes.
# ---------------------------------------------------------------------------
replace_once(
    "src/decode/read.rs",
    '''        const MAX_QUEUE_SIZE: usize = 15 * 1024 * 1024;
''',
    '''        // Absolute malformed/sparse-stream guard. Crossbeam's len() is a
        // packet count, not a byte count; the old "15 MiB" constant therefore
        // allowed ~15.7 million queued packets.
        const MAX_QUEUE_PACKETS: usize = 4096;
''',
)
replace_once(
    "src/decode/read.rs",
    '''            if (video_stream.packets.tx.len() + audio_stream.packets.tx.len()/*+ self.subtitle_tx.packets.len()*/)
                > MAX_QUEUE_SIZE
''',
    '''            if (video_stream.packets.tx.len() + audio_stream.packets.tx.len()/*+ self.subtitle_tx.packets.len()*/)
                > MAX_QUEUE_PACKETS
''',
)

# ---------------------------------------------------------------------------
# 6. Software pixel formats and YCbCr math.
# ---------------------------------------------------------------------------
replace_once(
    "src/software_video.rs",
    '''    /// 1-plane packed RGB24 (no alpha).
    Rgb24,
    /// 1-plane packed RGBA32.
    Rgba,
    /// 1-plane packed YUV422 (UYVY/VYUY/YUYV/YVYU). Decoder outputs
    /// this for some capture devices. `to_rgba()` returns a converted
    /// RGBA8 buffer (no resize — the consumer should resize via the
    /// engine's fused YUV→RGB bilinear helper if needed).
    Yuv422,
''',
    '''    /// 1-plane packed RGB24 (no alpha).
    Rgb24,
    /// 1-plane packed BGR24 (no alpha).
    Bgr24,
    /// 1-plane packed RGBA32.
    Rgba,
    /// Packed YUYV422 (`Y0 U Y1 V`). Kept as `Yuv422` for API compatibility.
    Yuv422,
    /// Packed UYVY422 (`U Y0 V Y1`).
    Uyvy422,
    /// Packed YVYU422 (`Y0 V Y1 U`).
    Yvyu422,
''',
)
replace_once(
    "src/software_video.rs",
    '''            // BGR24 is byte-identical to RGB24 for our purposes (we just
            // need to copy pixels; the renderer doesn't care about channel
            // order for 24-bit formats when wrapped in DynamicImageBuffer::RGB).
            x if x == ff::AVPixelFormat::AV_PIX_FMT_BGR24 => Some(Self::Rgb24),
''',
    '''            x if x == ff::AVPixelFormat::AV_PIX_FMT_BGR24 => Some(Self::Bgr24),
''',
)
regex_once(
    "src/software_video.rs",
    r'''            // YUV422 packed variants.*?            \{\n                Some\(Self::Yuv422\)\n            \}\n''',
    '''            // Preserve the packed byte order; the converter must not
            // reinterpret U/V or luma positions.
            x if x == ff::AVPixelFormat::AV_PIX_FMT_YUYV422 => Some(Self::Yuv422),
            x if x == ff::AVPixelFormat::AV_PIX_FMT_UYVY422 => Some(Self::Uyvy422),
            x if x == ff::AVPixelFormat::AV_PIX_FMT_YVYU422 => Some(Self::Yvyu422),
''',
)
replace_once(
    "src/software_video.rs",
    '''/// A single decoded video frame, ready for CPU consumption.
''',
    '''fn ycbcr_coefficients(matrix: ColorMatrix) -> (f32, f32, f32) {
    match matrix {
        ColorMatrix::Bt601 => (0.2990, 0.5870, 0.1140),
        ColorMatrix::Bt2020 => (0.2627, 0.6780, 0.0593),
        ColorMatrix::Bt709 | ColorMatrix::Identity => (0.2126, 0.7152, 0.0722),
    }
}

fn ycbcr_to_rgb8(
    y: u8,
    cb: u8,
    cr: u8,
    range: YuvRange,
    matrix: ColorMatrix,
) -> [u8; 3] {
    let (kr, kg, kb) = ycbcr_coefficients(matrix);
    let (luma, cb, cr) = match range {
        YuvRange::Limited => (
            (y as f32 - 16.0) / 219.0,
            (cb as f32 - 128.0) / 224.0,
            (cr as f32 - 128.0) / 224.0,
        ),
        YuvRange::Full => (
            y as f32 / 255.0,
            (cb as f32 - 128.0) / 255.0,
            (cr as f32 - 128.0) / 255.0,
        ),
    };

    let r = luma + 2.0 * (1.0 - kr) * cr;
    let b = luma + 2.0 * (1.0 - kb) * cb;
    let g = luma
        - 2.0 * kb * (1.0 - kb) / kg * cb
        - 2.0 * kr * (1.0 - kr) / kg * cr;

    [
        (r.clamp(0.0, 1.0) * 255.0).round() as u8,
        (g.clamp(0.0, 1.0) * 255.0).round() as u8,
        (b.clamp(0.0, 1.0) * 255.0).round() as u8,
    ]
}

/// A single decoded video frame, ready for CPU consumption.
''',
)
replace_once(
    "src/software_video.rs",
    '''            PixelFormat::Rgb24 => Some(self.rgb_to_rgba(false)),
            PixelFormat::Rgba => Some(self.rgb_to_rgba(true)),
            PixelFormat::Yuv422 => self.yuv422_to_rgba(),
''',
    '''            PixelFormat::Rgb24 => Some(self.rgb_to_rgba(false, false)),
            PixelFormat::Bgr24 => Some(self.rgb_to_rgba(false, true)),
            PixelFormat::Rgba => Some(self.rgb_to_rgba(true, false)),
            PixelFormat::Yuv422 | PixelFormat::Uyvy422 | PixelFormat::Yvyu422 => {
                self.yuv422_to_rgba()
            }
''',
)
regex_once(
    "src/software_video.rs",
    r'''    fn yuv422_to_rgba\(&self\) -> Option<Vec<u8>> \{.*?\n    \}\n\n    fn nv12_to_rgba''',
    '''    fn yuv422_to_rgba(&self) -> Option<Vec<u8>> {
        let (width, height) = (self.width as usize, self.height as usize);
        if self.y.len() < width * height * 2 {
            return None;
        }

        let macropixels_per_row = (width + 1) / 2;
        let mut rgba = vec![0u8; width * height * 4];
        for row in 0..height {
            for x in 0..width {
                let base = (row * macropixels_per_row + x / 2) * 4;
                if base + 3 >= self.y.len() {
                    return None;
                }
                let bytes = &self.y[base..base + 4];
                let (y0, cb, y1, cr) = match self.format {
                    PixelFormat::Yuv422 => (bytes[0], bytes[1], bytes[2], bytes[3]),
                    PixelFormat::Uyvy422 => (bytes[1], bytes[0], bytes[3], bytes[2]),
                    PixelFormat::Yvyu422 => (bytes[0], bytes[3], bytes[2], bytes[1]),
                    _ => return None,
                };
                let y_sample = if x & 1 == 0 { y0 } else { y1 };
                let rgb = ycbcr_to_rgb8(
                    y_sample,
                    cb,
                    cr,
                    self.yuv_range,
                    self.color_matrix,
                );
                let dst = (row * width + x) * 4;
                rgba[dst..dst + 3].copy_from_slice(&rgb);
                rgba[dst + 3] = 255;
            }
        }
        Some(rgba)
    }

    fn nv12_to_rgba''',
)
regex_once(
    "src/software_video.rs",
    r'''    fn nv12_to_rgba\(&self\) -> Option<Vec<u8>> \{.*?\n    \}\n\n    fn yuv444p_to_rgba''',
    '''    fn nv12_to_rgba(&self) -> Option<Vec<u8>> {
        let (width, height) = (self.width as usize, self.height as usize);
        if self.y.len() < width * height {
            return None;
        }
        let chroma_width = (width + 1) / 2;
        let chroma_height = (height + 1) / 2;
        let uv_row_bytes = chroma_width * 2;
        if self.uv.len() < uv_row_bytes * chroma_height {
            return None;
        }

        let mut rgba = vec![0u8; width * height * 4];
        for row in 0..height {
            for x in 0..width {
                let y_sample = self.y[row * width + x];
                let uv_idx = (row / 2) * uv_row_bytes + (x / 2) * 2;
                let rgb = ycbcr_to_rgb8(
                    y_sample,
                    self.uv[uv_idx],
                    self.uv[uv_idx + 1],
                    self.yuv_range,
                    self.color_matrix,
                );
                let dst = (row * width + x) * 4;
                rgba[dst..dst + 3].copy_from_slice(&rgb);
                rgba[dst + 3] = 255;
            }
        }
        Some(rgba)
    }

    fn yuv444p_to_rgba''',
)
regex_once(
    "src/software_video.rs",
    r'''    fn yuv444p_to_rgba\(&self\) -> Option<Vec<u8>> \{.*?\n    \}\n\n    fn rgb_to_rgba''',
    '''    fn yuv444p_to_rgba(&self) -> Option<Vec<u8>> {
        let (width, height) = (self.width as usize, self.height as usize);
        let pixels = width * height;
        if self.y.len() < pixels || self.uv.len() < pixels || self.v.len() < pixels {
            return None;
        }

        let mut rgba = vec![0u8; pixels * 4];
        for i in 0..pixels {
            let rgb = ycbcr_to_rgb8(
                self.y[i],
                self.uv[i],
                self.v[i],
                self.yuv_range,
                self.color_matrix,
            );
            rgba[i * 4..i * 4 + 3].copy_from_slice(&rgb);
            rgba[i * 4 + 3] = 255;
        }
        Some(rgba)
    }

    fn rgb_to_rgba''',
)
regex_once(
    "src/software_video.rs",
    r'''    fn rgb_to_rgba\(&self, has_alpha: bool\) -> Vec<u8> \{.*?\n    \}\n\}''',
    '''    fn rgb_to_rgba(&self, has_alpha: bool, bgr: bool) -> Vec<u8> {
        let pixel_count = (self.width * self.height) as usize;
        let channels = if has_alpha { 4 } else { 3 };
        if self.y.len() < pixel_count * channels {
            return Vec::new();
        }
        let mut rgba = vec![0u8; pixel_count * 4];
        for i in 0..pixel_count {
            if has_alpha {
                rgba[i * 4..i * 4 + 4].copy_from_slice(&self.y[i * 4..i * 4 + 4]);
            } else if bgr {
                rgba[i * 4] = self.y[i * 3 + 2];
                rgba[i * 4 + 1] = self.y[i * 3 + 1];
                rgba[i * 4 + 2] = self.y[i * 3];
                rgba[i * 4 + 3] = 255;
            } else {
                rgba[i * 4] = self.y[i * 3];
                rgba[i * 4 + 1] = self.y[i * 3 + 1];
                rgba[i * 4 + 2] = self.y[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
        }
        rgba
    }
}''',
)

# Replace the final extraction function wholesale. It is the last item in file.
regex_once(
    "src/software_video.rs",
    r'''fn extract_planes\(\n    frame: &ffn::Frame,.*\Z''',
    '''fn extract_planes(
    frame: &ffn::Frame,
    format: PixelFormat,
    y: &mut Vec<u8>,
    uv: &mut Vec<u8>,
    v: &mut Vec<u8>,
) -> Result<()> {
    unsafe {
        let raw = frame.as_ptr();
        let width = (*raw).width as usize;
        let height = (*raw).height as usize;

        let copy_rows = |dst: &mut Vec<u8>,
                         data_ptr: *const u8,
                         stride: i32,
                         row_bytes: usize,
                         rows: usize| {
            dst.clear();
            dst.reserve(row_bytes * rows);
            let stride = stride as isize;
            if stride == row_bytes as isize {
                dst.extend_from_slice(std::slice::from_raw_parts(data_ptr, row_bytes * rows));
            } else {
                for row in 0..rows {
                    let ptr = data_ptr.offset(row as isize * stride);
                    dst.extend_from_slice(std::slice::from_raw_parts(ptr, row_bytes));
                }
            }
        };

        match format {
            PixelFormat::Nv12 => {
                copy_rows(y, (*raw).data[0], (*raw).linesize[0], width, height);
                let chroma_width = (width + 1) / 2;
                let chroma_height = (height + 1) / 2;
                copy_rows(
                    uv,
                    (*raw).data[1],
                    (*raw).linesize[1],
                    chroma_width * 2,
                    chroma_height,
                );
                v.clear();
            }
            PixelFormat::Yuv420P => {
                copy_rows(y, (*raw).data[0], (*raw).linesize[0], width, height);
                let chroma_width = (width + 1) / 2;
                let chroma_height = (height + 1) / 2;
                let mut u = Vec::with_capacity(chroma_width * chroma_height);
                let mut vv = Vec::with_capacity(chroma_width * chroma_height);
                copy_rows(
                    &mut u,
                    (*raw).data[1],
                    (*raw).linesize[1],
                    chroma_width,
                    chroma_height,
                );
                copy_rows(
                    &mut vv,
                    (*raw).data[2],
                    (*raw).linesize[2],
                    chroma_width,
                    chroma_height,
                );
                uv.clear();
                uv.reserve(u.len() * 2);
                for (&u, &vv) in u.iter().zip(vv.iter()) {
                    uv.push(u);
                    uv.push(vv);
                }
                v.clear();
            }
            PixelFormat::Yuv444P => {
                copy_rows(y, (*raw).data[0], (*raw).linesize[0], width, height);
                copy_rows(uv, (*raw).data[1], (*raw).linesize[1], width, height);
                copy_rows(v, (*raw).data[2], (*raw).linesize[2], width, height);
            }
            PixelFormat::Rgb24 | PixelFormat::Bgr24 | PixelFormat::Rgba => {
                let channels = if format == PixelFormat::Rgba { 4 } else { 3 };
                copy_rows(
                    y,
                    (*raw).data[0],
                    (*raw).linesize[0],
                    width * channels,
                    height,
                );
                uv.clear();
                v.clear();
            }
            PixelFormat::Yuv422 | PixelFormat::Uyvy422 | PixelFormat::Yvyu422 => {
                copy_rows(
                    y,
                    (*raw).data[0],
                    (*raw).linesize[0],
                    width * 2,
                    height,
                );
                uv.clear();
                v.clear();
            }
        }
        Ok(())
    }
}
''',
)

# Add targeted unit KATs to the existing software_video module tests.
replace_once(
    "src/software_video.rs",
    '''mod tests {
    use super::frame_duration_seconds;
    use ffmpeg_next::Rational;
''',
    '''mod tests {
    use super::{
        frame_duration_seconds, ycbcr_to_rgb8, ColorMatrix, PixelFormat, SoftwareFrame, YuvRange,
    };
    use ffmpeg_next::Rational;
''',
)
replace_once(
    "src/software_video.rs",
    '''    #[test]
    fn software_pacing_prefers_pts_delta() {
''',
    '''    #[test]
    fn ycbcr_neutral_and_limited_range_kats() {
        for matrix in [ColorMatrix::Bt601, ColorMatrix::Bt709, ColorMatrix::Bt2020] {
            let gray = ycbcr_to_rgb8(128, 128, 128, YuvRange::Full, matrix);
            assert!(gray.iter().all(|&c| (c as i16 - 128).abs() <= 1), "{matrix:?}: {gray:?}");
            assert_eq!(ycbcr_to_rgb8(16, 128, 128, YuvRange::Limited, matrix), [0, 0, 0]);
            assert_eq!(ycbcr_to_rgb8(235, 128, 128, YuvRange::Limited, matrix), [255, 255, 255]);
        }
    }

    #[test]
    fn packed_byte_orders_are_preserved() {
        let bgr = SoftwareFrame {
            width: 1,
            height: 1,
            format: PixelFormat::Bgr24,
            y: vec![3, 2, 1],
            uv: vec![],
            v: vec![],
            yuv_range: YuvRange::Full,
            color_matrix: ColorMatrix::Identity,
        };
        assert_eq!(bgr.to_rgba().unwrap(), vec![1, 2, 3, 255]);

        let make_422 = |format| SoftwareFrame {
            width: 2,
            height: 1,
            format,
            y: match format {
                PixelFormat::Yuv422 => vec![81, 90, 81, 240],
                PixelFormat::Uyvy422 => vec![90, 81, 240, 81],
                PixelFormat::Yvyu422 => vec![81, 240, 81, 90],
                _ => unreachable!(),
            },
            uv: vec![],
            v: vec![],
            yuv_range: YuvRange::Limited,
            color_matrix: ColorMatrix::Bt601,
        };
        let reference = make_422(PixelFormat::Yuv422).to_rgba().unwrap();
        assert_eq!(make_422(PixelFormat::Uyvy422).to_rgba().unwrap(), reference);
        assert_eq!(make_422(PixelFormat::Yvyu422).to_rgba().unwrap(), reference);
    }

    #[test]
    fn software_pacing_prefers_pts_delta() {
''',
)

print("deterministic audit repair applied")
