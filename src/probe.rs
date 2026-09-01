//! Proactive hardware-decoding capability probe.
//!
//! Used by the engine (or any caller) to decide *upfront* whether to
//! attempt hardware acceleration for a given file on the current GPU,
//! before constructing a `Video`. The internal `ffgpu::Video` already
//! does a runtime fallback (see `decode/video.rs:408`), but that fallback
//! only triggers after the first frame — too late to avoid a one-time
//! startup cost and a possibly-visible glitch on unsupported codecs.
//!
//! This module is the public API for that pre-check. It mirrors
//! `VideoThread`'s internal `get_hw_format` callback logic by walking
//! `avcodec_get_hw_config` for the stream's codec and reporting whether
//! a `AVHWDeviceType` matching the wgpu backend's preferred device type
//! is available.

use crate::error::{Error, Result};
use ffmpeg_next::{self as ffn, sys as ff, media::Type};

/// Map a wgpu backend to the FFmpeg hardware device type that ffgpu
/// would try first. Mirrors `ffgpu::video::preferred_device_type_for_backend`.
#[cfg(target_os = "windows")]
pub fn preferred_device_type_for_backend(backend: wgpu::Backend) -> ff::AVHWDeviceType {
    match backend {
        wgpu::Backend::Vulkan => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        wgpu::Backend::Dx12 => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        _ => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
    }
}

#[cfg(target_os = "macos")]
pub fn preferred_device_type_for_backend(_backend: wgpu::Backend) -> ff::AVHWDeviceType {
    ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX
}

#[cfg(target_os = "linux")]
pub fn preferred_device_type_for_backend(backend: wgpu::Backend) -> ff::AVHWDeviceType {
    match backend {
        wgpu::Backend::Vulkan => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        _ => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn preferred_device_type_for_backend(_backend: wgpu::Backend) -> ff::AVHWDeviceType {
    ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
}

/// Proactively check whether the video stream's codec supports hardware
/// decoding on the current GPU/ASIC.
///
/// Returns:
/// - `Ok(true)`  — a hwaccel config exists for this codec on this backend;
///                 ffgpu::Video will attempt hwaccel at construction.
/// - `Ok(false)` — no hwaccel config; the engine should construct ffgpu::Video
///                 with `hw_device_ctx = None` so ffgpu skips hwaccel entirely
///                 and goes straight to software decoding.
/// - `Err(_)`    — file could not be opened or codec not found.
pub fn probe_hardware_decoding_support<P: AsRef<std::path::Path>>(
    path: P,
    backend: wgpu::Backend,
) -> Result<bool> {
    // ffmpeg_next::init is idempotent.
    ffn::init()?;

    let input = ffn::format::input(&path.as_ref())
        .map_err(|e| Error::Probe(format!("failed to open video for hw probe: {}", e)))?;

    let video_stream = input
        .streams()
        .best(Type::Video)
        .ok_or_else(|| Error::Probe("no video stream found for hw probe".to_string()))?;

    let codec_id = video_stream.parameters().id();
    let codec = ffn::decoder::find(codec_id).ok_or_else(|| {
        Error::Probe(format!("codec {:?} not found for hw probe", codec_id))
    })?;

    // FFmpeg can advertise a Vulkan VP9 hw config even when the active Vulkan
    // device has no VP9 decode extension. Let the GPU-backed software adapter
    // upload YUV420 planes instead of entering the failing Vulkan -> D3D11VA
    // fallback sequence.
    if backend == wgpu::Backend::Vulkan && codec_id == ffn::codec::Id::VP9 {
        return Ok(false);
    }

    let device_type = preferred_device_type_for_backend(backend);
    if device_type == ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
        return Ok(false);
    }

    for i in 0..16 {
        let config = unsafe { ff::avcodec_get_hw_config(codec.as_ptr(), i) };
        if config.is_null() {
            break;
        }
        let config = unsafe { &*config };
        if (config.methods & ff::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0
            && config.device_type == device_type
        {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Inverse of `probe_hardware_decoding_support` — returns `Ok(true)`
/// when the caller should construct `ffgpu::Video` with `hw_device_ctx = None`
/// (skip hwaccel entirely) and let ffgpu's software path do the work.
///
/// This is the helper the engine should use in its `HybridVideoPlayer::new`
/// pre-check. It is the inverse rather than an alias so the engine call site
/// reads naturally: "if software is required, take the software path".
///
/// On error (file unreadable, codec missing) the function returns `Ok(false)`
/// — the safe default is to let ffgpu try hwaccel and fall back internally.
pub fn probe_software_required<P: AsRef<std::path::Path>>(
    path: P,
    backend: wgpu::Backend,
) -> Result<bool> {
    match probe_hardware_decoding_support(path, backend) {
        Ok(true) => Ok(false),
        Ok(false) => Ok(true),
        Err(_) => Ok(false),
    }
}
