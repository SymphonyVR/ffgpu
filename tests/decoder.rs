//! Decoder-level integration tests for ffgpu.
//!
//! These tests exercise the FFmpeg decoder configuration and seek/discard
//! plumbing without requiring a wgpu device. They are the battle-tests
//! for the new APIs ported from video-rs's async_decode.rs:
//!
//! - `DiscardLevel` → `set_discard()` → `ffn::codec::discard::Discard` mapping
//! - `seek_forward` → ReadMessage routing
//! - `frame_rate` / `stream_time_base` / `duration_ms` getters
//!
//! The actual wgpu-bound tests live in `tests/wgpu_video.rs` (will be
//! added once the engine integration is done).

use std::path::PathBuf;

fn corpus_path(name: &str) -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.parent().expect("repo root").join(name)
}

fn has_corpus() -> bool {
    corpus_path("Test.mp4").exists()
}

#[test]
fn discard_level_converts_to_ffmpeg_default() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::Default.into();
    assert!(matches!(
        level,
        ffmpeg_next::codec::discard::Discard::Default
    ));
}

#[test]
fn discard_level_converts_to_ffmpeg_non_reference() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::NonRef.into();
    assert!(matches!(
        level,
        ffmpeg_next::codec::discard::Discard::NonReference
    ));
}

#[test]
fn discard_level_converts_to_ffmpeg_bidirectional() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::Bidir.into();
    assert!(matches!(
        level,
        ffmpeg_next::codec::discard::Discard::Bidirectional
    ));
}

#[test]
fn discard_level_converts_to_ffmpeg_non_key() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::NonKey.into();
    assert!(matches!(
        level,
        ffmpeg_next::codec::discard::Discard::NonKey
    ));
}

#[test]
fn discard_level_converts_to_ffmpeg_non_intra() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::NonIntra.into();
    assert!(matches!(
        level,
        ffmpeg_next::codec::discard::Discard::NonIntra
    ));
}

#[test]
fn discard_level_converts_to_ffmpeg_none() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::None.into();
    assert!(matches!(level, ffmpeg_next::codec::discard::Discard::None));
}

#[test]
fn test_mp4_metadata_round_trip() {
    if !has_corpus() {
        eprintln!("[decoder test] Test.mp4 missing — skipping");
        return;
    }

    // Open the file with ffmpeg-next directly and verify the metadata
    // we'd surface to callers matches what's actually in the container.
    ffmpeg_next::init().unwrap();
    let input = ffmpeg_next::format::input(&corpus_path("Test.mp4")).unwrap();
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .expect("no video stream");

    let codec_id = stream.parameters().id();
    assert!(
        matches!(codec_id, ffmpeg_next::codec::Id::H264),
        "Test.mp4 should be H.264, got {:?}",
        codec_id
    );

    // Stream time base should be 1/3000 or 1/15360 (H.264 60fps typical).
    let tb = stream.time_base();
    let tb_f = tb.0 as f64 / tb.1 as f64;
    assert!(
        tb_f > 0.0 && tb_f < 1.0,
        "stream time_base numerator/denominator should give a sub-second rational"
    );

    // Duration should be 9.28s ± 0.1s (matches ffprobe output).
    let duration_s = input.duration() as f64 / ffmpeg_next::ffi::AV_TIME_BASE as f64;
    assert!(
        (duration_s - 9.28).abs() < 0.1,
        "Test.mp4 duration should be ~9.28s, got {}",
        duration_s
    );
}

#[test]
fn test_noaudio_mp4_has_no_audio() {
    if !corpus_path("test-noaudio.mp4").exists() {
        eprintln!("[decoder test] test-noaudio.mp4 missing — skipping");
        return;
    }

    ffmpeg_next::init().unwrap();
    let input = ffmpeg_next::format::input(&corpus_path("test-noaudio.mp4")).unwrap();
    let audio_streams: Vec<_> = input
        .streams()
        .filter(|s| matches!(s.parameters().medium(), ffmpeg_next::media::Type::Audio))
        .collect();
    assert!(
        audio_streams.is_empty(),
        "test-noaudio.mp4 should have no audio streams, got {}",
        audio_streams.len()
    );
}

#[test]
fn av1_test_webm_codec_id() {
    if !corpus_path("4k test.webm").exists() {
        eprintln!("[decoder test] 4k test.webm missing — skipping");
        return;
    }

    ffmpeg_next::init().unwrap();
    let input = ffmpeg_next::format::input(&corpus_path("4k test.webm")).unwrap();
    let stream = input
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .expect("no video stream");

    let codec_id = stream.parameters().id();
    assert!(
        matches!(codec_id, ffmpeg_next::codec::Id::AV1),
        "4k test.webm should be AV1, got {:?}",
        codec_id
    );
}


#[test]
fn test_cuda_decode_tiny_mp4() {
    ffmpeg_next::init().unwrap();
    use ffmpeg_next::sys as ff;

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/data/tiny.mp4");
    if !path.exists() {
        eprintln!("[probe] tests/data/tiny.mp4 missing");
        return;
    }

    let mut input = ffmpeg_next::format::input(&path).unwrap();
    let (stream_index, codec_params) = {
        let stream = input.streams().best(ffmpeg_next::media::Type::Video).unwrap();
        (stream.index(), stream.parameters())
    };
    let codec_id = codec_params.id();
    let decoder = ffmpeg_next::decoder::find(codec_id).unwrap();

    let mut decoder_ctx = ffmpeg_next::codec::Context::new_with_codec(decoder).decoder();
    decoder_ctx.set_parameters(codec_params).unwrap();

    let mut hwctx: *mut ff::AVBufferRef = std::ptr::null_mut();
    let ret = unsafe {
        ff::av_hwdevice_ctx_create(
            &mut hwctx,
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_CUDA,
            std::ptr::null(),
            std::ptr::null_mut(),
            0,
        )
    };
    assert_eq!(ret, 0);

    struct ProbeData {
        hw_pix_fmt: ff::AVPixelFormat,
        chosen_fmt: ff::AVPixelFormat,
    }

    unsafe extern "C" fn probe_get_format(
        s: *mut ff::AVCodecContext,
        mut fmt: *const ff::AVPixelFormat,
    ) -> ff::AVPixelFormat {
        let data = unsafe { ((*s).opaque as *mut ProbeData).as_mut().unwrap() };
        eprintln!("[probe_get_format] offered formats:");
        unsafe {
            while (*fmt) != ff::AVPixelFormat::AV_PIX_FMT_NONE {
                eprintln!("  format: {:?}", *fmt);
                if *fmt == data.hw_pix_fmt {
                    data.chosen_fmt = *fmt;
                    return *fmt;
                }
                fmt = fmt.add(1);
            }
        }
        ff::AVPixelFormat::AV_PIX_FMT_NONE
    }

    let mut probe_data = ProbeData {
        hw_pix_fmt: ff::AVPixelFormat::AV_PIX_FMT_CUDA,
        chosen_fmt: ff::AVPixelFormat::AV_PIX_FMT_NONE,
    };

    unsafe {
        (*decoder_ctx.as_mut_ptr()).opaque = &mut probe_data as *mut _ as _;
        (*decoder_ctx.as_mut_ptr()).get_format = Some(probe_get_format);
        (*decoder_ctx.as_mut_ptr()).hw_device_ctx = hwctx;
    }

    let mut video_decoder = decoder_ctx.video().unwrap();

    let mut packet = ffmpeg_next::Packet::empty();
    let mut frame = ffmpeg_next::frame::Video::empty();

    let mut decoded_frames = 0;
    while packet.read(&mut input).is_ok() {
        if packet.stream() == stream_index && video_decoder.send_packet(&packet).is_ok() {
            while video_decoder.receive_frame(&mut frame).is_ok() {
                decoded_frames += 1;
            }
        }
    }

    assert_eq!(probe_data.chosen_fmt, ff::AVPixelFormat::AV_PIX_FMT_CUDA);
    assert!(decoded_frames > 0);
}

#[test]
fn test_video_player_playback_crash() {
    ffmpeg_next::init().unwrap();
    unsafe { ffmpeg_next::sys::av_log_set_level(ffmpeg_next::sys::AV_LOG_DEBUG); }
    let path = std::path::Path::new(r"E:\c0fa50060aaba663bab8d8bcb12b0b9b.mp4");
    if !path.exists() {
        return;
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: ffgpu::required_wgpu_device_features(&adapter),
            ..Default::default()
        },
    ))
    .expect("device");

    let mut ctx = ffgpu::Context::new(&instance, &adapter, &device, &queue).expect("context");
    let (mut video, _audio) = ctx.create_video(path).expect("create_video");
    video.set_looping(true);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    for i in 0..100 {
        match video.update(&mut encoder) {
            Ok((wait, decoded)) => {
                eprintln!("[test_playback] step {}: wait={:?}, decoded={}", i, wait, decoded);
                if decoded {
                    let _ = queue.submit(Some(encoder.finish()));
                    video.release_completed_frames().unwrap();
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("test_encoder"),
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            Err(e) => {
                panic!("[test_playback] step {} failed: {:?}", i, e);
            }
        }
    }
}

#[test]
fn test_video_player_playback_yuv444() {
    ffmpeg_next::init().unwrap();
    let path = std::path::Path::new(r"E:\2026-05-21 14-52-15.mp4");
    if !path.exists() {
        return;
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: ffgpu::required_wgpu_device_features(&adapter),
            ..Default::default()
        },
    ))
    .expect("device");

    let mut ctx = ffgpu::Context::new(&instance, &adapter, &device, &queue).expect("context");
    let (mut video, _audio) = ctx.create_video(path).expect("create_video");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let mut decoded_count = 0;
    for i in 0..60 {
        match video.update(&mut encoder) {
            Ok((_wait, decoded)) => {
                if decoded {
                    decoded_count += 1;
                    let views = video.plane_views().expect("plane_views");
                    assert_eq!(views.len(), 3, "YUV444 must have 3 plane views");
                    let _ = queue.submit(Some(encoder.finish()));
                    video.release_completed_frames().unwrap();
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("test_encoder"),
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            Err(e) => {
                panic!("[test_yuv444] step {} failed: {:?}", i, e);
            }
        }
    }
    assert!(decoded_count > 0, "Expected at least 1 decoded frame");
}

#[test]
fn test_video_player_playback_hevc_10bit() {
    ffmpeg_next::init().unwrap();
    let path = std::path::Path::new(r"E:\OP darkins.mov");
    if !path.exists() {
        return;
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: ffgpu::required_wgpu_device_features(&adapter),
            ..Default::default()
        },
    ))
    .expect("device");

    let mut ctx = ffgpu::Context::new(&instance, &adapter, &device, &queue).expect("context");
    let (mut video, _audio) = ctx.create_video(path).expect("create_video");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let mut decoded_count = 0;
    for i in 0..60 {
        match video.update(&mut encoder) {
            Ok((_wait, decoded)) => {
                if decoded {
                    decoded_count += 1;
                    let views = video.plane_views().expect("plane_views");
                    assert_eq!(views.len(), 2, "HEVC Main 10 (P010) must have 2 plane views (Y, UV)");
                    let _ = queue.submit(Some(encoder.finish()));
                    video.release_completed_frames().unwrap();
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("test_encoder"),
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            Err(e) => {
                panic!("[test_hevc_10bit] step {} failed: {:?}", i, e);
            }
        }
    }
    assert!(decoded_count > 0, "Expected at least 1 decoded frame");
}

#[test]
fn test_video_player_playback_hevc_10bit_dma() {
    unsafe { std::env::set_var("NEXA_CUDA_FORCE_DMA", "1"); }
    ffmpeg_next::init().unwrap();
    let path = std::path::Path::new(r"E:\OP darkins.mov");
    if !path.exists() {
        return;
    }

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }))
    .expect("adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            required_features: ffgpu::required_wgpu_device_features(&adapter),
            ..Default::default()
        },
    ))
    .expect("device");

    let mut ctx = ffgpu::Context::new(&instance, &adapter, &device, &queue).expect("context");
    let (mut video, _audio) = ctx.create_video(path).expect("create_video");

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("test_encoder"),
    });

    let mut decoded_count = 0;
    for i in 0..60 {
        match video.update(&mut encoder) {
            Ok((_wait, decoded)) => {
                if decoded {
                    decoded_count += 1;
                    let views = video.plane_views().expect("plane_views");
                    assert_eq!(views.len(), 2, "HEVC Main 10 (P010) must have 2 plane views (Y, UV)");
                    let _ = queue.submit(Some(encoder.finish()));
                    video.release_completed_frames().unwrap();
                    encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("test_encoder"),
                    });
                }
                std::thread::sleep(std::time::Duration::from_millis(16));
            }
            Err(e) => {
                panic!("[test_hevc_10bit_dma] step {} failed: {:?}", i, e);
            }
        }
    }
    assert!(decoded_count > 0, "Expected at least 1 decoded frame");
}

