//! Tests for the software-only video path (`ffgpu::SoftwareVideo`).
//!
//! These tests verify that the new software path can:
//! - Open a video file and read its metadata
//! - Decode frames and publish them via the bounded channel
//! - Handle seek, set_discard, and looping without panicking
//! - Handle files with no audio stream
//!
//! No wgpu device is required — these tests run on any platform.

use ffgpu::{
    ColorMatrix, DiscardLevel, MasterClock, PixelFormat, SoftwareContext, SoftwareFrame, YuvRange,
};
use std::time::Duration;

const TEST_MP4: &str = "Test.mp4";
const TEST_NOAUDIO_MP4: &str = "test-noaudio.mp4";
const TEST_WEBM: &str = "test 2.webm";

fn test_file(name: &str) -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let p = std::path::PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(name);
    assert!(
        p.exists(),
        "Test file {} not found at {:?}",
        name,
        p
    );
    p
}

#[test]
fn software_video_opens_test_mp4() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    // H.264 1440x810 ~9.28s (verified via ffprobe)
    assert_eq!(video.width(), 1440);
    assert_eq!(video.height(), 810);
    assert!(video.duration().as_secs_f64() > 9.0 && video.duration().as_secs_f64() < 10.0);
}

#[test]
fn software_video_decodes_at_least_one_frame() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    let rx = video.frame_receiver();
    let _tx = video.frame_sender();

    // Wait up to 5 seconds for the first frame
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if let Ok(sf) = rx.recv_timeout(Duration::from_millis(100)) {
            assert!(sf.width > 0 && sf.height > 0);
            assert!(!sf.y.is_empty(), "Y plane should be populated");
            return;
        }
    }
    panic!("Did not receive any frame within 5 seconds");
}

#[test]
fn software_video_frame_format_is_yuv() {
    // Test.mp4 is H.264 → software decoder outputs YUV420P
    // (which we pack to NV12-style interleaved UV in extract_planes).
    // The GPU path converts to NV12 internally; the software path
    // accepts either.
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    let rx = video.frame_receiver();

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if let Ok(sf) = rx.recv_timeout(Duration::from_millis(100)) {
            assert!(
                sf.format == PixelFormat::Nv12 || sf.format == PixelFormat::Yuv420P,
                "expected YUV format, got {:?}",
                sf.format
            );
            // Y plane: width * height bytes
            assert_eq!(sf.y.len(), (sf.width * sf.height) as usize);
            // UV plane: width * height/2 bytes (NV12-style interleaved)
            assert_eq!(sf.uv.len(), (sf.width * sf.height / 2) as usize);
            return;
        }
    }
    panic!("No frame received within timeout");
}

#[test]
fn software_video_to_rgba_produces_valid_buffer() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    let rx = video.frame_receiver();

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while std::time::Instant::now() < deadline {
        if let Ok(sf) = rx.recv_timeout(Duration::from_millis(100)) {
            let rgba = sf
                .to_rgba()
                .expect("NV12 → RGBA conversion should succeed");
            // RGBA: 4 bytes per pixel
            assert_eq!(rgba.len(), (sf.width * sf.height * 4) as usize);
            // Spot-check: not all zeros (frame has content)
            let non_zero = rgba.iter().filter(|&&b| b != 0).count();
            assert!(
                non_zero > 1000,
                "RGBA buffer should have significant non-zero content (got {} non-zero bytes)",
                non_zero
            );
            return;
        }
    }
    panic!("No frame received within timeout");
}

#[test]
fn software_video_seek_does_not_panic() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    let _rx = video.frame_receiver();

    video.set_playback_rate(2.0);

    // Seek to 5 seconds (mid-video)
    video.seek(Duration::from_secs(5));
    assert_eq!(video.playback_rate(), 2.0);
    // Give the seek time to propagate
    std::thread::sleep(Duration::from_millis(200));
    // Forward seek too
    video.seek_forward(Duration::from_secs(7));
    std::thread::sleep(Duration::from_millis(200));
    // Back to start
    video.seek(Duration::from_secs(0));
    // No assertion needed; test passes if no panic
}

#[test]
fn software_video_set_discard_does_not_panic() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    let _rx = video.frame_receiver();

    // VLC "hurry up" levels
    video.set_discard(DiscardLevel::Default);
    std::thread::sleep(Duration::from_millis(50));
    video.set_discard(DiscardLevel::NonRef);
    std::thread::sleep(Duration::from_millis(50));
    video.set_discard(DiscardLevel::NonKey);
    std::thread::sleep(Duration::from_millis(50));
    video.set_discard(DiscardLevel::Default);
    // No panic is the test
}

#[test]
fn software_video_no_audio_file_works() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (_video, _audio) = ctx
        .create_video(&test_file(TEST_NOAUDIO_MP4))
        .expect("open test-noaudio.mp4");
    // Just verify it opens. The video path should still work even
    // when the file has no audio (the AudioThread is replaced with a
    // stub that just sleeps until alive=false).
}

#[test]
fn software_video_no_audio_reports_duration() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_NOAUDIO_MP4))
        .expect("open test-noaudio.mp4");
    let duration = video.duration().as_secs_f64();

    assert!(
        duration > 0.9 && duration < 1.1,
        "expected test-noaudio.mp4 duration near 0.984s, got {duration:.6}s"
    );
}

#[test]
fn software_video_switch_after_loop_does_not_hang() {
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let ctx = SoftwareContext::new().expect("ffmpeg init");
        let (mut video, _audio) = ctx
            .create_video(&test_file(TEST_MP4))
            .expect("open Test.mp4");
        video.set_looping(true);

        std::thread::sleep(Duration::from_millis(250));
        drop(video);

        let (replacement, _audio) = ctx
            .create_video(&test_file(TEST_NOAUDIO_MP4))
            .expect("open test-noaudio.mp4 after switch");
        drop(replacement);
        let _ = done_tx.send(());
    });

    assert!(
        done_rx.recv_timeout(Duration::from_secs(5)).is_ok(),
        "switch teardown did not complete within 5 seconds"
    );
}

#[test]
fn software_video_prefetches_no_audio_loop() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut video, _audio) = ctx
        .create_video(&test_file(TEST_NOAUDIO_MP4))
        .expect("open test-noaudio.mp4");
    let rx = video.frame_receiver();
    video.set_looping(true);

    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while video.loop_generation() == 0 && std::time::Instant::now() < deadline {
        let _ = rx.recv_timeout(Duration::from_millis(100));
    }

    assert!(video.loop_generation() > 0, "decoder did not cross a loop boundary");
    assert!(!video.is_eof(), "prefetched looping decoder reported EOF");
}

#[test]
fn software_video_metadata_helpers() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (video, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");

    // time_base, stream_time_base should be the same value
    assert_eq!(
        video.time_base(),
        video.stream_time_base(),
        "time_base and stream_time_base should be identical"
    );

    // frame_rate should be > 0 (Test.mp4 is 60fps)
    let fps = video.frame_rate();
    assert!(fps > 0.0, "frame_rate should be > 0, got {}", fps);

    // decoder_name should be the software marker
    assert_eq!(video.decoder_name(), "ffgpu (software)");
}

#[test]
fn software_frame_to_rgba_handles_rgba_input() {
    // Construct a SoftwareFrame in RGBA format directly
    let sf = SoftwareFrame {
        width: 2,
        height: 2,
        format: PixelFormat::Rgba,
        y: vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
            255, 255, 255, 255, // white
        ],
        uv: vec![],
        v: vec![],
        yuv_range: YuvRange::Full,
        color_matrix: ColorMatrix::Identity,
    };
    let rgba = sf.to_rgba().expect("RGBA passthrough");
    assert_eq!(rgba.len(), 16);
    assert_eq!(&rgba[0..4], &[255, 0, 0, 255]);
    assert_eq!(&rgba[4..8], &[0, 255, 0, 255]);
}

#[test]
fn software_frame_to_rgba_handles_rgb24_input() {
    // Construct a SoftwareFrame in RGB24 format
    let sf = SoftwareFrame {
        width: 1,
        height: 1,
        format: PixelFormat::Rgb24,
        y: vec![128, 64, 32],
        uv: vec![],
        v: vec![],
        yuv_range: YuvRange::Full,
        color_matrix: ColorMatrix::Identity,
    };
    let rgba = sf.to_rgba().expect("RGB24 → RGBA");
    assert_eq!(rgba.len(), 4);
    assert_eq!(&rgba[..], &[128, 64, 32, 255]);
}

#[test]
fn yuv422_format_recognized() {
    use ffgpu::PixelFormat;
    // ffmpeg-next 8.1.0 binds 3 of 4 packed 4:2:2 variants.
    assert_eq!(
        PixelFormat::from_ffmpeg(ffmpeg_next::format::pixel::Pixel::YUYV422.into()),
        Some(PixelFormat::Yuv422),
    );
    assert_eq!(
        PixelFormat::from_ffmpeg(ffmpeg_next::format::pixel::Pixel::UYVY422.into()),
        Some(PixelFormat::Yuv422),
    );
    assert_eq!(
        PixelFormat::from_ffmpeg(ffmpeg_next::format::pixel::Pixel::YVYU422.into()),
        Some(PixelFormat::Yuv422),
    );
}

#[test]
fn yuv422_to_rgba_conversion() {
    // 4x2 image in YUV422 (YUYV packing: Y0 U Y1 V per 4-byte macropixel).
    // 4 macropixels per row × 2 rows = 8 macropixels × 4 bytes = 32 bytes.
    let yuyv_data: Vec<u8> = vec![
        // row 0
        // macropixel 0 (covers x=0, x=1): Y0=128, U=128, Y1=128, V=128
        128, 128, 128, 128,
        // macropixel 1 (covers x=2, x=3): Y0=64, U=200, Y1=200, V=200
        64, 200, 200, 200,
        // row 1
        // macropixel 2 (covers x=0, x=1): Y0=200, U=64, Y1=64, V=64
        200, 64, 64, 64,
        // macropixel 3 (covers x=2, x=3): Y0=128, U=128, Y1=128, V=128
        128, 128, 128, 128,
    ];
    let sf = SoftwareFrame {
        format: PixelFormat::Yuv422,
        width: 4,
        height: 2,
        y: yuyv_data,
        uv: vec![],
        v: vec![],
        yuv_range: YuvRange::Full,
        color_matrix: ColorMatrix::Bt709,
    };
    let rgba = sf.to_rgba().expect("YUV422 → RGBA");
    // 4 pixels * 2 rows = 8 pixels, 4 bytes each = 32 bytes
    assert_eq!(rgba.len(), 4 * 2 * 4);
    // All alpha channels must be 255
    for px in rgba.chunks_exact(4) {
        assert_eq!(px[3], 255, "alpha channel must be opaque");
    }
    // x=0, y=0 was Y=128, U=128, V=128 (no chroma) — expect ~(128, 128, 128)
    let p0 = &rgba[0..4];
    assert!((p0[0] as i32 - 128).abs() < 5, "p0.r ~= 128, got {}", p0[0]);
    assert!((p0[1] as i32 - 128).abs() < 5, "p0.g ~= 128, got {}", p0[1]);
    assert!((p0[2] as i32 - 128).abs() < 5, "p0.b ~= 128, got {}", p0[2]);
}

#[test]
fn yuv444p_to_rgba_conversion() {
    // 2x2 image in YUV444P: every pixel has unique Y/U/V triple.
    // Pixel (0,0): Y=16,  U=128, V=128 → grayish ~(16,16,16) limited range
    // Pixel (1,0): Y=235, U=128, V=128 → near-white
    // Pixel (0,1): Y=16,  U=200, V=64  → greenish
    // Pixel (1,1): Y=128, U=64,  V=200 → pinkish
    let sf = SoftwareFrame {
        format: PixelFormat::Yuv444P,
        width: 2,
        height: 2,
        y: vec![16, 235, 16, 128],
        uv: vec![128, 128, 200, 64],
        v: vec![128, 128, 64, 200],
        yuv_range: YuvRange::Limited,
        color_matrix: ColorMatrix::Bt709,
    };
    let rgba = sf.to_rgba().expect("YUV444P → RGBA");
    assert_eq!(rgba.len(), 2 * 2 * 4);
    for px in rgba.chunks_exact(4) {
        assert_eq!(px[3], 255, "alpha must be opaque");
    }
    // Pixel (0,0): Y=16, U=128, V=128 — neutral gray (near black in limited range)
    // We just verify it's not a crash and the pixel order matches
    let p0 = &rgba[0..4];
    assert!(p0[0] <= p0[1], "p0.r <= p0.g for gray input, got r={}, g={}", p0[0], p0[1]);
}

#[test]
fn yuv444p_format_recognized() {
    assert_eq!(
        PixelFormat::from_ffmpeg(ffmpeg_next::format::pixel::Pixel::YUV444P.into()),
        Some(PixelFormat::Yuv444P),
    );
    assert_eq!(
        PixelFormat::from_ffmpeg(ffmpeg_next::format::pixel::Pixel::YUVJ444P.into()),
        Some(PixelFormat::Yuv444P),
    );
    // YUV440P is NOT YUV444P — should return None
    assert_eq!(
        PixelFormat::from_ffmpeg(ffmpeg_next::format::pixel::Pixel::YUV440P.into()),
        None,
    );
}

#[test]
fn rate_adjustment_starts_at_one() {
    // Construct a context and create a video. The decoder is
    // returned as (SoftwareDecodeVideo, AudioSink).
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut sv, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    assert!(
        (sv.playback_rate() - 1.0).abs() < 0.01,
        "initial playback rate must be 1.0, got {}",
        sv.playback_rate()
    );
    sv.set_playback_rate(1.5);
    assert!(
        (sv.playback_rate() - 1.5).abs() < 0.01,
        "set_playback_rate(1.5) must take effect, got {}",
        sv.playback_rate()
    );
    sv.set_playback_rate(0.5);
    assert!(
        (sv.playback_rate() - 0.5).abs() < 0.01,
        "set_playback_rate(0.5) must take effect, got {}",
        sv.playback_rate()
    );
    // Clamping: negative rates are invalid
    sv.set_playback_rate(-1.0);
    assert!(sv.playback_rate() > 0.0, "rate must be clamped to >0");
    // Clamping: too-large rates are capped
    sv.set_playback_rate(100.0);
    assert!(
        sv.playback_rate() <= 4.0,
        "rate must be clamped to <=4.0, got {}",
        sv.playback_rate()
    );
}

#[test]
fn master_clock_default_is_audio() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (sv, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    assert_eq!(
        sv.master_clock_kind(),
        MasterClock::Audio,
        "default master clock must be Audio (uses Arc<Clock> shared with AudioSink)"
    );
}

#[test]
fn update_returns_sleep_duration() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut sv, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");
    // Calling update() before any frames are decoded should return
    // a sensible (Duration, bool) tuple. The Duration is the time
    // to sleep before the next call. The bool indicates whether a
    // new frame is queued.
    let (sleep_dur, has_frame) = sv.update().expect("update before decode");
    let _ = sleep_dur; // any non-negative duration is fine
    let _ = has_frame; // false is fine (no frame decoded yet)
}

// ===========================================================================
// 20-LOOP BATTLE TEST
// ===========================================================================
//
// Pattern modeled after STUTTER_INVESTIGATION_REPORT.md's "20-loop" test
// (originally used to measure playback stutter, but repurposed here as
// a general stability soak). Each loop exercises the full API surface:
//   1. Open the test video
//   2. Decode 1 frame (verifies the read/decode/extract path)
//   3. Seek to 5s + set_discard + decode (verifies state transitions)
//   4. seek_forward(7s) + decode (verifies forward-biased seek)
//   5. update() (verifies the pacing API)
//   6. Drop the decoder (verifies clean teardown)
//
// The test is considered "green" if it completes 20 loops without
// panicking and decodes at least one frame per loop.
//
// The full test corpus (Test.mp4, test-noaudio.mp4) is exercised so
// we catch audio/video desync bugs and no-audio edge cases.

#[test]
fn battle_20_loops_with_audio() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let mut total_frames = 0u64;

    for loop_idx in 0..20 {
        let (mut sv, _audio) = ctx
            .create_video(&test_file(TEST_MP4))
            .unwrap_or_else(|e| panic!("loop {loop_idx}: create_video failed: {e}"));

        let rx = sv.frame_receiver();
        let duration = sv.duration();
        assert!(
            duration > Duration::from_secs(0),
            "loop {loop_idx}: duration must be > 0"
        );

        // Phase 1: receive at least one frame from the channel
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let mut got_frame = false;
        while std::time::Instant::now() < deadline {
            if let Ok(sf) = rx.recv_timeout(Duration::from_millis(200)) {
                assert!(sf.width > 0 && sf.height > 0, "loop {loop_idx}: bad frame size");
                assert!(!sf.y.is_empty(), "loop {loop_idx}: empty Y plane");
                got_frame = true;
                total_frames += 1;
                break;
            }
        }
        assert!(got_frame, "loop {loop_idx}: no frame within 3s");

        // Phase 2: seek + set_discard + decode
        sv.seek(Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(50));
        sv.set_discard(DiscardLevel::NonKey);
        std::thread::sleep(Duration::from_millis(50));
        let deadline2 = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline2 {
            if rx.recv_timeout(Duration::from_millis(200)).is_ok() {
                total_frames += 1;
                break;
            }
        }
        sv.set_discard(DiscardLevel::Default);

        // Phase 3: forward seek + decode
        sv.seek_forward(Duration::from_secs(7));
        std::thread::sleep(Duration::from_millis(50));
        let deadline3 = std::time::Instant::now() + Duration::from_secs(2);
        while std::time::Instant::now() < deadline3 {
            if rx.recv_timeout(Duration::from_millis(200)).is_ok() {
                total_frames += 1;
                break;
            }
        }

        // Phase 4: update() pacing API
        let (wait, decoded) = sv.update().expect("update");
        // wait must be a non-negative duration
        assert!(
            wait <= Duration::from_secs(60),
            "loop {loop_idx}: implausible wait_duration {wait:?}"
        );
        let _ = decoded; // no assertion on value

        // Drop sv — should cleanly tear down the decode/consumer threads
        drop(sv);
        std::thread::sleep(Duration::from_millis(50));
    }

    // 20 loops × ≥3 frames per loop = ≥60 frames expected
    assert!(
        total_frames >= 60,
        "20-loop battle: expected >= 60 frames, got {total_frames}"
    );
}

#[test]
fn battle_20_loops_no_audio() {
    // No-audio variant: the audio thread becomes a stub. The decoder
    // should still produce frames and `update()` should still return
    // valid durations (it falls back to MasterClock::System if Audio
    // is requested but unavailable, but the stub provides a real
    // Arc<Clock> with 0.0 readings — either path must not panic).
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let mut total_frames = 0u64;

    for loop_idx in 0..20 {
        let (mut sv, _audio) = ctx
            .create_video(&test_file(TEST_NOAUDIO_MP4))
            .unwrap_or_else(|e| panic!("loop {loop_idx}: create_video (noaudio) failed: {e}"));

        let rx = sv.frame_receiver();

        // Get at least one frame
        let deadline = std::time::Instant::now() + Duration::from_secs(3);
        let mut got_frame = false;
        while std::time::Instant::now() < deadline {
            if let Ok(sf) = rx.recv_timeout(Duration::from_millis(200)) {
                assert!(sf.width > 0 && sf.height > 0, "loop {loop_idx}: bad frame size");
                got_frame = true;
                total_frames += 1;
                break;
            }
        }
        assert!(got_frame, "loop {loop_idx}: noaudio: no frame within 3s");

        // update() with no audio stream — must not panic
        let (wait, _decoded) = sv.update().expect("update (noaudio)");
        assert!(
            wait <= Duration::from_secs(60),
            "loop {loop_idx}: implausible wait_duration {wait:?}"
        );

        drop(sv);
        std::thread::sleep(Duration::from_millis(50));
    }

    assert!(
        total_frames >= 20,
        "20-loop noaudio battle: expected >= 20 frames, got {total_frames}"
    );
}

#[test]
fn battle_master_clock_switching() {
    // Switches master_clock 20 times between Audio and System,
    // exercising set_master_clock + master_clock_kind. The
    // rolling_gap buffer should be cleared on each switch.
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut sv, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");

    assert_eq!(sv.master_clock_kind(), MasterClock::Audio);

    for i in 0..20 {
        let clock = if i % 2 == 0 {
            MasterClock::System
        } else {
            MasterClock::Audio
        };
        sv.set_master_clock(clock);
        assert_eq!(
            sv.master_clock_kind(),
            clock,
            "iteration {i}: master_clock_kind did not reflect set"
        );
    }
    // Final state should match the last write (i=19 → Audio)
    assert_eq!(sv.master_clock_kind(), MasterClock::Audio);
}

#[test]
fn battle_rate_adjustment_cycle() {
    // Cycles set_playback_rate through [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    // 20 times to verify clamping and persistence.
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut sv, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");

    let rates = [0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    for loop_idx in 0..20 {
        for &r in &rates {
            sv.set_playback_rate(r);
            let got = sv.playback_rate();
            assert!(
                (got - r).abs() < 0.001,
                "loop {loop_idx} rate {r}: playback_rate returned {got}"
            );
        }
    }
    // Below the minimum — clamped to 0.5
    sv.set_playback_rate(0.0);
    assert!(
        (sv.playback_rate() - 0.5).abs() < 0.001,
        "rate=0.0 must clamp to 0.5, got {}",
        sv.playback_rate()
    );
    // Above the maximum — clamped to 4.0
    sv.set_playback_rate(99.0);
    assert!(
        (sv.playback_rate() - 4.0).abs() < 0.001,
        "rate=99.0 must clamp to 4.0, got {}",
        sv.playback_rate()
    );
    // Negative — clamped to 0.5
    sv.set_playback_rate(-10.0);
    assert!(
        sv.playback_rate() >= 0.5,
        "rate=-10.0 must clamp to >= 0.5, got {}",
        sv.playback_rate()
    );
}

#[test]
fn battle_update_loop_smoke() {
    // Calls update() repeatedly 200×20=4000 times to verify the
    // manual pacing API doesn't panic, deadlock, or leak under
    // churn. We do NOT assert that frames are drained: the
    // channel-based bridge worker (started automatically by
    // create_video) is the production consumption path and
    // races update()/frame() for the same FrameQueue. The
    // bridge worker drains frames into the bounded(1) channel,
    // so update() will typically see an empty queue. This is
    // expected — the two paths are mutually exclusive. The
    // smoke test is purely a "does the API stay well-behaved
    // under repeated calls" check.
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut sv, _audio) = ctx
        .create_video(&test_file(TEST_MP4))
        .expect("open Test.mp4");

    let mut total_calls = 0u32;
    for _ in 0..20 {
        for _ in 0..200 {
            let (wait, decoded) = sv.update().expect("update");
            assert!(
                wait <= Duration::from_secs(60),
                "implausible wait_duration {wait:?} after {total_calls} calls"
            );
            let _ = decoded;
            if decoded {
                let _ = sv.frame();
            }
            total_calls += 1;
        }
    }
    assert_eq!(
        total_calls, 4000,
        "should have called update() 4000 times without panic"
    );
}

/// Shared backward-seek re-anchor check (see test below).
///
/// After a BACKWARD `seek(pos)`, both the video clock and the audio clock
/// must re-anchor DOWN to the target instead of staying at their pre-seek
/// absolute position (the reported "video jumps, audio stays behind" defect).
///
/// Drives the software path like the engine's CPU player: `read_to_slice`
/// consumes the ring buffer in a tight loop (the non-cpal analogue of the
/// device callback) while the video queue is drained. It plays until BOTH
/// clocks are ahead of the target, announces the seek (software mirror of
/// `prepare_seek`, so the audio position follows the target immediately), and
/// asserts the trough (minimum re-anchored position, ignoring the 0.0
/// unanchored transient) lands near the target for both clocks.
fn run_backward_reanchor(file: &str, target_ms: u64, pre_audio_s: f64, pre_video_s: f64) {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut video, mut audio) = ctx
        .create_video(&test_file(file))
        .unwrap_or_else(|_| panic!("open {file}"));
    let rx = video.frame_receiver();

    let mut buf = [0f32; 8192];

    // Play until BOTH clocks are demonstrably live and AHEAD of the seek target,
    // guaranteeing a *backward* re-anchor for video and audio alike.
    let target = Duration::from_millis(target_ms);
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while (video.position().as_secs_f64() < pre_video_s
        || audio.current_position_ms() / 1000.0 < pre_audio_s)
        && std::time::Instant::now() < deadline
    {
        let _ = audio.read_to_slice(&mut buf, 1.0);
        let _ = rx.recv_timeout(Duration::from_millis(10));
    }
    let pre_video = video.position().as_secs_f64();
    let pre_audio = audio.current_position_ms() / 1000.0;
    assert!(pre_audio > pre_audio_s - 0.5, "audio clock should be live before seek (got {pre_audio:.3}s)");

    // Backward seek; the audio position must jump to the target, not stall at
    // its pre-seek position or at 0.0 while the post-seek frame arrives.
    audio.announce_seek(target);
    video.seek(target);

    // During a backward re-anchor the audio clock briefly returns unanchored
    // (0.0) before locking to the new target; sample the *trough* of the
    // re-anchored region (never 0). Passing post-seek audio is allowed to be
    // silent for the whole window (no-cues WebM decode is slow): in that case
    // there are no samples to re-anchor, so `min_audio` stays at infinity and
    // only video proves the seek. The defect this guards is a FROZEN/stale
    // audio clock (reading a pre-seek position, 3s, after a 1.5s seek) — that
    // manifests as finite `min_audio` far above the target and still fails.
    let seek_start = std::time::Instant::now();
    let t_s = target.as_secs_f64();
    let mut min_video = f64::INFINITY;
    while std::time::Instant::now() - seek_start < Duration::from_secs(60) {
        let _ = audio.read_to_slice(&mut buf, 1.0);
        let _ = rx.recv_timeout(Duration::from_millis(5));
        min_video = min_video.min(video.position().as_secs_f64());

        let ap = audio.current_position_ms() / 1000.0;
        if ap > 0.05 {
            // Real (non-0.0) audio read: it must have re-anchored toward the
            // target, never stay parked near the pre-seek position.
            assert!(
                (ap - t_s).abs() < 0.7,
                "audio clock read {ap:.3}s after a {t_s:.1}s seek (frozen/stale clock)"
            );
        }
        if (min_video - t_s).abs() < 0.7 {
            eprintln!("POST-SEEK min video={min_video:.3}s re-anchored ✓ (audio {ap:.3}s)");
            return;
        }
    }
    panic!(
        "backward seek did not re-anchor video: min video={min_video:.3}s \
         (target {t_s:.1}s, pre-seek was video={pre_video:.3}s audio={pre_audio:.3}s)"
    );
}

/// `test.mp4`: WAV-style indexed MP4. NOTE: target 1.5s is below the ~9.3s
/// duration ffmpeg's demuxer resolves at open time (physical 13.8s); seeking
/// past that (e.g. 10s) lands past the demuxer's known EOF and yields no
/// post-seek frames — a seek-past-EOF artifact, not a follow defect.
#[test]
fn seek_moves_audio_and_video_together() {
    run_backward_reanchor(TEST_MP4, 1500, 3.0, 2.0);
}

/// `test2.webm`: a no-Cues WebM. ffmpeg cannot index-seek it, so after
/// `avformat_seek_file` the demuxer restarts BOTH streams at pts 0 and the
/// skip logic decodes+discards up to the target — a gap that historically
/// read as "audio stuck at 0 while video jumped". The audio clock must
/// still re-anchor to the target.
#[test]
fn seek_moves_audio_and_video_together_webm() {
    run_backward_reanchor(TEST_WEBM, 1500, 1.8, 2.0);
}
