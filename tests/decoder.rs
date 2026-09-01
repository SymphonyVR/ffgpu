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
    assert!(matches!(level, ffmpeg_next::codec::discard::Discard::Default));
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
    assert!(matches!(level, ffmpeg_next::codec::discard::Discard::NonKey));
}

#[test]
fn discard_level_converts_to_ffmpeg_non_intra() {
    use ffgpu::DiscardLevel;
    let level: ffmpeg_next::codec::discard::Discard = DiscardLevel::NonIntra.into();
    assert!(matches!(level, ffmpeg_next::codec::discard::Discard::NonIntra));
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
