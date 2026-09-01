//! Integration tests for the public probe API.
//!
//! These tests verify that `ffgpu::probe_hardware_decoding_support` and
//! `ffgpu::probe_software_required` correctly introspect a video file
//! and report whether the codec is supported by the current GPU/ASIC.
//!
//! The test corpus is in the repo root:
//! - `Test.mp4`            — H.264 1440x810 @60fps, AAC, 9.28s
//! - `test-noaudio.mp4`    — H.264 1280x720 @59.94fps, no audio, 0.98s
//! - `4k test.webm`        — AV1 3840x1632 @60fps, Opus, 100.8s
//! - `test 2.webm`         — AV1 1920x1080 @30fps, Opus, 161.7s
//!
//! The test videos are at the repository root, not relative to this
//! file. We resolve them with `CARGO_MANIFEST_DIR/../<name>`.

use std::path::PathBuf;

fn corpus_path(name: &str) -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir.parent().expect("repo root").join(name)
}

fn has_corpus() -> bool {
    corpus_path("Test.mp4").exists() && corpus_path("test-noaudio.mp4").exists()
}

#[test]
fn probe_h264_no_hwaccel_when_unsupported_backend() {
    if !has_corpus() {
        eprintln!("[probe test] test corpus missing — skipping");
        return;
    }

    // WebGPU/OpenGL backend has no FFmpeg hardware device type — must
    // report `false` (no hwaccel available) regardless of codec.
    let result = ffgpu::probe_hardware_decoding_support(
        corpus_path("Test.mp4"),
        wgpu::Backend::BrowserWebGpu,
    );

    match result {
        Ok(false) => {} // expected
        Ok(true) => panic!("WebGPU backend should never report hwaccel support"),
        Err(e) => panic!("probe should not error for an unsupported backend: {}", e),
    }
}

#[test]
fn probe_software_required_is_inverse() {
    if !has_corpus() {
        eprintln!("[probe test] test corpus missing — skipping");
        return;
    }

    // For the unsupported backend, the two probes must be exact inverses.
    let hw_ok = ffgpu::probe_hardware_decoding_support(
        corpus_path("Test.mp4"),
        wgpu::Backend::BrowserWebGpu,
    )
    .unwrap();
    let sw_req =
        ffgpu::probe_software_required(corpus_path("Test.mp4"), wgpu::Backend::BrowserWebGpu)
            .unwrap();
    assert_eq!(hw_ok, !sw_req, "probe_software_required must be the inverse of probe_hardware_decoding_support");
}

#[test]
fn probe_av1_reports_correctly() {
    if !corpus_path("4k test.webm").exists() {
        eprintln!("[probe test] 4k test.webm missing — skipping");
        return;
    }

    // AV1 codec. With an unsupported backend the probe must report
    // no hwaccel; the test asserts the negative path which is stable
    // across machines and doesn't depend on actual GPU capabilities.
    let result = ffgpu::probe_hardware_decoding_support(
        corpus_path("4k test.webm"),
        wgpu::Backend::BrowserWebGpu,
    );
    assert!(
        matches!(result, Ok(false)),
        "AV1 + WebGPU backend should not report hwaccel: got {:?}",
        result
    );
}
