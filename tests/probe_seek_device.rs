use ffgpu::SoftwareContext;
use std::time::{Duration, Instant};

const TEST_WEBM: &str = "test 2.webm";

fn test_file(name: &str) -> std::path::PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join(name)
}

// Real device-path reproduction of the reported bug: "no audio after seeking
//     test 2.webm". Drives the same calls the engine's Software
//     HybridVideoPlayer uses: prepare_seek(position, …) then video.seek(…),
//     and polls video frames the way the engine's update loop does. Covers
//     BACKWARD, FORWARD and DEEP seeks (the file is ~99s long).
fn run_device_seek(seek_ms: u64, forward: bool) {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut video, audio) = ctx
        .create_video(&test_file(TEST_WEBM))
        .expect("open webm");
    let rx = video.frame_receiver();
    let sink = audio.into_device_sink();
    video.play();

    let target = Duration::from_millis(seek_ms);

    // Play until we have live video before the seek.
    let deadline = Instant::now() + Duration::from_secs(20);
    while video.position().as_secs_f64() < 0.5 && Instant::now() < deadline {
        let _ = rx.recv_timeout(Duration::from_millis(10));
    }
    eprintln!(
        "[PROBE-DEV] pre-seek video={:.3}s audio={:.3}s -> seek to {}s (forward={})",
        video.position().as_secs_f64(),
        sink.current_position_ms() / 1000.0,
        seek_ms / 1000,
        forward
    );

    // Engine HybridVideoPlayer::seek (Software): prepare_seek(pos,0) then video.seek.
    // A scrub slider (apply_pending_seek) also lands here with 75ms preview.
    sink.prepare_seek(target, 0);
    if forward {
        video.seek_forward(target);
    } else {
        video.seek(target);
    }
    let t0 = Instant::now();

    let seek_deadline = Instant::now() + Duration::from_secs(90);
    let mut video_min = f64::INFINITY;
    let mut audio_recovered_at: Option<f64> = None;
    let mut audio_peak = 0.0f64;
    while Instant::now() < seek_deadline {
        let _ = rx.recv_timeout(Duration::from_millis(5));
        video_min = video_min.min(video.position().as_secs_f64());

        let ap = sink.current_position_ms() / 1000.0;
        if ap > 0.05 && ap.is_finite() {
            audio_peak = audio_peak.max(ap);
            if audio_recovered_at.is_none() && (ap - target.as_secs_f64()).abs() < 2.0 {
                audio_recovered_at = Some(t0.elapsed().as_secs_f64());
            }
        }
        if audio_recovered_at.is_some() && (video_min - target.as_secs_f64()).abs() < 0.7 {
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "[PROBE-DEV] post-seek {elapsed:.2}s: video_min={video_min:.3}s last_audio={:.3}s audio_peak={audio_peak:.3}s audio_recovered_at={:?}",
        sink.current_position_ms() / 1000.0, audio_recovered_at
    );

    assert!(
        audio_recovered_at.is_some(),
        "device audio never re-anchored after {:?} seek (video_min={video_min:.3}s, audio_peak={audio_peak:.3}s)",
        if forward { "forward" } else { "backward" }
    );
}

// Non-cpal build: nothing to run.
#[cfg(not(feature = "cpal"))]
#[test]
fn noop() {}

#[cfg(feature = "cpal")]
#[test]
fn probe_device_audio_after_backward_seek() {
    // Entered view: an old bug exercised by the previous probe — seek backwards
    //     from ~2s to 1.5s (video had looped/pass the target).
    run_device_seek(1500, false);
}

#[cfg(feature = "cpal")]
#[test]
fn probe_device_audio_after_forward_seek() {
    // User-reported: seek ahead deep into a long file and audio stayed silent.
    run_device_seek(90_000, true);
}

#[cfg(feature = "cpal")]
#[test]
fn probe_device_audio_after_forward_mid_seek() {
    run_device_seek(45_000, true);
}