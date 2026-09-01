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

#[test]
fn probe_software_forward_seek_moves_video() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut video, mut audio) = ctx
        .create_video(&test_file(TEST_WEBM))
        .expect("open webm");
    let rx = video.frame_receiver();
    let mut buf = [0f32; 8192];

    let deadline = Instant::now() + Duration::from_secs(20);
    while video.position().as_secs_f64() < 0.5 && Instant::now() < deadline {
        let _ = audio.read_to_slice(&mut buf, 1.0);
        let _ = rx.recv_timeout(Duration::from_millis(10));
    }
    eprintln!(
        "[FWD] pre-seek video={:.3}s audio={:.3}s",
        video.position().as_secs_f64(),
        audio.current_position_ms() / 1000.0
    );

    // Forward seek deep into the file (the user's failing case).
    let target = Duration::from_secs(45);
    audio.announce_seek(target);
    video.seek_forward(target);
    let t0 = Instant::now();

    let mut audio_peak = 0.0f64;
    let mut video_peak = 0.0f64;
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        let _ = audio.read_to_slice(&mut buf, 1.0);
        let _ = rx.recv_timeout(Duration::from_millis(5));
        video_peak = video_peak.max(video.position().as_secs_f64());
        let ap = audio.current_position_ms() / 1000.0;
        if ap > 0.05 && ap.is_finite() {
            audio_peak = audio_peak.max(ap);
        }
        // Both must recover to near the 45s target (the user's failing case).
        if audio_peak > 44.0 && video_peak > 44.0 {
            break;
        }
    }
    eprintln!(
        "[FWD] post-seek {:.0}s: video_peak={video_peak:.3}s audio_peak={audio_peak:.3}s",
        t0.elapsed().as_secs_f64()
    );
    assert!(video_peak > 40.0, "video never reached 45s target (video_peak={video_peak:.3})");
    assert!(
        audio_peak > 44.0,
        "audio never recovered after forward seek: still {audio_peak:.3}s"
    );
}