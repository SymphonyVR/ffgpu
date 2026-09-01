use ffgpu::{SoftwareContext, YuvRange, PixelFormat, ColorMatrix, DiscardLevel, MasterClock};

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
fn probe_webm_audio_recovers_after_seek() {
    let ctx = SoftwareContext::new().expect("ffmpeg init");
    let (mut video, mut audio) = ctx
        .create_video(&test_file(TEST_WEBM))
        .expect("open webm");
    let rx = video.frame_receiver();

    let mut buf = [0f32; 8192];
    let target = std::time::Duration::from_millis(1500);

    // Play until both live and ahead of the seek target (backward seek).
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(40);
    while (video.position().as_secs_f64() < 2.0
        || audio.current_position_ms() / 1000.0 < 1.8)
        && std::time::Instant::now() < deadline
    {
        let _ = audio.read_to_slice(&mut buf, 1.0);
        let _ = rx.recv_timeout(std::time::Duration::from_millis(5));
    }
    let pre = (video.position().as_secs_f64(), audio.current_position_ms() / 1000.0);
    eprintln!("[PROBE] pre-seek video={:.3}s audio={:.3}s", pre.0, pre.1);

    // Seek backward.
    audio.announce_seek(target);
    video.seek(target);
    let t0 = std::time::Instant::now();

    // Poll: how long until audio re-anchors to ~target (not 0, not pre-seek)?
    let seek_deadline = std::time::Instant::now() + std::time::Duration::from_secs(90);
    let mut last_ap = 0.0f64;
    let mut video_min = f64::INFINITY;
    let mut audio_recovered_at: Option<f64> = None;
    let mut audio_peak = 0.0f64;
    while std::time::Instant::now() < seek_deadline {
        let _ = audio.read_to_slice(&mut buf, 1.0);
        let _ = rx.recv_timeout(std::time::Duration::from_millis(5));
        video_min = video_min.min(video.position().as_secs_f64());

        let ap = audio.current_position_ms() / 1000.0;
        if ap > 0.05 && ap.is_finite() {
            audio_peak = audio_peak.max(ap);
            if audio_recovered_at.is_none() && (ap - target.as_secs_f64()).abs() < 0.7 {
                audio_recovered_at = Some(t0.elapsed().as_secs_f64());
            }
        }
        last_ap = ap;

        if (video_min - target.as_secs_f64()).abs() < 0.7
            && audio_recovered_at.is_some()
        {
            break;
        }
        if video.position().as_secs_f64() > target.as_secs_f64() + 2.0
            && audio_recovered_at.is_some()
        {
            break;
        }
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "[PROBE] post-seek {elapsed:.2}s: video_min={video_min:.3}s last_audio={last_ap:.3}s audio_peak={audio_peak:.3}s audio_recovered_at={:?}",
        audio_recovered_at
    );

    assert!(
        audio_recovered_at.is_some(),
        "audio never re-anchored after backward seek (video_min={video_min:.3}s, audio_peak={audio_peak:.3}s)"
    );
}