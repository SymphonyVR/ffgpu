use std::time::Duration;

fn seek_and_resume(input: &mut ffmpeg_next::format::context::Input, ts: i64, n_before: u64) {
    let rc = unsafe {
        ffmpeg_next::sys::avformat_seek_file(
            input.as_mut_ptr(),
            -1,
            i64::MIN,
            ts,
            i64::MAX,
            ffmpeg_next::sys::AVSEEK_FLAG_BACKWARD,
        )
    };
    eprintln!("[RAW] avformat_seek_file rc={rc}");
    let mut n_after = 0u64;
    let t0 = std::time::Instant::now();
    loop {
        let mut packet = ffmpeg_next::Packet::empty();
        match packet.read(input) {
            Ok(_) => {
                n_after += 1;
                if n_after <= 6 {
                    eprintln!("[RAW] ({n_before}?) post-seek pkt stream={} pts_ms={:?}", packet.stream(), packet.pts());
                }
                if n_after >= 100 {
                    break;
                }
            }
            Err(e) => {
                eprintln!("[RAW] post-seek EOF at n={n_after} after {:?}: {e}", t0.elapsed());
                break;
            }
        }
    }
    eprintln!("[RAW] read {n_after} packets post-seek");
}

#[test]
fn probe_raw_seek_eof() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("test 2.webm");
    let input = ffmpeg_next::format::input(&path).expect("open");
    let mut input = input;
    let mut n_before = 0u64;
    loop {
        let mut packet = ffmpeg_next::Packet::empty();
        match packet.read(&mut input) {
            Ok(_) => {
                n_before += 1;
            }
            Err(e) => {
                eprintln!("[RAW] pre-seek EOF at n={n_before}: {e}");
                break;
            }
        }
    }
    eprintln!("[RAW] read {n_before} packets pre-seek");
    seek_and_resume(&mut input, 45_000_000i64, n_before);
    eprintln!("[RAW] --- now reopen and seek MID-STREAM ---");
    let input2 = ffmpeg_next::format::input(&path).expect("open");
    let mut input2 = input2;
    let mut n_mid = 0u64;
    loop {
        let mut packet = ffmpeg_next::Packet::empty();
        match packet.read(&mut input2) {
            Ok(_) => {
                n_mid += 1;
                if n_mid >= 1500 {
                    break;
                }
            }
            Err(e) => {
                eprintln!("[RAW] mid-stream EOF early at n={n_mid}: {e}");
                break;
            }
        }
    }
    eprintln!("[RAW] read {n_mid} packets then seek");
    seek_and_resume(&mut input2, 45_000_000i64, n_mid);
    let _ = Duration::ZERO;
}
