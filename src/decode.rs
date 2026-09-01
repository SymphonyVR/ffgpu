pub(crate) mod audio;
pub(crate) mod device_monitor;
pub(crate) mod frames;
pub(crate) mod read;
pub(crate) mod video;
pub(crate) mod vulkan_hwcontext;

use crate::decode::{audio::AudioStream, read::Metadata, video::VideoStream};
use arc_swap::ArcSwap;
use atomic_float::AtomicF64;
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use ffmpeg_next::{self as ffn, sys as ff};
use std::sync::{
    Arc,
    atomic::{
        AtomicBool, AtomicI64, AtomicPtr, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering,
    },
};
use std::time::Duration;

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PlayState {
    Playing = 0,
    Paused,
    Step,
}

/// High-level lifecycle of a decoder (Playing/Paused/Stopping/Stopped).
///
/// Independent of `play_state` (which only tracks Playing/Paused/Step). The
/// `kill()` call transitions Active → Stopping and the engine can then call
/// `wait_to_finish()` to block until every decoder thread has actually
/// returned. Calling `wait_to_finish()` while still in `Active` state would
/// deadlock (the threads never observe `alive=false` on their own), so the
/// function panics — `kill()` must be called first. Mirrors the Kira audio
/// player's `Stopping/Stopped` pattern.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Lifecycle {
    Active = 0, // Playing or Paused (play_state has the detail)
    Stopping,   // kill() was called, threads are winding down
    Stopped,    // all decoder threads have exited
}

pub(crate) struct DecoderState {
    pub metadata: ArcSwap<Metadata>,

    /// Shared with the ffmpeg interrupt callback so av_read_frame aborts
    /// when kill() is called. Wrapped in Arc because the callback's
    /// closure must be 'static and capture a reference to flip.
    pub alive: Arc<AtomicBool>,
    pub play_state: AtomicU8,
    pub lifecycle: AtomicU8,
    pub is_eof: AtomicBool,
    pub current_pts: AtomicI64,
    pub looping: AtomicBool,
    pub loop_index: AtomicU64,
    pub loop_events: Arc<AtomicU64>,

    /// Count of running decoder threads (read, video, audio). Incremented
    /// in each thread's `run()` before spawn, decremented by a ThreadGuard
    /// when the thread closure returns. `wait_to_finish` polls this.
    /// Wrapped in Arc so the guard can be sent across thread boundaries.
    pub thread_count: Arc<AtomicUsize>,

    pub video_stream: ArcSwap<VideoStream>,
    pub audio_stream: ArcSwap<AudioStream>,

    /// Current `AVCodecContext` of the video decoder. The `VideoThread` swaps
    /// decoders on hwaccel fallback (Vulkan → D3D11VA → software); consumers
    /// (ffgpu `Video::update`) must read the decoder through this shared
    /// pointer instead of a pointer captured at open time, which would dangle
    /// once the thread frees the old context.
    pub video_decoder: AtomicPtr<ff::AVCodecContext>,
}

impl DecoderState {
    /// Create a state with the given metadata and streams. Use this for
    /// the final construction after decoders are wired up. The `alive`
    /// flag is freshly initialized.
    pub fn new(metadata: Metadata, video: VideoStream, audio: AudioStream) -> Self {
        DecoderState {
            metadata: ArcSwap::new(Arc::new(metadata)),

            alive: Arc::new(AtomicBool::new(true)),
            play_state: AtomicU8::new(PlayState::Playing as u8),
            lifecycle: AtomicU8::new(Lifecycle::Active as u8),
            is_eof: AtomicBool::new(false),
            current_pts: AtomicI64::new(0),
            looping: AtomicBool::new(false),
            loop_index: AtomicU64::new(0),
            loop_events: Arc::new(AtomicU64::new(0)),
            thread_count: Arc::new(AtomicUsize::new(0)),

            video_stream: ArcSwap::new(Arc::new(video)),
            audio_stream: ArcSwap::new(Arc::new(audio)),
            video_decoder: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Create a bare-bones state with default metadata and dummy streams.
    /// Use this when you need a state *before* the streams are constructed
    /// (so the state can be passed to `Input::open_with_state` and the
    /// ffmpeg interrupt callback can be wired to `state.alive`). The
    /// caller MUST follow up with `set_metadata` + `set_streams` before
    /// spawning any threads.
    pub fn empty() -> Self {
        DecoderState {
            metadata: ArcSwap::new(Arc::new(Metadata {
                duration: Duration::ZERO,
                ..Default::default()
            })),

            alive: Arc::new(AtomicBool::new(true)),
            play_state: AtomicU8::new(PlayState::Playing as u8),
            lifecycle: AtomicU8::new(Lifecycle::Active as u8),
            is_eof: AtomicBool::new(false),
            current_pts: AtomicI64::new(0),
            looping: AtomicBool::new(false),
            loop_index: AtomicU64::new(0),
            loop_events: Arc::new(AtomicU64::new(0)),
            thread_count: Arc::new(AtomicUsize::new(0)),

            video_stream: ArcSwap::new(Arc::new(VideoStream::dummy())),
            audio_stream: ArcSwap::new(Arc::new(AudioStream::dummy())),
            video_decoder: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    /// Set the metadata + streams after the Input has been opened with
    /// the state's `alive` as the ffmpeg interrupt source. Called once
    /// during initialization.
    pub fn install(&self, metadata: Metadata, video: VideoStream, audio: AudioStream) {
        self.metadata.store(Arc::new(metadata));
        self.video_stream.store(Arc::new(video));
        self.audio_stream.store(Arc::new(audio));
    }

    pub fn play_state(&self) -> PlayState {
        match self.play_state.load(Ordering::Relaxed) {
            0 => PlayState::Playing,
            1 => PlayState::Paused,
            2 => PlayState::Step,
            _ => unreachable!(),
        }
    }

    pub fn lifecycle(&self) -> Lifecycle {
        match self.lifecycle.load(Ordering::Relaxed) {
            0 => Lifecycle::Active,
            1 => Lifecycle::Stopping,
            2 => Lifecycle::Stopped,
            _ => Lifecycle::Active,
        }
    }

    pub fn kill(&self) {
        // Transition Active → Stopping. Idempotent: re-calling kill() while
        // already Stopping/Stopped is a no-op.
        self.lifecycle
            .store(Lifecycle::Stopping as u8, Ordering::SeqCst);
        self.alive.store(false, Ordering::SeqCst);
    }

    /// Block until every decoder thread has returned, or `timeout` elapses.
    /// Returns `true` if all threads finished, `false` on timeout.
    ///
    /// **Caller contract:** `kill()` must be called first. Calling this on
    /// an `Active` decoder would deadlock (the threads never see `alive=false`
    /// on their own), so we panic loudly to surface the bug.
    pub fn wait_to_finish(&self, timeout: std::time::Duration) -> bool {
        match self.lifecycle() {
            Lifecycle::Stopped => return true,
            Lifecycle::Active => {
                panic!(
                    "ffgpu::DecoderState::wait_to_finish called on Active decoder; \
                     call kill() first"
                );
            }
            Lifecycle::Stopping => {}
        }
        let deadline = std::time::Instant::now() + timeout;
        while self.thread_count.load(Ordering::Acquire) > 0 {
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.lifecycle
            .store(Lifecycle::Stopped as u8, Ordering::SeqCst);
        true
    }
}

/// RAII guard that decrements the decoder's thread counter on drop. Held
/// inside each decoder thread's closure so the counter goes back to zero
/// when `run_thread()` returns (normally or via panic).
pub(crate) struct ThreadGuard(Arc<AtomicUsize>);

impl ThreadGuard {
    /// Increment `counter` and return a guard that decrements on drop.
    pub fn new(counter: Arc<AtomicUsize>) -> Self {
        counter.fetch_add(1, Ordering::AcqRel);
        ThreadGuard(counter)
    }
}

impl Drop for ThreadGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub(crate) struct PacketQueueMetadata {
    pub duration: AtomicI64,
    pub serial: AtomicU32,
    pub loop_index: AtomicU64,
}

pub(crate) struct Packet {
    pub packet: ffn::Packet,
    pub serial: u32,
    pub loop_index: u64,
}

#[derive(Clone)]
pub(crate) struct PacketSender {
    pub metadata: Arc<PacketQueueMetadata>,
    pub rx: Receiver<Packet>,
    pub tx: Sender<Packet>,
}

impl PacketSender {
    const MIN_FRAMES: usize = 25;

    fn push(&self, packet: ffn::Packet) {
        self.metadata
            .duration
            .fetch_add(packet.duration(), Ordering::SeqCst);
        let serial = self.metadata.serial.load(Ordering::SeqCst);
        let loop_index = self.metadata.loop_index.load(Ordering::SeqCst);
        self.tx
            .send(Packet {
                packet,
                serial,
                loop_index,
            })
            .unwrap();
    }

    pub(crate) fn push_null(&self, mut packet: ffn::Packet, stream_index: usize) {
        packet.set_stream(stream_index);
        self.metadata
            .duration
            .fetch_add(packet.duration(), Ordering::SeqCst);
        let serial = self.metadata.serial.load(Ordering::SeqCst);
        let loop_index = self.metadata.loop_index.load(Ordering::SeqCst);
        let _ = self.tx.send(Packet {
            packet,
            serial,
            loop_index,
        });
    }

    // Packet-count backpressure for the read thread. The old
    // `buffered duration > 1.0s` term is unreachable for clips shorter than
    // one second, and ANDing with the audio queue meant a no-audio stream
    // never throttled the reader at all. Either way a short looping video
    // grew the unbounded packet queue forever (memory leak). A plain count
    // bound works for any duration.
    fn has_enough_packets(&self) -> bool {
        self.tx.len() > Self::MIN_FRAMES
    }

    fn flush(&self) {
        while let Ok(_) = self.rx.try_recv() {}
        self.metadata.duration.store(0, Ordering::SeqCst);
        self.metadata.serial.fetch_add(1, Ordering::SeqCst);
    }

    fn set_loop_index(&self, loop_index: u64) {
        self.metadata.loop_index.store(loop_index, Ordering::SeqCst);
    }
}

unsafe impl Send for PacketSender {}
unsafe impl Sync for PacketSender {}

#[derive(Clone)]
pub(crate) struct PacketReceiver {
    metadata: Arc<PacketQueueMetadata>,
    rx: Receiver<Packet>,
}

impl PacketReceiver {
    fn receive(&self) -> Option<Packet> {
        let Ok(recv) = self.rx.recv() else {
            return None;
        };
        self.metadata
            .duration
            .fetch_sub(recv.packet.duration(), Ordering::SeqCst);
        Some(recv)
    }
}

unsafe impl Send for PacketReceiver {}
unsafe impl Sync for PacketReceiver {}

pub(crate) fn packet_queue() -> (PacketSender, PacketReceiver, Arc<PacketQueueMetadata>) {
    let metadata = Arc::new(PacketQueueMetadata {
        duration: AtomicI64::new(0),
        serial: AtomicU32::new(0),
        loop_index: AtomicU64::new(0),
    });
    let (tx, rx) = unbounded();
    let tx = PacketSender {
        metadata: metadata.clone(),
        rx: rx.clone(),
        tx,
    };
    let rx = PacketReceiver {
        metadata: metadata.clone(),
        rx,
    };
    (tx, rx, metadata)
}

pub(crate) struct Frame {
    pub frame: ffn::Frame,
    pub serial: u32,
}

#[derive(Clone)]
pub(crate) struct FrameQueue {
    free_tx: Sender<Frame>,
    free_rx: Receiver<Frame>,
    queue_tx: Sender<Frame>,
    queue_rx: Receiver<Frame>,
}

impl FrameQueue {
    pub fn new(capacity: usize) -> Self {
        let (free_tx, free_rx) = bounded(capacity);
        let (queue_tx, queue_rx) = bounded(capacity);

        for _ in 0..capacity {
            let frame = unsafe { ffn::Frame::empty() };
            free_tx
                .send(Frame {
                    frame,
                    serial: u32::MAX,
                })
                .unwrap();
        }

        FrameQueue {
            free_tx,
            free_rx,
            queue_tx,
            queue_rx,
        }
    }

    pub fn send(&self, frame: &mut ffn::Frame, serial: u32, alive: &AtomicBool) -> bool {
        let mut dst = loop {
            match self.free_rx.recv_timeout(Duration::from_millis(10)) {
                Ok(dst) => break dst,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                    if !alive.load(Ordering::Relaxed) {
                        return false;
                    }
                }
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return false,
            }
        };

        unsafe {
            ff::av_frame_unref(dst.frame.as_mut_ptr());
            ff::av_frame_move_ref(dst.frame.as_mut_ptr(), frame.as_mut_ptr());
        }
        dst.serial = serial;
        self.queue_tx.send(dst).is_ok()
    }

    pub fn queued_len(&self) -> usize {
        self.queue_rx.len()
    }

    pub fn try_next(&self) -> Option<Frame> {
        self.queue_rx.try_recv().ok()
    }

    pub fn release(&self, mut frame: Frame) {
        unsafe { ff::av_frame_unref(frame.frame.as_mut_ptr()) };
        self.free_tx.send(frame).unwrap();
    }

    pub fn flush(&self) {
        while let Some(frame) = self.try_next() {
            self.release(frame);
        }
    }
}

unsafe impl Send for FrameQueue {}
unsafe impl Sync for FrameQueue {}

// complete atomic abuse
// but we really don't want to use locking primitives
// because the clock is updated in the audio callback
pub(crate) struct Clock {
    pub pts: AtomicF64,
    pub pts_drift: AtomicF64,
    pub last_updated: AtomicF64,
    pub speed: AtomicF64,
    pub serial: AtomicU32,
    pub paused: AtomicBool,
    pub queue: Arc<PacketQueueMetadata>,
}

impl Clock {
    #[allow(dead_code)] // ffplay-parity clock constants; consumed by sync_to_slave and future A/V sync
    pub const NO_SYNC_THRESHOLD: f64 = 10.;
    #[allow(dead_code)]
    pub const SYNC_MIN: f64 = 0.04;
    pub const SYNC_MAX: f64 = 0.1;
    #[allow(dead_code)]
    pub const FRAME_DUPLICATION_THRESHOLD: f64 = 0.1;

    pub fn new(queue: Arc<PacketQueueMetadata>) -> Self {
        let clock = Clock {
            pts: AtomicF64::new(0.),
            pts_drift: AtomicF64::new(0.),
            last_updated: AtomicF64::new(0.),
            speed: AtomicF64::new(1.),
            serial: AtomicU32::new(0),
            paused: AtomicBool::new(false),
            queue,
        };
        clock.set(f64::NAN, u32::MAX, None);
        clock
    }

    pub fn get(&self) -> Option<f64> {
        let queue_serial = self.queue.serial.load(Ordering::Relaxed);
        if self.serial.load(Ordering::Relaxed) != queue_serial {
            None
        } else if self.paused.load(Ordering::Relaxed) {
            let pts = self.pts.load(Ordering::Relaxed);
            if pts.is_nan() { None } else { Some(pts) }
        } else {
            let t = ffn::time::relative() as f64 / 1000000.;
            let pts = self.pts_drift.load(Ordering::Relaxed) + t
                - (t - self.last_updated.load(Ordering::Relaxed))
                    * (1. - self.speed.load(Ordering::Relaxed));
            if pts.is_nan() { None } else { Some(pts) }
        }
    }

    pub fn set(&self, pts: f64, serial: u32, time: Option<f64>) {
        let time = time.unwrap_or_else(|| ffn::time::relative() as f64 / 1000000.);
        self.pts.store(pts, Ordering::Relaxed);
        self.last_updated.store(time, Ordering::Relaxed);
        self.pts_drift.store(pts - time, Ordering::Relaxed);
        self.serial.store(serial, Ordering::Relaxed);
    }

    #[allow(dead_code)] // ffplay-parity slave-clock sync; reserved for multi-stream playback
    pub fn sync_to_slave(&self, slave: &Clock) {
        let clock = self.get();
        let slave_clock = slave.get();
        if let Some(slave_clock) = slave_clock
            && clock.is_none_or(|clock| (clock - slave_clock).abs() > Clock::NO_SYNC_THRESHOLD)
        {
            self.set(slave_clock, slave.serial.load(Ordering::Relaxed), None);
        }
    }
}

pub(crate) fn sink_thread(state: Arc<DecoderState>, packets: PacketReceiver) {
    while state.alive.load(Ordering::Relaxed) {
        packets.receive();
    }
}

pub(crate) fn loop_timestamp_offset(
    state: &DecoderState,
    loop_index: u64,
    time_base: ffn::Rational,
) -> i64 {
    let duration_us = state.metadata.load().duration.as_micros();
    let total_us = duration_us
        .saturating_mul(u128::from(loop_index))
        .min(i64::MAX as u128) as i64;

    unsafe { ff::av_rescale_q(total_us, ff::AV_TIME_BASE_Q, time_base.into()) }
}

/// Convert a public (start-time-relative) position into the absolute
/// AV_TIME_BASE timestamp that `avformat_seek_file()` and the decoder
/// `SkipToTimestamp` messages expect. ffplay contract: `abs = rel +
/// container.start_time`. A relative position below the origin is clamped
/// up to it so we never seek before the start of the media.
pub(crate) fn seek_target_us(state: &DecoderState, position: Duration) -> i64 {
    let rel = (position.as_secs_f64() * ff::AV_TIME_BASE as f64) as i64;
    let start = state.metadata.load().start_time;
    if start != ff::AV_NOPTS_VALUE {
        rel.saturating_add(start).max(start)
    } else {
        rel
    }
}

/// Container start offset in seconds for the public timeline (0 when the
/// format carries none). Public position = decoded absolute PTS (seconds)
/// minus this value, per the ffplay model.
pub(crate) fn container_start_seconds(state: &DecoderState) -> f64 {
    let start = state.metadata.load().start_time;
    if start != ff::AV_NOPTS_VALUE && start > 0 {
        start as f64 / ff::AV_TIME_BASE as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DecoderState, PacketSender, audio::AudioStream, packet_queue, read::Metadata,
        video::VideoStream,
    };
    use ffmpeg_next::Packet;
    use ffmpeg_next::Rational;
    use std::time::Duration;

    fn state_with_start_time(start_time: i64) -> DecoderState {
        let state = DecoderState::empty();
        state.metadata.store(std::sync::Arc::new(Metadata {
            duration: Duration::from_secs(120),
            start_time,
        }));
        state
    }

    #[test]
    fn seek_target_adds_start_time_and_clamps() {
        let no_start = state_with_start_time(ffmpeg_next::sys::AV_NOPTS_VALUE);
        assert_eq!(
            super::seek_target_us(&no_start, Duration::from_secs(5)),
            5_000_000
        );

        let shifted = state_with_start_time(4_000_000);
        assert_eq!(
            super::seek_target_us(&shifted, Duration::from_secs(5)),
            9_000_000
        );
        // Relative position below the origin is clamped up to the origin.
        assert_eq!(super::seek_target_us(&shifted, Duration::ZERO), 4_000_000);
    }

    #[test]
    fn container_start_seconds_projects_reports_offset_only_when_positive() {
        assert_eq!(
            super::container_start_seconds(&state_with_start_time(4_000_000)),
            4.0
        );
        assert_eq!(
            super::container_start_seconds(&state_with_start_time(0)),
            0.0
        );
        assert_eq!(
            super::container_start_seconds(&state_with_start_time(
                ffmpeg_next::sys::AV_NOPTS_VALUE
            )),
            0.0,
        );
    }

    #[test]
    fn install_publishes_input_duration() {
        let state = DecoderState::empty();
        let duration = Duration::from_secs(2142);

        state.install(
            Metadata {
                duration,
                ..Default::default()
            },
            VideoStream::dummy(),
            AudioStream::dummy(),
        );

        assert_eq!(state.metadata.load().duration, duration);
    }

    #[test]
    fn loop_timestamp_offset_uses_container_duration() {
        let state = DecoderState::empty();
        state.metadata.store(std::sync::Arc::new(Metadata {
            duration: Duration::from_secs(2),
            ..Default::default()
        }));

        assert_eq!(super::loop_timestamp_offset(&state, 3, Rational(1, 1)), 6,);
    }

    #[test]
    fn packet_queue_tags_prefetched_loop_packets() {
        let (sender, receiver, _) = packet_queue();
        sender.set_loop_index(2);
        sender.push_null(Packet::empty(), 0);

        assert_eq!(receiver.receive().unwrap().loop_index, 2);
    }

    #[test]
    fn has_enough_packets_throttles_by_count_independent_of_duration() {
        // Regression: the old gate required `buffered duration > 1.0s`, which
        // is unreachable for clips shorter than one second (and ANDing with
        // the audio queue meant a no-audio stream never throttled). A short
        // looping video therefore grew the unbounded packet queue forever.
        // This is a packet-count bound now, so it must engage regardless of
        // how short the clip is.
        let (sender, _receiver, _) = packet_queue();
        assert!(!sender.has_enough_packets());
        for _ in 0..(PacketSender::MIN_FRAMES + 1) {
            sender.push_null(Packet::empty(), 0);
        }
        assert!(sender.has_enough_packets());
    }
}
