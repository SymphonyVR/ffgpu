use crate::{
    context::pipeline_cache::PipelineCache,
    decode::{
        audio::{self, AudioSink, AudioThread},
        frames, packet_queue,
        read::{Input, ReadMessage, ReadThread},
        sink_thread, video, Clock, DecoderState, Frame, FrameQueue, PlayState,
    },
    error::{Error, Result},
};
use crossbeam_channel::{unbounded, Sender};
use ffmpeg_next::{self as ffn, sys as ff};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::{
    ops::Add,
    path::Path,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SeekMode {
    Fast,
    Accurate,
}

enum FrameResponse {
    Continue,
    Retry,
    #[allow(dead_code)] // reserved re-decode response for the seek path
    Requeue,
}

#[cfg(target_os = "windows")]
fn preferred_device_type_for_backend(backend: wgpu::Backend) -> ff::AVHWDeviceType {
    match backend {
        wgpu::Backend::Vulkan => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        wgpu::Backend::Dx12 => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
        _ => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA,
    }
}

#[cfg(target_os = "macos")]
fn preferred_device_type_for_backend(_backend: wgpu::Backend) -> ff::AVHWDeviceType {
    ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VIDEOTOOLBOX
}

#[cfg(target_os = "linux")]
fn preferred_device_type_for_backend(backend: wgpu::Backend) -> ff::AVHWDeviceType {
    match backend {
        wgpu::Backend::Vulkan => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VULKAN,
        _ => ff::AVHWDeviceType::AV_HWDEVICE_TYPE_VAAPI,
    }
}

pub struct Statistics {
    pub video_clock: f64,
    pub audio_clock: f64,
    pub sync_latency: f64,
    pub decoder_name: &'static str,
    // TODO: add dropped frames and whatnot
}

pub struct Video {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    state: Arc<DecoderState>,
    frame_decoder: frames::FrameDecoder,
    video_decoder: NonNull<ff::AVCodecContext>,
    read_thread: Option<JoinHandle<()>>,
    video_thread: Option<JoinHandle<()>>,
    audio_thread: Option<JoinHandle<()>>,
    video_clock: Arc<Clock>,
    audio_clock: Arc<Clock>,
    has_audio: bool,

    read_messages: Sender<ReadMessage>,

    /// Shared with VideoThread's Decoder. Set to true when the hardware decoder
    /// rejects the pixel format (get_hw_format callback returns AV_PIX_FMT_NONE),
    /// meaning the VideoThread has fallen back to software decoding.
    hw_unsupported: Arc<std::sync::atomic::AtomicBool>,

    looping: bool,
    frame_timer: f64,
    last_pts: i64,
    last_serial: u32,
    queued_frame: Option<Frame>,
    step_needs_copy: u8,
}

impl Video {
    pub(crate) fn new<P>(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline_cache: Arc<Mutex<PipelineCache>>,
        hw_device_ctx: Option<NonNull<ff::AVBufferRef>>,
        path: &P,
    ) -> Result<(Self, AudioSink)>
    where
        P: AsRef<Path> + ?Sized,
    {
        Self::new_with_options(
            instance,
            adapter,
            device,
            queue,
            pipeline_cache,
            hw_device_ctx,
            false,
            path,
        )
    }

    pub(crate) fn new_software_planes<P>(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline_cache: Arc<Mutex<PipelineCache>>,
        path: &P,
    ) -> Result<(Self, AudioSink)>
    where
        P: AsRef<Path> + ?Sized,
    {
        Self::new_with_options(
            instance,
            adapter,
            device,
            queue,
            pipeline_cache,
            None,
            true,
            path,
        )
    }

    fn new_with_options<P>(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline_cache: Arc<Mutex<PipelineCache>>,
        hw_device_ctx: Option<NonNull<ff::AVBufferRef>>,
        force_software: bool,
        path: &P,
    ) -> Result<(Self, AudioSink)>
    where
        P: AsRef<Path> + ?Sized,
    {
        // Create the state first with empty streams so the ffmpeg
        // interrupt callback (set when we open the input) can reference
        // `state.alive`. The real streams are installed via
        // `state.install()` after the decoders are wired.
        let state = Arc::new(DecoderState::empty());

        let mut input = Input::open_with_state(path, state.clone())?;

        let backend = adapter.get_info().backend;
        let device_type = if force_software {
            ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
        } else {
            preferred_device_type_for_backend(backend)
        };

        let video_decoder = video::Decoder::new(&mut input.format_ctx, device_type, hw_device_ctx)?;
        let hw_unsupported = video_decoder.unsupported.clone();
        let audio_decoder = audio::Decoder::new(&mut input)?;
        let has_audio = audio_decoder.is_some();
        let frame_decoder = frames::FrameDecoder::new(
            &device,
            pipeline_cache.clone(),
            &video_decoder.metadata,
            adapter
                .get_downlevel_capabilities()
                .flags
                .contains(wgpu::DownlevelFlags::VIEW_FORMATS),
        )?;

        let (video_tx, video_rx, video_queue) = packet_queue();
        let (audio_tx, audio_rx, audio_queue) = packet_queue();
        let video_frame_queue = FrameQueue::new(8);
        let audio_frame_queue = FrameQueue::new(16);

        let video_clock = Arc::new(Clock::new(video_queue.clone()));
        let audio_clock = Arc::new(Clock::new(audio_queue.clone()));

        let (read_msg_tx, read_msg_rx) = unbounded();
        let (video_msg_tx, video_msg_rx) = unbounded();
        let (audio_msg_tx, audio_msg_rx) = unbounded();

        let video_stream = video::VideoStream {
            metadata: video_decoder.metadata,
            messages: video_msg_tx,
            packets: video_tx,
            frames: video_frame_queue.clone(),
        };

        let audio_stream = audio::AudioStream {
            metadata: audio_decoder
                .as_ref()
                .map(|decoder| decoder.metadata)
                .unwrap_or_default(),
            messages: audio_msg_tx.clone(),
            packets: audio_tx,
            frames: audio_frame_queue.clone(),
        };

        // Install the real streams on the (previously empty) state. The
        // alive flag and lifecycle are already set; we only swap in the
        // stream channels. The ffmpeg interrupt callback (set when
        // open_with_state was called) is already wired to state.alive,
        // so any subsequent kill() will unblock av_read_frame.
        state.install(input.metadata, video_stream, audio_stream);

        let read_thread = ReadThread::new(input, state.clone(), read_msg_rx).run();

        let video_decoder_ptr =
            NonNull::new(unsafe { video_decoder.decoder.as_ptr() as _ }).unwrap();
        let video_thread = video::VideoThread::new(
            video_decoder,
            state.clone(),
            video_rx,
            video_frame_queue,
            video_msg_rx,
            read_msg_tx.clone(),
            video_clock.clone(),
            audio_clock.clone(),
        )
        .run();

        let audio_sink = AudioSink::new(
            state.clone(),
            audio_frame_queue.clone(),
            audio_msg_tx.clone(),
            audio_queue.clone(),
            audio_clock.clone(),
        );

        let audio_thread = audio_decoder
            .map(|audio_decoder| {
                AudioThread::new(
                    audio_decoder,
                    state.clone(),
                    audio_rx.clone(),
                    audio_frame_queue.clone(),
                    audio_msg_rx,
                )
                .run()
            })
            .unwrap_or_else(|| {
                let state = state.clone();
                std::thread::spawn(move || sink_thread(state, audio_rx))
            });

        Ok((
            Video {
                instance,
                adapter,
                device,
                queue,

                state,
                frame_decoder,
                video_decoder: video_decoder_ptr,
                read_thread: Some(read_thread),
                video_thread: Some(video_thread),
                audio_thread: Some(audio_thread),
                video_clock,
                audio_clock,
                has_audio,

                read_messages: read_msg_tx,

                hw_unsupported,

                looping: false,
                frame_timer: 0.0,
                last_pts: 0,
                last_serial: 0,
                queued_frame: None,
                step_needs_copy: 0,
            },
            audio_sink,
        ))
    }

    pub fn texture(&self) -> &wgpu::Texture {
        self.frame_decoder.texture()
    }

    /// Whether the first decoded frame has reached the GPU frame adapter.
    pub fn has_frame(&self) -> bool {
        self.frame_decoder.has_frame()
    }

    /// Opt the engine into sampling YUV planes directly (skipping the
    /// `copy_to_rgb` RGBA8 pass). Safe to call before the first frame import;
    /// the engine must only sample `yuv_bind_group()` when `direct_yuv()` is
    /// true, and must keep consuming `texture()` otherwise.
    pub fn set_direct_yuv(&mut self, enabled: bool) {
        self.frame_decoder.set_copy_to_rgb(!enabled);
    }

    /// YUV plane bind group for direct sampling. `Some` once a frame has been
    /// imported and the active adapter exposes planes.
    pub fn yuv_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.frame_decoder.yuv_bind_group()
    }

    /// YUV plane layout/format descriptor used to pick the engine-side
    /// YUV→RGB conversion (NV12, planar I420, YUV444, bit depth, ...).
    pub fn layout_identity(&self) -> Option<crate::context::layout::FrameDescriptor<()>> {
        self.frame_decoder.layout_identity()
    }

    /// Per-plane texture views for direct engine sampling (Y, then UV/planes).
    pub fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.frame_decoder.plane_views()
    }

    /// Release imported frames after the queue submission that sampled them.
    pub fn release_completed_frames(&mut self) -> Result<()> {
        self.frame_decoder.release_completed_frames(&self.device)
    }

    /// Whether direct YUV sampling is active (RGBA8 copy disabled and planes
    pub fn color_space(&self) -> ffn::color::Space {
        self.frame_decoder.color_space()
    }

    /// Color range detected from the stream metadata (MPEG/limited vs JPEG/full).
    pub fn color_range(&self) -> ffn::color::Range {
        self.frame_decoder.color_range()
    }

    /// Whether direct YUV sampling is active (RGBA8 copy disabled and planes
    /// are available).
    pub fn direct_yuv(&self) -> bool {
        self.frame_decoder.direct_yuv()
    }

    /// Drain GL interop tickets produced during the most recent
    /// `update`/`decode_frame`. Empty on non-interop backends.
    pub fn take_pending_gl_tickets(&mut self) -> Vec<frames::GlInteropTicket> {
        self.frame_decoder.take_pending_gl_tickets()
    }

    /// Order/flush the WGL interop GL work for the given tickets. MUST be
    /// called AFTER the `queue.submit` that contains the video draws, so the
    /// GL flush ordering is established after the wgpu copy reads the shared
    /// D3D11 texture. No-op (returns Ok) for empty tickets.
    pub fn finish_gl_frames(&mut self, tickets: &[frames::GlInteropTicket]) -> Result<()> {
        self.frame_decoder.finish_gl_frames(tickets)
    }

    fn update_frame(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        _queued_len: usize,
        wait_duration: &mut Duration,
    ) -> Result<FrameResponse> {
        if frame.serial
            != self
                .state
                .video_stream
                .load()
                .packets
                .metadata
                .serial
                .load(Ordering::SeqCst)
        {
            return Ok(FrameResponse::Retry);
        }

        let best_effort_timestamp = unsafe { (*frame.frame.as_ptr()).best_effort_timestamp };

        let video_info = self.video_info();

        let duration = if frame.serial == self.last_serial {
            (best_effort_timestamp as f64 * f64::from(video_info.time_base))
                - (self.last_pts as f64 * f64::from(video_info.time_base))
        } else {
            0.0
        };
        let duration = frame_duration_seconds(duration, video_info.framerate);

        // Update the wait duration so the background thread knows exactly how long to sleep
        *wait_duration = Duration::from_secs_f64(duration);

        let pts_sec = best_effort_timestamp as f64
            * f64::from(self.state.video_stream.load().metadata.time_base);
        self.video_clock.set(pts_sec, frame.serial, None);

        unsafe {
            // Read the current codec context through the shared atomic: the
            // VideoThread swaps decoders on hwaccel fallback, and the pointer
            // captured at open time would dangle once the old context is freed.
            let decoder_ptr = self.state.video_decoder.load(Ordering::Acquire);
            let decoder = if decoder_ptr.is_null() {
                self.video_decoder
            } else {
                NonNull::new_unchecked(decoder_ptr)
            };

            match self.frame_decoder.decode_frame(
                &self.instance,
                &self.adapter,
                &self.device,
                &self.queue,
                encoder,
                decoder,
                &frame.frame,
            ) {
                Ok(()) => {}
                Err(Error::UnsupportedBackend) | Err(Error::Probe(_)) => {
                    eprintln!("[Video] hardware decode not viable (probe), switching decoder");
                    self.hw_unsupported.store(true, Ordering::Relaxed);
                    return Ok(FrameResponse::Retry);
                }
                Err(e) => return Err(e),
            }
        };

        if self.state.play_state() == PlayState::Step {
            self.step_needs_copy = self.step_needs_copy.add(4).min(16);
            self.set_paused(true);
        }

        Ok(FrameResponse::Continue)
    }

    fn flush(&mut self) {
        if let Some(queued_frame) = self.queued_frame.take() {
            self.state.video_stream.load().frames.release(queued_frame);
        }

        self.state.video_stream.load().frames.flush();
        self.state.audio_stream.load().frames.flush();

        // TODO: flush samples in AudioSink
    }

    pub fn update(&mut self, encoder: &mut wgpu::CommandEncoder) -> Result<(Duration, bool)> {
        let play_state = self.state.play_state();

        if play_state == PlayState::Playing {
            self.step_needs_copy = 0;
        }

        if self.step_needs_copy > 0 && self.frame_decoder.copy_to_rgb_enabled {
            // NOTE: this is a bit of a hack;
            // on certain backends (namely D3D11VA) we have no way of syncing the D3D11 frame copy with the wgpu YUV->RGB copy.
            // (well, of course it is possible, but wgpu doesn't enable the necessary device extensions...)
            // to counteract this, when stepping a frame (e.g., during accurate seek while paused), we perform the YUV->RGB copy a few more times after the seek.
            // Skipped when direct YUV sampling is active (engine reads planes directly).
            self.step_needs_copy -= 1;
            // TODO: actually get the color space
            self.frame_decoder
                .copy_to_rgb(encoder, ffn::color::Space::BT709);
        }

        let video_frame_queue = self.state.video_stream.load().frames.clone();

        // Normal looping is handled solely by the read thread's EOF rewind
        // (decode/read.rs) so the loop stays gapless and decoder pools stay
        // put. BUT on the very first boundary the reader's `avformat_seek_file`
        // can fail at initial EOF, leaving is_eof stuck true and the queue
        // permanently empty — playback freezes at the last frame until the
        // user seeks. This fallback is exactly the "manual seek" that unsticks
        // it: a pure rewind message to the read thread. It fires at most once
        // (the handler clears is_eof), does NOT call self.flush(), so it does
        // not drain the frame queues or re-expand the decoder's frame-thread
        // buffer set (the previous memory leak).
        if self.queued_frame.is_none()
            && video_frame_queue.queued_len() == 0
            && self.state.is_eof.load(Ordering::SeqCst)
            && self.looping
            && play_state != PlayState::Paused
        {
            let _ = self.read_messages.send(ReadMessage::SeekStream {
                ts: 0,
                mode: SeekMode::Fast,
                forward: false,
            });
        }

        if play_state == PlayState::Paused
            || (self.queued_frame.is_none() && video_frame_queue.queued_len() == 0)
        {
            return Ok((Duration::from_millis(50), false));
        }

        let video_info = self.video_info();
        let time_base = f64::from(video_info.time_base);

        // Audio is the only external clock; no-audio playback uses frame duration.
        // Comparing loop-offset PTS values to a second wall clock can requeue
        // the first frame of a new loop.
        let sync_ref = if self.has_audio {
            self.audio_clock.get()
        } else {
            None
        };

        let mut duration = Duration::from_secs_f64(f64::from(video_info.framerate.invert()));
        let mut frame_decoded = false;
        loop {
            let queued_len = video_frame_queue.queued_len();
            let frame = self
                .queued_frame
                .take()
                .or_else(|| video_frame_queue.try_next());
            if let Some(frame) = frame {
                let current_serial = self
                    .state
                    .video_stream
                    .load()
                    .packets
                    .metadata
                    .serial
                    .load(Ordering::SeqCst);
                if frame.serial == current_serial {
                    if let Some(sync_ref_sec) = sync_ref {
                        let pts_sec = unsafe { (*frame.frame.as_ptr()).best_effort_timestamp }
                            as f64
                            * time_base;
                        // Clamp sync_ref to prevent runaway clock from evicting entire queue
                        let last_video_pts_sec = self.last_pts as f64 * time_base;
                        let sync_ref_sec = sync_ref_sec.min(last_video_pts_sec + 0.5);
                        let diff = pts_sec - sync_ref_sec;

                        // Audio is the master clock. Hold early frames and drop late
                        // frames only when a newer frame is available, matching the
                        // software consumer's policy without allowing queue backlog
                        // to become visible A/V lag.
                        if diff > Clock::SYNC_MAX {
                            let retry = (diff - Clock::SYNC_MAX).clamp(0.001, 0.050);
                            self.queued_frame = Some(frame);
                            duration = Duration::from_secs_f64(retry);
                            break;
                        }
                        if diff < -Clock::SYNC_MAX && queued_len > 0 {
                            self.last_pts =
                                unsafe { (*frame.frame.as_ptr()).best_effort_timestamp };
                            self.last_serial = frame.serial;
                            video_frame_queue.release(frame);
                            continue;
                        }
                    }
                }

                let response = self.update_frame(encoder, &frame, queued_len, &mut duration)?;
                match response {
                    FrameResponse::Continue => {
                        self.last_pts = unsafe { (*frame.frame.as_ptr()).best_effort_timestamp };
                        self.last_serial = frame.serial;
                        video_frame_queue.release(frame);
                        frame_decoded = true;
                        break;
                    }
                    FrameResponse::Retry => {
                        self.last_pts = unsafe { (*frame.frame.as_ptr()).best_effort_timestamp };
                        self.last_serial = frame.serial;
                        video_frame_queue.release(frame);
                    }
                    FrameResponse::Requeue => {
                        self.queued_frame = Some(frame);
                        break;
                    }
                }
            } else {
                break;
            }
        }

        Ok((duration, frame_decoded))
    }

    fn video_info(&self) -> video::VideoMetadata {
        self.state.video_stream.load().metadata
    }

    pub fn statistics(&self) -> Statistics {
        let video_clock = self.video_clock.get().unwrap_or(0.);
        let audio_clock = self.audio_clock.get().unwrap_or(0.);
        let sync_latency = video_clock - audio_clock;
        let decoder_name = self.decoder_name();

        Statistics {
            video_clock,
            audio_clock,
            sync_latency,
            decoder_name,
        }
    }

    #[inline]
    pub fn width(&self) -> u32 {
        self.video_info().width
    }

    #[inline]
    pub fn height(&self) -> u32 {
        self.video_info().height
    }

    #[inline]
    pub fn duration(&self) -> Duration {
        self.state.metadata.load().duration
    }

    #[inline]
    pub fn framerate(&self) -> f64 {
        let video_info = self.video_info();
        video_info.framerate.0 as f64 / video_info.framerate.1 as f64
    }

    /// Stream time base (used to convert packet PTS values into seconds
    /// for A/V sync and wall-clock math). Mirrors `video_rs::AsyncVideoDecoder::stream_time_base`.
    #[inline]
    pub fn stream_time_base(&self) -> ffn::Rational {
        self.video_info().time_base
    }

    /// Decoder time base (alias of `stream_time_base` — kept for
    /// compatibility with `video_rs::AsyncVideoDecoder::time_base`).
    #[inline]
    pub fn time_base(&self) -> ffn::Rational {
        self.video_info().time_base
    }

    /// Frame rate as f32. Mirrors `video_rs::AsyncVideoDecoder::frame_rate`.
    #[inline]
    pub fn frame_rate(&self) -> f32 {
        let fr = self.video_info().framerate;
        if fr.1 == 0 {
            0.0
        } else {
            fr.0 as f32 / fr.1 as f32
        }
    }

    /// Duration in milliseconds, or `None` if the file does not advertise one.
    /// Mirrors `video_rs::AsyncVideoDecoder::duration`.
    #[inline]
    pub fn duration_ms(&self) -> Option<f64> {
        let d = self.state.metadata.load().duration;
        if d.is_zero() {
            None
        } else {
            Some(d.as_secs_f64() * 1000.0)
        }
    }

    #[inline]
    pub fn decoder_name(&self) -> &'static str {
        self.frame_decoder
            .adapter
            .as_ref()
            .map(|adapter| adapter.name())
            .unwrap_or("Unknown")
    }

    /// Codec short name (e.g. "h264", "hevc", "vp9", "av1") read from the
    /// live codec context (the VideoThread swaps decoders on hwaccel
    /// fallback, so this is read through the shared atomic, never a stale
    /// pointer).
    pub fn codec_name(&self) -> String {
        let ptr = self.state.video_decoder.load(Ordering::Acquire);
        if ptr.is_null() {
            return "?".to_string();
        }
        let name = unsafe { ff::avcodec_get_name((*ptr).codec_id) };
        if name.is_null() {
            "?".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Software pixel format of the stream (e.g. "yuv420p", "nv12", "p010le")
    /// read from the live codec context's `sw_pix_fmt`.
    pub fn pixel_format_name(&self) -> String {
        let ptr = self.state.video_decoder.load(Ordering::Acquire);
        if ptr.is_null() {
            return "?".to_string();
        }
        let name = unsafe { ff::av_get_pix_fmt_name((*ptr).sw_pix_fmt) };
        if name.is_null() {
            "?".to_string()
        } else {
            unsafe { std::ffi::CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        }
    }

    /// Compact `codec pixfmt` descriptor for HUD labels (e.g. "vp9 yuv420p").
    pub fn format_label(&self) -> String {
        format!("{} {}", self.codec_name(), self.pixel_format_name())
    }

    /// Returns true if the hardware decoder rejected the video's pixel format
    /// and the VideoThread fell back to software decoding. The application
    /// should check this after the first frame is decoded and switch to a
    /// different software decoder (e.g. video-rs) if desired, since ffgpu's
    /// software path is slower than video-rs's async decoder.
    #[inline]
    pub fn is_software_fallback(&self) -> bool {
        self.hw_unsupported
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn position(&self) -> Duration {
        let secs = self.last_pts as f64 * f64::from(self.video_info().time_base)
            - crate::decode::container_start_seconds(&self.state);
        Duration::from_secs_f64(secs.max(0.0))
    }

    pub fn seek(&mut self, position: Duration, mode: SeekMode) {
        self.seek_with_direction(position, mode, false);
    }

    /// Forward-biased seek: lands on the *next* keyframe after the target
    /// instead of the previous one. Eliminates the expensive catch-up decode
    /// walk that follows a backward seek. Use this when the player is
    /// already behind real-time (the catch-up from the new keyframe is
    /// effectively free).
    pub fn seek_forward(&mut self, position: Duration, mode: SeekMode) {
        self.seek_with_direction(position, mode, true);
    }

    fn seek_with_direction(&mut self, position: Duration, mode: SeekMode, forward: bool) {
        let position = position.min(self.duration());
        let ts = crate::decode::seek_target_us(&self.state, position);

        if let Err(error) = self
            .read_messages
            .send(ReadMessage::SeekStream { ts, mode, forward })
        {
            log::error!("failed to send seek message: {}", error);
        }

        self.flush();
    }

    /// Set the codec-level discard level. VLC/MPV-style "hurry up": when
    /// the presentation thread is behind, raise the discard level so
    /// libavcodec skips non-reference or non-keyframe packets entirely.
    /// Far cheaper than decoding and throwing away, and safe for all
    /// codecs including AV1/dav1d (no EOF, no decoder reset).
    ///
    /// Pass [`DiscardLevel::Default`] once the player has caught up.
    pub fn set_discard(&self, level: crate::decode::video::DiscardLevel) {
        if let Err(error) = self
            .state
            .video_stream
            .load()
            .messages
            .send(crate::decode::video::Message::SetDiscard(level))
        {
            log::error!("failed to send set_discard message: {}", error);
        }
    }

    pub fn paused(&self) -> bool {
        self.state.play_state() != PlayState::Playing
    }

    pub fn set_paused(&mut self, paused: bool) {
        if !paused {
            self.frame_timer += ffn::time::relative() as f64 / 1000000.0
                - self.video_clock.last_updated.load(Ordering::Relaxed);
            self.video_clock.set(
                self.video_clock.get().unwrap_or(0.),
                self.video_clock.serial.load(Ordering::Relaxed),
                None,
            );
        }

        self.video_clock.paused.store(paused, Ordering::Relaxed);
        self.audio_clock.paused.store(paused, Ordering::Relaxed);

        self.state.play_state.store(
            if paused {
                PlayState::Paused
            } else {
                PlayState::Playing
            } as _,
            Ordering::Relaxed,
        );
    }

    pub fn looping(&self) -> bool {
        self.looping
    }

    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
        self.state.looping.store(looping, Ordering::SeqCst);
    }

    pub fn loop_generation(&self) -> Arc<AtomicU64> {
        self.state.loop_events.clone()
    }

    pub fn step_one_frame(&mut self) {
        self.set_paused(false);
        self.state
            .play_state
            .store(PlayState::Step as _, Ordering::Relaxed);
    }

    /// Signal all decoder threads to stop.
    ///
    /// This transitions the lifecycle to `Stopping` and sets
    /// `state.alive = false`, which the read/video/audio threads poll on
    /// every loop iteration. It does NOT block; call `wait_to_finish` to
    /// block until the threads have actually returned. The threads can
    /// take up to ~one buffer worth of work to exit (the read thread may
    /// be blocked in `av_read_frame`).
    pub fn kill(&mut self) {
        self.state.kill();
    }

    /// Block until all decoder threads have actually exited, or `timeout`
    /// elapses. Returns `true` if every thread finished, `false` on
    /// timeout. Panics if the decoder is still `Active` — `kill()` must
    /// be called first.
    ///
    /// This is the proper sync primitive for "I want this video gone
    /// before I construct the next one. The destructor performs the same
    /// shutdown sequence and joins all decoder threads before returning.
    pub fn wait_to_finish(&self, timeout: std::time::Duration) -> bool {
        self.state.wait_to_finish(timeout)
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        // Transition lifecycle Active → Stopping and signal threads.
        self.state.kill();

        // Packet consumers can be blocked in recv() while the read thread is
        // still inside FFmpeg. Wake them directly so shutdown does not depend
        // on the read thread reaching its normal end-of-loop sentinels first.
        let video_stream = self.state.video_stream.load().clone();
        video_stream
            .packets
            .push_null(ffn::Packet::empty(), video_stream.metadata.index);
        let audio_stream = self.state.audio_stream.load().clone();
        audio_stream
            .packets
            .push_null(ffn::Packet::empty(), audio_stream.metadata.index);
        self.flush();

        // Join the reader first: it owns the FFmpeg input and its interrupt
        // callback is what releases av_read_frame. The wake packets above let
        // the consumers observe alive=false without waiting for that join.
        if let Some(handle) = self.read_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.video_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.audio_thread.take() {
            let _ = handle.join();
        }
    }
}

fn frame_duration_seconds(pts_delta: f64, frame_rate: ffn::Rational) -> f64 {
    if pts_delta.is_finite() && pts_delta > 0.0 && pts_delta <= 3600.0 {
        pts_delta
    } else if frame_rate.0 > 0 && frame_rate.1 > 0 {
        f64::from(frame_rate.invert())
    } else {
        1.0 / 60.0
    }
}

#[cfg(test)]
mod tests {
    use super::frame_duration_seconds;
    use ffmpeg_next::Rational;

    #[test]
    fn invalid_pts_delta_uses_stream_frame_rate() {
        let expected = 1.0 / 30.0;
        assert!((frame_duration_seconds(0.0, Rational(30, 1)) - expected).abs() < f64::EPSILON);
        assert!((frame_duration_seconds(-1.0, Rational(30, 1)) - expected).abs() < f64::EPSILON);
        assert!(
            (frame_duration_seconds(f64::NAN, Rational(30, 1)) - expected).abs() < f64::EPSILON
        );
    }

    #[test]
    fn valid_pts_delta_is_preserved() {
        assert_eq!(frame_duration_seconds(0.04, Rational(30, 1)), 0.04);
    }
}
