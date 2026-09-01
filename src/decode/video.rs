use crate::{
    SeekMode,
    decode::{
        Clock, DecoderState, FrameQueue, PacketReceiver, PacketSender, PlayState,
        loop_timestamp_offset, packet_queue, read::ReadMessage,
    },
    error::{Error, Result},
};
use crossbeam_channel::{Receiver, Sender};
use ffmpeg_next::{self as ffn, sys as ff};
use std::{
    mem::ManuallyDrop,
    pin::Pin,
    ptr::{NonNull, null, null_mut},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
    time::Duration,
};

struct DecoderData {
    hw_pixel_format: ff::AVPixelFormat,
    unsupported: Arc<AtomicBool>,
}

unsafe extern "C" fn get_hw_format(
    decoder_ctx: *mut ff::AVCodecContext,
    mut px_fmts: *const ff::AVPixelFormat,
) -> ff::AVPixelFormat {
    unsafe {
        let decoder_data = ((*decoder_ctx).opaque as *mut DecoderData)
            .as_mut()
            .unwrap();
        while (*px_fmts) != ff::AVPixelFormat::AV_PIX_FMT_NONE {
            if (*px_fmts) == decoder_data.hw_pixel_format {
                return *px_fmts;
            }
            px_fmts = px_fmts.add(1);
        }
        decoder_data.unsupported.store(true, Ordering::Relaxed);
        ff::AVPixelFormat::AV_PIX_FMT_NONE
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoMetadata {
    pub index: usize,
    pub time_base: ffn::Rational,
    pub framerate: ffn::Rational,
    pub color_space: ffn::color::Space,
    pub color_range: ffn::color::Range,
    pub width: u32,
    pub height: u32,
}

impl Default for VideoMetadata {
    fn default() -> Self {
        Self {
            index: usize::MAX,
            time_base: ffn::Rational::new(0, 1),
            framerate: ffn::Rational::new(0, 1),
            color_space: ffn::color::Space::Unspecified,
            color_range: ffn::color::Range::Unspecified,
            width: 0,
            height: 0,
        }
    }
}

pub(crate) struct VideoStream {
    pub metadata: VideoMetadata,
    pub messages: Sender<Message>,
    pub packets: PacketSender,
    pub frames: FrameQueue,
}

impl VideoStream {
    /// Placeholder stream for `DecoderState::empty()`. The real stream
    /// is installed via `DecoderState::install()` before any thread is
    /// spawned, so this is only ever observed for the brief window
    /// between `empty()` and `install()`. Uses real channels so Send +
    /// Sync bounds are satisfied.
    pub(crate) fn dummy() -> Self {
        use crossbeam_channel::unbounded;
        let (packets, _packets_rx, _packets_meta) = packet_queue();
        VideoStream {
            metadata: VideoMetadata::default(),
            messages: unbounded().0,
            packets,
            frames: FrameQueue::new(1),
        }
    }
}

pub(crate) struct Decoder {
    pub decoder: ffn::decoder::Video,
    pub metadata: VideoMetadata,
    pub unsupported: Arc<AtomicBool>,
    pub format_ctx: NonNull<ff::AVFormatContext>,
    pub device_type: ff::AVHWDeviceType,
    _decoder_data: Option<Pin<Box<DecoderData>>>,
}

// Multi-threaded software decoder configuration
//
// Ported from video-rs's async_decode.rs. The key insight: ffgpu previously set
// `count: 0` (auto-detect) for ALL decoders, including software. FFmpeg's
// auto-detect is conservative (typically 4 threads). For 4K software decoding
// (e.g. when hwaccel rejects the video), we need explicit thread counts tuned
// per codec and resolution, matching Firefox/Chromium's approach.
//
// When hwaccel is active, threading is disabled entirely (the GPU handles
// parallelism). This also avoids thread-safety issues with hardware contexts.

/// Get CPU core count for thread calculation
fn get_cpu_cores() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Calculate optimal thread counts for software video decoding.
/// Matches Firefox's approach: maximize parallelism for 4K 60fps.
///
/// Returns `(frame_threads, tile_threads)` where `tile_threads` is 0 for
/// non-AV1 codecs (only dav1d uses tile threading).
fn calculate_thread_counts(
    codec_id: ffn::codec::Id,
    width: u32,
    height: u32,
) -> (usize, usize) {
    let cpu_cores = get_cpu_cores();
    let is_4k = width >= 3840 || height >= 2160;

    match codec_id {
        ffn::codec::Id::AV1 => {
            // Firefox-style thread distribution for dav1d:
            // Tile threads handle frame sub-sections (latency critical)
            // Frame threads handle multiple frames (throughput critical)
            let tile_threads = if is_4k { 4usize } else { 2usize };
            let frame_threads = (cpu_cores - tile_threads).max(4);
            eprintln!(
                "[ffgpu] AV1 threading: {} frame threads, {} tile threads ({} cores, {}K)",
                frame_threads,
                tile_threads,
                cpu_cores,
                if is_4k { "4" } else { "1080" }
            );
            (frame_threads, tile_threads)
        }
        ffn::codec::Id::H264 | ffn::codec::Id::HEVC | ffn::codec::Id::VP9 => {
            // Frame threading only (these codecs don't use tile threads)
            let frame_threads = if is_4k {
                cpu_cores.min(12).max(8)
            } else {
                cpu_cores.min(6).max(4)
            };
            eprintln!(
                "[ffgpu] {:?} threading: {} frame threads ({}K)",
                codec_id,
                frame_threads,
                if is_4k { "4" } else { "1080" }
            );
            (frame_threads, 0)
        }
        _ => {
            let frame_threads = cpu_cores.min(4).max(2);
            (frame_threads, 0)
        }
    }
}

/// Configure threading on a raw AVCodecContext for software decoding.
///
/// - Sets `thread_count` and `thread_type` via raw FFI (like ffplay/video-rs)
/// - For AV1/dav1d: sets `dav1d_frame_threads` and `dav1d_tile_threads` options
/// - Sets `AV_CODEC_FLAG2_FAST` (matches ffplay's `-fast` behavior)
///
/// # Safety
/// `ctx` must point to a valid AVCodecContext that has not yet been opened
/// (avcodec_open2 not yet called).
unsafe fn configure_software_threading(
    ctx: *mut ff::AVCodecContext,
    codec_id: ffn::codec::Id,
    width: u32,
    height: u32,
) {
    let (frame_threads, tile_threads) = calculate_thread_counts(codec_id, width, height);

    // Set thread type to Frame+Slice (FF_THREAD_FRAME | FF_THREAD_SLICE)
    // This matches video-rs's "Both" threading kind.
    unsafe {
        (*ctx).thread_type = (ff::FF_THREAD_FRAME | ff::FF_THREAD_SLICE) as i32;
    }

    if codec_id == ffn::codec::Id::AV1 {
        // For dav1d, set both standard thread_count and dav1d-specific options
        let total_threads = if tile_threads > 0 {
            frame_threads + tile_threads
        } else {
            frame_threads
        };
        unsafe {
            (*ctx).thread_count = total_threads as i32;
            ff::av_opt_set_int(
                ctx as *mut _,
                b"dav1d_frame_threads\0".as_ptr() as *const _,
                frame_threads as i64,
                0,
            );
            if tile_threads > 0 {
                ff::av_opt_set_int(
                    ctx as *mut _,
                    b"dav1d_tile_threads\0".as_ptr() as *const _,
                    tile_threads as i64,
                    0,
                );
            }
        }
    } else {
        // For H.264/HEVC/VP9 and others: just set thread_count
        unsafe {
            (*ctx).thread_count = frame_threads as i32;
        }
    }

    // Enable fast mode (matches ffplay's -fast flag)
    // Allows decoders to skip some expensive operations that don't affect
    // visual quality significantly (e.g. fast bilinear motion estimation).
    unsafe {
        (*ctx).flags2 |= ff::AV_CODEC_FLAG2_FAST as i32;
    }

    let actual = unsafe { (*ctx).thread_count };
    eprintln!(
        "[ffgpu] Software decoder threading configured: thread_count={}, codec={:?}, {}x{}",
        actual, codec_id, width, height
    );
}

impl Decoder {
    pub fn new(
        input: &mut ffn::format::context::Input,
        device_type: ff::AVHWDeviceType,
        hw_device_ctx: Option<NonNull<ff::AVBufferRef>>,
    ) -> Result<Self> {
        let video_stream = input
            .streams()
            .best(ffn::media::Type::Video)
            .ok_or(Error::InvalidStream)?;

        let video_stream_index = video_stream.index();

        let video_codec = video_stream.parameters().id();
        let decoder =
            ffn::decoder::find(video_codec).ok_or(Error::MissingCodec(video_codec.name()))?;

        // Get dimensions from stream parameters before decoder creation
        // (needed for thread count calculation, like video-rs does)
        let (stream_width, stream_height) = unsafe {
            let params = video_stream.parameters().as_ptr();
            ((*params).width as u32, (*params).height as u32)
        };

        let mut decoder_ctx = ffn::codec::Context::new_with_codec(decoder).decoder();
        unsafe { (*decoder_ctx.as_mut_ptr()).extra_hw_frames = 8 };
        decoder_ctx.set_parameters(video_stream.parameters())?;

        let mut is_hwaccel = device_type != ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
        let mut hw_pixel_format = ff::AVPixelFormat::AV_PIX_FMT_NONE;

        if is_hwaccel {
            for i in 0..16 {
                let Some(config) =
                    (unsafe { ff::avcodec_get_hw_config(decoder.as_ptr(), i).as_ref() })
                else {
                    break;
                };

                if (config.methods & ff::AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX as i32) != 0
                    && config.device_type == device_type
                {
                    hw_pixel_format = config.pix_fmt;
                    break;
                }
            }

            if hw_pixel_format == ff::AVPixelFormat::AV_PIX_FMT_NONE {
                log::error!(
                    "{:#?} does not support codec {}",
                    device_type,
                    video_codec.name()
                );
                log::warn!("using software decode");
                is_hwaccel = false;
            }
        }

        // Threading configuration:
        // - Hardware decoding: no CPU threading (GPU handles parallelism).
        //   We still set Frame threading with count=0 as before — FFmpeg will
        //   effectively use 1 thread for hwaccel since the decode happens on GPU.
        // - Software decoding: explicit multi-threaded config ported from video-rs
        //   (per-codec thread counts, dav1d options for AV1, AV_CODEC_FLAG2_FAST).
        if is_hwaccel {
            decoder_ctx.set_threading(ffn::threading::Config {
                kind: ffn::threading::Type::Frame,
                count: 0,
            });
        } else {
            // Software path: configure explicit multi-threading before open.
            // set_threading with count=0 first (FFmpeg will override via raw FFI),
            // then apply our explicit thread_count/thread_type/dav1d options.
            decoder_ctx.set_threading(ffn::threading::Config {
                kind: ffn::threading::Type::Frame,
                count: 0,
            });
            unsafe {
                configure_software_threading(
                    decoder_ctx.as_mut_ptr(),
                    video_codec,
                    stream_width,
                    stream_height,
                );
            }
        }

        let unsupported = Arc::new(AtomicBool::new(false));
        let decoder_data = if is_hwaccel {
            let mut decoder_data = Box::pin(DecoderData {
                hw_pixel_format,
                unsupported: unsupported.clone(),
            });
            unsafe {
                (*decoder_ctx.as_mut_ptr()).opaque = (&mut *decoder_data) as *mut _ as _;
                (*decoder_ctx.as_mut_ptr()).get_format = Some(get_hw_format);
            };

            // The codec context owns one ref on the hardware device context.
            //
            // Caller-supplied ctx (shared Vulkan hwctx): the caller keeps its
            // own ref (unref'd by ffgpu::Context::drop), so take a second one
            // here — avcodec_free_context balances it.
            //
            // Locally created ctx (D3D11VA/VAAPI): av_hwdevice_ctx_create
            // returned a fresh ref (count 1). Transfer it into the codec
            // context directly — avcodec_free_context unrefs it. Do NOT
            // av_buffer_ref here: the original ref would otherwise never be
            // unref'd and the whole AVHWDeviceContext (D3D11 device + video
            // device + immediate context) would leak on every video open —
            // the DX12/D3D11VA memory-leak this fixes.
            unsafe {
                if let Some(ctx) = hw_device_ctx {
                    (*decoder_ctx.as_mut_ptr()).hw_device_ctx = ff::av_buffer_ref(ctx.as_ptr());
                } else {
                    let mut hwctx = null_mut();
                    if device_type == ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA {
                        let mut opts: *mut ff::AVDictionary = null_mut();
                        let dict_ret = ff::av_dict_set(
                            &mut opts,
                            c"SHADER".as_ptr(),
                            c"1".as_ptr(),
                            0,
                        );
                        if dict_ret < 0 {
                            ff::av_dict_free(&mut opts);
                            return Err(Error::FFmpeg(dict_ret.into()));
                        }
                        let create_ret = ff::av_hwdevice_ctx_create(
                            &mut hwctx,
                            device_type,
                            null(),
                            opts,
                            0,
                        );
                        ff::av_dict_free(&mut opts);
                        if create_ret < 0 {
                            return Err(Error::FFmpeg(create_ret.into()));
                        }
                        eprintln!("[ffgpu] D3D11VA device created with SHADER option");
                    } else {
                        let create_ret = ff::av_hwdevice_ctx_create(
                            &mut hwctx,
                            device_type,
                            null(),
                            null_mut(),
                            0,
                        );
                        if create_ret < 0 {
                            return Err(Error::FFmpeg(create_ret.into()));
                        }
                    }
                    if hwctx.is_null() {
                        return Err(Error::HardwareContext);
                    }
                    (*decoder_ctx.as_mut_ptr()).hw_device_ctx = hwctx;
                }
            }

            Some(decoder_data)
        } else {
            None
        };

        let decoder_ctx = decoder_ctx.video()?;

        let width = decoder_ctx.width();
        let height = decoder_ctx.height();
        let color_space = decoder_ctx.color_space();
        let color_range = decoder_ctx.color_range();

        let metadata = VideoMetadata {
            index: video_stream_index,
            time_base: video_stream.time_base(),
            framerate: video_stream.avg_frame_rate(),
            color_space,
            color_range,
            width: width,
            height: height,
        };

        Ok(Decoder {
            decoder: decoder_ctx,
            metadata,
            unsupported,
            format_ctx: NonNull::new(unsafe { input.as_mut_ptr() }).unwrap(),
            device_type,
            _decoder_data: decoder_data,
        })
    }
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

pub(crate) enum Message {
    SkipToTimestamp(i64),
    /// Set the codec-level discard level (VLC "hurry up" mode). Cheap and
    /// safe to set mid-stream for all codecs including AV1/dav1d — unlike
    /// decoder reset, it does not corrupt reference frames.
    SetDiscard(DiscardLevel),
}

/// Codec-level discard level. Mirrors libavcodec's `AVDiscard` enum but
/// is exposed as a stable public type so callers don't depend on FFI.
///
/// Used to implement VLC/MPV-style "hurry up" catch-up: when the
/// presentation thread falls behind, raise the discard level so libavcodec
/// skips B-frames (or eventually everything but keyframes) to regain sync
/// without seeking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscardLevel {
    /// Decode every packet. Normal playback.
    Default,
    /// Drop non-reference packets (B-frames). ~30-50% CPU savings, no
    /// visible artefacts since B-frames are not referenced by any other frame.
    NonRef,
    /// Drop bidirectional packets. Subset of `NonRef` on most codecs;
    /// kept for completeness.
    Bidir,
    /// Drop non-keyframes. Severe judder but stream stays coherent.
    /// Used only when the player is many hundreds of ms behind.
    NonKey,
    /// Drop everything but intra frames. No frame at all for inter frames.
    NonIntra,
    /// Discard nothing. Identical to `Default` but kept for parity with
    /// the underlying FFmpeg API.
    None,
}

impl From<DiscardLevel> for ffn::codec::discard::Discard {
    #[inline]
    fn from(level: DiscardLevel) -> Self {
        match level {
            DiscardLevel::Default => ffn::codec::discard::Discard::Default,
            DiscardLevel::NonRef => ffn::codec::discard::Discard::NonReference,
            DiscardLevel::Bidir => ffn::codec::discard::Discard::Bidirectional,
            DiscardLevel::NonKey => ffn::codec::discard::Discard::NonKey,
            DiscardLevel::NonIntra => ffn::codec::discard::Discard::NonIntra,
            DiscardLevel::None => ffn::codec::discard::Discard::None,
        }
    }
}

pub(crate) struct VideoThread {
    decoder: Decoder,
    state: Arc<DecoderState>,
    video_rx: PacketReceiver,
    frame_queue: FrameQueue,
    messages: Receiver<Message>,
    read_messages: Sender<ReadMessage>,
    clock: Arc<Clock>,
    master_clock: Arc<Clock>,
}

impl VideoThread {
    pub fn new(
        decoder: Decoder,
        pbs: Arc<DecoderState>,
        video_rx: PacketReceiver,
        frame_queue: FrameQueue,
        messages: Receiver<Message>,
        read_messages: Sender<ReadMessage>,
        clock: Arc<Clock>,
        master_clock: Arc<Clock>,
    ) -> Self {
        VideoThread {
            decoder,
            state: pbs,
            video_rx,
            frame_queue,
            messages,
            read_messages,
            clock,
            master_clock,
        }
    }

    fn run_thread(&mut self) {
        eprintln!("[VideoThread] STARTED — device_type={:?}", self.decoder.device_type);
        // Publish the codec context so ffgpu's consumer (Video::update) never
        // dereferences the stale pointer captured at open time. Re-published
        // after every hwaccel-fallback decoder swap below.
        self.state
            .video_decoder
            .store(unsafe { self.decoder.decoder.as_mut_ptr() }, Ordering::Release);
        let mut packet_serial = 0;
        let mut packet_loop_index = 0;

        let mut frame = unsafe { ffn::Frame::empty() };

        let mut skip_to_ts = None;
        let mut diag_count: u32 = 0;

        // Pipeline timing (logged every 30 frames)
        let mut t_recv_sum: u64 = 0;
        let mut t_push_sum: u64 = 0;
        let mut frame_count: u32 = 0;

        'exit: while self.state.alive.load(Ordering::Relaxed) {
            if self.decoder.device_type != ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
                && self.decoder.unsupported.load(Ordering::Relaxed)
            {
                // Try D3D11VA first on Windows (works on Vulkan backend via shared handles).
                // If D3D11VA is already the current device type and it failed, fall back
                // to software decoding to avoid an infinite retry loop.
                #[cfg(target_os = "windows")]
                let new_device_type = if self.decoder.device_type == ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA {
                    ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE
                } else {
                    ff::AVHWDeviceType::AV_HWDEVICE_TYPE_D3D11VA
                };
                #[cfg(not(target_os = "windows"))]
                let new_device_type = ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;

                eprintln!("[VideoThread] hardware decode failed, trying {:?}", new_device_type);
                log::error!("hardware decode failed, trying {:?}", new_device_type);
                let mut input = unsafe {
                    ManuallyDrop::new(ffn::format::context::Input::wrap(
                        self.decoder.format_ctx.as_ptr(),
                    ))
                };
                match Decoder::new(&mut input, new_device_type, None) {
                    Ok(new_decoder) => {
                        self.decoder = new_decoder;
                        self.state.video_decoder.store(
                            unsafe { self.decoder.decoder.as_mut_ptr() },
                            Ordering::Release,
                        );
                    }
                    Err(e) => {
                        eprintln!("[VideoThread] decoder recreation with {:?} failed: {}, falling back to software", new_device_type, e);
                        if new_device_type != ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE {
                            match Decoder::new(&mut input, ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE, None) {
                                Ok(sw_decoder) => {
                                    self.decoder = sw_decoder;
                                    self.state.video_decoder.store(
                                        unsafe { self.decoder.decoder.as_mut_ptr() },
                                        Ordering::Release,
                                    );
                                }
                                Err(e2) => {
                                    eprintln!("[VideoThread] software decoder also failed: {}", e2);
                                    break 'exit;
                                }
                            }
                        } else {
                            break 'exit;
                        }
                    }
                }

                // The failed hardware decoder may have consumed packets from
                // the middle of the stream before the format was rejected.
                // Rewind the demuxer to the start so the replacement decoder
                // begins at frame 0 instead of the next keyframe (the visible
                // "starts at 0:04" jump). The read thread's seek handler
                // flushes both packet queues and bumps the serial, which
                // discards any stale hardware-era frames.
                let _ = self
                    .read_messages
                    .send(ReadMessage::SeekStream {
                        ts: 0,
                        mode: SeekMode::Fast,
                        forward: false,
                    });
                self.frame_queue.flush();
            }

            while let Ok(message) = self.messages.try_recv() {
                match message {
                    Message::SkipToTimestamp(ts) => {
                        skip_to_ts = Some(ts);
                    }
                    Message::SetDiscard(level) => {
                        // Cheap, mid-stream-safe. AV1/dav1d does not corrupt
                        // reference state when this changes (unlike draining).
                        self.decoder.decoder.skip_frame(level.into());
                    }
                }
            }

            let mut prev_frame = None;

            while self.state.alive.load(Ordering::Relaxed) {
                if self.state.play_state() == PlayState::Paused {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }

                if self.frame_queue.free_rx.is_empty() {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }

                let t_recv = std::time::Instant::now();
                let frame = match self.decoder.decoder.receive_frame(&mut frame) {
                    Ok(_) => {
                        if diag_count < 20 {
                            diag_count += 1;
                            let fmt = unsafe { (*frame.as_ptr()).format };
                            eprintln!("[VideoThread] receive_frame OK: format={} (VULKAN={})", fmt, ff::AVPixelFormat::AV_PIX_FMT_VULKAN as i32);
                        }
                        if let Some(pts) = frame.pts() {
                            let offset = loop_timestamp_offset(
                                &self.state,
                                packet_loop_index,
                                self.decoder.metadata.time_base,
                            );
                            let pts = pts.saturating_add(offset);
                            let best_effort_timestamp =
                                unsafe { (*frame.as_ptr()).best_effort_timestamp };
                            if offset != 0 {
                                unsafe {
                                    (*frame.as_mut_ptr()).pts = pts;
                                    if best_effort_timestamp != ff::AV_NOPTS_VALUE {
                                        (*frame.as_mut_ptr()).best_effort_timestamp =
                                            best_effort_timestamp.saturating_add(offset);
                                    }
                                }
                            }
                            let av_pts = unsafe {
                                ff::av_rescale_q(
                                    pts,
                                    self.decoder.metadata.time_base.into(),
                                    ff::AV_TIME_BASE_Q,
                                )
                            };

                            if let Some(ts) = skip_to_ts
                                && av_pts < ts
                            {
                                // discard frame
                                // but keep a ref in case next receive is EOF
                                // in which case TS is past last frame
                                let mut frame_ref = unsafe { ffn::Frame::empty() };
                                unsafe {
                                    ff::av_frame_move_ref(
                                        frame_ref.as_mut_ptr(),
                                        frame.as_mut_ptr(),
                                    )
                                };
                                prev_frame = Some(frame_ref);
                                continue;
                            } else {
                                prev_frame = None;
                            }

                            let pts_sec = pts as f64 * f64::from(self.decoder.metadata.time_base);


                        }
                        Some(&mut frame)
                    }
                    Err(ffn::Error::Eof) => {
                        if diag_count < 20 { eprintln!("[VideoThread] receive_frame EOF"); }
                        if let Some(mut prev_frame) = prev_frame.take() {
                            unsafe {
                                ff::av_frame_move_ref(frame.as_mut_ptr(), prev_frame.as_mut_ptr())
                            };
                            Some(&mut frame)
                        } else {
                            self.decoder.decoder.flush();
                            break;
                        }
                    }
                    Err(ffn::Error::Other { errno: ff::EAGAIN }) => {
                        if diag_count < 20 { eprintln!("[VideoThread] receive_frame EAGAIN"); }
                        break;
                    }
                    Err(ref e) => {
                        if diag_count < 20 {
                            diag_count += 1;
                            eprintln!("[VideoThread] receive_frame ERROR: {:?}", e);
                        }
                        // Treat like EAGAIN: without a new packet the decoder
                        // will not recover, and retrying in a tight loop here
                        // would spin one core at 100% on a corrupt stream.
                        break;
                    }
                };

                if let Some(frame) = frame {
                    prev_frame = None;

                    let mut step = false;
                    let presentation_pts = unsafe {
                        let best_effort = (*frame.as_ptr()).best_effort_timestamp;
                        if best_effort != ff::AV_NOPTS_VALUE {
                            best_effort
                        } else {
                            frame.pts().unwrap_or(ff::AV_NOPTS_VALUE)
                        }
                    };
                    if presentation_pts != ff::AV_NOPTS_VALUE {
                        self.state
                            .current_pts
                            .store(presentation_pts, Ordering::Relaxed);
                        if skip_to_ts.is_some() {
                            step = self.state.play_state() == PlayState::Paused;
                        }
                    }
                    skip_to_ts = None;

                    let t_push = std::time::Instant::now();
                    if !self
                        .frame_queue
                        .send(frame, packet_serial, &self.state.alive)
                    {
                        break 'exit;
                    }
                    let recv_us = t_recv.elapsed().as_micros() as u64;
                    let push_us = t_push.elapsed().as_micros() as u64;
                    t_recv_sum += recv_us;
                    t_push_sum += push_us;
                    frame_count += 1;
                    if frame_count >= 30 {
                        let avg_recv = t_recv_sum as f64 / frame_count as f64 / 1000.0;
                        let avg_push = t_push_sum as f64 / frame_count as f64 / 1000.0;
                        eprintln!(
                            "[VideoThread] Pipe avg ({} frames): recv_frame={:.2}ms push={:.2}ms",
                            frame_count, avg_recv, avg_push
                        );
                        t_recv_sum = 0;
                        t_push_sum = 0;
                        frame_count = 0;
                    }
                    if step {
                        self.state
                            .play_state
                            .store(PlayState::Step as _, Ordering::Relaxed);
                    }
                }
            }

            let packet = loop {
                if !self.state.alive.load(Ordering::Relaxed) {
                    break 'exit;
                }

                let Some(packet) = self.video_rx.receive() else {
                    continue;
                };

                if packet_serial != packet.serial {
                    self.decoder.decoder.flush();
                    packet_serial = packet.serial;
                    packet_loop_index = packet.loop_index;
                } else if packet_loop_index != packet.loop_index {
                    self.decoder.decoder.flush();
                    packet_loop_index = packet.loop_index;
                }

                if packet_serial == self.video_rx.metadata.serial.load(Ordering::Relaxed) {
                    break packet;
                }
            };

            let is_eof_packet = packet.packet.data().is_none();
            if is_eof_packet {
                if let Err(error) = self.decoder.decoder.send_eof() {
                    eprintln!("[VideoThread] send_eof ERROR: {}", error);
                    log::error!("failed to send EOF to video decoder: {}", error);
                }
            } else if let Err(error) = self.decoder.decoder.send_packet(&packet.packet) {
                eprintln!("[VideoThread] send_packet ERROR: {}", error);
                log::error!("failed to send packet: {}", error);
            }
        }
    }

    pub fn run(mut self) -> JoinHandle<()> {
        let guard =
            crate::decode::ThreadGuard::new(self.state.thread_count.clone());
        std::thread::spawn(move || {
            let _guard = guard;
            self.run_thread();
        })
    }
}
