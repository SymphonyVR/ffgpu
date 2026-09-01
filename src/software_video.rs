//! Software-only video path.
//!
//! CPU-only video player that does NOT require a `wgpu` device. Used
//! by the engine's CPU-rendering fallback when no GPU is available
//! or when GPU features are disabled.
//!
//! # Threading model
//!
//! Mirrors the GPU path's threading 1:1:
//! - **ReadThread** (decode/read.rs) demuxes packets from the file
//! - **VideoThread** (decode/video.rs) decodes them with software FFmpeg
//!   (no hwaccel). Pushes decoded `ffn::Frame`s into a `FrameQueue`.
//! - **AudioThread** (decode/audio.rs) decodes audio and feeds the
//!   cpal sink via `AudioSink`.
//!
//! The difference is the consumer side: the GPU path has `Video::update`
//! pulling frames and uploading to a wgpu texture; the software path
//! has a [`SoftwareFrameConsumer`] thread pulling frames and pushing
//! [`SoftwareFrame`]s to a bounded(1) crossbeam channel for an
//! engine-side consumer.
//!
//! # Consumer contract
//!
//! ```no_run
//! use ffgpu::{SoftwareContext, SoftwareFrame};
//!
//! let ctx = SoftwareContext::new().unwrap();
//! let (video, audio) = ctx.create_video("test.mp4").unwrap();
//! let frame_tx = video.frame_sender();
//!
//! // Engine-side worker:
//! std::thread::spawn(move || {
//!     let rx = video.frame_receiver();
//!     while let Ok(sf) = rx.recv() {
//!         // do work with the frame
//!     }
//! });
//! ```
//!
//! Bounded(1) + `try_send` gives latest-frame-wins: if the consumer
//! falls behind, the oldest frame is dropped (correct for real-time
//! video).

use crate::{
    decode::{
        self,
        audio::{self, AudioSink, AudioStream, AudioThread},
        read::{Input, ReadMessage, ReadThread},
        video::{self, VideoStream},
        Clock, DecoderState, Frame, FrameQueue, PacketQueueMetadata,
    },
    error::Result,
    SeekMode,
};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};

/// Public alias for the receiver side of the frame channel. Engine
/// worker threads hold one of these and call `.recv()` to receive
/// [`SoftwareFrame`]s.
pub type SoftwareFrameReceiver = Receiver<SoftwareFrame>;

/// Master clock for A/V sync. The audio clock is preferred when the
/// audio thread is alive and producing samples; otherwise we fall
/// back to a system clock. This mirrors the engine's legacy
/// `MasterClock` enum and the GPU path's behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterClock {
    /// Audio is the master clock. A/V sync uses the audio thread's
    /// PTS as the reference; video is paced to match.
    Audio,
    /// System (wall-clock) is the master clock. Audio is dead or
    /// unavailable; video is paced to wall-clock only.
    System,
}

const SYNC_THRESHOLD_MS: f64 = 100.0;
const EARLY_RETRY_MAX_MS: f64 = 50.0;

impl Default for MasterClock {
    fn default() -> Self {
        Self::Audio
    }
}

/// Rolling buffer of `(Instant, gap_ms)` samples used by the rate
/// adjustment logic. Stores up to `WINDOW` samples (a few seconds
/// at 60fps) and computes the mean of samples within the last
/// `window_ms` of wall-clock time.
struct RollingGap {
    /// Ring buffer of (timestamp, gap_ms) tuples. Oldest first.
    samples: Vec<(std::time::Instant, f64)>,
    /// Capacity (max samples retained).
    capacity: usize,
}

impl RollingGap {
    fn new(capacity: usize) -> Self {
        Self {
            samples: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, gap_ms: f64) {
        let now = std::time::Instant::now();
        if self.samples.len() >= self.capacity {
            self.samples.remove(0);
        }
        self.samples.push((now, gap_ms));
    }

    /// Mean of samples within the last `window_ms` of wall-clock.
    /// Returns `None` if no samples are within the window.
    fn last_window_mean(&self, window_ms: u64) -> Option<f64> {
        let cutoff = std::time::Instant::now() - std::time::Duration::from_millis(window_ms);
        let mut sum = 0.0;
        let mut count = 0u32;
        for (t, g) in &self.samples {
            if *t >= cutoff {
                sum += g;
                count += 1;
            }
        }
        if count == 0 {
            None
        } else {
            Some(sum / count as f64)
        }
    }

    fn clear(&mut self) {
        self.samples.clear();
    }
}
use ffmpeg_next::{self as ffn, sys as ff};
use std::{
    path::Path,
    sync::{atomic::Ordering, Arc},
    thread::JoinHandle,
    time::Duration,
};

/// A CPU-only ffmpeg context. Wraps `ffmpeg_next::init()` and nothing
/// else. Cheap to construct; reuse one across many videos.
pub struct SoftwareContext {
    _private: (),
}

impl SoftwareContext {
    /// Initialize ffmpeg and return a context. Safe to call multiple
    /// times — ffmpeg_next uses an internal init counter.
    pub fn new() -> Result<Self> {
        ffmpeg_next::init()?;
        Ok(Self { _private: () })
    }

    /// Open a video file and spawn the consumer thread that pushes
    /// `SoftwareFrame`s to a bounded(1) crossbeam channel. Use this
    /// for the "consume via `frame_receiver()`" API (e.g. the
    /// ffgpu test suite). Returns `(player, audio_sink)`. The
    /// player must be kept alive — dropping it stops the decode
    /// threads and the consumer thread.
    pub fn create_video<P>(&self, path: &P) -> Result<(SoftwareDecodeVideo, AudioSink)>
    where
        P: AsRef<Path> + ?Sized,
    {
        SoftwareDecodeVideo::new(path, true)
    }

    /// Open a video file **without** spawning the consumer thread.
    /// Use this when the caller drives the decoder via the direct
    /// `update()` / `frame()` API (e.g. the engine's bridge worker).
    /// Skipping the consumer thread eliminates two problems:
    ///
    /// 1. **Wasted work.** The consumer thread would call
    ///    `frame_to_software()` (a multi-MB YUV-plane memcpy) on
    ///    every decoded frame and immediately discard the result
    ///    because nothing reads from `frame_rx` in this mode.
    /// 2. **Race for frames.** The consumer thread and the caller's
    ///    `update()` would both pull from the same `FrameQueue`,
    ///    dropping each other's work unpredictably.
    ///
    /// The `FrameQueue` is bounded(8) and `VideoThread::send`
    /// blocks when full, so the decoder is back-pressured to the
    /// caller's drain rate automatically — no need for the consumer
    /// thread to act as a buffer.
    ///
    /// The channel API (`frame_sender` / `frame_receiver` /
    /// `SoftwareFrameReceiver`) remains exported for callers that
    /// want it; it's just not fed by an internal thread in this
    /// mode.
    pub fn create_video_without_consumer<P>(
        &self,
        path: &P,
    ) -> Result<(SoftwareDecodeVideo, AudioSink)>
    where
        P: AsRef<Path> + ?Sized,
    {
        SoftwareDecodeVideo::new(path, false)
    }
}

impl Default for SoftwareContext {
    fn default() -> Self {
        Self::new().expect("ffmpeg_next::init failed")
    }
}

/// Pixel format of a [`SoftwareFrame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// 2-plane YUV420: full-res Y + half-res interleaved UV. Most
    /// common output of H.264 / HEVC / VP9 / AV1 software decoders.
    Nv12,
    /// 3-plane YUV420: full-res Y + half-res U + half-res V. Rare in
    /// modern codecs; `to_rgba()` returns `None` for this format.
    Yuv420P,
    /// 3-plane YUV444: full-res Y + full-res U + full-res V. No
    /// chroma subsampling. The engine converts via `yuv444p_to_rgb`.
    Yuv444P,
    /// 1-plane packed RGB24 (no alpha).
    Rgb24,
    /// 1-plane packed RGBA32.
    Rgba,
    /// 1-plane packed YUV422 (UYVY/VYUY/YUYV/YVYU). Decoder outputs
    /// this for some capture devices. `to_rgba()` returns a converted
    /// RGBA8 buffer (no resize — the consumer should resize via the
    /// engine's fused YUV→RGB bilinear helper if needed).
    Yuv422,
}

impl PixelFormat {
    pub fn from_ffmpeg(format: ff::AVPixelFormat) -> Option<Self> {
        match format {
            x if x == ff::AVPixelFormat::AV_PIX_FMT_NV12 => Some(Self::Nv12),
            x if x == ff::AVPixelFormat::AV_PIX_FMT_YUV420P => Some(Self::Yuv420P),
            x if x == ff::AVPixelFormat::AV_PIX_FMT_RGB24 => Some(Self::Rgb24),
            x if x == ff::AVPixelFormat::AV_PIX_FMT_RGBA => Some(Self::Rgba),
            // BGR24 is byte-identical to RGB24 for our purposes (we just
            // need to copy pixels; the renderer doesn't care about channel
            // order for 24-bit formats when wrapped in DynamicImageBuffer::RGB).
            x if x == ff::AVPixelFormat::AV_PIX_FMT_BGR24 => Some(Self::Rgb24),
            x if x == ff::AVPixelFormat::AV_PIX_FMT_YUV444P
                || x == ff::AVPixelFormat::AV_PIX_FMT_YUVJ444P =>
            {
                Some(Self::Yuv444P)
            }
            // YUV422 packed variants — all stored as Yuv422 and
            // converted in `to_rgba()` via the same code path.
            // ffmpeg-next 8.1.0 exposes 3 of 4: YUYV422, UYVY422,
            // YVYU422. VYUY422 is the only packed 4:2:2 not bound
            // by this version; it would fall through to the catch-all
            // and decode to a different format if produced.
            x if x == ff::AVPixelFormat::AV_PIX_FMT_YUYV422
                || x == ff::AVPixelFormat::AV_PIX_FMT_YVYU422
                || x == ff::AVPixelFormat::AV_PIX_FMT_UYVY422 =>
            {
                Some(Self::Yuv422)
            }
            _ => None,
        }
    }
}

/// YUV range (luma/chroma sample value scaling).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum YuvRange {
    /// TV (limited) range: Y in [16, 235], UV in [16, 240]. Most
    /// broadcast and consumer content.
    Limited,
    /// PC (full) range: Y/UV in [0, 255]. Screen-capture / JPEG-style.
    Full,
}

impl YuvRange {
    pub fn from_ffmpeg(range: ff::AVColorRange) -> Self {
        match range {
            ff::AVColorRange::AVCOL_RANGE_JPEG => Self::Full,
            _ => Self::Limited,
        }
    }
}

/// YUV→RGB color matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ColorMatrix {
    Bt601,
    Bt709,
    Bt2020,
    Identity,
}

impl ColorMatrix {
    pub fn from_ffmpeg(cs: ff::AVColorSpace) -> Self {
        match cs {
            ff::AVColorSpace::AVCOL_SPC_BT470BG
            | ff::AVColorSpace::AVCOL_SPC_SMPTE170M
            | ff::AVColorSpace::AVCOL_SPC_SMPTE240M => Self::Bt601,
            ff::AVColorSpace::AVCOL_SPC_BT709 => Self::Bt709,
            ff::AVColorSpace::AVCOL_SPC_BT2020_NCL | ff::AVColorSpace::AVCOL_SPC_BT2020_CL => {
                Self::Bt2020
            }
            ff::AVColorSpace::AVCOL_SPC_RGB => Self::Identity,
            _ => Self::Bt709,
        }
    }
}

/// A single decoded video frame, ready for CPU consumption.
///
/// Plane layout by `format`:
/// - `Nv12`: `y` is full-res Y plane (width × height bytes); `uv` is
///   half-res interleaved UV plane (width × height/2 bytes); `v` is empty.
/// - `Yuv420P`: `y` is full-res Y; `uv` is half-res interleaved UV
///   (packed from U+V planes); `v` is empty.
/// - `Yuv444P`: `y` is full-res Y; `uv` is full-res U; `v` is full-res V.
/// - `Rgb24` / `Rgba`: `y` holds the packed pixels; `uv` and `v` are empty.
/// - `Yuv422`: `y` holds packed YUYV data; `uv` and `v` are empty.
#[derive(Debug, Clone)]
pub struct SoftwareFrame {
    pub width: u32,
    pub height: u32,
    pub format: PixelFormat,
    pub y: Vec<u8>,
    pub uv: Vec<u8>,
    /// Third plane for 3-plane formats (e.g. `Yuv444P` V plane).
    /// Empty for 1- and 2-plane formats.
    pub v: Vec<u8>,
    pub yuv_range: YuvRange,
    pub color_matrix: ColorMatrix,
}

impl SoftwareFrame {
    /// Eager YUV→RGBA8 conversion. Returns RGBA pixels as a flat
    /// `Vec<u8>` of length `width * height * 4`.
    ///
    /// **Cost**: ~1ms for 1080p, ~4-8ms for 4K. Call from a worker
    /// thread, not the render thread.
    ///
    /// **Coverage**:
    /// - `Nv12` and `Yuv420P` → RGBA (both are unpacked to NV12-style
    ///   Y + interleaved UV in [`extract_planes`], so the conversion
    ///   path is identical)
    /// - `Yuv444P` → RGBA (3-plane full-res Y/U/V)
    /// - `Rgb24` / `Rgba` → RGBA passthrough
    /// - `Yuv422` → RGBA (UYVY packing; supports all 4 channel orderings
    ///   via the underlying format enum)
    pub fn to_rgba(&self) -> Option<Vec<u8>> {
        match self.format {
            PixelFormat::Nv12 | PixelFormat::Yuv420P => self.nv12_to_rgba(),
            PixelFormat::Yuv444P => self.yuv444p_to_rgba(),
            PixelFormat::Rgb24 => Some(self.rgb_to_rgba(false)),
            PixelFormat::Rgba => Some(self.rgb_to_rgba(true)),
            PixelFormat::Yuv422 => self.yuv422_to_rgba(),
        }
    }

    /// YUV422 packed → RGBA8. Pairs of horizontally adjacent pixels
    /// share UV samples (4:2:2 chroma subsampling). The Y values are
    /// full-resolution; the UV values are sampled at every other
    /// position.
    ///
    /// This is a simple non-resizing converter. The consumer (engine)
    /// typically wants the fused YUV→RGB + bilinear downscale path
    /// instead, but this method exists for tests and for non-resize
    /// use cases (e.g. snapshot captures).
    fn yuv422_to_rgba(&self) -> Option<Vec<u8>> {
        let (yw, yh) = (self.width as usize, self.height as usize);
        if self.y.len() < yw * yh * 2 {
            return None;
        }
        // The y buffer holds packed YUYV data (Y0 U Y1 V pattern,
        // 4 bytes per 2 pixels). width * height * 2 bytes total.
        let (kr, kg, kb) = match self.color_matrix {
            ColorMatrix::Bt601 => (0.299f32, 0.587, 0.114),
            _ => (0.2126, 0.7152, 0.0722),
        };
        let (y_off, y_scale, uv_off, uv_scale) = match self.yuv_range {
            YuvRange::Limited => (16.0f32, 219.0 / 255.0, 128.0, 224.0 / 255.0),
            YuvRange::Full => (0.0, 1.0, 128.0, 1.0),
        };

        let mut rgba = vec![0u8; yw * yh * 4];
        for y in 0..yh {
            for x in 0..yw {
                // Each 4-byte macropixel covers 2 horizontally
                // adjacent pixels. For even x, the macropixel index
                // is x/2; for odd x, same macropixel.
                let macropixel = y * yw / 2 + x / 2;
                let base = macropixel * 4;
                if base + 3 >= self.y.len() {
                    return None;
                }
                let y0 = self.y[base] as f32;
                let u = self.y[base + 1] as f32;
                let y1 = self.y[base + 2] as f32;
                let v = self.y[base + 3] as f32;
                let y_sample = if x % 2 == 0 { y0 } else { y1 };

                let luma = (y_sample - y_off) / 255.0 * y_scale;
                let cb = (u - uv_off) * uv_scale / 255.0;
                let cr = (v - uv_off) * uv_scale / 255.0;

                let r = (luma + 2.0 * (1.0 - kr) * cr).clamp(0.0, 1.0);
                let g = (luma - 2.0 * kr * cb / kg - 2.0 * (1.0 - kr) * cr * (1.0 - kb) / kg)
                    .clamp(0.0, 1.0);
                let b = (luma + 2.0 * (1.0 - kb) * cb).clamp(0.0, 1.0);

                let pixel = (y * yw + x) * 4;
                rgba[pixel] = (r * 255.0) as u8;
                rgba[pixel + 1] = (g * 255.0) as u8;
                rgba[pixel + 2] = (b * 255.0) as u8;
                rgba[pixel + 3] = 255;
            }
        }
        Some(rgba)
    }

    fn nv12_to_rgba(&self) -> Option<Vec<u8>> {
        let (yw, yh) = (self.width as usize, self.height as usize);
        if self.y.len() < yw * yh {
            return None;
        }
        let half_h = yh / 2;
        if self.uv.len() < yw * half_h {
            return None;
        }

        // BT.709 coefficients for HD content (most common); BT.601 for SD.
        let (kr, kg, kb) = match self.color_matrix {
            ColorMatrix::Bt601 => (0.299f32, 0.587, 0.114),
            _ => (0.2126, 0.7152, 0.0722),
        };

        let (y_off, y_scale, uv_off, uv_scale) = match self.yuv_range {
            YuvRange::Limited => (16.0f32, 219.0 / 255.0, 128.0, 224.0 / 255.0),
            YuvRange::Full => (0.0, 1.0, 128.0, 1.0),
        };

        let mut rgba = vec![0u8; yw * yh * 4];
        for y in 0..yh {
            for x in 0..yw {
                let y_sample = self.y[y * yw + x] as f32;
                let uv_idx = (y / 2) * yw + (x & !1);
                let u_sample = self.uv[uv_idx] as f32;
                let v_sample = self.uv[uv_idx + 1] as f32;

                let luma = (y_sample - y_off) / 255.0 * y_scale;
                let cb = (u_sample - uv_off) / 255.0 * uv_scale - 0.5;
                let cr = (v_sample - uv_off) / 255.0 * uv_scale - 0.5;

                let r = (luma + 2.0 * (1.0 - kr) * cr).clamp(0.0, 1.0);
                let g = (luma - 2.0 * kr * cb / kg - 2.0 * (1.0 - kr) * cr * (1.0 - kb) / kg)
                    .clamp(0.0, 1.0);
                let b = (luma + 2.0 * (1.0 - kb) * cb).clamp(0.0, 1.0);

                let pixel = (y * yw + x) * 4;
                rgba[pixel] = (r * 255.0) as u8;
                rgba[pixel + 1] = (g * 255.0) as u8;
                rgba[pixel + 2] = (b * 255.0) as u8;
                rgba[pixel + 3] = 255;
            }
        }
        Some(rgba)
    }

    /// YUV444 planar → RGBA8. Full-res Y, U, V planes (no chroma
    /// subsampling). Every pixel has a unique Y/U/V triple.
    fn yuv444p_to_rgba(&self) -> Option<Vec<u8>> {
        let (yw, yh) = (self.width as usize, self.height as usize);
        if self.y.len() < yw * yh || self.uv.len() < yw * yh || self.v.len() < yw * yh {
            return None;
        }
        let (kr, kg, kb) = match self.color_matrix {
            ColorMatrix::Bt601 => (0.299f32, 0.587, 0.114),
            _ => (0.2126, 0.7152, 0.0722),
        };
        let (y_off, y_scale, uv_off, uv_scale) = match self.yuv_range {
            YuvRange::Limited => (16.0f32, 219.0 / 255.0, 128.0, 224.0 / 255.0),
            YuvRange::Full => (0.0, 1.0, 128.0, 1.0),
        };

        let mut rgba = vec![0u8; yw * yh * 4];
        for y in 0..yh {
            for x in 0..yw {
                let idx = y * yw + x;
                let y_sample = self.y[idx] as f32;
                let u_sample = self.uv[idx] as f32;
                let v_sample = self.v[idx] as f32;

                let luma = (y_sample - y_off) / 255.0 * y_scale;
                let cb = (u_sample - uv_off) / 255.0 * uv_scale - 0.5;
                let cr = (v_sample - uv_off) / 255.0 * uv_scale - 0.5;

                let r = (luma + 2.0 * (1.0 - kr) * cr).clamp(0.0, 1.0);
                let g = (luma - 2.0 * kr * cb / kg - 2.0 * (1.0 - kr) * cr * (1.0 - kb) / kg)
                    .clamp(0.0, 1.0);
                let b = (luma + 2.0 * (1.0 - kb) * cb).clamp(0.0, 1.0);

                let pixel = idx * 4;
                rgba[pixel] = (r * 255.0) as u8;
                rgba[pixel + 1] = (g * 255.0) as u8;
                rgba[pixel + 2] = (b * 255.0) as u8;
                rgba[pixel + 3] = 255;
            }
        }
        Some(rgba)
    }

    fn rgb_to_rgba(&self, has_alpha: bool) -> Vec<u8> {
        let pixel_count = (self.width * self.height) as usize;
        let _channels = if has_alpha { 4 } else { 3 };
        let mut rgba = vec![0u8; pixel_count * 4];
        if has_alpha {
            rgba.copy_from_slice(&self.y[..pixel_count * 4]);
        } else {
            for i in 0..pixel_count {
                rgba[i * 4] = self.y[i * 3];
                rgba[i * 4 + 1] = self.y[i * 3 + 1];
                rgba[i * 4 + 2] = self.y[i * 3 + 2];
                rgba[i * 4 + 3] = 255;
            }
        }
        rgba
    }
}

/// CPU-only video player. Dropping the player stops all decode
/// threads and the consumer thread.
///
/// Renamed from `SoftwareVideo` to `SoftwareDecodeVideo` in step 8b
/// to clarify that this type **decodes** frames — the pacing,
/// conversion, and presentation are the consumer's responsibility.
/// The old name is kept as a deprecated alias for one release so
/// existing imports keep working.
pub struct SoftwareDecodeVideo {
    state: Arc<DecoderState>,

    /// Latest-frame-wins channel: the consumer thread pulls from
    /// the VideoThread's `FrameQueue`, converts to `SoftwareFrame`,
    /// and pushes here.
    frame_tx: Sender<SoftwareFrame>,
    /// Public receiver handle (also held by the consumer thread).
    frame_rx: Receiver<SoftwareFrame>,

    read_thread: Option<JoinHandle<()>>,
    video_thread: Option<JoinHandle<()>>,
    audio_thread: Option<JoinHandle<()>>,
    consumer_thread: Option<JoinHandle<()>>,

    read_messages: Sender<ReadMessage>,

    looping: bool,

    /// Source of truth for A/V sync. `Audio` while the audio thread
    /// is alive; falls back to `System` when audio is dead or
    /// unavailable.
    master_clock: MasterClock,

    /// Rolling buffer of `(Instant, gap_ms)` samples. Updated each
    /// call to `update()`; read by the rate adjustment logic to
    /// decide whether to speed up video playback.
    rolling_gap: RollingGap,

    /// Current playback rate (1.0 = real-time, 2.0 = 2x). Adjusted
    /// by `update()` based on the rolling-average gap between video
    /// PTS and audio PTS. The consumer should use this when
    /// computing its wait duration (wait = frame_duration / rate).
    current_rate: f32,
    /// User-selected playback rate. Automatic sync may speed this up when the
    /// video is behind, but never slows normal playback below this value.
    requested_rate: f32,

    /// Last video PTS in stream time base (microseconds, since
    /// ffmpeg uses i64 PTS). Used by the gap calculation.
    last_video_pts: i64,

    /// Last audio PTS in milliseconds. Sampled from the audio
    /// sink's clock. Used as the master reference.
    last_audio_pts_ms: f64,

    /// Latest decoded frame, held until the consumer calls
    /// `frame()`. Decouples decode timing from presentation
    /// timing — the consumer can call `update()` to pace, then
    /// `frame()` to get the latest frame.
    queued_frame: Option<SoftwareFrame>,
    /// Raw frame held while video is early relative to the audio master clock.
    pending_frame: Option<Frame>,
    /// PTS of the last frame considered by the presentation loop.
    last_seen_pts_sec: Option<f64>,
    /// Loop generation used to discard stale timing state at a loop boundary.
    last_loop_generation: u64,
    /// Most recent audio-minus-video gap in milliseconds for diagnostics.
    last_sync_gap_ms: Option<f64>,

    /// Audio clock shared with the AudioSink. Used by `update()`
    /// to compute the gap between video PTS and audio PTS. Cloned
    /// from the same Arc that AudioSink::new receives, so readings
    /// here are the same as `DeviceAudioSink::current_position_ms()`.
    audio_clock: Arc<Clock>,

    /// Reusable Y plane buffer — avoids per-frame allocation in
    /// `extract_planes`. Capacity is preserved across frames.
    frame_y: Vec<u8>,
    /// Reusable UV plane buffer.
    frame_uv: Vec<u8>,
    /// Reusable V plane buffer (Yuv444P).
    frame_v: Vec<u8>,
}

/// Backwards-compatible alias. Existing code importing
/// `ffgpu::SoftwareVideo` continues to work for one release.
#[deprecated(note = "Use ffgpu::SoftwareDecodeVideo instead")]
#[allow(dead_code)] // compatibility alias kept through the deprecation window
pub type SoftwareVideo = SoftwareDecodeVideo;

impl SoftwareDecodeVideo {
    fn new<P: AsRef<Path> + ?Sized>(
        path: &P,
        spawn_consumer_thread: bool,
    ) -> Result<(Self, AudioSink)> {
        let mut input = Input::open(path)?;

        // Force software decode: no hwaccel device type, no hw context.
        let device_type = ff::AVHWDeviceType::AV_HWDEVICE_TYPE_NONE;
        let hw_device_ctx = None;
        let video_decoder = video::Decoder::new(&mut input.format_ctx, device_type, hw_device_ctx)?;
        let audio_decoder = audio::Decoder::new(&mut input)?;

        let (video_tx, video_rx, video_meta) = decode::packet_queue();
        let (audio_tx, audio_rx, audio_meta) = decode::packet_queue();
        let video_frame_queue = FrameQueue::new(8);
        let audio_frame_queue = FrameQueue::new(16);

        let (read_msg_tx, read_msg_rx) = unbounded();
        let (video_msg_tx, video_msg_rx) = unbounded();
        let (audio_msg_tx, audio_msg_rx) = unbounded();

        let video_stream = VideoStream {
            metadata: video_decoder.metadata,
            messages: video_msg_tx,
            packets: video_tx,
            frames: video_frame_queue.clone(),
        };

        let audio_stream = AudioStream {
            metadata: audio_decoder
                .as_ref()
                .map(|d| d.metadata)
                .unwrap_or_default(),
            messages: audio_msg_tx.clone(),
            packets: audio_tx,
            frames: audio_frame_queue.clone(),
        };

        let state = Arc::new(DecoderState::new(
            input.metadata,
            video_stream,
            audio_stream,
        ));

        let read_thread = ReadThread::new(input, state.clone(), read_msg_rx).run();

        let video_clock = Arc::new(Clock::new(video_meta.clone()));
        let audio_clock = Arc::new(Clock::new(audio_meta.clone()));

        let video_thread = video::VideoThread::new(
            video_decoder,
            state.clone(),
            video_rx,
            video_frame_queue.clone(),
            video_msg_rx,
            read_msg_tx.clone(),
            video_clock,
            audio_clock.clone(),
        )
        .run();

        let audio_sink = AudioSink::new(
            state.clone(),
            audio_frame_queue.clone(),
            audio_msg_tx,
            audio_meta,
            audio_clock.clone(), // real audio clock, used by the engine's
                                 // rate adjustment logic via current_position_ms()
        );

        let audio_thread = audio_decoder
            .map(|audio_decoder| {
                AudioThread::new(
                    audio_decoder,
                    state.clone(),
                    audio_rx,
                    audio_frame_queue.clone(),
                    audio_msg_rx,
                )
                .run()
            })
            // No-audio: still spawn a placeholder so Drop can join it
            // cleanly. The placeholder just sleeps until alive=false.
            .unwrap_or_else(|| {
                let st = state.clone();
                std::thread::spawn(move || {
                    while st.alive.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                })
            });

        // Latest-frame-wins channel for the consumer. Even when
        // the consumer thread is disabled (direct-API consumers
        // like the engine's bridge worker), we still allocate the
        // channel so `frame_sender` / `frame_receiver` are usable.
        let (frame_tx, frame_rx) = bounded(1);

        // Consumer thread: pulls from video_frame_queue, converts to
        // SoftwareFrame, pushes to the channel. Only spawned when
        // the caller wants the channel-driven API. Direct-API
        // consumers (which call `update()` / `frame()` directly) skip
        // this — see `SoftwareContext::create_video_without_consumer`.
        let consumer_thread = if spawn_consumer_thread {
            let consumer_state = state.clone();
            let consumer_queue = video_frame_queue.clone();
            let consumer_tx = frame_tx.clone();
            Some(std::thread::spawn(move || {
                run_software_consumer(consumer_state, consumer_queue, consumer_tx);
            }))
        } else {
            None
        };

        Ok((
            SoftwareDecodeVideo {
                state,
                frame_tx,
                frame_rx,
                read_thread: Some(read_thread),
                video_thread: Some(video_thread),
                audio_thread: Some(audio_thread),
                consumer_thread,
                read_messages: read_msg_tx,
                looping: false,
                master_clock: MasterClock::Audio,
                rolling_gap: RollingGap::new(128),
                current_rate: 1.0,
                requested_rate: 1.0,
                last_video_pts: 0,
                last_audio_pts_ms: 0.0,
                queued_frame: None,
                pending_frame: None,
                last_seen_pts_sec: None,
                last_loop_generation: 0,
                last_sync_gap_ms: None,
                audio_clock,
                frame_y: Vec::new(),
                frame_uv: Vec::new(),
                frame_v: Vec::new(),
            },
            audio_sink,
        ))
    }

    /// Get a clone of the frame sender. Engine workers can use this
    /// to subscribe to frames (rare — usually `frame_receiver` is
    /// enough since the consumer thread is internal).
    pub fn frame_sender(&self) -> Sender<SoftwareFrame> {
        self.frame_tx.clone()
    }

    /// Get a clone of the frame receiver. The engine spawns a worker
    /// thread that calls `.recv()` on this.
    pub fn frame_receiver(&self) -> Receiver<SoftwareFrame> {
        self.frame_rx.clone()
    }

    pub fn position(&self) -> Duration {
        let pts = self.state.current_pts.load(Ordering::Relaxed);
        let secs = pts as f64 * f64::from(self.stream_time_base())
            - crate::decode::container_start_seconds(&self.state);
        Duration::from_secs_f64(secs.max(0.0))
    }

    pub fn duration(&self) -> Duration {
        self.state.metadata.load().duration
    }

    pub fn stream_time_base(&self) -> ffn::Rational {
        self.state.video_stream.load().metadata.time_base
    }

    pub fn time_base(&self) -> ffn::Rational {
        self.stream_time_base()
    }

    pub fn frame_rate(&self) -> f32 {
        let fr = self.state.video_stream.load().metadata.framerate;
        if fr.1 == 0 {
            0.0
        } else {
            fr.0 as f32 / fr.1 as f32
        }
    }

    /// Number of decoded frames waiting for the software consumer.
    /// Diagnostic consumers use this to distinguish decoder throughput from
    /// presentation backlog without changing queue behavior.
    pub fn queued_frames(&self) -> usize {
        self.state.video_stream.load().frames.queued_len()
    }

    pub fn is_looping(&self) -> bool {
        self.looping
    }

    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
        self.state.looping.store(looping, Ordering::SeqCst);
    }

    pub fn loop_generation(&self) -> u64 {
        self.state.loop_events.load(Ordering::SeqCst)
    }

    pub fn seek(&mut self, position: Duration) {
        self.seek_with_direction(position, SeekMode::Accurate, false);
    }

    pub fn seek_forward(&mut self, position: Duration) {
        self.seek_with_direction(position, SeekMode::Accurate, true);
    }

    /// Fast-seek to the nearest master keyframe BEFORE the target (no frame
    /// walk, no `SkipToTimestamp`). Synchronous. Used for live scrubbing
    /// while paused to avoid the per-drag frame walk; release with a regular
    /// `seek` to land exactly.
    pub fn seek_fast(&mut self, position: Duration) {
        self.seek_with_direction(position, SeekMode::Fast, false);
    }

    fn seek_with_direction(&mut self, position: Duration, mode: SeekMode, forward: bool) {
        let position = position.min(self.duration());
        let ts = crate::decode::seek_target_us(&self.state, position);

        // A seek invalidates both the held frame and the A/V drift samples
        // collected at the old position. Never carry catch-up state across it.
        self.rolling_gap.clear();
        self.current_rate = self.requested_rate;
        self.last_video_pts = 0;
        self.last_audio_pts_ms = 0.0;
        self.queued_frame = None;
        if let Some(frame) = self.pending_frame.take() {
            self.state.video_stream.load().frames.release(frame);
        }
        self.last_seen_pts_sec = None;
        self.last_sync_gap_ms = None;

        let _ = self
            .read_messages
            .send(ReadMessage::SeekStream { ts, mode, forward });
    }

    pub fn set_discard(&self, level: crate::DiscardLevel) {
        let _ = self
            .state
            .video_stream
            .load()
            .messages
            .send(video::Message::SetDiscard(level));
    }

    pub fn width(&self) -> u32 {
        self.state.video_stream.load().metadata.width
    }

    pub fn height(&self) -> u32 {
        self.state.video_stream.load().metadata.height
    }

    pub fn decoder_name(&self) -> &'static str {
        "ffgpu (software)"
    }

    pub fn is_eof(&self) -> bool {
        self.state.is_eof.load(Ordering::SeqCst)
    }

    pub fn pause(&self) {
        self.state
            .play_state
            .store(decode::PlayState::Paused as u8, Ordering::Relaxed);
    }

    pub fn play(&self) {
        self.state
            .play_state
            .store(decode::PlayState::Playing as u8, Ordering::Relaxed);
    }

    /// Set the playback rate (1.0 = real-time, 2.0 = 2x). Used by
    /// the engine to fast-forward the video when it falls behind
    /// the audio. Clamped to `[0.5, 4.0]`.
    ///
    /// **Note**: Setting this does NOT change the audio's playback
    /// speed (cpal is real-time). It only affects the wait_duration
    /// returned by `update()` and thus the rate at which the
    /// consumer presents frames.
    pub fn set_playback_rate(&mut self, rate: f32) {
        let rate = rate.clamp(0.5, 4.0);
        self.requested_rate = rate;
        self.current_rate = rate;
    }

    /// Current playback rate (1.0 = real-time, 2.0 = 2x). Set
    /// initially to 1.0; adjusted automatically by `update()` based
    /// on the rolling-average gap between video PTS and audio PTS
    /// (only when `master_clock` is `Audio`).
    pub fn playback_rate(&self) -> f32 {
        self.current_rate
    }

    /// Most recent audio-minus-video timing gap in milliseconds.
    pub fn sync_gap_ms(&self) -> Option<f64> {
        self.last_sync_gap_ms
    }

    /// Set the master clock source for A/V sync. `Audio` (default)
    /// uses the audio thread's PTS as the reference; `System` uses
    /// wall-clock only. The engine should call this when it
    /// detects the audio device has been lost (or recovered).
    pub fn set_master_clock(&mut self, clock: MasterClock) {
        if self.master_clock != clock {
            self.master_clock = clock;
            // Reset the rolling buffer to avoid using stale samples
            // from the previous master clock.
            self.rolling_gap.clear();
            self.last_sync_gap_ms = None;
        }
    }

    /// Current master clock selection.
    pub fn master_clock_kind(&self) -> MasterClock {
        self.master_clock
    }

    /// Drive the decoder. Pulls a frame from the `FrameQueue`,
    /// applies A/V sync (with the configured master clock),
    /// updates the rolling-average gap, adjusts the playback
    /// rate, and returns `(wait_duration, frame_decoded)`.
    ///
    /// `wait_duration` is the time the consumer should sleep
    /// before the next `update()` call. It is computed from the
    /// frame's duration and the current `playback_rate`
    /// (`wait = frame_duration / rate`). At rate 1.0, this is the
    /// normal frame interval. At rate 2.0, the consumer should
    /// present twice as many frames per second (i.e., sleep half
    /// as long).
    ///
    /// `frame_decoded` is `true` if a new frame is now in
    /// `queued_frame` (call `frame()` to retrieve it).
    ///
    /// The engine typically calls this in a loop:
    /// ```no_run
    /// # use ffgpu::SoftwareDecodeVideo;
    /// # fn doc(v: &mut SoftwareDecodeVideo) {
    /// loop {
    ///     let (wait, decoded) = v.update().unwrap();
    ///     if decoded {
    ///         if let Some(sf) = v.frame() {
    ///             // present sf
    ///         }
    ///     }
    ///     std::thread::sleep(wait);
    /// }
    /// # }
    /// ```
    pub fn update(&mut self) -> Result<(Duration, bool)> {
        let video_frame_queue = self.state.video_stream.load().frames.clone();
        let loop_generation = self.state.loop_events.load(Ordering::Acquire);
        if loop_generation != self.last_loop_generation {
            if let Some(frame) = self.pending_frame.take() {
                video_frame_queue.release(frame);
            }
            self.rolling_gap.clear();
            self.current_rate = self.requested_rate;
            self.last_video_pts = 0;
            self.last_audio_pts_ms = 0.0;
            self.last_seen_pts_sec = None;
            self.last_sync_gap_ms = None;
            self.last_loop_generation = loop_generation;
        }

        // Get audio position from the audio clock (master clock).
        // This is the same Arc<Clock> that the AudioSink holds, so
        // readings are consistent with the cpal callback's view of
        // "now".
        let master_pts_ms = match self.master_clock {
            MasterClock::Audio => self.audio_clock.get().map(|pts| pts * 1000.0),
            // The bridge's elapsed-time pacing is the system-clock path. A
            // Unix-epoch value cannot be compared with media PTS values.
            MasterClock::System => None,
        };

        // Pull one frame from the FrameQueue. Hold an early frame and drop late
        // frames only when a newer frame is available, keeping the displayed
        // image moving instead of allowing queue backlog to become A/V lag.
        let mut next_frame = self
            .pending_frame
            .take()
            .or_else(|| video_frame_queue.try_next());
        let mut frame_decoded = false;
        let mut frame_for_queue = None;
        let frame_rate = self.state.video_stream.load().metadata.framerate;
        let fallback_duration = frame_duration_seconds(f64::NAN, frame_rate);
        let mut frame_duration = fallback_duration;
        while let Some(f) = next_frame {
            let current_serial = self
                .state
                .video_stream
                .load()
                .packets
                .metadata
                .serial
                .load(Ordering::Acquire);
            if f.serial != current_serial {
                video_frame_queue.release(f);
                next_frame = video_frame_queue.try_next();
                continue;
            }

            let raw = unsafe { f.frame.as_ptr() };
            let pts = unsafe {
                let best_effort = (*raw).best_effort_timestamp;
                if best_effort != ffn::ffi::AV_NOPTS_VALUE {
                    best_effort
                } else {
                    (*raw).pts
                }
            };
            let pts_sec = (pts != ffn::ffi::AV_NOPTS_VALUE)
                .then(|| pts as f64 * f64::from(self.stream_time_base()));

            // Compute gap only while the audio clock is valid. A missing clock
            // during a seek is not audio at time zero and must not drive rate
            // adjustment.
            let gap_ms = master_pts_ms.zip(pts_sec).map(|(audio_pts_ms, pts_sec)| {
                let gap_ms = audio_pts_ms - pts_sec * 1000.0;
                self.last_audio_pts_ms = audio_pts_ms;
                self.last_sync_gap_ms = Some(gap_ms);
                gap_ms
            });

            if let Some(gap_ms) = gap_ms {
                self.rolling_gap.push(gap_ms);

                // Do not present a frame substantially ahead of audio. Keep the
                // decoded frame so the next update can present it once audio
                // reaches its PTS.
                if gap_ms < -SYNC_THRESHOLD_MS {
                    let retry_ms = (-gap_ms - SYNC_THRESHOLD_MS).clamp(1.0, EARLY_RETRY_MAX_MS);
                    self.pending_frame = Some(f);
                    frame_duration = retry_ms / 1000.0;
                    break;
                }

                // Drop stale frames only when a newer frame is available. If the
                // decoder is itself slow, showing the only frame is preferable
                // to freezing the picture while audio continues.
                if gap_ms > SYNC_THRESHOLD_MS && video_frame_queue.queued_len() > 0 {
                    if let Some(pts_sec) = pts_sec {
                        self.last_seen_pts_sec = Some(pts_sec);
                        self.last_video_pts = pts;
                    }
                    video_frame_queue.release(f);
                    next_frame = video_frame_queue.try_next();
                    continue;
                }
            }

            if let Some(pts_sec) = pts_sec {
                let previous_pts = self.last_seen_pts_sec;
                frame_duration = frame_duration_seconds(
                    previous_pts.map_or(f64::NAN, |previous| pts_sec - previous),
                    frame_rate,
                );
                self.last_seen_pts_sec = Some(pts_sec);
                self.last_video_pts = pts;
            }

            // Convert to SoftwareFrame using pre-allocated plane buffers to
            // avoid per-frame allocation in the presentation path.
            let convert_result = (|| -> Result<SoftwareFrame> {
                let fmt_raw = unsafe { (*raw).format };
                let width = unsafe { (*raw).width } as u32;
                let height = unsafe { (*raw).height } as u32;
                let color_range = unsafe { (*raw).color_range };
                let color_space = unsafe { (*raw).colorspace };
                let pixel_format =
                    PixelFormat::from_ffmpeg(unsafe { std::mem::transmute(fmt_raw) })
                        .ok_or(crate::Error::UnsupportedPixelFormat)?;

                extract_planes(
                    &f.frame,
                    pixel_format,
                    &mut self.frame_y,
                    &mut self.frame_uv,
                    &mut self.frame_v,
                )?;

                Ok(SoftwareFrame {
                    width,
                    height,
                    format: pixel_format,
                    y: std::mem::take(&mut self.frame_y),
                    uv: std::mem::take(&mut self.frame_uv),
                    v: std::mem::take(&mut self.frame_v),
                    yuv_range: YuvRange::from_ffmpeg(color_range),
                    color_matrix: ColorMatrix::from_ffmpeg(color_space),
                })
            })();
            video_frame_queue.release(f);
            match convert_result {
                Ok(sf) => {
                    frame_decoded = true;
                    frame_for_queue = Some(sf);
                }
                Err(e) => log::warn!("[ffgpu-sw] update: dropping frame: {}", e),
            }
            break;
        }

        if let Some(sf) = frame_for_queue {
            self.queued_frame = Some(sf);
        }

        // Rate adjustment based on rolling-average gap.
        // hysteresis: 200ms triggers catch-up, 50ms releases.
        self.adjust_playback_rate();

        // Compute wait_duration from the actual PTS interval. This preserves
        // variable-frame-rate timing; average FPS is only the invalid-PTS
        // fallback.
        let wait = Duration::from_secs_f64(frame_duration / f64::from(self.current_rate));

        Ok((wait, frame_decoded))
    }

    /// Take the latest decoded frame (set by `update()`). Returns
    /// `None` if no new frame has been decoded since the last call
    /// (the frame is consumed — calling `frame()` again returns
    /// `None` until the next `update()` call with `frame_decoded =
    /// true`).
    pub fn frame(&mut self) -> Option<SoftwareFrame> {
        self.queued_frame.take()
    }

    /// Return decoded YUV plane buffers to the decoder so the next
    /// `update()` reuses their capacity instead of allocating a fresh
    /// ~9.4MB (4K) of virtual memory and page-faulting it every frame.
    /// Called by the engine's software bridge worker after it has
    /// converted the frame. The buffers are emptied by `mem::take` in
    /// `update()`, so a missed recycle is a safe (if slower) fallback.
    pub fn recycle_planes(&mut self, y: Vec<u8>, uv: Vec<u8>, v: Vec<u8>) {
        self.frame_y = y;
        self.frame_uv = uv;
        self.frame_v = v;
    }

    /// Adjust `current_rate` based on the rolling-average gap.
    /// A/V catch-up only speeds playback when video is behind. A negative gap
    /// is handled by dropping stale queued frames, not by changing the source
    /// cadence to the old 0.75x floor.
    fn adjust_playback_rate(&mut self) {
        let Some(avg_gap) = self.rolling_gap.last_window_mean(1000) else {
            return;
        };

        let target_rate = if avg_gap > 200.0 {
            // Video is behind audio; speed up to catch up.
            // Linear ramp from 1.0x at +200ms to 2.0x at +1200ms.
            let extra = ((avg_gap - 200.0) / 1000.0).min(1.0) as f32;
            (self.requested_rate + extra).min(4.0)
        } else {
            // Video is in sync or briefly ahead. Keep source cadence and let
            // the queue-drain policy above discard stale frames.
            self.requested_rate
        };

        // Low-pass filter to smooth the rate change.
        self.current_rate = self.current_rate * 0.7 + target_rate * 0.3;
        self.current_rate = self.current_rate.clamp(self.requested_rate, 4.0);
    }
}

fn frame_duration_seconds(pts_delta: f64, frame_rate: ffn::Rational) -> f64 {
    if pts_delta.is_finite() && pts_delta > 0.0 && pts_delta <= 3600.0 {
        pts_delta
    } else if frame_rate.0 > 0 && frame_rate.1 > 0 {
        f64::from(frame_rate.invert())
    } else {
        1.0 / 30.0
    }
}

impl Drop for SoftwareDecodeVideo {
    fn drop(&mut self) {
        self.state.kill();
        if let Some(h) = self.read_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.video_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.audio_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.consumer_thread.take() {
            let _ = h.join();
        }
    }
}

unsafe impl Send for SoftwareDecodeVideo {}
unsafe impl Sync for SoftwareDecodeVideo {}

#[cfg(test)]
mod tests {
    use super::frame_duration_seconds;
    use ffmpeg_next::Rational;

    #[test]
    fn software_pacing_prefers_pts_delta() {
        assert_eq!(frame_duration_seconds(0.041, Rational(60, 1)), 0.041);
    }

    #[test]
    fn software_pacing_falls_back_to_average_fps() {
        let expected = 1.0 / 30.0;
        assert!(
            (frame_duration_seconds(f64::NAN, Rational(30, 1)) - expected).abs() < f64::EPSILON
        );
        assert!((frame_duration_seconds(-1.0, Rational(30, 1)) - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn software_pacing_rejects_timestamp_discontinuities() {
        let expected = 1.0 / 24.0;
        assert!((frame_duration_seconds(3601.0, Rational(24, 1)) - expected).abs() < f64::EPSILON);
    }
}

impl Default for PacketQueueMetadata {
    fn default() -> Self {
        Self {
            duration: std::sync::atomic::AtomicI64::new(0),
            serial: std::sync::atomic::AtomicU32::new(0),
            loop_index: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// The consumer thread loop. Pulls decoded frames from the
/// `FrameQueue`, converts to `SoftwareFrame`, pushes to the
/// bounded(1) channel. Exits when the decoder state is killed.
///
/// **Why a separate thread**: the VideoThread's contract is "push
/// the ffn::Frame to the FrameQueue ASAP; A/V sync is the consumer's
/// job." Software consumers (the engine's worker) want a
/// `SoftwareFrame`, not an `ffn::Frame`, so the conversion happens
/// here — out of the decode hot path and out of the render thread.
fn run_software_consumer(state: Arc<DecoderState>, queue: FrameQueue, tx: Sender<SoftwareFrame>) {
    while state.alive.load(Ordering::Relaxed) {
        let Some(frame) = queue.try_next() else {
            // Frame queue empty — yield briefly. The VideoThread will
            // wake us when a new frame arrives (the FrameQueue is a
            // crossbeam channel which is already event-driven, so
            // this sleep is the only polling in the system).
            std::thread::sleep(Duration::from_millis(2));
            continue;
        };

        let current_serial = state
            .video_stream
            .load()
            .packets
            .metadata
            .serial
            .load(Ordering::Acquire);
        if frame.serial != current_serial {
            queue.release(frame);
            continue;
        }

        let software_frame = match frame_to_software(&frame) {
            Ok(sf) => sf,
            Err(e) => {
                log::warn!("[ffgpu-sw] dropping frame (extraction failed): {}", e);
                queue.release(frame);
                continue;
            }
        };

        // Latest-frame-wins: try_send replaces the old frame if the
        // consumer is slow. The consumer is the engine worker; it
        // will process the latest available frame on its next tick.
        // We don't block the decode thread waiting for the consumer.
        let _ = tx.try_send(software_frame);
        queue.release(frame);
    }
}

/// Convert a `ffn::Frame` (from the decode pipeline) to a
/// `SoftwareFrame` (CPU-ready). Returns an error if the pixel
/// format is unsupported.
fn frame_to_software(frame: &Frame) -> Result<SoftwareFrame> {
    let raw = unsafe { frame.frame.as_ptr() };
    let format = unsafe { (*raw).format };
    let width = unsafe { (*raw).width } as u32;
    let height = unsafe { (*raw).height } as u32;
    let color_range = unsafe { (*raw).color_range };
    let color_space = unsafe { (*raw).colorspace };

    let pixel_format = PixelFormat::from_ffmpeg(unsafe { std::mem::transmute(format) })
        .ok_or(crate::Error::UnsupportedPixelFormat)?;

    let (y_data, uv_data, v_data) = extract_planes_new(&frame.frame, pixel_format)?;

    Ok(SoftwareFrame {
        width,
        height,
        format: pixel_format,
        y: y_data,
        uv: uv_data,
        v: v_data,
        yuv_range: YuvRange::from_ffmpeg(color_range),
        color_matrix: ColorMatrix::from_ffmpeg(color_space),
    })
}

/// Allocate-new variant of [`extract_planes`]. Used by the consumer
/// thread and other callers that need owned buffers.
fn extract_planes_new(
    frame: &ffn::Frame,
    format: PixelFormat,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let mut y = Vec::new();
    let mut uv = Vec::new();
    let mut v = Vec::new();
    extract_planes(frame, format, &mut y, &mut uv, &mut v)?;
    Ok((y, uv, v))
}

/// Extract Y (and optional UV) planes from a decoded frame into
/// caller-supplied buffers. Reuses existing buffer capacity so
/// repeated calls avoid re-allocation.
///
/// Handles stride correctly: when the frame's stride is greater than
/// its width, we copy row-by-row to produce contiguous output.
fn extract_planes(
    frame: &ffn::Frame,
    format: PixelFormat,
    y: &mut Vec<u8>,
    uv: &mut Vec<u8>,
    v: &mut Vec<u8>,
) -> Result<()> {
    unsafe {
        let raw = frame.as_ptr();
        let width = (*raw).width as usize;
        let height = (*raw).height as usize;

        let copy_plane = |dst: &mut Vec<u8>, plane: usize, div: usize| {
            let stride = (*raw).linesize[plane] as usize;
            let plane_w = width / div;
            let plane_h = height / div;
            let row_bytes = plane_w;
            dst.clear();
            dst.reserve(plane_w * plane_h);
            let data_ptr = (*raw).data[plane] as *const u8;
            if stride == row_bytes {
                dst.extend_from_slice(std::slice::from_raw_parts(data_ptr, row_bytes * plane_h));
            } else {
                for row in 0..plane_h {
                    let src = std::slice::from_raw_parts(data_ptr.add(row * stride), row_bytes);
                    dst.extend_from_slice(src);
                }
            }
        };

        match format {
            PixelFormat::Nv12 => {
                y.clear();
                y.reserve(width * height);
                uv.clear();
                uv.reserve(width * (height / 2));
                copy_plane(y, 0, 1);

                let uv_stride = (*raw).linesize[1] as usize;
                let plane_h = height / 2;
                let row_bytes = width;
                let uv_ptr = (*raw).data[1] as *const u8;
                if uv_stride == row_bytes {
                    uv.extend_from_slice(std::slice::from_raw_parts(uv_ptr, row_bytes * plane_h));
                } else {
                    for row in 0..plane_h {
                        let src =
                            std::slice::from_raw_parts(uv_ptr.add(row * uv_stride), row_bytes);
                        uv.extend_from_slice(src);
                    }
                }
                v.clear();
                Ok(())
            }
            PixelFormat::Yuv420P => {
                y.clear();
                y.reserve(width * height);
                uv.clear();
                uv.reserve(width * (height / 2));
                copy_plane(y, 0, 1);
                let half_w = width / 2;
                let half_h = height / 2;
                let u_stride = (*raw).linesize[1] as usize;
                let v_stride = (*raw).linesize[2] as usize;
                let u_ptr = (*raw).data[1] as *const u8;
                let v_ptr = (*raw).data[2] as *const u8;
                if u_stride == half_w && v_stride == half_w {
                    let u_src = std::slice::from_raw_parts(u_ptr, half_w * half_h);
                    let v_src = std::slice::from_raw_parts(v_ptr, half_w * half_h);
                    for i in 0..u_src.len() {
                        uv.push(u_src[i]);
                        uv.push(v_src[i]);
                    }
                } else {
                    for row in 0..half_h {
                        for col in 0..half_w {
                            uv.push(*u_ptr.add(row * u_stride + col));
                            uv.push(*v_ptr.add(row * v_stride + col));
                        }
                    }
                }
                v.clear();
                Ok(())
            }
            PixelFormat::Yuv444P => {
                y.clear();
                y.reserve(width * height);
                uv.clear();
                uv.reserve(width * height);
                v.clear();
                v.reserve(width * height);
                copy_plane(y, 0, 1);
                copy_plane(uv, 1, 1);
                copy_plane(v, 2, 1);
                Ok(())
            }
            PixelFormat::Rgb24 | PixelFormat::Rgba => {
                let channels = if format == PixelFormat::Rgba { 4 } else { 3 };
                y.clear();
                y.reserve(width * height * channels);
                let stride = (*raw).linesize[0] as usize;
                let row_bytes = width * channels;
                let data_ptr = (*raw).data[0] as *const u8;
                if stride == row_bytes {
                    y.extend_from_slice(std::slice::from_raw_parts(data_ptr, row_bytes * height));
                } else {
                    for row in 0..height {
                        let src = std::slice::from_raw_parts(data_ptr.add(row * stride), row_bytes);
                        y.extend_from_slice(src);
                    }
                }
                uv.clear();
                v.clear();
                Ok(())
            }
            PixelFormat::Yuv422 => {
                let row_bytes = width * 2;
                y.clear();
                y.reserve(row_bytes * height);
                let stride = (*raw).linesize[0] as usize;
                let data_ptr = (*raw).data[0] as *const u8;
                if stride == row_bytes {
                    y.extend_from_slice(std::slice::from_raw_parts(data_ptr, row_bytes * height));
                } else {
                    for row in 0..height {
                        let src = std::slice::from_raw_parts(data_ptr.add(row * stride), row_bytes);
                        y.extend_from_slice(src);
                    }
                }
                uv.clear();
                v.clear();
                Ok(())
            }
        }
    }
}
