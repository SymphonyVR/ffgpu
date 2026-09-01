use crate::{
    context::pipeline_cache::PipelineCache,
    decode::{
        Clock, DecoderState, Frame, FrameQueue, PlayState,
        audio::{self, AudioSink, AudioThread},
        frames, packet_queue,
        read::{Input, ReadMessage, ReadThread},
        sink_thread, video,
    },
    error::Result,
};
use crossbeam_channel::{Sender, unbounded};
use ffmpeg_next::{self as ffn, sys as ff};
use std::ptr::NonNull;
use std::sync::Mutex;
use std::{
    ops::Add,
    path::Path,
    sync::{Arc, atomic::Ordering},
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

    read_messages: Sender<ReadMessage>,

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
        let mut input = Input::open(path)?;

        let backend = adapter.get_info().backend;
        let device_type = preferred_device_type_for_backend(backend);

        let video_decoder =
            video::Decoder::new(&mut input.format_ctx, device_type, hw_device_ctx)?;
        let audio_decoder = audio::Decoder::new(&mut input)?;
        let frame_decoder =
            frames::FrameDecoder::new(&device, pipeline_cache.clone(), &video_decoder.metadata)?;

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

        let state = Arc::new(DecoderState::new(
            input.metadata,
            video_stream,
            audio_stream,
        ));

        let read_thread = ReadThread::new(input, state.clone(), read_msg_rx).run();

        let video_decoder_ptr =
            NonNull::new(unsafe { video_decoder.decoder.as_ptr() as _ }).unwrap();
        let video_thread = video::VideoThread::new(
            video_decoder,
            state.clone(),
            video_rx,
            video_frame_queue,
            video_msg_rx,
            video_clock.clone(),
            audio_clock.clone(),
        )
        .run();

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

        let audio_sink = AudioSink::new(
            state.clone(),
            audio_frame_queue.clone(),
            audio_msg_tx.clone(),
            audio_queue.clone(),
            audio_clock.clone(),
        );

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

                read_messages: read_msg_tx,

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
        let duration = if duration < 0.0 || duration > 3600.0 {
            if video_info.framerate.0 > 0 && video_info.framerate.1 > 0 {
                f64::from(video_info.framerate.invert())
            } else {
                0.0
            }
        } else {
            duration
        };

        // Update the wait duration so the background thread knows exactly how long to sleep
        *wait_duration = Duration::from_secs_f64(duration);

        let pts_sec = best_effort_timestamp as f64
            * f64::from(self.state.video_stream.load().metadata.time_base);
        self.video_clock.set(pts_sec, frame.serial, None);

        unsafe {
            self.frame_decoder.decode_frame(
                &self.instance,
                &self.adapter,
                &self.device,
                &self.queue,
                encoder,
                self.video_decoder,
                &frame.frame,
            )?
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

        if self.step_needs_copy > 0 {
            // NOTE: this is a bit of a hack;
            // on certain backends (namely D3D11VA) we have no way of syncing the D3D11 frame copy with the wgpu YUV->RGB copy.
            // (well, of course it is possible, but wgpu doesn't enable the necessary device extensions...)
            // to counteract this, when stepping a frame (e.g., during accurate seek while paused), we perform the YUV->RGB copy a few more times after the seek.
            self.step_needs_copy -= 1;
            // TODO: actually get the color space
            self.frame_decoder
                .copy_to_rgb(encoder, ffn::color::Space::BT709);
        }

        let video_frame_queue = self.state.video_stream.load().frames.clone();

        if video_frame_queue.queued_len() == 0
            && self.state.is_eof.load(Ordering::SeqCst)
            && self.looping
            && play_state != PlayState::Paused
        {
            // eof reached
            self.seek(Duration::ZERO, SeekMode::Fast);
        }

        if play_state == PlayState::Paused || video_frame_queue.queued_len() == 0 {
            return Ok((Duration::from_millis(50), false));
        }

        let video_info = self.video_info();
        let time_base = f64::from(video_info.time_base);
        let audio_time = self.audio_clock.get();

        let mut duration = Duration::from_secs_f64(f64::from(video_info.framerate.invert()));
        let mut frame_decoded = false;
        loop {
            let queued_len = video_frame_queue.queued_len();
            let was_requeued = self.queued_frame.is_some();
            let frame = self
                .queued_frame
                .take()
                .or_else(|| video_frame_queue.try_next());
            if let Some(frame) = frame {
                let current_serial = self.state.video_stream.load().packets.metadata.serial.load(Ordering::SeqCst);
                if frame.serial == current_serial {
                    if let Some(audio_time_sec) = audio_time {
                        let pts_sec = unsafe { (*frame.frame.as_ptr()).best_effort_timestamp } as f64 * time_base;
                        // Clamp audio_time_sec to prevent runaway audio clock from evicting entire queue
                        let last_video_pts_sec = self.last_pts as f64 * time_base;
                        let audio_time_sec = audio_time_sec.min(last_video_pts_sec + 1.0);
                        let diff = pts_sec - audio_time_sec;

                        // 1. Frame is in the future: requeue it and wait.
                        // Cap the wait to 100 ms so we never stall longer than that even
                        // if the audio clock is lagging or hasn't started yet (e.g. during
                        // software-decode startup, codec pipeline warm-up, etc.).
                        if diff > 0.015 {
                            // If this frame was already requeued once and audio still hasn't
                            // caught up, decode it anyway. In software decoding, audio may
                            // never catch up at real-time rates, causing an infinite stall.
                            if was_requeued {
                                // fall through to update_frame()
                            } else {
                                self.queued_frame = Some(frame);
                                duration = Duration::from_secs_f64(diff.min(0.100));
                                break;
                            }
                        }

                        // 2. Frame is too late: drop/skip it to catch up
                        // Tightened threshold to avoid dropping on micro-jitter
                        if diff < -0.200 && queued_len > 2 {
                            self.last_pts = unsafe { (*frame.frame.as_ptr()).best_effort_timestamp };
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
        self.state.metadata.duration
    }

    #[inline]
    pub fn framerate(&self) -> f64 {
        let video_info = self.video_info();
        video_info.framerate.0 as f64 / video_info.framerate.1 as f64
    }

    #[inline]
    pub fn decoder_name(&self) -> &'static str {
        self.frame_decoder
            .adapter
            .as_ref()
            .map(|adapter| adapter.name())
            .unwrap_or("Unknown")
    }

    pub fn position(&self) -> Duration {
        Duration::from_secs_f64(self.last_pts as f64 * f64::from(self.video_info().time_base))
    }

    pub fn seek(&mut self, position: Duration, mode: SeekMode) {
        let position = position.min(self.duration());
        let ts = (position.as_secs_f64() * ff::AV_TIME_BASE as f64) as i64;

        if let Err(error) = self
            .read_messages
            .send(ReadMessage::SeekStream { ts, mode })
        {
            log::error!("failed to send seek message: {}", error);
        }

        self.flush();
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
    }

    pub fn step_one_frame(&mut self) {
        self.set_paused(false);
        self.state
            .play_state
            .store(PlayState::Step as _, Ordering::Relaxed);
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        self.flush();

        self.state.kill();

        if let Some(audio_thread) = self.audio_thread.take() {
            audio_thread.join().unwrap();
        }

        if let Some(video_thread) = self.video_thread.take() {
            video_thread.join().unwrap();
        }

        if let Some(read_thread) = self.read_thread.take() {
            read_thread.join().unwrap();
        }
    }
}
