use crate::{
    decode::{
        Clock, DecoderState, Frame, FrameQueue, PacketQueueMetadata, PacketReceiver, PacketSender,
        PlayState, loop_timestamp_offset, packet_queue, read::Input,
    },
    error::{Error, Result},
};
use atomic_float::AtomicF32;
use crossbeam_channel::{Receiver, Sender};
use ffmpeg_next::{self as ffn, sys as ff};
use std::{
    mem::ManuallyDrop,
    sync::{Arc, atomic::{Ordering, AtomicU32, AtomicU64}},
    thread::JoinHandle,
    time::Duration,
};

/// RAII guard that promotes the current thread to real-time audio priority
/// for its lifetime, then demotes on drop. Uses `audio_thread_priority`
/// (Windows MMCSS "Pro Audio", Linux rtkit/SCHED_FIFO, macOS Mach
/// time-constraint). Promotion is best-effort: failure logs a warning and
/// falls back to normal priority, never panicking or blocking audio.
struct AudioThreadBoost {
    handle: Option<audio_thread_priority::RtPriorityHandle>,
}

impl AudioThreadBoost {
    fn new() -> Self {
        let handle = match audio_thread_priority::promote_current_thread_to_real_time(512, 44100) {
            Ok(h) => Some(h),
            Err(e) => {
                log::warn!("[Audio] real-time thread promote failed: {e}");
                None
            }
        };
        Self { handle }
    }
}

impl Drop for AudioThreadBoost {
    fn drop(&mut self) {
        if let Some(h) = self.handle.take() {
            if let Err(e) = audio_thread_priority::demote_current_thread_from_real_time(h) {
                log::warn!("[Audio] real-time thread demote failed: {e}");
            }
        }
    }
}

pub(crate) enum Message {
    SkipToTimestamp(i64),
    UpdateParameters(AudioParameters),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AudioParameters {
    pub sample_rate: u32,
    pub channels: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioMetadata {
    pub index: usize,
    pub time_base: ffn::Rational,
    pub sample_rate: u32,
    pub channels: u16,
    pub(crate) format: ffn::format::Sample,
    pub(crate) channel_layout: ffn::ChannelLayout,
    pub(crate) frame_size: u32,
}

unsafe impl Send for AudioMetadata {}
unsafe impl Sync for AudioMetadata {}

impl Default for AudioMetadata {
    fn default() -> Self {
        AudioMetadata {
            index: usize::MAX,
            time_base: ffn::Rational::new(0, 1),
            sample_rate: 0,
            channels: 0,
            format: ffn::format::Sample::None,
            channel_layout: ffn::ChannelLayout::STEREO,
            frame_size: 0,
        }
    }
}

pub(crate) struct AudioStream {
    pub metadata: AudioMetadata,
    pub messages: Sender<Message>,
    pub packets: PacketSender,
    pub frames: FrameQueue,
}

impl AudioStream {
    /// Placeholder stream for `DecoderState::empty()`. See
    /// `VideoStream::dummy()`.
    pub(crate) fn dummy() -> Self {
        use crossbeam_channel::unbounded;
        let (packets, _packets_rx, _packets_meta) = packet_queue();
        AudioStream {
            metadata: AudioMetadata::default(),
            messages: unbounded().0,
            packets,
            frames: FrameQueue::new(1),
        }
    }
}

pub(crate) struct Decoder {
    pub decoder: ffn::decoder::Audio,
    pub metadata: AudioMetadata,
}

impl Decoder {
    pub fn new(input: &mut Input) -> Result<Option<Self>> {
        let Some(stream) = input.format_ctx.streams().best(ffn::media::Type::Audio) else {
            return Ok(None);
        };

        let stream_index = stream.index();

        let codec = stream.parameters().id();
        let decoder = ffn::decoder::find(codec).ok_or(Error::MissingCodec(codec.name()))?;

        let mut decoder = ffn::codec::Context::new_with_codec(decoder).decoder();
        decoder.set_parameters(stream.parameters())?;
        decoder.set_threading(ffn::threading::Config {
            kind: ffn::threading::Type::Frame,
            count: 0,
        });

        let decoder = decoder.audio()?;

        let sample_rate = decoder.rate();
        let channels = decoder.channels();
        let format = decoder.format();
        let channel_layout = decoder.channel_layout();
        let frame_size = decoder.frame_size();

        let metadata = AudioMetadata {
            index: stream_index,
            time_base: stream.time_base(),
            sample_rate,
            channels,
            format,
            channel_layout,
            frame_size,
        };

        Ok(Some(Decoder { decoder, metadata }))
    }
}

unsafe impl Send for Decoder {}
unsafe impl Sync for Decoder {}

struct ResamplerState {
    parameters: AudioParameters,
    resampler: ffn::software::resampling::Context,
}

pub(crate) struct AudioThread {
    decoder: Decoder,
    state: Arc<DecoderState>,
    audio_rx: PacketReceiver,
    frame_queue: FrameQueue,
    resampler: Option<ResamplerState>,
    messages: Receiver<Message>,
}

impl AudioThread {
    pub fn new(
        decoder: Decoder,
        pbs: Arc<DecoderState>,
        audio_rx: PacketReceiver,
        frame_queue: FrameQueue,
        messages: Receiver<Message>,
    ) -> Self {
        AudioThread {
            decoder,
            state: pbs,
            audio_rx,
            frame_queue,
            resampler: None,
            messages,
        }
    }

    fn flush(&mut self) {
        self.decoder.decoder.flush();

        if let Some(resampler) = &mut self.resampler {
            let mut frame = ffn::util::frame::Audio::empty();
            while let Ok(Some(_)) = resampler.resampler.flush(&mut frame) {}
        }
    }

    fn update_parameters(&mut self, parameters: AudioParameters) -> Result<()> {
        if self
            .resampler
            .as_ref()
            .is_none_or(|resampler| resampler.parameters != parameters)
        {
            let stream = self.state.audio_stream.load().clone();

            let format = ffn::format::Sample::F32(ffn::format::sample::Type::Packed);
            let channel_layout = ffn::ChannelLayout(ff::AVChannelLayout {
                order: ff::AVChannelOrder::AV_CHANNEL_ORDER_NATIVE,
                nb_channels: parameters.channels as _,
                u: ff::AVChannelLayout__bindgen_ty_1 {
                    mask: (1 << parameters.channels) - 1,
                },
                opaque: std::ptr::null_mut(),
            });

            let resampler = ffn::software::resampler(
                (
                    stream.metadata.format,
                    stream.metadata.channel_layout,
                    stream.metadata.sample_rate,
                ),
                (format, channel_layout, parameters.sample_rate),
            )?;

            self.resampler = Some(ResamplerState {
                parameters,
                resampler,
            });
        }
        Ok(())
    }

    fn run_thread(&mut self) {
        let mut packet_serial = 0;
        let mut packet_loop_index = 0;

        let mut frame = unsafe { ffn::Frame::empty() };

        let mut skip_to_ts = None;

        'exit: while self.state.alive.load(Ordering::Relaxed) {
            let mut prev_frame = None;

            while self.state.alive.load(Ordering::Relaxed) {
                // Drain seek/control messages here, not only at the outer loop.
                // The inner loop remains active while paused, so handling them
                // only outside it leaves SkipToTimestamp queued throughout a scrub.
                while let Ok(message) = self.messages.try_recv() {
                    match message {
Message::SkipToTimestamp(ts) => {
                            skip_to_ts = Some(ts);
                        }
                        Message::UpdateParameters(parameters) => {
                            if let Err(error) = self.update_parameters(parameters) {
                                log::error!("audio thread failed to update parameters: {}", error);
                            }
                        }
                    }
                }

                if self.state.play_state() == PlayState::Paused {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }

                if packet_serial != self.audio_rx.metadata.serial.load(Ordering::Relaxed) {
                    self.flush();
                }

                if self.frame_queue.free_rx.is_empty() {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }

let frame = match self.decoder.decoder.receive_frame(&mut frame) {
                    Ok(_) => {
                        if let Some(pts) = frame.pts() {
                            let offset = loop_timestamp_offset(
                                &self.state,
                                packet_loop_index,
                                self.decoder.metadata.time_base,
                            );
                            let pts = pts.saturating_add(offset);
                            if offset != 0 {
                                unsafe {
                                    (*frame.as_mut_ptr()).pts = pts;
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
                        }
                        Some(&mut frame)
                    }
                    Err(ffn::Error::Eof) => {
                        if let Some(mut prev_frame) = prev_frame.take() {
                            unsafe {
                                ff::av_frame_move_ref(frame.as_mut_ptr(), prev_frame.as_mut_ptr())
                            };
                            Some(&mut frame)
                        } else {
                            self.flush();
                            break;
                        }
                    }
                    Err(ffn::Error::Other { errno: ff::EAGAIN }) => {
                        break;
                    }
                    _ => None,
                };

                if let Some(frame) = frame {
                    prev_frame = None;
                    skip_to_ts = None;

                    let mut audio_frame = ManuallyDrop::new(unsafe {
                        ffn::util::frame::Audio::wrap(frame.as_mut_ptr())
                    });

                    // TODO: automatically recreate sampler if input frame format changed
                    if let Some(resampler) = &mut self.resampler {
                        let resampled_pts = unsafe {
                            ff::swr_next_pts(
                                resampler.resampler.as_mut_ptr(),
                                (*frame.as_ptr()).pts * resampler.parameters.sample_rate as i64,
                            )
                        };

                        let resampled_pts =
                            resampled_pts / self.decoder.metadata.sample_rate as i64;

                        if audio_frame.channel_layout().0.order
                            == ff::AVChannelOrder::AV_CHANNEL_ORDER_UNSPEC
                        {
                            let channels = audio_frame.channels();
                            audio_frame
                                .set_channel_layout(ffn::ChannelLayout::default(channels as _));
                        }

                        let mut resampled_frame = ffn::util::frame::Audio::empty();
                        if let Err(error) =
                            resampler.resampler.run(&audio_frame, &mut resampled_frame)
                        {
                            log::error!("audio resampler failed: {}", error);
                        } else {
                            unsafe {
                                ff::av_frame_unref(audio_frame.as_mut_ptr());
                                audio_frame.alloc(
                                    resampled_frame.format(),
                                    resampled_frame.samples(),
                                    resampled_frame.channel_layout(),
                                );
                                ff::av_frame_copy(
                                    audio_frame.as_mut_ptr(),
                                    resampled_frame.as_mut_ptr(),
                                );
                                ff::av_frame_copy_props(
                                    audio_frame.as_mut_ptr(),
                                    resampled_frame.as_mut_ptr(),
                                );
                                audio_frame.set_pts(Some(resampled_pts));
                            }
                        }
                    }

if !self
                        .frame_queue
                        .send(frame, packet_serial, &self.state.alive)
                    {
                        if std::env::var("P").is_ok() { eprintln!("[A] send(frame) FAILED serial={packet_serial}"); }
                        break 'exit;
                    }
                    if std::env::var("P").is_ok() { eprintln!("[A] sent frame serial={packet_serial}"); }
                }
            }

            let packet = loop {
                if !self.state.alive.load(Ordering::Relaxed) {
                    break 'exit;
                }

                let Some(packet) = self.audio_rx.receive() else {
                    continue;
                };

if packet_serial != packet.serial {
                    if std::env::var("P").is_ok() { eprintln!("[A] outer adopt serial {} (was {})", packet.serial, packet_serial); }
                    self.flush();
                    packet_serial = packet.serial;
                    packet_loop_index = packet.loop_index;
                } else if packet_loop_index != packet.loop_index {
                    self.flush();
                    packet_loop_index = packet.loop_index;
                }

                if packet_serial == self.audio_rx.metadata.serial.load(Ordering::SeqCst) {
                    break packet;
                }
            };

            let is_eof_packet = packet.packet.data().is_none();
            if is_eof_packet {
                if let Err(error) = self.decoder.decoder.send_eof() {
                    log::error!("failed to send EOF to audio decoder: {}", error);
                }
            } else {
                if let Err(error) = self.decoder.decoder.send_packet(&packet.packet) {
                    log::error!("failed to send packet: {}", error);
                }
            }
        }
    }

    pub fn run(mut self) -> JoinHandle<()> {
        let guard =
            crate::decode::ThreadGuard::new(self.state.thread_count.clone());
        std::thread::spawn(move || {
            let _guard = guard;
            let _boost = AudioThreadBoost::new();
            self.run_thread();
        })
    }
}

#[derive(Debug, Clone)]
struct AudioClockAnchor {
    pts: f64,
    write_count: u64,
    serial: u32,
}

pub struct AudioSink {
    state: Arc<DecoderState>,
    messages: Sender<Message>,
    parameters: AudioParameters,
    queue: Arc<PacketQueueMetadata>,
    clock: Arc<Clock>,
    consumer: ringbuf::HeapCons<f32>,
    last_pts: Arc<AtomicU64>,
    last_serial: Arc<AtomicU32>,
    write_count: Arc<AtomicU64>,
    read_count: Arc<AtomicU64>,
    anchor: Arc<arc_swap::ArcSwap<AudioClockAnchor>>,
    required_serial: Arc<AtomicU32>,
    preview_samples: Arc<AtomicU64>,
    preview_armed: Arc<std::sync::atomic::AtomicBool>,
}

fn initial_audio_parameters() -> AudioParameters {
    #[cfg(feature = "cpal")]
    {
        use cpal::traits::{DeviceTrait, HostTrait};

        if let Some(config) = cpal::default_host()
            .default_output_device()
            .and_then(|device| device.default_output_config().ok())
        {
            return AudioParameters {
                sample_rate: config.sample_rate(),
                channels: config.channels(),
            };
        }
    }

    AudioParameters {
        sample_rate: 48_000,
        channels: 2,
    }
}

impl AudioSink {
    pub(crate) fn new(
        pbs: Arc<DecoderState>,
        frame_queue: FrameQueue,
        messages: Sender<Message>,
        queue: Arc<PacketQueueMetadata>,
        clock: Arc<Clock>,
    ) -> Self {
        use ringbuf::traits::{Producer, Split, Observer};

        // Configure the decoder before it can produce its first frame. The
        // old order let the producer reinterpret source-format samples as
        // packed f32 until the asynchronous parameter message was handled.
        let initial_parameters = initial_audio_parameters();

        // 2 seconds buffer for 48kHz stereo
        let rb = ringbuf::HeapRb::<f32>::new(48000 * 2 * 2);
        let (mut prod, cons) = rb.split();

        let last_pts = Arc::new(AtomicU64::new(f64::NAN.to_bits()));
        let last_serial = Arc::new(AtomicU32::new(u32::MAX));
        let write_count = Arc::new(AtomicU64::new(0));
        let read_count = Arc::new(AtomicU64::new(0));
        let anchor = Arc::new(arc_swap::ArcSwap::new(Arc::new(AudioClockAnchor {
            pts: f64::NAN,
            write_count: 0,
            serial: u32::MAX,
        })));
        let required_serial = Arc::new(AtomicU32::new(u32::MAX));
        let preview_samples = Arc::new(AtomicU64::new(0));
        let preview_armed = Arc::new(std::sync::atomic::AtomicBool::new(false));

        let last_pts_clone = last_pts.clone();
        let last_serial_clone = last_serial.clone();
        let write_count_clone = write_count.clone();
        let read_count_clone = read_count.clone();
        let anchor_clone = anchor.clone();
        let required_serial_clone = required_serial.clone();
        let state_clone = pbs.clone();
        let queue_clone = queue.clone();

        let mut sink = AudioSink {
            state: pbs,
            messages,
            parameters: AudioParameters {
                sample_rate: 0,
                channels: 0,
            },
            queue,
            clock,
            consumer: cons,
            last_pts,
            last_serial,
            write_count,
            read_count,
            anchor,
            required_serial,
            preview_samples,
            preview_armed,
        };
        sink.set_parameters(initial_parameters);

// Background thread to continuously pull decoded frames into the lock-free ringbuffer
        std::thread::spawn(move || {
            let _boost = AudioThreadBoost::new();
            let mut last_serial = u32::MAX;
            let mut local_write_count = 0u64;

            while state_clone.alive.load(Ordering::Relaxed) {
                if state_clone.play_state() != PlayState::Playing {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }

                // Pacing: only pull frame if we have < 100ms buffered
                // Target fill: 100ms of output audio (keeps latency low)
                let target_fill_samples = (initial_parameters.sample_rate
                    * initial_parameters.channels as u32
                    / 10) as usize;
                
                let fill_level = prod.occupied_len();
                if fill_level >= target_fill_samples {
                    // Buffer is full enough, sleep and check again
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }

                let Some(frame) = frame_queue.try_next() else {
                    std::thread::sleep(Duration::from_millis(2));
                    continue;
                };

                let serial = frame.serial;
                if serial != queue_clone.serial.load(Ordering::Relaxed) {
                    if std::env::var("P").is_ok() { eprintln!("[P] DROP frame serial={serial} queue={}", queue_clone.serial.load(Ordering::Relaxed)); }
                    frame_queue.release(frame);
                    continue;
                }

                // Reset counts on seek/serial change
                if serial != last_serial {
                    read_count_clone.store(0, Ordering::SeqCst);
                    write_count_clone.store(0, Ordering::SeqCst);
                    local_write_count = 0;
                    last_serial = serial;
                }

                let mut frame = ManuallyDrop::new(ffn::util::frame::Audio::from(frame.frame));
                
                let pts = if let Some(pts) = frame.pts() {
                    pts as f64 * f64::from(state_clone.audio_stream.load().metadata.time_base)
                        + frame.samples() as f64 / frame.rate() as f64
                } else {
                    f64::NAN
                };

                // Discard trailing padding bytes by slicing precisely using frame.samples() * frame.channels()
                let num_samples = frame.samples() * frame.channels() as usize;
                let samples_ptr = frame.data(0).as_ptr() as *const f32;
                let samples = unsafe { std::slice::from_raw_parts(samples_ptr, num_samples) };

                let mut pushed = 0;
                while pushed < samples.len() && state_clone.alive.load(Ordering::Relaxed) {
                    let current_serial = queue_clone.serial.load(Ordering::Relaxed);
                    if serial != current_serial {
                        break;
                    }
                    
                    let n = prod.push_slice(&samples[pushed..]);
                    pushed += n;
                    if pushed < samples.len() {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }

if pushed == samples.len() {
                    if std::env::var("P").is_ok() { eprintln!("[P] PUSH serial={serial} pts={pts:.3} pushed={pushed}"); }
                    local_write_count += pushed as u64;
                    write_count_clone.store(local_write_count, Ordering::SeqCst);
                    last_pts_clone.store(pts.to_bits(), Ordering::SeqCst);

                    // A preview seek and its final seek can be processed
                    // together, so the actual serial may be greater than the
                    // predicted current+1 value from prepare_seek(). Publish
                    // the serial that really supplied the audio before making
                    // it eligible for the device callback.
                    required_serial_clone.store(serial, Ordering::Release);
                    last_serial_clone.store(serial, Ordering::SeqCst);
                    
                    // Update anchor atomically
                    anchor_clone.store(Arc::new(AudioClockAnchor {
                        pts,
                        write_count: local_write_count,
                        serial,
                    }));
                }

                frame_queue.release(Frame {
                    frame: unsafe { ffn::Frame::wrap(frame.as_mut_ptr()) },
                    serial,
                });
            }
        });
        sink
    }

    pub fn set_parameters(&mut self, parameters: AudioParameters) {
        if self.parameters != parameters {
            self.parameters = parameters;
            if let Err(_) = self.messages.send(Message::UpdateParameters(parameters)) {
                log::error!("cannot update parameters, audio thread closed");
            }
        }
    }

    #[inline]
    pub fn parameters(&self) -> AudioParameters {
        self.parameters
    }

    pub fn sample_rate(&self) -> u32 {
        self.state.audio_stream.load().metadata.sample_rate
    }

    pub fn channels(&self) -> u16 {
        self.state.audio_stream.load().metadata.channels
    }

    pub fn read_to_slice(&mut self, out: &mut [f32], gain: f32) -> Result<()> {
        pump_output(
            out,
            gain,
            self.state.play_state(),
            self.queue.serial.load(Ordering::Relaxed),
            &mut self.consumer,
            &self.clock,
            &self.last_serial,
            &self.read_count,
            &self.anchor,
            &self.required_serial,
            &self.preview_samples,
            &self.preview_armed,
            self.parameters.sample_rate * u32::from(self.parameters.channels),
        );
        Ok(())
    }

/// Software analogue of [`DeviceAudioSink::prepare_seek`]: prune the ring
    /// buffer and gate the reader on our predicted serial so the audio clock
    /// re-anchors at the seek target instead of holding the pre-seek position.
    /// Must be called *before* the matching `read::Message::SeekStream`.
    pub fn announce_seek(&mut self, _position: Duration) {
        use ringbuf::traits::Consumer;

        let next_serial = self.queue.serial.load(Ordering::Acquire).wrapping_add(1);

        // Invalidate the clock during the seek (NAN + predicted serial).
        self.clock.set(f64::NAN, next_serial, None);
        self.anchor.store(Arc::new(AudioClockAnchor {
            pts: f64::NAN,
            write_count: 0,
            serial: next_serial,
        }));
        self.read_count.store(0, Ordering::SeqCst);
        self.write_count.store(0, Ordering::SeqCst);
        self.required_serial.store(next_serial, Ordering::Release);

        // Prune any pre-seek samples so stale audio cannot leak past the seek.
        while let Some(_) = self.consumer.try_pop() {}
    }

    /// Current audio playback position in milliseconds, read from the shared
    /// audio clock. Returns `0.0` when the clock is in an invalid state (e.g.
    /// during a seek re-anchor or before the first sample is written).
    pub fn current_position_ms(&self) -> f64 {
        match self.clock.get() {
            Some(secs) => secs * 1000.0,
            None => 0.0,
        }
    }


    /// Convert this `AudioSink` into a device-backed, self-healing `DeviceAudioSink`.
    ///
    /// - Wraps the ring-buffer consumer in `Arc<Mutex<…>>` so CPAL stream callbacks
    ///   can be recreated without touching the producer thread.
    /// - Spawns a background monitor that detects default-device changes via both
    ///   native OS notifications and a CPAL name-comparison fallback.
    /// - On device change or stream error: flushes stale samples, sends
    ///   `Message::UpdateParameters` to the FFmpeg SWR resampler, and rebuilds
    ///   the CPAL stream — zero producer-side interruption.
    #[cfg(feature = "cpal")]
    pub fn into_device_sink(self) -> DeviceAudioSink {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        use std::sync::Mutex;
        use std::sync::atomic::AtomicBool;

        // Wrap the consumer so multiple stream generations can share it.
        let consumer = Arc::new(Mutex::new(self.consumer));
        let gain     = Arc::new(AtomicF32::new(1.0));
        let alive    = Arc::new(AtomicBool::new(false));
        let stream   = Arc::new(Mutex::new(None::<cpal::Stream>));

        // Build the initial CPAL stream.
        match build_cpal_stream(
            &self.state,
            &self.messages,
            &consumer,
            &gain,
            &self.last_pts,
            &self.clock,
            &self.last_serial,
            &self.write_count,
            &self.read_count,
            &self.anchor,
            &self.required_serial,
            &self.preview_samples,
            &self.preview_armed,
        ) {
            Ok(s) => {
                *stream.lock().unwrap() = Some(s);
                alive.store(true, Ordering::Relaxed);
            }
            Err(e) => log::error!("[Audio] Failed to build initial CPAL stream: {}", e),
        }

        // Kick off the native OS device monitor (one global thread per process).
        super::device_monitor::start_device_monitor();

        // ── monitor thread ────────────────────────────────────────────────────
        let stop      = Arc::new(AtomicBool::new(false));
        let stop_c    = stop.clone();
        let stream_c  = stream.clone();
        let alive_c   = alive.clone();
        let state_c   = self.state.clone();
        let msgs_c    = self.messages.clone();
        let consumer_c = consumer.clone();
        let gain_c    = gain.clone();
        let last_pts_c   = self.last_pts.clone();
        let clock_c      = self.clock.clone();
        let last_serial_c = self.last_serial.clone();
        let write_count_c = self.write_count.clone();
        let read_count_c = self.read_count.clone();
        let anchor_c = self.anchor.clone();
        let required_serial_c = self.required_serial.clone();
        let preview_samples_c = self.preview_samples.clone();
        let preview_armed_c = self.preview_armed.clone();

        let monitor = std::thread::spawn(move || {
            // Fallback: remember the device name so we detect changes without
            // a native listener (e.g. on unsupported platforms or if COM fails).
            let mut last_name: Option<String> = cpal::default_host()
                .default_output_device()
                .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));

            loop {
                // Interruptible sleep: 5 × 100 ms so Drop exits quickly.
                for _ in 0..5 {
                    if stop_c.load(Ordering::Relaxed) { return; }
                    std::thread::sleep(Duration::from_millis(100));
                }

                // Stop if the whole player was killed.
                if !state_c.alive.load(Ordering::Relaxed) { break; }

                // ── detect change ─────────────────────────────────────────────
                let native_changed   = super::device_monitor::audio_device_changed();
                let current_name: Option<String> = cpal::default_host()
                    .default_output_device()
                    .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));
                let fallback_changed = current_name != last_name;
                let stream_dead      = !alive_c.load(Ordering::Relaxed)
                    || stream_c.lock().unwrap().is_none();

                if !native_changed && !fallback_changed && !stream_dead {
                    continue;
                }

                log::info!(
                     "[Audio] Device change detected (native={}, name={}, dead={}). Rebuilding stream…",
                     native_changed, fallback_changed, stream_dead
                );
                last_name = current_name;

                // ── stop old stream ───────────────────────────────────────────
                if let Some(old) = stream_c.lock().unwrap().take() {
                    let _ = old.pause();
                }
                alive_c.store(false, Ordering::Relaxed);

                // ── flush stale samples from the ring buffer ─────────────────
                // Mandatory: old-rate samples must not reach the new stream.
                {
                    use ringbuf::traits::Consumer;
                    let mut cons = consumer_c.lock().unwrap();
                    while cons.try_pop().is_some() {}
                }

                // ── rebuild ───────────────────────────────────────────────────
                match build_cpal_stream(
                    &state_c,
                    &msgs_c,
                    &consumer_c,
                    &gain_c,
                    &last_pts_c,
                    &clock_c,
                    &last_serial_c,
                    &write_count_c,
                    &read_count_c,
                    &anchor_c,
                    &required_serial_c,
                    &preview_samples_c,
                    &preview_armed_c,
                ) {
                    Ok(new_stream) => {
                        log::info!("[Audio] CPAL stream successfully rebuilt");
                        *stream_c.lock().unwrap() = Some(new_stream);
                        alive_c.store(true, Ordering::Relaxed);
                    }
                    Err(e) => log::error!("[Audio] Failed to rebuild CPAL stream: {}", e),
                }
            }
        });

        DeviceAudioSink {
            state: self.state,
            messages: self.messages,
            clock: self.clock,
            queue: self.queue,
            consumer,
            last_pts: self.last_pts,
            last_serial: self.last_serial,
            write_count: self.write_count,
            read_count: self.read_count,
            anchor: self.anchor,
            required_serial: self.required_serial,
            preview_samples: self.preview_samples,
            preview_armed: self.preview_armed,
            gain,
            stream,
            alive,
            monitor: Some(monitor),
            stop,
        }
    }
}

// ── Free helper: negotiate device format, update resampler, build stream ──────

/// Shared audio pump used by both the software `read_to_slice` (tests,
/// `SoftwareDecodeVideo`) and the real-time cpal device callback (the path
/// the engine actually ships). Both consumers must behave identically, or the
/// ring buffer deadlock shows up on only one of them.
///
/// Serial gate: audio is dropped while a seek is in flight, but the ring is
/// still DRAINED rather than left full — a consumer that returns without
/// popping lets the buffer fill, the producer blocks on a full ring,
/// `last_serial` never advances, and the gate stays dead forever (the
/// "no audio after seeking" stall). Draining frees the ring so the producer
/// can publish the post-seek serial and unblock the consumer.
fn pump_output(
    out: &mut [f32],
    gain: f32,
    play_state: PlayState,
    current_serial: u32,
    consumer: &mut ringbuf::HeapCons<f32>,
    clock: &Arc<Clock>,
    last_serial: &Arc<AtomicU32>,
    read_count: &Arc<AtomicU64>,
    anchor: &Arc<arc_swap::ArcSwap<AudioClockAnchor>>,
    required_serial: &Arc<AtomicU32>,
    preview_samples: &Arc<AtomicU64>,
    preview_armed: &std::sync::atomic::AtomicBool,
    target_rate: u32,
) {
    use ringbuf::traits::Consumer;

    out.fill(0.0);

    let active_serial = last_serial.load(Ordering::Acquire);
    let required_serial = required_serial.load(Ordering::Acquire);

    let gated = active_serial != current_serial
        || (required_serial != u32::MAX && active_serial != required_serial);
    if gated {
        while consumer.try_pop().is_some() {}
        return;
    }

    let g = gain.max(0.0);

    if preview_armed.swap(false, Ordering::AcqRel) {
        while consumer.try_pop().is_some() {}
        return;
    }

    let max_samples = if play_state == PlayState::Playing {
        out.len()
    } else {
        preview_samples.load(Ordering::Acquire).min(out.len() as u64) as usize
    };
    if max_samples == 0 {
        return;
    }

    let popped = consumer.pop_slice(&mut out[..max_samples]);
    if play_state != PlayState::Playing {
        preview_samples.fetch_sub(popped as u64, Ordering::AcqRel);
    }
    for s in out[..popped].iter_mut() {
        *s *= g;
    }

    read_count.fetch_add(popped as u64, Ordering::SeqCst);

    // FIX: only update the clock when we actually consumed audio.
    // On underflow (popped < out.len()), the clock must not advance,
    // otherwise it drifts ahead of real audio at wall-clock rate.
    if popped > 0 {
        let anchor = anchor.load();
        let serial = last_serial.load(Ordering::Relaxed);
        if anchor.serial == serial && !anchor.pts.is_nan() {
            let r_count = read_count.load(Ordering::SeqCst);
            let unread = anchor.write_count.saturating_sub(r_count);
            let time = ffn::time::relative() as f64 / 1_000_000.0;
            clock.set(
                anchor.pts - unread as f64 / target_rate as f64,
                serial,
                Some(time),
            );
        }
    }
}

#[cfg(feature = "cpal")]
fn build_cpal_stream(
    state:       &Arc<crate::decode::DecoderState>,
    messages:    &crossbeam_channel::Sender<Message>,
    consumer:    &Arc<std::sync::Mutex<ringbuf::HeapCons<f32>>>,
    gain:        &Arc<AtomicF32>,
    _last_pts:    &Arc<AtomicU64>,
    clock:       &Arc<crate::decode::Clock>,
    last_serial: &Arc<AtomicU32>,
    _write_count: &Arc<AtomicU64>,
    read_count:  &Arc<AtomicU64>,
    anchor:      &Arc<arc_swap::ArcSwap<AudioClockAnchor>>,
    required_serial: &Arc<AtomicU32>,
    preview_samples: &Arc<AtomicU64>,
    preview_armed: &Arc<std::sync::atomic::AtomicBool>,
) -> crate::error::Result<cpal::Stream> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let device = cpal::default_host()
        .default_output_device()
        .ok_or(crate::error::Error::Unknown)?;

    let config = device
        .default_output_config()
        .map_err(|_| crate::error::Error::Unknown)?;

    let sample_rate = config.sample_rate();
    let channels    = config.channels() as u16;

    // Tell the FFmpeg SWR resampler to target the new device format.
    // AudioThread will rebuild its resampler context on the next loop tick.
    let _ = messages.send(Message::UpdateParameters(AudioParameters {
        sample_rate,
        channels,
    }));

    // ── build the output callback ─────────────────────────────────────────────
    let consumer_cb    = consumer.clone();
    let gain_cb        = gain.clone();
    let clock_cb       = clock.clone();
    let last_serial_cb = last_serial.clone();
    let state_cb       = state.clone();
    let read_count_cb  = read_count.clone();
    let anchor_cb      = anchor.clone();
    let required_serial_cb = required_serial.clone();
    let preview_samples_cb = preview_samples.clone();
    let preview_armed_cb = preview_armed.clone();
    let target_rate    = sample_rate * channels as u32;

    let read_samples = move |out: &mut [f32]| {
        // NOTE: cpal's `realtime` feature already promotes this (cpal-owned)
        // output thread via audio_thread_priority. Do not promote manually.

        let play_state = state_cb.play_state();
        let current_serial = state_cb
            .audio_stream
            .load()
            .packets
            .metadata
            .serial
            .load(Ordering::Relaxed);
        let mut cons = consumer_cb.lock().unwrap();
        pump_output(
            out,
            gain_cb.load(Ordering::Relaxed),
            play_state,
            current_serial,
            &mut cons,
            &clock_cb,
            &last_serial_cb,
            &read_count_cb,
            &anchor_cb,
            &required_serial_cb,
            &preview_samples_cb,
            &preview_armed_cb,
            target_rate,
        );
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            config.config(),
            move |data: &mut [f32], _| read_samples(data),
            |e| log::error!("[Audio] CPAL error: {e}"),
            None,
        ),
        cpal::SampleFormat::I16 => {
            let mut buf = Vec::<f32>::new();
            device.build_output_stream(
                config.config(),
                move |data: &mut [i16], _| {
                    buf.resize(data.len(), 0.0);
                    read_samples(&mut buf);
                    for (d, &s) in data.iter_mut().zip(buf.iter()) {
                        *d = cpal::Sample::from_sample(s);
                    }
                },
                |e| log::error!("[Audio] CPAL error: {e}"),
                None,
            )
        }
        other => {
            log::warn!("[Audio] Unsupported CPAL format {other:?}, falling back to f32");
            device.build_output_stream(
                config.config(),
                move |data: &mut [f32], _| read_samples(data),
                |e| log::error!("[Audio] CPAL error: {e}"),
                None,
            )
        }
    }
    .map_err(|_| crate::error::Error::Unknown)?;

    stream
        .play()
        .map_err(|_| crate::error::Error::Unknown)?;

    Ok(stream)
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "cpal")]
pub struct DeviceAudioSink {
    // Shared context needed to rebuild the stream on device change.
    state:       Arc<crate::decode::DecoderState>,
    messages:    crossbeam_channel::Sender<Message>,
    clock:       Arc<crate::decode::Clock>,
    queue:       Arc<crate::decode::PacketQueueMetadata>,
    consumer:    Arc<std::sync::Mutex<ringbuf::HeapCons<f32>>>,
    last_pts:    Arc<AtomicU64>,
    last_serial: Arc<AtomicU32>,
    write_count: Arc<AtomicU64>,
    read_count:  Arc<AtomicU64>,
    anchor:      Arc<arc_swap::ArcSwap<AudioClockAnchor>>,
    required_serial: Arc<AtomicU32>,
    preview_samples: Arc<AtomicU64>,
    preview_armed: Arc<std::sync::atomic::AtomicBool>,

    // Atomically shared gain applied inside the CPAL callback.
    gain: Arc<AtomicF32>,

    // Active CPAL stream (None while being rebuilt).
    stream: Arc<std::sync::Mutex<Option<cpal::Stream>>>,
    // True while the stream is playing and healthy.
    alive:  Arc<std::sync::atomic::AtomicBool>,

    // Background monitor thread control.
    monitor: Option<std::thread::JoinHandle<()>>,
    stop:    Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(feature = "cpal")]
impl DeviceAudioSink {
    /// Drop pre-seek device samples and rebase audio to the seek target.
    /// While paused, at most `preview_ms` of aligned audio is emitted.
    pub fn prepare_seek(&self, position: Duration, preview_ms: u64) {
        use ringbuf::traits::Consumer;
        let next_serial = self.queue.serial.load(Ordering::Acquire).wrapping_add(1);
        let timestamp = (position.as_secs_f64() * ff::AV_TIME_BASE as f64) as i64;

        // Tell the audio decoder to skip to the seek target.
        let _ = self.messages.send(Message::SkipToTimestamp(timestamp));

        // Invalidate the clock during the seek (NAN + predicted serial).
        // The clock.get() returns None while clock.serial != queue.serial,
        // so A/V sync stops driving the video until post-seek audio actually
        // flows. The cpal callback's required_serial gate then mutes the
        // device until the producer has pumped a real post-seek frame and
        // updated last_serial; the callback then re-arms the clock from the
        // real anchor PTS. Speculatively setting the clock to `seek_pts`
        // here would make get() advance with wall-clock before any audio
        // is consumed, which the video sync reads as "audio ahead of video"
        // and accelerates playback to "catch up" — visible as a runaway rate.
        self.clock.set(f64::NAN, next_serial, None);
        self.anchor.store(Arc::new(AudioClockAnchor {
            pts: f64::NAN,
            write_count: 0,
            serial: next_serial,
        }));
        self.read_count.store(0, Ordering::SeqCst);
        self.write_count.store(0, Ordering::SeqCst);

        self.required_serial.store(next_serial, Ordering::Release);
        let samples = u64::from(self.parameters().sample_rate)
            * u64::from(self.parameters().channels)
            * preview_ms
            / 1000;
        self.preview_samples.store(samples, Ordering::Release);
        self.preview_armed.store(true, Ordering::Release);
        let mut consumer = self.consumer.lock().unwrap();
        while consumer.try_pop().is_some() {}
    }

    fn parameters(&self) -> AudioParameters {
        initial_audio_parameters()
    }

    /// Volume multiplier for the CPAL output callback (0.0 = mute, 1.0 = unity).
    pub fn set_gain(&self, gain: f32) {
        self.gain.store(gain.max(0.0), Ordering::Relaxed);
    }

    /// Current audio playback position in milliseconds. This is the
    /// PTS of the audio sample currently being played by cpal. Returns
    /// `0.0` if no audio is playing or the clock is in an invalid
    /// state (e.g. before the first sample is written).
    ///
    /// Used by [`SoftwareDecodeVideo::update`](crate::SoftwareDecodeVideo::update)
    /// as the master clock for A/V sync rate adjustment.
    pub fn current_position_ms(&self) -> f64 {
        match self.clock.get() {
            Some(secs) => secs * 1000.0,
            None => 0.0,
        }
    }

    pub fn gain(&self) -> f32 {
        self.gain.load(Ordering::Relaxed)
    }

    /// Returns `true` when the CPAL stream is active and playing.
    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    /// Manually trigger stream recreation (e.g. after caller detects a problem).
    /// Returns `true` if the rebuild succeeded.
    pub fn try_recreate(&self) -> bool {
        use cpal::traits::StreamTrait;
        use ringbuf::traits::Consumer;

        if self.is_alive() {
            return true;
        }

        // Stop and drop the old stream.
        if let Some(old) = self.stream.lock().unwrap().take() {
            let _ = old.pause();
        }

        // Flush stale samples before switching format.
        {
            let mut cons = self.consumer.lock().unwrap();
            while cons.try_pop().is_some() {}
        }

        match build_cpal_stream(
            &self.state,
            &self.messages,
            &self.consumer,
            &self.gain,
            &self.last_pts,
            &self.clock,
            &self.last_serial,
            &self.write_count,
            &self.read_count,
            &self.anchor,
            &self.required_serial,
            &self.preview_samples,
            &self.preview_armed,
        ) {
            Ok(s) => {
                *self.stream.lock().unwrap() = Some(s);
                self.alive.store(true, Ordering::Relaxed);
                true
            }
            Err(e) => {
                log::error!("[Audio] try_recreate failed: {e}");
                false
            }
        }
    }
}

#[cfg(feature = "cpal")]
impl Drop for DeviceAudioSink {
    fn drop(&mut self) {
        // Signal the monitor thread to exit.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.monitor.take() {
            // Bounded join: the monitor thread normally sleeps in
            // 100ms intervals, but it can be in a slow `build_cpal_stream`
            // call (audio device rebuild) when we signal it. A bare
            // `t.join()` would block the close path for the full
            // duration of the rebuild, which can be hundreds of ms.
            // 100ms timeout caps the wait; the OS reaps the leaked
            // thread on process exit.
            join_with_timeout(t, 100);
        }
        // Pause and drop the active stream.
        if let Some(s) = self.stream.lock().unwrap().take() {
            use cpal::traits::StreamTrait;
            let _ = s.pause();
        }
    }
}

/// Join a thread with a hard timeout. Mirrors the helper in
/// `patches/ffgpu/src/video.rs`. Returns `true` if the thread finished
/// in time, `false` otherwise. The handle is consumed in both cases;
/// on timeout the thread is detached (leaked, but the OS reaps it on
/// process exit).
fn join_with_timeout(handle: JoinHandle<()>, timeout_ms: u64) -> bool {
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    let waiter = std::thread::spawn(move || {
        let _ = handle.join();
        let _ = tx.send(());
    });
    match rx.recv_timeout(Duration::from_millis(timeout_ms)) {
        Ok(()) => {
            let _ = waiter.join();
            true
        }
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            // Waiter is now stuck inside `handle.join()`. Detach it.
            std::mem::forget(waiter);
            false
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            let _ = waiter.join();
            false
        }
    }
}

