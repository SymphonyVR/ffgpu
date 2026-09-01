use crate::{
    decode::{
        Clock, DecoderState, Frame, FrameQueue, PacketQueueMetadata, PacketReceiver, PacketSender,
        PlayState, read::Input,
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
            time_base: decoder.time_base(),
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

        let mut frame = unsafe { ffn::Frame::empty() };

        let mut skip_to_ts = None;

        'exit: while self.state.alive.load(Ordering::Relaxed) {
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

            let mut prev_frame = None;

            while self.state.alive.load(Ordering::Relaxed) {
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

                    self.frame_queue.send(frame, packet_serial);
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
                    self.flush();
                    packet_serial = packet.serial;
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
        std::thread::spawn(move || {
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

        let last_pts_clone = last_pts.clone();
        let last_serial_clone = last_serial.clone();
        let write_count_clone = write_count.clone();
        let read_count_clone = read_count.clone();
        let anchor_clone = anchor.clone();
        let state_clone = pbs.clone();
        let queue_clone = queue.clone();
        
        // Background thread to continuously pull decoded frames into the lock-free ringbuffer
        std::thread::spawn(move || {
            let mut last_serial = u32::MAX;
            let mut local_write_count = 0u64;

            while state_clone.alive.load(Ordering::Relaxed) {
                if state_clone.play_state() == PlayState::Paused {
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }

                // Pacing: only pull frame if we have < 100ms buffered
                let rate = state_clone.audio_stream.load().metadata.sample_rate;
                let channels = state_clone.audio_stream.load().metadata.channels;
                
                // Target fill: 100ms of audio (keeps latency low)
                let target_fill_samples = (rate * channels as u32 / 10) as usize;
                
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
                    local_write_count += pushed as u64;
                    write_count_clone.store(local_write_count, Ordering::SeqCst);
                    last_pts_clone.store(pts.to_bits(), Ordering::SeqCst);
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
        };
        sink.set_parameters(AudioParameters {
            sample_rate: 48000,
            channels: 2,
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
        use ringbuf::traits::Consumer;
        let gain = gain.max(0.);
        out.fill(0.);

        let time = ffn::time::relative() as f64 / 1000000.;

        if self.state.play_state() == PlayState::Paused {
            return Ok(());
        }

        let current_serial = self.queue.serial.load(Ordering::Relaxed);
        let active_serial = self.last_serial.load(Ordering::Relaxed);
        
        if active_serial != current_serial {
            // Drop audio if we are seeking
            return Ok(());
        }

        let popped = self.consumer.pop_slice(out);
        
        for i in 0..popped {
            out[i] *= gain;
        }

        self.read_count.fetch_add(popped as u64, Ordering::SeqCst);

        // FIX: only update the clock when we actually consumed audio.
        // On underflow (popped < out.len()), the clock must not advance,
        // otherwise it drifts ahead of real audio at wall-clock rate.
        if popped > 0 {
            let anchor = self.anchor.load();
            if anchor.serial == current_serial && !anchor.pts.is_nan() {
                let r_count = self.read_count.load(Ordering::SeqCst);
                let unread = anchor.write_count.saturating_sub(r_count);
                let target_rate = self.parameters.sample_rate * self.parameters.channels as u32;
                self.clock.set(
                    anchor.pts - unread as f64 / target_rate as f64,
                    current_serial,
                    Some(time),
                );
            }
        }

        Ok(())
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
            gain,
            stream,
            alive,
            monitor: Some(monitor),
            stop,
        }
    }
}

// ── Free helper: negotiate device format, update resampler, build stream ──────

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
    let target_rate    = sample_rate * channels as u32;

    let read_samples = move |out: &mut [f32]| {
        use ringbuf::traits::Consumer;

        out.fill(0.0);

        if state_cb.play_state() == crate::decode::PlayState::Paused {
            return;
        }

        let g = gain_cb.load(Ordering::Relaxed).max(0.0);
        let mut cons = consumer_cb.lock().unwrap();

        let popped = cons.pop_slice(out);
        for s in out[..popped].iter_mut() {
            *s *= g;
        }

        read_count_cb.fetch_add(popped as u64, Ordering::SeqCst);

        // FIX: only update the clock when we actually consumed audio.
        // On underflow (popped < out.len()), the clock must not advance,
        // otherwise it drifts ahead of real audio at wall-clock rate.
        if popped > 0 {
            let anchor = anchor_cb.load();
            let serial = last_serial_cb.load(Ordering::Relaxed);
            if anchor.serial == serial && !anchor.pts.is_nan() {
                let r_count = read_count_cb.load(Ordering::SeqCst);
                let unread = anchor.write_count.saturating_sub(r_count);
                let time = ffmpeg_next::time::relative() as f64 / 1_000_000.0;
                clock_cb.set(
                    anchor.pts - unread as f64 / target_rate as f64,
                    serial,
                    Some(time),
                );
            }
        }
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_output_stream(
            &config.config(),
            move |data: &mut [f32], _| read_samples(data),
            |e| log::error!("[Audio] CPAL error: {e}"),
            None,
        ),
        cpal::SampleFormat::I16 => {
            let mut buf = Vec::<f32>::new();
            device.build_output_stream(
                &config.config(),
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
                &config.config(),
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
    /// Volume multiplier for the CPAL output callback (0.0 = mute, 1.0 = unity).
    pub fn set_gain(&self, gain: f32) {
        self.gain.store(gain.max(0.0), Ordering::Relaxed);
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
            // The thread sleeps in 100 ms intervals, so this joins in ≤100 ms.
            let _ = t.join();
        }
        // Pause and drop the active stream.
        if let Some(s) = self.stream.lock().unwrap().take() {
            use cpal::traits::StreamTrait;
            let _ = s.pause();
        }
    }
}

