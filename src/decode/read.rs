use crate::{
    SeekMode,
    decode::{DecoderState, PlayState, audio, video},
    error::Result,
};
use crossbeam_channel::Receiver;
use ffmpeg_next::{self as ffn, sys as ff};
use std::{
    path::Path,
    sync::{Arc, atomic::Ordering},
    thread::JoinHandle,
    time::Duration,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Metadata {
    pub duration: Duration,
    /// Container start offset in AV_TIME_BASE units (`AV_NOPTS_VALUE` when
    /// the format carries none). Public positions are relative to this, per
    /// the ffplay model.
    pub start_time: i64,
}

pub(crate) struct Input {
    pub format_ctx: ffn::format::context::Input,
    pub metadata: Metadata,
}

impl Input {
    pub fn open<P>(path: &P) -> Result<Self>
    where
        P: AsRef<Path> + ?Sized,
    {
        let format_ctx = ffn::format::input(path)?;

        let metadata = metadata_from_format(&format_ctx);

        Ok(Input {
            format_ctx,
            metadata,
        })
    }

    /// Open a video with an ffmpeg interrupt callback wired to
    /// `state.alive`. When `state.kill()` (or any code that flips
    /// `state.alive` to false) is called, the next ffmpeg blocking I/O
    /// call (av_read_frame) returns AVERROR(EINTR) immediately, unblocking
    /// the read thread so it can observe alive=false and exit. Without
    /// this, the read thread can sit inside av_read_frame for many
    /// seconds (disk I/O, looping video) and defeat `wait_to_finish()`.
    ///
    /// The caller must have already constructed `state` (use
    /// `DecoderState::empty()` and follow up with `state.install()` once
    /// the real streams are wired).
    pub fn open_with_state<P>(path: &P, state: Arc<DecoderState>) -> Result<Self>
    where
        P: AsRef<Path> + ?Sized,
    {
        let alive = state.alive.clone();
        // ffmpeg-next's interrupt callback: returns `true` to abort,
        // `false` to continue. We abort when alive is false.
        let format_ctx = ffn::format::input_with_interrupt(path, move || {
            !alive.load(Ordering::Relaxed)
        })?;

        let metadata = metadata_from_format(&format_ctx);

        Ok(Input {
            format_ctx,
            metadata,
        })
    }
}

fn metadata_from_av_duration(duration: i64) -> Metadata {
    Metadata {
        duration: if duration > 0 {
            Duration::from_micros(duration as u64)
        } else {
            Duration::ZERO
        },
        start_time: ff::AV_NOPTS_VALUE,
    }
}

fn stream_duration_us(stream: &ffn::format::stream::Stream<'_>) -> Option<i64> {
    let duration = stream.duration();
    if duration > 0 {
        return Some(unsafe {
            ff::av_rescale_q(duration, stream.time_base().into(), ff::AV_TIME_BASE_Q)
        });
    }

    let frames = stream.frames();
    let frame_rate = stream.avg_frame_rate();
    if frames <= 0 || frame_rate.0 <= 0 || frame_rate.1 <= 0 {
        return None;
    }

    let seconds = frames as f64 * f64::from(frame_rate.invert());
    (seconds.is_finite() && seconds > 0.0).then(|| (seconds * 1_000_000.0).round() as i64)
}

fn metadata_from_format(format_ctx: &ffn::format::context::Input) -> Metadata {
    // Container start offset (AV_TIME_BASE units) drives the public timeline:
    // `seek_target_us` adds it to relative positions and `video.position()`
    // subtracts it from absolute PTS. Absent without a timer overlay; strictly
    // add `start_time` only when positive (ffplay treats it as AV_NOPTS).
    let start_time = unsafe { (*format_ctx.as_ptr()).start_time };

    let duration = format_ctx.duration();
    if duration > 0 {
        let mut metadata = metadata_from_av_duration(duration);
        metadata.start_time = start_time;
        return metadata;
    }

    let video_duration = format_ctx
        .streams()
        .filter(|stream| stream.parameters().medium() == ffn::media::Type::Video)
        .filter_map(|stream| stream_duration_us(&stream))
        .max();

    if let Some(duration) = video_duration {
        let mut metadata = metadata_from_av_duration(duration);
        metadata.start_time = start_time;
        return metadata;
    }

    let stream_duration = format_ctx
        .streams()
        .filter_map(|stream| stream_duration_us(&stream))
        .max()
        .unwrap_or(0);

    let mut metadata = metadata_from_av_duration(stream_duration);
    metadata.start_time = start_time;
    metadata
}

#[cfg(test)]
mod tests {
    use super::metadata_from_av_duration;
    use ffmpeg_next::sys as ff;
    use std::time::Duration;

    #[test]
    fn av_duration_uses_microseconds_and_zero_for_unknown() {
        assert_eq!(
            metadata_from_av_duration(2_142_890_000).duration,
            Duration::from_micros(2_142_890_000),
        );
        assert_eq!(
            metadata_from_av_duration(ff::AV_NOPTS_VALUE).duration,
            Duration::ZERO,
        );
    }
}

#[derive(Debug)]
pub(crate) enum ReadMessage {
    SeekStream { ts: i64, mode: SeekMode, forward: bool },
}

pub(crate) struct ReadThread {
    input: Input,
    state: Arc<DecoderState>,
    messages: Receiver<ReadMessage>,
}

impl ReadThread {
    pub fn new(input: Input, pbs: Arc<DecoderState>, messages: Receiver<ReadMessage>) -> Self {
        ReadThread {
            input,
            state: pbs,
            messages,
        }
    }

    fn run_thread(&mut self) {
        const MAX_QUEUE_SIZE: usize = 15 * 1024 * 1024;

        while self.state.alive.load(Ordering::Relaxed) {
            let play_state = self.state.play_state();
            // TODO: for network streams
            /*if self.was_paused != paused {
                self.was_paused = paused;
                if let Err(error) = if self.was_paused {
                    format_ctx.pause()
                } else {
                    format_ctx.play()
                } {
                    dbg!(std::ffi::c_int::from(error));
                    log::error!("failed to play/pause stream: {}", error);
                }
            }*/

            let video_stream = self.state.video_stream.load().clone();
            let audio_stream = self.state.audio_stream.load().clone();

            while let Ok(message) = self.messages.try_recv() {
                match message {
                    ReadMessage::SeekStream { ts, mode, forward: _ } => {
                        // Always seek BACKWARD (like ffplay). A "forward"
                        // seek (seek flags = 0, next keyframe after target)
                        // only helps when the container has a real index
                        // (mp4/mkv Cues). For Cue-less WebM there is no future
                        // index yet, so avformat_seek_file cannot advance: the
                        // demuxer stays near the start and the audio decoder
                        // must walk the whole clip discarding frames — the
                        // "no audio after seek" stall. BACKWARD lands on the
                        // previous keyframe (always seekable) and the decode
                        // walk with SkipToTimestamp handles the rest.
                        let seek_flags = ff::AVSEEK_FLAG_BACKWARD;
                        if let Err(error) = {
                            let err = unsafe {
                                ff::avformat_seek_file(
                                    self.input.format_ctx.as_mut_ptr(),
                                    -1,
                                    i64::MIN,
                                    ts,
                                    i64::MAX,
                                    seek_flags,
                                )
                            };

                            (err == 0).then_some(()).ok_or(ffn::Error::from(err))
                        } {
                            log::error!("failed to seek stream: {}", error);
                        } else {
                            self.state.is_eof.store(false, Ordering::SeqCst);

video_stream.packets.flush();
                            video_stream.packets.set_loop_index(0);

                            audio_stream.packets.flush();
                            audio_stream.packets.set_loop_index(0);
                            self.state.loop_index.store(0, Ordering::SeqCst);

                            match mode {
                                SeekMode::Accurate => {
                                    let _ = video_stream
                                        .messages
                                        .send(video::Message::SkipToTimestamp(ts));
                                    let _ = audio_stream
                                        .messages
                                        .send(audio::Message::SkipToTimestamp(ts));
                                }
                                _ => {}
                            }

                            if play_state == PlayState::Paused {
                                self.state
                                    .play_state
                                    .store(PlayState::Step as _, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }

// Backpressure gate: sleep once BOTH the video AND audio
            // pipelines are primed (count-based, so short clips throttle too).
            // A no-audio stream has `metadata.index == usize::MAX` and is
            // treated as always-primed — otherwise it would gate forever and
            // the queue would grow unbounded (the original leak).
            // `has_enough_packets` is count-only; the old `buffered > 1.0s`
            // bound was unreachable for clips shorter than a second.
            let video_primed = video_stream.packets.has_enough_packets();
            let audio_primed = audio_stream.metadata.index == usize::MAX
                || audio_stream.packets.has_enough_packets();
            if (video_stream.packets.tx.len() + audio_stream.packets.tx.len()/*+ self.subtitle_tx.packets.len()*/)
                > MAX_QUEUE_SIZE
                || (video_primed && audio_primed)
            {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            let is_eof = self.state.is_eof.load(Ordering::SeqCst);

            if play_state == PlayState::Playing && is_eof && video_stream.frames.queue_rx.is_empty()
            {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }

            let mut packet = ffn::Packet::empty();
            match packet.read(&mut self.input.format_ctx) {
                Ok(_) => {
                    self.state.is_eof.store(false, Ordering::SeqCst);
                }
                Err(error) => {
                    if !is_eof
                        && (error == ffn::Error::Eof
                            || unsafe { ff::avio_feof((*self.input.format_ctx.as_ptr()).pb) != 0 })
                    {
                        // Keep the end marker in front of the next iteration so decoder
                        // delay is drained before the first frame of the loop is sent.
                        video_stream
                            .packets
                            .push_null(ffn::Packet::empty(), video_stream.metadata.index);
                        audio_stream
                            .packets
                            .push_null(ffn::Packet::empty(), audio_stream.metadata.index);

                        if self.state.looping.load(Ordering::Relaxed) {
                            let rewind = unsafe {
                                ff::avformat_seek_file(
                                    self.input.format_ctx.as_mut_ptr(),
                                    -1,
                                    i64::MIN,
                                    0,
                                    i64::MAX,
                                    ff::AVSEEK_FLAG_BACKWARD,
                                )
                            } == 0;

                            if rewind {
                                let next_loop = self.state.loop_index.fetch_add(1, Ordering::SeqCst) + 1;
                                video_stream.packets.set_loop_index(next_loop);
                                audio_stream.packets.set_loop_index(next_loop);
                                self.state.loop_events.fetch_add(1, Ordering::SeqCst);
                                self.state.is_eof.store(false, Ordering::SeqCst);
                                continue;
                            }
                        }

                        self.state.is_eof.store(true, Ordering::SeqCst);
                    }

                    let avio_error = unsafe { (*(*self.input.format_ctx.as_ptr()).pb).error };
                    if avio_error != 0 {
                        log::error!("AVIOContext error {}", avio_error);
                    }

                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
            }

            let stream_index = packet.stream();
            if stream_index == video_stream.metadata.index {
                video_stream.packets.push(packet);
            } else if stream_index == audio_stream.metadata.index {
                audio_stream.packets.push(packet);
            }

            // TODO: handle subtitle packets
        }

        // force the other threads to wake up in order to exit from alive=false
        let video_stream = self.state.video_stream.load().clone();
        let audio_stream = self.state.audio_stream.load().clone();
        video_stream
            .packets
            .push_null(ffn::Packet::empty(), video_stream.metadata.index);
        audio_stream
            .packets
            .push_null(ffn::Packet::empty(), audio_stream.metadata.index);
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

unsafe impl Send for ReadThread {}
unsafe impl Sync for ReadThread {}
