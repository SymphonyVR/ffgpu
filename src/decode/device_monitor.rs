//! Cross-platform audio device change detection for ffgpu.
//!
//! Mirrors the technique in `video-rs/src/audio/device_monitor.rs`.
//!
//! Platform strategy:
//! - Windows : WASAPI `IMMNotificationClient` via `windows-core` (primary)
//!             + CPAL device-name polling (fallback, always active in monitor thread)
//! - macOS   : CoreAudio `AudioObjectAddPropertyListener`
//! - Linux   : PulseAudio/PipeWire subscription callbacks
//!
//! On Windows we use the CPAL name-comparison fallback (built into the monitor
//! thread in `audio.rs`) as the primary detection path, since it requires no
//! additional crate re-exports and works perfectly at a 500 ms polling interval.
//! The native listener would add a faster path but is left as a future
//! improvement once `windows-core` stabilises its proc-macro ABI.

use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag set when the default audio output device changes.
static AUDIO_DEVICE_CHANGED: AtomicBool = AtomicBool::new(false);

/// Called by platform-specific listeners when a change is detected.
pub fn mark_audio_device_changed() {
    AUDIO_DEVICE_CHANGED.store(true, Ordering::Relaxed);
}

/// Check (and atomically clear) the device-changed flag.
/// Returns `true` once per change event.
pub fn audio_device_changed() -> bool {
    AUDIO_DEVICE_CHANGED.swap(false, Ordering::Relaxed)
}

/// Spawn a background thread that registers native OS audio listeners where
/// supported. On Windows the CPAL name-comparison fallback in the monitor
/// thread is the primary mechanism; this function is a no-op there.
pub fn start_device_monitor() {
    // macOS and Linux have blocking / callback-based listeners that need their
    // own thread. Windows relies on the CPAL name-comparison poll in audio.rs.
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    std::thread::spawn(|| {
        let _ = register_audio_listener();
    });
}

// ─── macOS ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
mod macos_impl {
    use coreaudio_sys::{
        AudioObjectAddPropertyListener, AudioObjectID, AudioObjectPropertyAddress,
        OSStatus, kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal,
        kAudioObjectSystemObject,
    };
    use std::os::raw::c_void;
    use super::mark_audio_device_changed;

    extern "C" fn device_changed(
        _object: AudioObjectID,
        _n: u32,
        _addrs: *const AudioObjectPropertyAddress,
        _data: *mut c_void,
    ) -> OSStatus {
        log::debug!("[AudioMonitor] macOS: default playback device changed");
        mark_audio_device_changed();
        0
    }

    pub fn register_audio_listener() {
        let addr = AudioObjectPropertyAddress {
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain,
        };
        unsafe {
            AudioObjectAddPropertyListener(
                kAudioObjectSystemObject,
                &addr,
                Some(device_changed),
                std::ptr::null_mut(),
            );
        }
        // Keep thread alive so the listener stays registered.
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
        }
    }
}

// ─── Linux ───────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod linux_impl {
    use libpulse_binding::{
        context::{
            subscribe::{Facility, InterestMaskSet, Operation},
            Context, FlagSet,
        },
        mainloop::standard::Mainloop,
    };
    use super::mark_audio_device_changed;

    pub fn register_audio_listener() {
        let mut mainloop = Mainloop::new().expect("PulseAudio mainloop");
        let mut ctx = Context::new(&mainloop, "ffgpu-audio-monitor")
            .expect("PulseAudio context");
        ctx.connect(None, FlagSet::NOFLAGS, None)
            .expect("PulseAudio connect");

        ctx.set_subscribe_callback(Some(Box::new(|facility, op, _idx| {
            let relevant_facility =
                facility == Some(Facility::Sink) || facility == Some(Facility::Server);
            let relevant_op =
                op == Some(Operation::Changed) || op == Some(Operation::New);
            if relevant_facility && relevant_op {
                log::debug!("[AudioMonitor] Linux: audio device/server changed");
                mark_audio_device_changed();
            }
        })));

        ctx.subscribe(InterestMaskSet::SINK | InterestMaskSet::SERVER, |_| {});

        // Blocks until PulseAudio shuts down.
        mainloop.run().expect("PulseAudio mainloop run");
    }
}

// ─── Platform dispatcher ─────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn register_audio_listener() -> Result<(), Box<dyn std::error::Error>> {
    macos_impl::register_audio_listener();
    Ok(())
}

#[cfg(target_os = "linux")]
fn register_audio_listener() -> Result<(), Box<dyn std::error::Error>> {
    linux_impl::register_audio_listener();
    Ok(())
}
