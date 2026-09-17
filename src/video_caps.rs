//! Vulkan video decode admission: stream-profile classification checked
//! against the device's ACTUAL decode capabilities.
//!
//! Origin (bug-1789674498884): an HEVC Range-Extensions stream (yuv444p)
//! handed to the Vulkan hwaccel deadlocked inside `avcodec_open2` on an
//! RTX 2060 / driver 596.36 — the driver blocks during its own
//! session/capability setup for that stream and never returns an error, so
//! neither FFmpeg's in-decoder probe (`get_hw_format` negotiation) nor the
//! engine's `Err`-driven software fallback ever gets a chance to run. The
//! admission here runs BEFORE any of that: the device's supported decode
//! profiles are queried once per device via
//! `vkGetPhysicalDeviceVideoCapabilitiesKHR`, and any stream whose profile
//! the device did not explicitly report is denied hardware decode up front
//! and takes the software pipeline (which is always correct and never
//! deadlocks).
//!
//! v1 policy — default-deny, deliberately conservative:
//! - Only 4:2:0 H.264 / H.265 profiles are ever admitted, because those
//!   are the only profiles v1 queries (every Vulkan-Video driver must
//!   support them to be usable by FFmpeg's vulkan hwaccel at all).
//! - Rext / 4:2:2 / 4:4:4 (any HEVC profile beyond Main/Main10/Main12, any
//!   H.264 profile beyond High/High10), unknown profiles, and non-H264/H265
//!   codecs are all denied to software. Silicon capability (Turing NVDEC
//!   CAN decode HEVC 4:4:4) does not imply the Vulkan driver exposes those
//!   profiles; when a future driver does, extend the caps query and add
//!   an SPS-level classifier instead of loosening the default-deny rule.

/// Capabilities queried once per Vulkan device via
/// `vkGetPhysicalDeviceVideoCapabilitiesKHR`. All-false means "deny
/// everything" (also the `Default` used when the query fails — fail-closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VulkanVideoDecodeCaps {
    /// H.264 High, 4:2:0, 8-bit.
    pub h264_420_8: bool,
    /// H.264 High 10, 4:2:0, 10-bit.
    pub h264_420_10: bool,
    /// HEVC Main, 4:2:0, 8-bit.
    pub h265_420_8: bool,
    /// HEVC Main 10, 4:2:0, 10-bit.
    pub h265_420_10: bool,
    /// HEVC Main 12, 4:2:0, 12-bit.
    pub h265_420_12: bool,
}

/// Codec class used by admission. Only H.264/HEVC are ever candidates for
/// Vulkan hw decode in v1; everything else is software.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCodecClass {
    H264,
    H265,
    Other,
}

/// Chroma subsampling class. v1 admits only `Chroma420`; `Unknown` (any
/// profile whose chroma cannot be determined from the codec parameters)
/// denies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamChroma {
    Chroma420,
    /// 4:2:2 / 4:4:4 / monochrome — never admitted in v1.
    Other,
    Unknown,
}

/// A stream's decode-relevant profile, classified from codec parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamProfile {
    pub codec: StreamCodecClass,
    pub chroma: StreamChroma,
    pub bit_depth: u8,
}

// libavcodec FF_PROFILE_* ids (defs.h). Local copies so the classifier and
// its tests do not need FFI constants at the pure layer.
pub const FF_PROFILE_UNKNOWN: i32 = -99;
pub const FF_PROFILE_H264_BASELINE: i32 = 66;
pub const FF_PROFILE_H264_MAIN: i32 = 77;
pub const FF_PROFILE_H264_EXTENDED: i32 = 88;
pub const FF_PROFILE_H264_HIGH: i32 = 100;
pub const FF_PROFILE_H264_HIGH10: i32 = 110;
pub const FF_PROFILE_H264_HIGH422: i32 = 122;
pub const FF_PROFILE_H264_HIGH444_PREDICTIVE: i32 = 244;
pub const FF_PROFILE_HEVC_MAIN: i32 = 1;
pub const FF_PROFILE_HEVC_MAIN10: i32 = 2;
pub const FF_PROFILE_HEVC_MAIN_STILL: i32 = 3;
pub const FF_PROFILE_HEVC_MAIN12: i32 = 4;
// HEVC 5..10 are the Main422/Main444 Rext family — all denied in v1.
pub const FF_PROFILE_HEVC_MAIN422: i32 = 5;
pub const FF_PROFILE_HEVC_MAIN444: i32 = 8;

// libavcodec AV_CODEC_ID_* (all codecs, defs.h) for the raw classifier.
pub const AV_CODEC_ID_H264: i32 = 27;
pub const AV_CODEC_ID_HEVC: i32 = 173;

/// Classify a stream from its `AVCodecID` and `AVCodecParameters::profile`
/// (the only metadata available before a decoder is opened). Coarse by
/// design: anything ambiguous maps to `Unknown` chroma and denies.
pub fn classify(codec_id: i32, profile_id: i32) -> StreamProfile {
    match codec_id {
        AV_CODEC_ID_H264 => classify_h264(profile_id),
        AV_CODEC_ID_HEVC => classify_h265(profile_id),
        _ => StreamProfile {
            codec: StreamCodecClass::Other,
            chroma: StreamChroma::Unknown,
            bit_depth: 0,
        },
    }
}

fn classify_h264(profile_id: i32) -> StreamProfile {
    let (chroma, depth) = match profile_id {
        FF_PROFILE_H264_BASELINE
        | FF_PROFILE_H264_MAIN
        | FF_PROFILE_H264_EXTENDED
        | FF_PROFILE_H264_HIGH => (StreamChroma::Chroma420, 8),
        FF_PROFILE_H264_HIGH10 => (StreamChroma::Chroma420, 10),
        // High422 / High444 and unknown H264 profiles: deny (either a
        // non-4:2:0 variant or unclassifiable — software is always safe).
        _ => (StreamChroma::Unknown, 0),
    };
    StreamProfile {
        codec: StreamCodecClass::H264,
        chroma,
        bit_depth: depth,
    }
}

fn classify_h265(profile_id: i32) -> StreamProfile {
    let (chroma, depth) = match profile_id {
        FF_PROFILE_HEVC_MAIN => (StreamChroma::Chroma420, 8),
        FF_PROFILE_HEVC_MAIN10 => (StreamChroma::Chroma420, 10),
        FF_PROFILE_HEVC_MAIN12 => (StreamChroma::Chroma420, 12),
        // Main-still, the entire Rext 422/444 family, and unknown profiles
        // (mp4/hvcC streams often report FF_PROFILE_UNKNOWN until the SPS
        // is parsed — v1 does not parse bitstreams) all deny.
        _ => (StreamChroma::Unknown, 0),
    };
    StreamProfile {
        codec: StreamCodecClass::H265,
        chroma,
        bit_depth: depth,
    }
}

/// Pure admission decision: does the device's capability table explicitly
/// support this stream's profile? Default-deny — `Unknown` chroma, unknown
/// bit depth, or a codec without a caps entry always returns `false`.
pub fn admit_vulkan_decode(caps: &VulkanVideoDecodeCaps, stream: &StreamProfile) -> bool {
    if stream.chroma != StreamChroma::Chroma420 {
        return false;
    }
    match (stream.codec, stream.bit_depth) {
        (StreamCodecClass::H264, 8) => caps.h264_420_8,
        (StreamCodecClass::H264, 10) => caps.h264_420_10,
        (StreamCodecClass::H265, 8) => caps.h265_420_8,
        (StreamCodecClass::H265, 10) => caps.h265_420_10,
        (StreamCodecClass::H265, 12) => caps.h265_420_12,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caps shaped like the RTX 2060 / 596.36 driver observed in
    /// bug-1789674498884: 4:2:0 profiles exposed, no 422/444 profiles.
    fn rtx2060_caps() -> VulkanVideoDecodeCaps {
        VulkanVideoDecodeCaps {
            h264_420_8: true,
            h264_420_10: true,
            h265_420_8: true,
            h265_420_10: true,
            h265_420_12: true,
        }
    }

    #[test]
    fn default_caps_deny_everything_fail_closed() {
        let caps = VulkanVideoDecodeCaps::default();
        for (codec_id, profile_id) in [
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN10),
        ] {
            let profile = classify(codec_id, profile_id);
            assert!(
                !admit_vulkan_decode(&caps, &profile),
                "{codec_id}/{profile_id} must deny on all-false caps"
            );
        }
    }

    #[test]
    fn rtx2060_admits_only_420_profiles() {
        let caps = rtx2060_caps();
        let admitted = [
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_MAIN),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH10),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN10),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN12),
        ];
        for (codec_id, profile_id) in admitted {
            let profile = classify(codec_id, profile_id);
            assert!(
                admit_vulkan_decode(&caps, &profile),
                "{codec_id}/{profile_id} (420) must admit with 420-only caps"
            );
        }
    }

    #[test]
    fn rtx2060_denies_444_rext_and_422() {
        // The bug-1789674498884 family: HEVC Rext/422/444 profiles must
        // never reach the Vulkan hwaccel on a 420-only driver.
        let caps = rtx2060_caps();
        let denied = [
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN422),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN444),
            (AV_CODEC_ID_HEVC, FF_PROFILE_UNKNOWN),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN_STILL),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH422),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH444_PREDICTIVE),
            (AV_CODEC_ID_H264, FF_PROFILE_UNKNOWN),
        ];
        for (codec_id, profile_id) in denied {
            let profile = classify(codec_id, profile_id);
            assert!(
                !admit_vulkan_decode(&caps, &profile),
                "{codec_id}/{profile_id} must deny with 420-only caps"
            );
        }
    }

    #[test]
    fn unknown_profile_id_classifies_to_unknown_and_denies() {
        // mp4/hvcC streams frequently report FF_PROFILE_UNKNOWN until the
        // SPS is parsed; v1 must deny (software) rather than guess.
        let profile = classify(AV_CODEC_ID_HEVC, FF_PROFILE_UNKNOWN);
        assert_eq!(profile.chroma, StreamChroma::Unknown);
        assert!(!admit_vulkan_decode(&rtx2060_caps(), &profile));

        let h264 = classify(AV_CODEC_ID_H264, FF_PROFILE_UNKNOWN);
        assert_eq!(h264.chroma, StreamChroma::Unknown);
        assert!(!admit_vulkan_decode(&rtx2060_caps(), &h264));
    }

    #[test]
    fn non_h264_h265_codecs_deny() {
        // VP9 / AV1 / anything else: software in v1.
        let profile = classify(-1, FF_PROFILE_UNKNOWN); // any foreign codec id
        assert_eq!(profile.codec, StreamCodecClass::Other);
        assert!(!admit_vulkan_decode(&rtx2060_caps(), &profile));
    }

    #[test]
    fn partial_caps_selectively_admit() {
        // A driver exposing only 8-bit 420 HEVC: H264 denies, HEVC Main
        // admits, HEVC Main10 denies.
        let caps = VulkanVideoDecodeCaps {
            h265_420_8: true,
            ..Default::default()
        };
        assert!(!admit_vulkan_decode(
            &caps,
            &classify(AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH)
        ));
        assert!(admit_vulkan_decode(
            &caps,
            &classify(AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN)
        ));
        assert!(!admit_vulkan_decode(
            &caps,
            &classify(AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN10)
        ));
    }
}
