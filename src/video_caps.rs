//! Vulkan video decode admission: stream-profile classification checked
//! against the device's ACTUAL Vulkan decode capabilities, and the
//! decode-route policy that turns a Vulkan denial into the right fallback.
//!
//! Origin (bug-1789674498884): an HEVC Range-Extensions stream (yuv444p)
//! handed to the Vulkan hwaccel deadlocked inside `avcodec_open2` on an
//! RTX 2060 / driver 596.36 — the driver blocks during its own
//! session/capability setup for that stream and never returns an error, so
//! neither FFmpeg's in-decoder probe (`get_hw_format` negotiation) nor the
//! engine's `Err`-driven software fallback ever gets a chance to run. The
//! admission here runs BEFORE any of that: the device's supported Vulkan
//! decode profiles are queried once per device via
//! `vkGetPhysicalDeviceVideoCapabilitiesKHR`, and any stream whose profile
//! the device did not explicitly report is routed away from the Vulkan
//! hwaccel up front.
//!
//! Routing policy (user ruling 2026-09-17): a Vulkan denial is
//! BACKEND-LOCAL. A profile missing from the Vulkan matrix does not imply
//! no hardware decoder can handle it — the pre-fix logs prove the Windows
//! D3D11VA/NVDEC path decoded these exact REXT 4:4:4 streams. So:
//! admitted -> Vulkan; unreported on Windows -> D3D11VA (which carries its
//! own probe ladder down to software); elsewhere -> software. The
//! deadlock stays impossible because the blocking path was the Vulkan
//! handoff, which a denial never reaches.
//!
//! v1 classification policy — deliberately coarse and fail-closed:
//! - Only 4:2:0 H.264 (Baseline/Main/Extended/High/High10, including
//!   FFmpeg's constraint-flag variants) and HEVC Main/Main10 profiles ever
//!   admit. HEVC REXT (FFmpeg profile 4) covers 4:2:0/4:2:2/4:4:4 at
//!   8-12 bits and its chroma is NOT determinable from the profile id
//!   alone without SPS parsing — v1 does not parse bitstreams, so REXT
//!   (and Main-Still, unknowns, non-H264/H265 codecs) deny Vulkan.
//! - When a future driver exposes 4:4:4 decode profiles, extend the caps
//!   query AND add an SPS-level classifier instead of loosening the deny.

/// Capabilities queried once per Vulkan device via
/// `vkGetPhysicalDeviceVideoCapabilitiesKHR`. All-false means "deny
/// everything" (also the `Default` used when the query fails — fail-closed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VulkanVideoDecodeCaps {
    /// H.264 High (or constraint-masked equivalent), 4:2:0, 8-bit.
    pub h264_420_8: bool,
    /// H.264 High 10, 4:2:0, 10-bit.
    pub h264_420_10: bool,
    /// HEVC Main, 4:2:0, 8-bit.
    pub h265_420_8: bool,
    /// HEVC Main 10, 4:2:0, 10-bit.
    pub h265_420_10: bool,
}

/// Codec class used by admission. Only H.264/HEVC are ever candidates for
/// Vulkan hw decode in v1; everything else routes away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamCodecClass {
    H264,
    H265,
    Other,
}

/// Chroma subsampling class. v1 admits only `Chroma420`; `Unknown` (any
/// profile whose chroma cannot be determined from the codec parameters
/// alone — e.g. HEVC REXT) routes away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamChroma {
    Chroma420,
    /// 4:2:2 / 4:4:4 / monochrome — never Vulkan-admitted in v1.
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

/// The hardware-decode route for one stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeRoute {
    /// Device explicitly reported the stream's profile: use the Vulkan
    /// hwaccel.
    Vulkan,
    /// Vulkan denial on Windows: try the D3D11VA/NVDEC path next (it owns
    /// its own probe ladder down to software). User ruling 2026-09-17.
    D3d11va,
    /// Vulkan denial elsewhere (or no Vulkan context at all): software.
    Software,
}

// libavcodec FF_PROFILE_* ids (defs.h). Local copies so the classifier and
// its tests do not need FFI constants at the pure layer.
pub const FF_PROFILE_UNKNOWN: i32 = -99;
// H.264 base profiles. FFmpeg OR-s constraint bits into codecpar.profile:
// CONSTRAINED = 1<<9 (constrained baseline 66|512 = 578), INTRA = 1<<11
// (High10 Intra = 2158 etc.). FFmpeg's own Vulkan hwaccel masks these bits
// before querying the driver (vulkan_decode.c), because drivers answer the
// BASE profiles; the classifier mirrors that mask.
pub const FF_PROFILE_H264_CONSTRAINED: i32 = 1 << 9;
pub const FF_PROFILE_H264_INTRA: i32 = 1 << 11;
pub const FF_PROFILE_H264_BASELINE: i32 = 66;
pub const FF_PROFILE_H264_MAIN: i32 = 77;
pub const FF_PROFILE_H264_EXTENDED: i32 = 88;
pub const FF_PROFILE_H264_HIGH: i32 = 100;
pub const FF_PROFILE_H264_HIGH10: i32 = 110;
// Stereo High (128) and Multiview High (118) are High-profile toolset
// variants FFmpeg maps onto the High std profile; they admit as 420/8.
pub const FF_PROFILE_H264_MULTIVIEW_HIGH: i32 = 118;
pub const FF_PROFILE_H264_STEREO_HIGH: i32 = 128;
pub const FF_PROFILE_H264_HIGH422: i32 = 122;
pub const FF_PROFILE_H264_HIGH444_PREDICTIVE: i32 = 244;
// HEVC profiles: the FFmpeg enum ends at REXT — there is no MAIN12 nor
// MAIN422/MAIN444 id (an earlier revision of this file fabricated those;
// corrected 2026-09-18 per the red-team audit).
pub const FF_PROFILE_HEVC_MAIN: i32 = 1;
pub const FF_PROFILE_HEVC_MAIN10: i32 = 2;
pub const FF_PROFILE_HEVC_MAIN_STILL: i32 = 3;
pub const FF_PROFILE_HEVC_REXT: i32 = 4;

// libavcodec AV_CODEC_ID_* (defs.h) for the raw classifier.
pub const AV_CODEC_ID_H264: i32 = 27;
pub const AV_CODEC_ID_HEVC: i32 = 173;

/// Classify a stream from its `AVCodecID` and `AVCodecParameters::profile`
/// (the only metadata available before a decoder is opened). Coarse by
/// design: anything ambiguous maps to `Unknown` chroma and routes away
/// from Vulkan.
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
    // Mirror FFmpeg's constraint mask (vulkan_decode.c): drivers answer
    // the base profiles, so 578 (constrained baseline) / 2158 (High10
    // Intra) etc. reduce to their bases before classification.
    let masked = profile_id & !(FF_PROFILE_H264_CONSTRAINED | FF_PROFILE_H264_INTRA);
    let (chroma, depth) = match masked {
        FF_PROFILE_H264_BASELINE
        | FF_PROFILE_H264_MAIN
        | FF_PROFILE_H264_EXTENDED
        | FF_PROFILE_H264_HIGH
        | FF_PROFILE_H264_MULTIVIEW_HIGH
        | FF_PROFILE_H264_STEREO_HIGH => (StreamChroma::Chroma420, 8),
        FF_PROFILE_H264_HIGH10 => (StreamChroma::Chroma420, 10),
        // High422 / High444 and anything unclassifiable: route away from
        // Vulkan (D3D11VA's probe ladder decides the rest).
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
        // Main-Still, REXT (4:2:0/4:2:2/4:4:4 at 8-12 bits — chroma not
        // determinable from the id without SPS parsing), and unknown
        // profiles (mp4/hvcC streams often report FF_PROFILE_UNKNOWN
        // until the SPS is parsed) all route away from Vulkan.
        _ => (StreamChroma::Unknown, 0),
    };
    StreamProfile {
        codec: StreamCodecClass::H265,
        chroma,
        bit_depth: depth,
    }
}

/// Pure admission decision: did the device's Vulkan capability table
/// explicitly report this stream's profile? Default-deny — `Other`
/// chroma, unknown bit depth, or a codec without a caps entry is `false`.
pub fn admit_vulkan_decode(caps: &VulkanVideoDecodeCaps, stream: &StreamProfile) -> bool {
    if stream.chroma != StreamChroma::Chroma420 {
        return false;
    }
    match (stream.codec, stream.bit_depth) {
        (StreamCodecClass::H264, 8) => caps.h264_420_8,
        (StreamCodecClass::H264, 10) => caps.h264_420_10,
        (StreamCodecClass::H265, 8) => caps.h265_420_8,
        (StreamCodecClass::H265, 10) => caps.h265_420_10,
        _ => false,
    }
}

/// Decode-route policy (bug-1789674498884, user ruling 2026-09-17).
/// A Vulkan denial is backend-local: on Windows the stream goes to the
/// D3D11VA path (which owns its own probe ladder down to software), on
/// other platforms straight to software. No caps data at all (broken
/// driver / missing Vulkan context) also routes away — never Vulkan.
pub fn route_stream(
    caps: Option<&VulkanVideoDecodeCaps>,
    stream: &StreamProfile,
    windows: bool,
) -> DecodeRoute {
    let admitted = caps.is_some_and(|c| admit_vulkan_decode(c, stream));
    match (admitted, windows) {
        (true, _) => DecodeRoute::Vulkan,
        (false, true) => DecodeRoute::D3d11va,
        (false, false) => DecodeRoute::Software,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Caps shaped like a healthy NVIDIA driver: 4:2:0 profiles exposed,
    /// no 422/444 profiles.
    fn healthy_420_caps() -> VulkanVideoDecodeCaps {
        VulkanVideoDecodeCaps {
            h264_420_8: true,
            h264_420_10: true,
            h265_420_8: true,
            h265_420_10: true,
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
    fn healthy_driver_admits_only_420_profiles() {
        let caps = healthy_420_caps();
        let admitted = [
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_MAIN),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH10),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN10),
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
    fn rext_routes_away_even_with_all_caps_true() {
        // TRAP GUARD (red-team finding 2026-09-18): FFmpeg profile 4 is
        // HEVC REXT, not "MAIN12" — its chroma (420/422/444) and bit depth
        // are NOT determinable from the profile id alone. It must route
        // away from Vulkan on ANY caps table, including a hypothetical
        // future driver with every reported cap true. If this test fails,
        // someone added caps without adding the SPS-level classifier.
        let all_true = healthy_420_caps();
        let rext = classify(AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_REXT);
        assert_eq!(rext.chroma, StreamChroma::Unknown);
        assert!(
            !admit_vulkan_decode(&all_true, &rext),
            "REXT (profile 4) must never Vulkan-admit in v1"
        );
    }

    #[test]
    fn known_444_rext_family_values_route_away() {
        // The bug-1789674498884 family: HEVC REXT / MAIN_STILL / unknown
        // and H264 High422 / High444 route away from a 420-only driver.
        let caps = healthy_420_caps();
        let denied = [
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_REXT),
            (AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN_STILL),
            (AV_CODEC_ID_HEVC, FF_PROFILE_UNKNOWN),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH422),
            (AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH444_PREDICTIVE),
            (AV_CODEC_ID_H264, FF_PROFILE_UNKNOWN),
        ];
        for (codec_id, profile_id) in denied {
            let profile = classify(codec_id, profile_id);
            assert!(
                !admit_vulkan_decode(&caps, &profile),
                "{codec_id}/{profile_id} must Vulkan-deny with 420-only caps"
            );
        }
    }

    #[test]
    fn h264_constraint_flags_reduce_to_base_profiles() {
        // FFmpeg OR-s constraint bits into codecpar.profile; the classifier
        // must mirror FFmpeg's mask so constrained-baseline (578), High10
        // Intra (2158) and friends keep the Vulkan path they had pre-fix
        // on healthy drivers.
        let caps = healthy_420_caps();
        let constrained_baseline = FF_PROFILE_H264_BASELINE | FF_PROFILE_H264_CONSTRAINED; // 578
        let high10_intra = FF_PROFILE_H264_HIGH10 | FF_PROFILE_H264_INTRA; // 2158
        for profile_id in [constrained_baseline, high10_intra] {
            let profile = classify(AV_CODEC_ID_H264, profile_id);
            assert!(
                admit_vulkan_decode(&caps, &profile),
                "constraint-flagged H264 profile {profile_id} must admit like its base"
            );
        }
    }

    #[test]
    fn h264_stereo_and_multiview_high_admit_as_high() {
        let caps = healthy_420_caps();
        for profile_id in [FF_PROFILE_H264_MULTIVIEW_HIGH, FF_PROFILE_H264_STEREO_HIGH] {
            let profile = classify(AV_CODEC_ID_H264, profile_id);
            assert_eq!(profile.chroma, StreamChroma::Chroma420);
            assert_eq!(profile.bit_depth, 8);
            assert!(admit_vulkan_decode(&caps, &profile));
        }
    }

    #[test]
    fn route_policy_user_ruling_cases() {
        // User's three routing cases (2026-09-17 ruling):
        let caps = healthy_420_caps();
        // 1. Known Vulkan-supported profile -> Vulkan.
        let main = classify(AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN);
        assert_eq!(route_stream(Some(&caps), &main, true), DecodeRoute::Vulkan);
        assert_eq!(route_stream(Some(&caps), &main, false), DecodeRoute::Vulkan);
        // 2. Unreported Vulkan profile on Windows -> D3D11VA (its own
        //    probe ladder handles the hardware/software split).
        let rext = classify(AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_REXT);
        assert_eq!(route_stream(Some(&caps), &rext, true), DecodeRoute::D3d11va);
        // 3. Unreported profile elsewhere -> software.
        assert_eq!(
            route_stream(Some(&caps), &rext, false),
            DecodeRoute::Software
        );
        // No caps data at all (broken driver: all-false matrix, or no
        // Vulkan context) -> never Vulkan, anywhere.
        let none_caps = VulkanVideoDecodeCaps::default();
        assert_eq!(
            route_stream(Some(&none_caps), &main, true),
            DecodeRoute::D3d11va
        );
        assert_eq!(route_stream(None, &main, true), DecodeRoute::D3d11va);
        assert_eq!(route_stream(None, &main, false), DecodeRoute::Software);
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
