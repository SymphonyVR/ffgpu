# ffgpu

A small experiment to bridge libavcodec/FFmpeg to WGPU for zero (CPU) copy GPU-accelerated video decoding.

The primary goal of this library is to bring simple and fast video playback to WGPU-based applications.

> [!IMPORTANT]
> This fork is an **experimental development branch**. It extends the original ffgpu experiment with an A/V playback stack, software-video fallback, direct YUV consumption, Vulkan Video decoding, and new OpenGL interop paths. The APIs and backend behavior may still change while synchronization, fallback, and cross-vendor coverage are hardened.

## Experimental additions

- Audio decode/output and A/V playback coordination, including seeking and clock recovery.
- Vulkan Video decode on a WGPU-created Vulkan device.
- Direct sampling of FFmpeg Vulkan hardware frames, with a GPU-copy staging fallback.
- Windows OpenGL interop for D3D11VA frames without CPU readback.
- Linux OpenGL zero-copy import from VA-API/DRM PRIME through EGL.
- Software-only video playback for CPU-renderer/headless fallback paths.
- Direct YUV plane access so an application can fuse YUV→RGB conversion into its own renderer.

## Backend matrix

The table below describes the paths implemented in this experimental branch. `GPU copy` means decoding remains hardware-accelerated and no CPU readback occurs, but an intermediate GPU copy is used. Availability also depends on FFmpeg build options, codec support, driver extensions, and the selected WGPU backend.

| Decode path | **Vulkan** | **DX12** | **Metal** | **OpenGL** | **CPU / no WGPU** |
|---|---|---|---|---|---|
| **Vulkan Video (Windows/Linux)** | **Direct sample / GPU-copy fallback** | N/A | N/A | N/A | N/A |
| **Windows D3D11VA** | Supported interop | **Supported** | N/A | **GPU plane-copy + GL external-memory interop** | N/A |
| **Linux VA-API / DRM PRIME** | Prefer Vulkan Video on Vulkan backend | N/A | N/A | **Direct EGL/DRM PRIME import** | N/A |
| **macOS VideoToolbox** | CPU fallback | N/A | **Supported** | CPU fallback | N/A |
| **Software decode** | CPU upload | CPU upload | CPU upload | CPU upload | **Supported** |

The Vulkan direct-sampling and cross-API OpenGL paths are experimental and should be validated on the target driver/GPU combination before being treated as production-safe.

## Work in progress

This library is still incomplete. Important remaining/hardening work includes:

- Wider YUV format coverage and additional 10-bit validation.
- Hardware integration tests for Vulkan Video and cross-API synchronization.
- Network streams.
- Stream query and selection.
- Subtitle decoding, including external subtitle files.
- Fast thumbnailing directly into an RGB texture atlas.
- Broader cross-vendor and multi-GPU validation.

## License

Licensed under either

- [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [MIT](http://opensource.org/licenses/MIT)

at your option.
