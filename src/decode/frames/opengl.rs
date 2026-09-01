#![allow(unsafe_op_in_unsafe_fn)]
//! OpenGL zero-copy video frame import.
//!
//! Two production paths, selected at runtime by the engine's wgpu backend and
//! the FFmpeg hardware pixel format:
//!
//! * **Linux** (`OpenGlLinuxFrameAdapter`): VA-API decoder → `av_hwframe_map`
//!   to `AV_PIX_FMT_DRM_PRIME` → read [`ff::AVDRMFrameDescriptor`] →
//!   `EGL_EXT_image_dma_buf_import` → `glEGLImageTargetTexture2DOES` →
//!   `wgpu::hal::gles::Device::texture_from_raw` →
//!   `wgpu::Device::create_texture_from_hal`. True zero-copy: the decoded
//!   planes are sampled directly by wgpu with no intermediate copy. (The
//!   VA-API→DRM PRIME descriptor parsing already existed for the Vulkan
//!   backend; this module adds the GL import leg.)
//!
//! * **Windows** (`OpenGlWindowsFrameAdapter`): D3D11VA decoder → a D3D11
//!   plane-copy shader extracts the NV12/P010 Y and UV planes into shared
//!   `R8`/`Rg8` (or `R16`/`Rg16`) textures → `GL_EXT_memory_object_win32`
//!   import executed *inside wgpu's own WGL context* (via
//!   `device.as_hal::<gles::Api>().context().lock()`) → `texture_from_raw` →
//!   `create_texture_from_hal`. One GPU copy, no CPU readback, no wgpu-hal fork.
//!
//! A reusable texture pool is created once at first frame and re-pointed at the
//! new memory object / EGL image each frame; the wgpu `Texture` wrappers and the
//! bind group are never recreated.
//!
//! Synchronization on Windows uses the `GL_EXT_win32_keyed_mutex` cycle
//! (D3D11 `AcquireSync(0)` → plane-copy → D3D11 `ReleaseSync(1)` →
//! `glAcquireKeyedMutexWin32EXT(1)` → sample → `glReleaseKeyedMutexWin32EXT(0)`).
//! No `glFinish`.

use super::FrameAdapter;
use super::GlInteropTicket;
use crate::{
    context::{layout, pipeline_cache::PipelineCache},
    decode::frames::FrameAdapterBuilder,
    error::{Error, Result},
};
use ffmpeg_next::sys as ff;
use std::ffi::c_void;
use std::{ptr::NonNull, sync::OnceLock};

// `gl` 0.14 only generates core GL entry points, not the EXT_memory_object /
// EXT_external_objects extensions we need. We declare the extension function
// pointer types and load them at runtime from the live GL context via the
// platform proc-address API. This is the standard, fork-free way to call these
// extensions.

// --- glext.h constant values (authoritative, from Khronos registry) ---
const GL_R8: gl::types::GLenum = 0x8229;
const GL_RG8: gl::types::GLenum = 0x822B;
const GL_R16: gl::types::GLenum = 0x822A;
const GL_RG16: gl::types::GLenum = 0x822C;
const GL_TEXTURE_2D: gl::types::GLenum = 0x0DE1;
const GL_NEAREST: gl::types::GLenum = 0x2600;
const GL_TRUE: gl::types::GLboolean = 1;
// GL_EXT_memory_object_win32 handle types
const GL_HANDLE_TYPE_D3D11_IMAGE_EXT: gl::types::GLenum = 0x958B;
#[allow(dead_code)] // sibling handle type kept for the opaque-win32 import variant
const GL_HANDLE_TYPE_OPAQUE_WIN32_EXT: gl::types::GLenum = 0x9587;
// GL_EXT_memory_object pname values
const GL_DEDICATED_MEMORY_OBJECT_EXT: gl::types::GLenum = 0x9581;
// GL_EXT_external_objects_win32
const GL_DEVICE_LUID_EXT: gl::types::GLenum = 0x9599;
const GL_DEVICE_NODE_MASK_EXT: gl::types::GLenum = 0x959A;

#[cfg(target_os = "windows")]
mod win32_gl_ext {
    use std::ffi::c_void;
    use windows::{Win32::Graphics::OpenGL::wglGetProcAddress, core::PCSTR};

    type PfnCreateMemoryObjectsEXT =
        unsafe extern "C" fn(gl::types::GLsizei, *mut gl::types::GLuint);
    type PfnDeleteMemoryObjectsEXT =
        unsafe extern "C" fn(gl::types::GLsizei, *const gl::types::GLuint);
    type PfnTexStorageMem2DEXT = unsafe extern "C" fn(
        gl::types::GLuint,
        gl::types::GLsizei,
        gl::types::GLenum,
        gl::types::GLsizei,
        gl::types::GLsizei,
        gl::types::GLuint,
        gl::types::GLuint64,
    );
    type PfnImportMemoryWin32HandleEXT = unsafe extern "C" fn(
        gl::types::GLuint,
        gl::types::GLuint64,
        gl::types::GLenum,
        *mut c_void,
    );
    type PfnAcquireKeyedMutexWin32EXT = unsafe extern "C" fn(
        gl::types::GLuint,
        gl::types::GLuint64,
        gl::types::GLuint64,
    ) -> gl::types::GLboolean;
    type PfnReleaseKeyedMutexWin32EXT =
        unsafe extern "C" fn(gl::types::GLuint, gl::types::GLuint64) -> gl::types::GLboolean;
    type PfnMemoryObjectParameterivEXT =
        unsafe extern "C" fn(gl::types::GLuint, gl::types::GLenum, *const gl::types::GLint);
    type PfnGetUnsignedBytevEXT = unsafe extern "C" fn(gl::types::GLenum, *mut gl::types::GLubyte);

    fn load_fn<T>(name: &[u8]) -> Option<T> {
        let cstr = std::ffi::CStr::from_bytes_with_nul(name).ok()?;
        let proc = unsafe { wglGetProcAddress(PCSTR(cstr.as_ptr() as *const u8)) };
        let ptr = unsafe { std::mem::transmute::<_, *const c_void>(proc) };
        if ptr.is_null() {
            None
        } else {
            // SAFETY: caller only invokes the pointer after verifying it is non-null
            // and the context that produced it is current.
            Some(unsafe { std::mem::transmute_copy(&ptr) })
        }
    }

    pub struct Win32GlExt {
        pub create_memory_objects: PfnCreateMemoryObjectsEXT,
        pub delete_memory_objects: PfnDeleteMemoryObjectsEXT,
        pub tex_storage_mem_2d: PfnTexStorageMem2DEXT,
        pub import_memory_win32: PfnImportMemoryWin32HandleEXT,
        pub acquire_keyed_mutex: PfnAcquireKeyedMutexWin32EXT,
        pub release_keyed_mutex: PfnReleaseKeyedMutexWin32EXT,
        pub memory_object_parameteriv: PfnMemoryObjectParameterivEXT,
        pub get_unsigned_bytev: PfnGetUnsignedBytevEXT,
    }

    /// Load the extension entry points. Must be called while wgpu's WGL context
    /// is current (inside `AdapterContext::lock()`).
    ///
    /// The six function pointers together indicate that
    /// `GL_EXT_memory_object`, `GL_EXT_memory_object_win32` and
    /// `GL_EXT_win32_keyed_mutex` are all present. If any one is missing the
    /// caller must fall back to a different interop strategy.
    pub unsafe fn load() -> Option<Win32GlExt> {
        Some(Win32GlExt {
            create_memory_objects: load_fn(b"glCreateMemoryObjectsEXT\0")?,
            delete_memory_objects: load_fn(b"glDeleteMemoryObjectsEXT\0")?,
            tex_storage_mem_2d: load_fn(b"glTexStorageMem2DEXT\0")?,
            import_memory_win32: load_fn(b"glImportMemoryWin32HandleEXT\0")?,
            acquire_keyed_mutex: load_fn(b"glAcquireKeyedMutexWin32EXT\0")?,
            release_keyed_mutex: load_fn(b"glReleaseKeyedMutexWin32EXT\0")?,
            memory_object_parameteriv: load_fn(b"glMemoryObjectParameterivEXT\0")?,
            get_unsigned_bytev: load_fn(b"glGetUnsignedBytevEXT\0")?,
        })
    }
}

/// `WGL_NV_DX_interop2` entry points (Windows GL ↔ D3D11 zero-copy handoff).
///
/// Unlike `GL_EXT_memory_object_win32` (which imports external memory into a
/// freshly allocated GL texture every frame), WGL interop registers a
/// persistent D3D11 resource into a persistent GL texture name *once*, then
/// hands ownership back and forth with `wglDXLockObjectsNV` /
/// `wglDXUnlockObjectsNV` each frame. No per-frame pixel copy or texture
/// re-allocation — the cross-API transfer is a driver-level page-table swap.
///
/// Quarantined behind the `experimental-wgl-interop` Cargo feature. The
/// cross-vendor `GL_EXT_memory_object_win32` path is the production default.
#[cfg(all(target_os = "windows", feature = "experimental-wgl-interop"))]
mod wgl_nv_dx_interop {
    use std::ffi::c_void;
    use windows::Win32::Graphics::OpenGL::wglGetProcAddress;
    use windows::core::PCSTR;

    type PfnDxOpenDeviceNV = unsafe extern "C" fn(d3d_device: *mut c_void) -> *mut c_void;
    type PfnDxCloseDeviceNV = unsafe extern "C" fn(d3d_device: *mut c_void) -> gl::types::GLboolean;
    type PfnDxRegisterObjectNV = unsafe extern "C" fn(
        d3d_device: *mut c_void,
        d3d_resource: *mut c_void,
        gl_name: gl::types::GLuint,
        gl_type: gl::types::GLenum,
        access: gl::types::GLenum,
    ) -> *mut c_void;
    type PfnDxUnregisterObjectNV =
        unsafe extern "C" fn(d3d_device: *mut c_void, object: *mut c_void) -> gl::types::GLboolean;
    type PfnDxLockObjectsNV = unsafe extern "C" fn(
        d3d_device: *mut c_void,
        count: gl::types::GLint,
        gl_names: *const gl::types::GLuint,
        dx_resources: *const *mut c_void,
    ) -> gl::types::GLboolean;
    type PfnDxUnlockObjectsNV = unsafe extern "C" fn(
        d3d_device: *mut c_void,
        count: gl::types::GLint,
        gl_names: *const gl::types::GLuint,
        dx_resources: *const *mut c_void,
    ) -> gl::types::GLboolean;

    fn load_fn<T>(name: &[u8]) -> Option<T> {
        let cstr = std::ffi::CStr::from_bytes_with_nul(name).ok()?;
        let proc = unsafe { wglGetProcAddress(PCSTR(cstr.as_ptr() as *const u8)) };
        let ptr = unsafe { std::mem::transmute::<_, *const c_void>(proc) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { std::mem::transmute_copy(&ptr) })
        }
    }

    #[derive(Clone, Copy)]
    pub struct WglNvDxInterop {
        pub dx_open_device: PfnDxOpenDeviceNV,
        pub dx_close_device: PfnDxCloseDeviceNV,
        pub dx_register_object: PfnDxRegisterObjectNV,
        pub dx_unregister_object: PfnDxUnregisterObjectNV,
        pub dx_lock_objects: PfnDxLockObjectsNV,
        pub dx_unlock_objects: PfnDxUnlockObjectsNV,
    }

    /// Load the `WGL_NV_DX_interop2` entry points. Must be called while wgpu's
    /// WGL context is current (inside `AdapterContext::lock()`).
    pub unsafe fn load() -> Option<WglNvDxInterop> {
        Some(WglNvDxInterop {
            dx_open_device: load_fn(b"wglDXOpenDeviceNV\0")?,
            dx_close_device: load_fn(b"wglDXCloseDeviceNV\0")?,
            dx_register_object: load_fn(b"wglDXRegisterObjectNV\0")?,
            dx_unregister_object: load_fn(b"wglDXUnregisterObjectNV\0")?,
            dx_lock_objects: load_fn(b"wglDXLockObjectsNV\0")?,
            dx_unlock_objects: load_fn(b"wglDXUnlockObjectsNV\0")?,
        })
    }

    /// Log which `WGL_NV_DX_interop2` entry points the driver exposes.
    /// Call while the WGL context is current (inside `AdapterContext::lock()`).
    pub unsafe fn log_extension_presence() {
        let mut missing = false;
        for &(name, label) in &[
            (b"wglDXOpenDeviceNV\0".as_slice(), "wglDXOpenDeviceNV"),
            (b"wglDXCloseDeviceNV\0".as_slice(), "wglDXCloseDeviceNV"),
            (
                b"wglDXRegisterObjectNV\0".as_slice(),
                "wglDXRegisterObjectNV",
            ),
            (
                b"wglDXUnregisterObjectNV\0".as_slice(),
                "wglDXUnregisterObjectNV",
            ),
            (b"wglDXLockObjectsNV\0".as_slice(), "wglDXLockObjectsNV"),
            (b"wglDXUnlockObjectsNV\0".as_slice(), "wglDXUnlockObjectsNV"),
        ] {
            let cstr = std::ffi::CStr::from_bytes_with_nul(name).unwrap();
            let proc = unsafe { wglGetProcAddress(PCSTR(cstr.as_ptr() as *const u8)) };
            let ptr: *const c_void = unsafe { std::mem::transmute(proc) };
            if ptr.is_null() {
                eprintln!("[opengl] WGL_NV_DX_interop2: {} MISSING", label);
                missing = true;
            }
        }
        if !missing {
            eprintln!("[opengl] WGL_NV_DX_interop2: all 6 entry points present");
        }
    }
}

// `WGL_NV_DX_interop2` access flags (Khronos registry).
#[cfg(all(target_os = "windows", feature = "experimental-wgl-interop"))]
const WGL_ACCESS_READ_ONLY_NV: gl::types::GLenum = 0x0000;
#[cfg(all(target_os = "windows", feature = "experimental-wgl-interop"))]
const WGL_ACCESS_READ_WRITE_NV: gl::types::GLenum = 0x0001;

/// Which import strategy an OpenGL adapter uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenGlInteropPath {
    /// Linux: VA-API → DRM PRIME → EGLImage. True zero-copy.
    #[allow(dead_code)] // Linux-only zero-copy path; inactive on Windows builds
    DirectPlaneImport,
    /// Windows: D3D11VA → D3D11 plane-copy shader → GL external memory import.
    /// One GPU copy, no CPU readback.
    GpuPlaneCopyThenImport,
    /// Windows experimental: import the D3D11VA decoder surface directly via
    /// `GL_EXT_memory_object_win32` with no plane copy. Capability-gated.
    #[allow(dead_code)]
    DirectWin32Import,
    /// Fallback: CPU upload via `av_hwframe_transfer_data`.
    #[allow(dead_code)]
    CpuUpload,
}

/// Compile probe: verifies that the high-level `wgpu` 29.0.4 API exposes
/// `gles::Device::context()` on the current target. On native Windows WGL this
/// is what lets us run GL extension calls inside wgpu's own context — the only
/// capability we need for the OpenGL zero-copy path, and no wgpu-hal fork is
/// required. If this fails to compile/link, the project's GL target does not
/// match upstream 29.0.4 and must be revisited.
#[allow(dead_code)]
unsafe fn gl_context_probe(device: &wgpu::Device) {
    // Compile probe: verifies that the high-level `wgpu` 29.0.4 API exposes
    // `gles::Device::context()` on the current target. On native Windows WGL
    // this is what lets us run GL extension calls inside wgpu's own context —
    // the only capability we need for the OpenGL zero-copy path, and no
    // wgpu-hal fork is required. If this fails to compile/link, the project's
    // GL target does not match upstream 29.0.4 and must be revisited.
    let hal = unsafe { device.as_hal::<wgpu::hal::gles::Api>() }.expect("GL backend");
    let _ctx = hal.context();
}

/// Holds the two wgpu textures (Y, UV), their bind group, and the raw GL /
/// platform handles needed to re-point them at each new frame's memory object.
#[allow(dead_code)] // draft GL-import frame holder; wired up with the memory-object path
struct GlImportedFrame {
    y_texture: wgpu::Texture,
    uv_texture: wgpu::Texture,
    y_gl: gl::types::GLuint,
    uv_gl: gl::types::GLuint,
    /// Cached wgpu texture wrapper created from the stable GL name. Reused
    /// across frames instead of being re-allocated (and discarded) on every
    /// `wrap()` call — removing per-frame wgpu texture churn at 4K.
    y_wgpu: Option<wgpu::Texture>,
    uv_wgpu: Option<wgpu::Texture>,
    #[cfg(target_os = "windows")]
    y_mem: gl::types::GLuint,
    #[cfg(target_os = "windows")]
    uv_mem: gl::types::GLuint,
    identity: layout::FrameDescriptor<()>,
    bg0: wgpu::BindGroup,
}

#[allow(dead_code)] // draft GL-import frame holder; wired up with the memory-object path
impl GlImportedFrame {
    /// Create the wgpu texture wrappers + bind group once. The GL objects are
    /// created lazily on first `import` because their dimensions are only known
    /// once a frame arrives.
    unsafe fn new(
        device: &wgpu::Device,
        pipeline_cache: &mut PipelineCache,
        width: u32,
        height: u32,
        depth: layout::Depth,
        color_space: ffmpeg_next::color::Space,
    ) -> Self {
        let (y_fmt, uv_fmt) = match depth {
            layout::Depth::D16 => (
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
            ),
            _ => (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm),
        };

        let y_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: y_fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let uv_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: uv_fmt,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let layout = layout::FrameDescriptor {
            planes: layout::PlaneLayout::PackedYUV420([y_fmt, uv_fmt]),
            depth,
        };
        let bg0 = pipeline_cache.bind_frame_textures(
            &layout::FrameDescriptor {
                planes: layout::PlaneLayout::PackedYUV420([
                    y_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    uv_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ]),
                depth,
            },
            color_space,
        );

        GlImportedFrame {
            y_texture,
            uv_texture,
            y_gl: 0,
            uv_gl: 0,
            y_wgpu: None,
            uv_wgpu: None,
            #[cfg(target_os = "windows")]
            y_mem: 0,
            #[cfg(target_os = "windows")]
            uv_mem: 0,
            identity: layout.as_identity(),
            bg0,
        }
    }

    /// Wrap an already-created GL texture (whose storage is backed by external
    /// memory / an EGL image) as a wgpu texture via `texture_from_raw`, and
    /// replace the pool's wgpu texture. The bind group stays valid because it
    /// references the *view* of the new texture with the same format/dimensions.
    ///
    /// NOTE: the first call creates the wgpu texture; subsequent calls recreate
    /// it from the same GLuint (the GL object identity is stable, so the bind
    /// group created against the original view remains valid).
    unsafe fn wrap(&mut self, device: &wgpu::Device, is_y: bool) -> Result<()> {
        unsafe {
            let hal = device
                .as_hal::<wgpu::hal::gles::Api>()
                .ok_or(Error::UnsupportedBackend)?;
            let (_tex, fmt, w, h) = if is_y {
                (
                    &self.y_texture,
                    self.y_texture.format(),
                    self.y_texture.width(),
                    self.y_texture.height(),
                )
            } else {
                (
                    &self.uv_texture,
                    self.uv_texture.format(),
                    self.uv_texture.width(),
                    self.uv_texture.height(),
                )
            };
            let gl_name = if is_y { self.y_gl } else { self.uv_gl };

            // The GL name is stable across frames (set once in
            // `import_win32_plane` when it was 0). Re-create + cache the wgpu
            // wrapper only when we don't already hold one, instead of
            // allocating and immediately discarding a texture every frame.
            let cache = if is_y {
                &mut self.y_wgpu
            } else {
                &mut self.uv_wgpu
            };
            if cache.is_some() {
                return Ok(());
            }

            let hal_tex = hal.texture_from_raw(
                std::num::NonZeroU32::new(gl_name).ok_or(Error::Unknown)?,
                &wgpu::hal::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_DST,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                None,
            );
            let tex = device.create_texture_from_hal::<wgpu::hal::gles::Api>(
                hal_tex,
                &wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: fmt,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            );
            *cache = Some(tex);
            Ok(())
        }
    }
}

impl Drop for GlImportedFrame {
    fn drop(&mut self) {
        unsafe {
            if self.y_gl != 0 {
                gl::DeleteTextures(1, &self.y_gl);
            }
            if self.uv_gl != 0 {
                gl::DeleteTextures(1, &self.uv_gl);
            }
            #[cfg(target_os = "windows")]
            {
                if let Some(ext) = GL_WIN32_EXT.get() {
                    if self.y_mem != 0 {
                        (ext.delete_memory_objects)(1, &self.y_mem);
                    }
                    if self.uv_mem != 0 {
                        (ext.delete_memory_objects)(1, &self.uv_mem);
                    }
                }
            }
        }
    }
}

#[cfg(target_os = "windows")]
static GL_WIN32_EXT: OnceLock<win32_gl_ext::Win32GlExt> = OnceLock::new();

// ---------------------------------------------------------------------------
// Linux: VA-API → DRM PRIME → EGL DMA-BUF → GL
// ---------------------------------------------------------------------------

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::ffi::c_void;

    // EGL_DMA_BUF_EXT attribute tokens (stable values from the extension spec;
    // khronos-egl does not enumerate every dma-buf attribute).
    const EGL_LINUX_DMA_BUF_EXT: gl::types::GLenum = 0x3033;
    const EGL_DMA_BUF_PLANE0_FD_EXT: gl::types::GLenum = 0x3272;
    const EGL_DMA_BUF_PLANE0_OFFSET_EXT: gl::types::GLenum = 0x3273;
    const EGL_DMA_BUF_PLANE0_PITCH_EXT: gl::types::GLenum = 0x3274;
    const EGL_DMA_BUF_PLANE1_FD_EXT: gl::types::GLenum = EGL_DMA_BUF_PLANE0_FD_EXT + 3;
    const EGL_DMA_BUF_PLANE1_OFFSET_EXT: gl::types::GLenum = EGL_DMA_BUF_PLANE0_OFFSET_EXT + 3;
    const EGL_DMA_BUF_PLANE1_PITCH_EXT: gl::types::GLenum = EGL_DMA_BUF_PLANE0_PITCH_EXT + 3;
    const EGL_WIDTH: gl::types::GLenum = 0x3057;
    const EGL_HEIGHT: gl::types::GLenum = 0x3056;
    const EGL_NONE: gl::types::GLenum = 0x3038;

    // Raw libEGL functions (always present when wgpu uses the GL/EGL backend).
    extern "C" {
        fn eglGetCurrentDisplay() -> *mut c_void;
        fn eglCreateImage(
            display: *mut c_void,
            ctx: *mut c_void,
            target: gl::types::GLenum,
            buffer: *mut c_void,
            attrib_list: *const gl::types::GLint,
        ) -> *mut c_void;
        fn eglDestroyImage(display: *mut c_void, image: *mut c_void) -> gl::types::GLboolean;
    }

    type PfnEGLImageTargetTexture2DOES = unsafe extern "C" fn(gl::types::GLenum, *mut c_void);

    unsafe fn load_egl_image_target() -> Option<PfnEGLImageTargetTexture2DOES> {
        // EGL and GL share the same proc-address space on EGL platforms.
        let name = std::ffi::CStr::from_bytes_with_nul(b"glEGLImageTargetTexture2DOES\0").ok()?;
        let ptr = egl_get_proc_address(name.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(std::mem::transmute(ptr))
        }
    }

    unsafe fn egl_get_proc_address(name: *const std::os::raw::c_char) -> *mut c_void {
        // Pulled from libEGL via the standard `eglGetProcAddress` symbol.
        unsafe extern "C" fn eglGetProcAddress(_name: *const std::os::raw::c_char) -> *mut c_void {
            // Resolved below through the real libEGL symbol.
            resolve_egl_proc_address(_name)
        }
        let _ = eglGetProcAddress;
        resolve_egl_proc_address(name)
    }

    // `resolve_egl_proc_address` is provided by linking libEGL; declare it.
    extern "C" {
        fn resolve_egl_proc_address(name: *const std::os::raw::c_char) -> *mut c_void;
    }

    /// EGL interop handles for a single imported plane.
    struct EglPlane {
        image: *mut c_void,
        gl_tex: gl::types::GLuint,
    }

    impl Drop for EglPlane {
        fn drop(&mut self) {
            unsafe {
                if self.gl_tex != 0 {
                    gl::DeleteTextures(1, &self.gl_tex);
                }
                let display = eglGetCurrentDisplay();
                if !self.image.is_null() && !display.is_null() {
                    eglDestroyImage(display, self.image);
                }
            }
        }
    }

    pub(super) struct VaapiEglImport {
        y: EglPlane,
        uv: EglPlane,
        imported: Option<GlImportedFrame>,
    }

    /// Import one DRM PRIME plane (R8 / GR88) as an EGLImage-backed GL texture.
    unsafe fn import_drm_plane(
        object_fd: i32,
        offset: i32,
        pitch: i32,
        width: i32,
        height: i32,
    ) -> Result<EglPlane> {
        unsafe {
            let attrs = [
                EGL_WIDTH as gl::types::GLint,
                width,
                EGL_HEIGHT as gl::types::GLint,
                height,
                EGL_LINUX_DMA_BUF_EXT as gl::types::GLint,
                0,
                EGL_DMA_BUF_PLANE0_FD_EXT as gl::types::GLint,
                object_fd,
                EGL_DMA_BUF_PLANE0_OFFSET_EXT as gl::types::GLint,
                offset,
                EGL_DMA_BUF_PLANE0_PITCH_EXT as gl::types::GLint,
                pitch,
                EGL_NONE as gl::types::GLint,
                0,
            ];

            let display = eglGetCurrentDisplay();
            let image = eglCreateImage(
                display,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                attrs.as_ptr(),
            );
            if image.is_null() {
                return Err(Error::TextureShare);
            }

            let mut gl_tex = 0u32;
            gl::GenTextures(1, &mut gl_tex);
            gl::BindTexture(GL_TEXTURE_2D, gl_tex);
            gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MAG_FILTER, GL_NEAREST as i32);

            let target = load_egl_image_target().ok_or(Error::TextureShare)?;
            target(GL_TEXTURE_2D, image);

            Ok(EglPlane { image, gl_tex })
        }
    }

    impl VaapiEglImport {
        pub(super) unsafe fn new(
            device: &wgpu::Device,
            pipeline_cache: &mut PipelineCache,
            frame: NonNull<ff::AVFrame>,
        ) -> Result<Self> {
            unsafe {
                let frame_ref = frame.as_ref();

                let drm_frame = NonNull::new(ff::av_frame_alloc()).ok_or(Error::Unknown)?;
                (*drm_frame.as_ptr()).format = ff::AVPixelFormat::AV_PIX_FMT_DRM_PRIME as i32;
                if ff::av_hwframe_map(
                    drm_frame.as_ptr(),
                    frame.as_ptr(),
                    ff::AV_HWFRAME_MAP_READ as i32,
                ) != 0
                {
                    ff::av_frame_free(&mut drm_frame.as_ptr());
                    return Err(Error::TextureShare);
                }

                let drm_desc = ((*drm_frame.as_ptr()).data[0] as *const ff::AVDRMFrameDescriptor)
                    .as_ref()
                    .ok_or(Error::TextureShare)?;

                if drm_desc.nb_layers != 2
                    || drm_desc.layers[0].format != (538982482/* DRM_FORMAT_R8 */)
                    || drm_desc.layers[1].format != (943215175/* DRM_FORMAT_GR88 */)
                {
                    ff::av_frame_unref(drm_frame.as_ptr());
                    ff::av_frame_free(&mut drm_frame.as_ptr());
                    return Err(Error::TextureShare);
                }

                let display = eglGetCurrentDisplay();

                let r8 = &drm_desc.layers[0].planes[0];
                let gr88 = &drm_desc.layers[1].planes[0];
                let r8_obj = &drm_desc.objects[r8.object_index as usize];
                let gr88_obj = &drm_desc.objects[gr88.object_index as usize];

                let y = import_drm_plane(
                    libc::dup(r8_obj.fd),
                    r8.offset as i32,
                    r8.pitch as i32,
                    frame_ref.width,
                    frame_ref.height,
                )?;
                let uv = import_drm_plane(
                    libc::dup(gr88_obj.fd),
                    gr88.offset as i32,
                    gr88.pitch as i32,
                    frame_ref.width / 2,
                    frame_ref.height / 2,
                )?;

                ff::av_frame_unref(drm_frame.as_ptr());
                ff::av_frame_free(&mut drm_frame.as_ptr());

                let mut imported = GlImportedFrame::new(
                    device,
                    pipeline_cache,
                    frame_ref.width as u32,
                    frame_ref.height as u32,
                    layout::Depth::D8,
                    frame_ref.colorspace.into(),
                );
                imported.y_gl = y.gl_tex;
                imported.uv_gl = uv.gl_tex;
                // The GL textures are now EGLImage-backed; wrap them once.
                imported.wrap(device, true)?;
                imported.wrap(device, false)?;

                let _ = display;
                Ok(VaapiEglImport {
                    y,
                    uv,
                    imported: Some(imported),
                })
            }
        }

        /// Re-point the reusable wgpu textures at the EGL images (they are
        /// already imported into `y`/`uv` GL textures during `new`).
        pub(super) unsafe fn attach(&mut self, device: &wgpu::Device) -> Result<()> {
            unsafe {
                let imported = self.imported.as_mut().ok_or(Error::Unknown)?;
                imported.y_gl = self.y.gl_tex;
                imported.uv_gl = self.uv.gl_tex;
                imported.wrap(device, true)?;
                imported.wrap(device, false)?;
                Ok(())
            }
        }

        pub(super) fn frame(&self) -> &GlImportedFrame {
            self.imported.as_ref().unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// Windows: D3D11VA → plane-copy shader → GL_EXT_memory_object_win32
// ---------------------------------------------------------------------------

#[cfg(target_os = "windows")]
mod win {
    use super::super::GlInteropTicket;
    use super::*;
    use std::ffi::c_void;
    use windows::{
        Win32::{
            Foundation,
            Graphics::{Direct3D, Direct3D11 as D3D11, Dxgi, OpenGL::wglGetProcAddress},
            System::LibraryLoader::{GetProcAddress, LoadLibraryA},
        },
        core::{Interface, PCSTR},
    };

    /// Resolve a GL entry point for the `gl` crate loader. `wglGetProcAddress`
    /// only returns *extension* functions; the core 1.1 entry points
    /// (`glGenTextures`, `glBindTexture`, `glTexParameteri`, `glDeleteTextures`,
    /// `glGetString`, `glFlush`, ...) must be pulled from `opengl32.dll` via
    /// `GetProcAddress`. Without the second leg every core `gl::*` call panics
    /// with "gl function was not loaded".
    unsafe fn gl_resolve(name: &str) -> *const std::ffi::c_void {
        let cname = match std::ffi::CString::new(name) {
            Ok(c) => c,
            Err(_) => return std::ptr::null(),
        };
        let proc = wglGetProcAddress(PCSTR(cname.as_ptr() as *const u8));
        let ptr = std::mem::transmute::<_, *const std::ffi::c_void>(proc);
        if !ptr.is_null() {
            return ptr;
        }
        if let Ok(lib) = LoadLibraryA(PCSTR(b"opengl32.dll\0".as_ptr() as *const u8)) {
            if !lib.is_invalid() {
                let proc = GetProcAddress(lib, PCSTR(cname.as_ptr() as *const u8));
                let ptr = std::mem::transmute::<_, *const std::ffi::c_void>(proc);
                if !ptr.is_null() {
                    return ptr;
                }
            }
        }
        std::ptr::null()
    }

    static GL_LOADER_INSTALLED: std::sync::Once = std::sync::Once::new();

    /// Install the `gl` crate's global function loader while wgpu's GL context
    /// is current. Every `gl::*` call in this module relies on it; without it the
    /// `gl` crate leaves all entry points as `missing_fn_panic` stubs — which is
    /// exactly what crashed OpenGL playback. Runs once per process.
    unsafe fn ensure_gl_loaded(device: &wgpu::Device) {
        GL_LOADER_INSTALLED.call_once(|| {
            if let Some(hal) = device.as_hal::<wgpu::hal::gles::Api>() {
                let _guard = hal.context().lock();
                gl::load_with(|name| unsafe { gl_resolve(name) });
            }
        });
    }

    /// Kept as a marker for any external debug symbol that still names it.
    /// The persistent memory-object ring built by `build_memory_object_slot`
    /// owns the D3D11 R8/RG8 textures directly; there is no per-frame shared
    /// copy object any more.
    #[allow(dead_code)]
    struct D3D11PlaneCopy {}

    /// Fullscreen-triangle vertex shader (no vertex buffer; generates clip-space
    /// positions from SV_VertexID). Shared by both plane-copy passes.
    const PLANE_COPY_VS: &str = r#"
        struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
        VsOut main(uint vid : SV_VertexID) {
            // Single oversized triangle covering the viewport.
            float2 p = float2((vid == 2) ? 3.0 : -1.0, (vid == 1) ? 3.0 : -1.0);
            VsOut o;
            o.pos = float4(p, 0.0, 1.0);
            // Map clip-space XY (-1..1) to UV (0..1), flip Y for texture space.
            o.uv = float2((p.x + 1.0) * 0.5, 1.0 - (p.y + 1.0) * 0.5);
            return o;
        }
    "#;

    /// Pixel shader that writes the luma (Y) plane. The input is NV12 bound as a
    /// single-slice `Texture2DArray` (z = 0), which D3D11 samples as a
    /// single-channel R texture for the luma half.
    const PLANE_COPY_PS_Y: &str = r#"
        Texture2DArray<float> src : register(t0);
        SamplerState samp : register(s0);
        struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
        float4 main(VsOut i) : SV_Target {
            return float4(src.Sample(samp, float3(i.uv, 0)).r, 0.0, 0.0, 1.0);
        }
    "#;

    /// Pixel shader that writes the interleaved chroma (UV) plane. The D3D11
    /// `R8G8_UNORM` SRV is already a view of NV12's chroma plane at half
    /// resolution, so coordinates stay in the normal 0..1 range.
    const PLANE_COPY_PS_UV: &str = r#"
        Texture2DArray<float2> src : register(t0);
        SamplerState samp : register(s0);
        struct VsOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
        float4 main(VsOut i) : SV_Target {
            float2 c = src.Sample(samp, float3(i.uv, 0));
            return float4(c, 0.0, 1.0);
        }
    "#;

    /// `ClearState` resets the D3D11 rasterizer state, including the viewport
    /// and input-assembler topology.  Keep the plane-copy dimensions in one
    /// pure helper so both interop implementations configure the exact target
    /// extent before drawing.
    fn plane_copy_viewport(width: u32, height: u32) -> D3D11::D3D11_VIEWPORT {
        D3D11::D3D11_VIEWPORT {
            TopLeftX: 0.0,
            TopLeftY: 0.0,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.0,
            MaxDepth: 1.0,
        }
    }

    /// Compiled shader pipeline for the NV12 -> R8/RG8 plane split. Built once
    /// and reused for every frame (the SRV/RTV are swapped per frame).
    struct PlaneCopyPipeline {
        vs: D3D11::ID3D11VertexShader,
        ps_y: D3D11::ID3D11PixelShader,
        ps_uv: D3D11::ID3D11PixelShader,
        layout: D3D11::ID3D11InputLayout,
        sampler: D3D11::ID3D11SamplerState,
    }

    impl Drop for PlaneCopyPipeline {
        fn drop(&mut self) {
            // The five owned COM interfaces (vs/ps_y/ps_uv/layout/sampler) are
            // released automatically by the `windows` crate's COM smart-pointer
            // `Drop` impl when this struct is dropped alongside the player.
        }
    }

    /// Compile an HLSL shader via d3dcompiler_47 (loaded by the `windows` crate
    /// from the system DLL). Returns raw bytecode. On failure we surface a
    /// `Probe` error — the caller should fall back to software decoding rather
    /// than crash, since a missing/old d3dcompiler is a non-fatal environment
    /// issue.
    unsafe fn compile_hlsl(source: &str, entry: &str, target: &str) -> Result<Vec<u8>> {
        unsafe {
            let src_bytes: Vec<u8> = source
                .as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0))
                .collect();
            let entry_bytes: Vec<u8> = entry
                .as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0))
                .collect();
            let target_bytes: Vec<u8> = target
                .as_bytes()
                .iter()
                .copied()
                .chain(std::iter::once(0))
                .collect();
            let mut code: Option<Direct3D::ID3DBlob> = None;
            let mut err: Option<Direct3D::ID3DBlob> = None;
            // d3dcompiler_47 ships with Windows 10+; load it via the OS.
            let hr = windows::Win32::Graphics::Direct3D::Fxc::D3DCompile(
                src_bytes.as_ptr().cast(),
                src_bytes.len(),
                None,
                None,
                None,
                windows::core::PCSTR(entry_bytes.as_ptr()),
                windows::core::PCSTR(target_bytes.as_ptr()),
                windows::Win32::Graphics::Direct3D::Fxc::D3DCOMPILE_OPTIMIZATION_LEVEL3,
                0,
                &mut code,
                Some(&mut err),
            );
            if let Some(err_blob) = err {
                let p = err_blob.GetBufferPointer();
                let len = err_blob.GetBufferSize();
                let msg = if len > 0 {
                    String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len))
                        .to_string()
                } else {
                    String::new()
                };
                return Err(Error::Probe(format!(
                    "HLSL compile ({entry}) failed: {msg}"
                )));
            }
            hr.map_err(|e| Error::Probe(format!("D3DCompile load/invoke failed: {e}")))?;
            let code = code.ok_or(Error::Probe("D3DCompile returned no bytecode".into()))?;
            let p = code.GetBufferPointer();
            let len = code.GetBufferSize();
            let bytes = std::slice::from_raw_parts(p as *const u8, len).to_vec();
            Ok(bytes)
        }
    }

    unsafe fn build_plane_copy_pipeline(device: &D3D11::ID3D11Device) -> Result<PlaneCopyPipeline> {
        unsafe {
            let vs = compile_hlsl(PLANE_COPY_VS, "main", "vs_5_0")?;
            let ps_y = compile_hlsl(PLANE_COPY_PS_Y, "main", "ps_5_0")?;
            let ps_uv = compile_hlsl(PLANE_COPY_PS_UV, "main", "ps_5_0")?;

            let mut vs_obj = None;
            device
                .CreateVertexShader(&vs, None, Some(&mut vs_obj))
                .map_err(|e| Error::Probe(format!("CreateVertexShader failed: {e}")))?;
            let vs_obj = vs_obj.ok_or(Error::Probe("CreateVertexShader null".into()))?;

            let mut ps_y_obj = None;
            device
                .CreatePixelShader(&ps_y, None, Some(&mut ps_y_obj))
                .map_err(|e| Error::Probe(format!("CreatePixelShader(Y) failed: {e}")))?;
            let ps_y_obj = ps_y_obj.ok_or(Error::Probe("CreatePixelShader(Y) null".into()))?;

            let mut ps_uv_obj = None;
            device
                .CreatePixelShader(&ps_uv, None, Some(&mut ps_uv_obj))
                .map_err(|e| Error::Probe(format!("CreatePixelShader(UV) failed: {e}")))?;
            let ps_uv_obj = ps_uv_obj.ok_or(Error::Probe("CreatePixelShader(UV) null".into()))?;

            // Empty input layout (positions come from SV_VertexID).
            let mut layout = None;
            device
                .CreateInputLayout(&[], &vs, Some(&mut layout))
                .map_err(|e| Error::Probe(format!("CreateInputLayout failed: {e}")))?;
            let layout = layout.ok_or(Error::Probe("CreateInputLayout null".into()))?;

            let mut sampler = None;
            device
                .CreateSamplerState(
                    &D3D11::D3D11_SAMPLER_DESC {
                        Filter: D3D11::D3D11_FILTER_MIN_MAG_MIP_POINT,
                        AddressU: D3D11::D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressV: D3D11::D3D11_TEXTURE_ADDRESS_CLAMP,
                        AddressW: D3D11::D3D11_TEXTURE_ADDRESS_CLAMP,
                        ComparisonFunc: D3D11::D3D11_COMPARISON_NEVER,
                        ..Default::default()
                    },
                    Some(&mut sampler),
                )
                .map_err(|e| Error::Probe(format!("CreateSamplerState failed: {e}")))?;
            let sampler = sampler.ok_or(Error::Probe("CreateSamplerState null".into()))?;

            Ok(PlaneCopyPipeline {
                vs: vs_obj,
                ps_y: ps_y_obj,
                ps_uv: ps_uv_obj,
                layout,
                sampler,
            })
        }
    }

    #[allow(dead_code)]
    impl Drop for D3D11PlaneCopy {
        fn drop(&mut self) {}
    }

    #[allow(dead_code)] // format introspection helper for future shared-handle validation
    unsafe fn decoder_texture_format(
        tex: &D3D11::ID3D11Texture2D,
    ) -> Result<Dxgi::Common::DXGI_FORMAT> {
        unsafe {
            let mut desc = D3D11::D3D11_TEXTURE2D_DESC::default();
            tex.GetDesc(&mut desc);
            Ok(desc.Format)
        }
    }

    unsafe fn create_shared(
        device: &D3D11::ID3D11Device,
        desc: &D3D11::D3D11_TEXTURE2D_DESC,
    ) -> Result<(D3D11::ID3D11Texture2D, Foundation::HANDLE)> {
        unsafe {
            let mut tex = None;
            device
                .CreateTexture2D(desc, None, Some(&mut tex))
                .map_err(|_| Error::TextureShare)?;
            let tex = tex.ok_or(Error::TextureShare)?;
            let dxgi = tex
                .cast::<Dxgi::IDXGIResource1>()
                .map_err(|_| Error::TextureShare)?;
            let handle = dxgi
                .CreateSharedHandle(None, Dxgi::DXGI_SHARED_RESOURCE_READ.0, None)
                .map_err(|_| Error::TextureShare)?;
            Ok((tex, handle))
        }
    }

    /// One set of D3D11 R8/RG8 plane textures (with the
    /// `SHARED_NTHANDLE | SHARED_KEYEDMUTEX` misc flag pair) plus its RTVs and
    /// NT handles. Created once per ring slot, never reallocated.
    struct PlaneCopyTextures {
        y: D3D11::ID3D11Texture2D,
        uv: D3D11::ID3D11Texture2D,
        y_rtv: D3D11::ID3D11RenderTargetView,
        uv_rtv: D3D11::ID3D11RenderTargetView,
        y_handle: Foundation::HANDLE,
        uv_handle: Foundation::HANDLE,
    }

    /// Create the per-slot D3D11 R8/RG8 textures. Each pair carries
    /// `SHARED_NTHANDLE | SHARED_KEYEDMUTEX` so the same resource can be both
    /// the plane-copy target and the GL memory-object import source.
    unsafe fn create_plane_copy_textures(
        d3d11_device: &D3D11::ID3D11Device,
        width: u32,
        height: u32,
        depth: layout::Depth,
    ) -> Result<PlaneCopyTextures> {
        unsafe {
            let (y_fmt, uv_fmt) = match depth {
                layout::Depth::D16 => (
                    Dxgi::Common::DXGI_FORMAT_R16_UNORM,
                    Dxgi::Common::DXGI_FORMAT_R16G16_UNORM,
                ),
                _ => (
                    Dxgi::Common::DXGI_FORMAT_R8_UNORM,
                    Dxgi::Common::DXGI_FORMAT_R8G8_UNORM,
                ),
            };

            let shared_desc = D3D11::D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: y_fmt,
                SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11::D3D11_USAGE_DEFAULT,
                BindFlags: D3D11::D3D11_BIND_RENDER_TARGET.0 as u32
                    | D3D11::D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11::D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 as u32
                    | D3D11::D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX.0 as u32,
            };
            let shared_desc_uv = D3D11::D3D11_TEXTURE2D_DESC {
                Width: width / 2,
                Height: height / 2,
                Format: uv_fmt,
                ..shared_desc
            };

            let (y, y_handle) = create_shared(d3d11_device, &shared_desc)?;
            let (uv, uv_handle) = create_shared(d3d11_device, &shared_desc_uv)?;

            let mut y_rtv = None;
            d3d11_device
                .CreateRenderTargetView(&y, None, Some(&mut y_rtv))
                .map_err(|_| Error::TextureShare)?;
            let y_rtv = y_rtv.ok_or(Error::TextureShare)?;

            let mut uv_rtv = None;
            d3d11_device
                .CreateRenderTargetView(&uv, None, Some(&mut uv_rtv))
                .map_err(|_| Error::TextureShare)?;
            let uv_rtv = uv_rtv.ok_or(Error::TextureShare)?;

            Ok(PlaneCopyTextures {
                y,
                uv,
                y_rtv,
                uv_rtv,
                y_handle,
                uv_handle,
            })
        }
    }

    const INTEROP_RING_SIZE: usize = 3;

    /// Which GL interop strategy an `OpenGlWindowsFrameAdapter` should use.
    /// Selected at runtime from `FFGPU_GL_INTEROP` (default `auto`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum InteropPreference {
        Auto,
        Wgl,
        MemoryObject,
        Cpu,
    }

    fn interop_preference() -> InteropPreference {
        match std::env::var("FFGPU_GL_INTEROP").ok().as_deref() {
            Some("wgl") => InteropPreference::Wgl,
            Some("memory-object") => InteropPreference::MemoryObject,
            Some("cpu") => InteropPreference::Cpu,
            _ => InteropPreference::Auto,
        }
    }

    /// State of a single WGL-interop ring slot.
    #[cfg(feature = "experimental-wgl-interop")]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum WglSlotState {
        Free,
        GlLockedReady,
        AwaitingSubmit,
        Submitted,
    }

    /// State of a single memory-object ring slot.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum MemoryObjectSlotState {
        /// Slot is available for the next `import_frame`.
        Free,
        /// Slot is locked: the D3D11 plane copy has released the keyed mutex
        /// at key 1, the GL side has enqueued a `glAcquireKeyedMutexWin32EXT`
        /// wait, and the slot is waiting to be sampled by the engine's draw
        /// and then released after its GL completion fence signals.
        Locked,
        /// The video submission has been fenced, but the GL commands have not
        /// completed yet. The slot cannot return to D3D11 ownership.
        Submitted,
    }

    /// One persistent WGL-interop slot. The D3D11 Y/UV textures, the GL texture
    /// names, the WGL registrations and the cached wgpu wrappers are all created
    /// ONCE; only ownership (via WGL lock/unlock) changes per frame.
    #[cfg(feature = "experimental-wgl-interop")]
    struct GlInteropSlot {
        slot_id: u8,
        y_d3d11: D3D11::ID3D11Texture2D,
        uv_d3d11: D3D11::ID3D11Texture2D,
        y_rtv: D3D11::ID3D11RenderTargetView,
        uv_rtv: D3D11::ID3D11RenderTargetView,
        y_gl: gl::types::GLuint,
        uv_gl: gl::types::GLuint,
        y_wgl: *mut c_void,
        uv_wgl: *mut c_void,
        y_wgpu: Option<wgpu::Texture>,
        uv_wgpu: Option<wgpu::Texture>,
        y_view: Option<wgpu::TextureView>,
        uv_view: Option<wgpu::TextureView>,
        state: WglSlotState,
        /// GL sync object inserted after the submitting `queue.submit`. `None`
        /// until the slot enters `Submitted`.
        fence: Option<gl::types::GLsync>,
    }

    /// One persistent memory-object ring slot. D3D11 R8/RG8 textures with
    /// `SHARED_NTHANDLE | SHARED_KEYEDMUTEX`, paired GL memory objects and GL
    /// texture names (imported once), and the cached wgpu wrappers. Only the
    /// keyed-mutex key state changes per frame.
    struct MemoryObjectSlot {
        slot_id: u8,
        y_d3d11: D3D11::ID3D11Texture2D,
        uv_d3d11: D3D11::ID3D11Texture2D,
        y_rtv: D3D11::ID3D11RenderTargetView,
        uv_rtv: D3D11::ID3D11RenderTargetView,
        /// Persistent GL memory objects; storage is bound once via
        /// `glTexStorageMem2DEXT` and reused for every frame.
        y_mem: gl::types::GLuint,
        uv_mem: gl::types::GLuint,
        /// Persistent GL texture names (target = GL_TEXTURE_2D); backed by
        /// the imported memory objects. The wgpu `Texture` is created once
        /// from this name and never re-wrapped.
        #[allow(dead_code)] // GL names kept for debugging/interop; draws go through the views
        y_gl: gl::types::GLuint,
        #[allow(dead_code)]
        uv_gl: gl::types::GLuint,
        #[allow(dead_code)] // cached wrappers kept alive alongside their views
        y_wgpu: Option<wgpu::Texture>,
        #[allow(dead_code)]
        uv_wgpu: Option<wgpu::Texture>,
        y_view: Option<wgpu::TextureView>,
        uv_view: Option<wgpu::TextureView>,
        state: MemoryObjectSlotState,
        /// Fence inserted after the wgpu GL submission that sampled this slot.
        fence: Option<gl::types::GLsync>,
    }

    /// Cross-vendor persistent ring built on `GL_EXT_memory_object_win32` and
    /// `GL_EXT_win32_keyed_mutex`. Production default. The D3D11 side uses
    /// the COM `IDXGIKeyedMutex` interface (Acquire/Release); the GL side
    /// uses `glAcquireKeyedMutexWin32EXT` / `glReleaseKeyedMutexWin32EXT` to
    /// insert the corresponding waits into the GL command stream.
    // pub(super) to match InteropMode, whose variant exposes this type.
    pub(super) struct MemoryObjectRing {
        core: PlaneCopyCore,
        ext: &'static win32_gl_ext::Win32GlExt,
        #[allow(dead_code)] // held to keep the wgpu device alive as long as the ring
        device: wgpu::Device,
        slots: Vec<MemoryObjectSlot>,
        generation: u32,
        /// Slot whose views were last locked and should be sampled by the draw.
        current_slot: Option<u8>,
        #[allow(dead_code)] // cached format identity; consumed by future resize handling
        depth: layout::Depth,
        width: u32,
        height: u32,
        /// FFmpeg hwctx lock/unlock (serialize D3D11 immediate-context access).
        lock: Option<unsafe extern "C" fn(*mut c_void)>,
        unlock: Option<unsafe extern "C" fn(*mut c_void)>,
        lock_ctx: *mut c_void,
        /// Cached YUV bind group built from the first slot's views; used
        /// only to keep the engine's `direct_yuv()` gate returning `true`.
        /// The actual draw uses `plane_views()` to build a fresh bind group
        /// from the current slot's views via `PipelineCache`.
        bind_group: Option<wgpu::BindGroup>,
        /// Layout identity of the YUV plane formats (R8/RG8 or R16/RG16,
        /// packed NV12-style). Returned by `layout_identity()`.
        layout_identity: layout::FrameDescriptor<()>,
    }

    /// Shader pipeline + decoder SRV shared by every ring slot (the plane-copy
    /// pass renders the decoder's NV12 slice into each slot's R8/RG8 targets).
    /// Both the WGL and the memory-object interop strategies use the same
    /// core: only the per-slot ownership of the R8/RG8 output differs.
    struct PlaneCopyCore {
        #[allow(dead_code)] // held to keep the D3D11 device alive as long as the core
        device: D3D11::ID3D11Device,
        context: D3D11::ID3D11DeviceContext,
        pipeline: PlaneCopyPipeline,
        srv_y: D3D11::ID3D11ShaderResourceView,
        srv_uv: D3D11::ID3D11ShaderResourceView,
    }

    impl PlaneCopyCore {
        /// Build the shared plane-copy state: device, immediate context, the
        /// decoder's NV12 SRV (one slice of the array), and the compiled
        /// vertex / pixel shaders. Independent of the chosen GL interop path.
        unsafe fn new(
            d3d11_device: &D3D11::ID3D11Device,
            decoder_texture: &D3D11::ID3D11Texture2D,
            array_slice: u32,
        ) -> Result<Self> {
            unsafe {
                let context = d3d11_device
                    .GetImmediateContext()
                    .map_err(|_| Error::Unknown)?;
                let mut tex_desc = D3D11::D3D11_TEXTURE2D_DESC::default();
                decoder_texture.GetDesc(&mut tex_desc);
                eprintln!(
                    "[opengl] PlaneCopyCore decoder texture: {}x{} fmt={:?} array_size={} mip_levels={} bind=0x{:08X} misc=0x{:08X} (array_slice={})",
                    tex_desc.Width,
                    tex_desc.Height,
                    tex_desc.Format,
                    tex_desc.ArraySize,
                    tex_desc.MipLevels,
                    tex_desc.BindFlags,
                    tex_desc.MiscFlags,
                    array_slice,
                );
                if tex_desc.Format != Dxgi::Common::DXGI_FORMAT_NV12 {
                    eprintln!(
                        "[opengl] native WGL interop supports NV12 only, got {:?}",
                        tex_desc.Format
                    );
                    return Err(Error::UnsupportedPixelFormat);
                }
                if tex_desc.Width % 2 != 0 || tex_desc.Height % 2 != 0 {
                    eprintln!(
                        "[opengl] NV12 decoder texture has odd dimensions: {}x{}",
                        tex_desc.Width, tex_desc.Height
                    );
                    return Err(Error::UnsupportedPixelFormat);
                }
                if array_slice >= tex_desc.ArraySize {
                    eprintln!(
                        "[opengl] decoder array slice {} is outside array size {}",
                        array_slice, tex_desc.ArraySize
                    );
                    return Err(Error::InvalidFrame);
                }
                if tex_desc.BindFlags & D3D11::D3D11_BIND_SHADER_RESOURCE.0 as u32 == 0 {
                    eprintln!(
                        "[opengl] decoder texture is not shader-readable (bind=0x{:08X})",
                        tex_desc.BindFlags
                    );
                    return Err(Error::TextureShare);
                }
                let srv_desc_y = D3D11::D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: Dxgi::Common::DXGI_FORMAT_R8_UNORM,
                    ViewDimension: Direct3D::D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
                    Anonymous: D3D11::D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2DArray: D3D11::D3D11_TEX2D_ARRAY_SRV {
                            MostDetailedMip: 0,
                            MipLevels: tex_desc.MipLevels,
                            FirstArraySlice: array_slice,
                            ArraySize: 1,
                        },
                    },
                };
                let mut srv_y = None;
                d3d11_device
                    .CreateShaderResourceView(decoder_texture, Some(&srv_desc_y), Some(&mut srv_y))
                    .map_err(|e| {
                        eprintln!("[opengl] CreateShaderResourceView(Y) FAILED: {:?}", e);
                        Error::TextureShare
                    })?;
                let srv_y = srv_y.ok_or_else(|| {
                    eprintln!("[opengl] CreateShaderResourceView(Y) returned null SRV");
                    Error::TextureShare
                })?;
                let srv_desc_uv = D3D11::D3D11_SHADER_RESOURCE_VIEW_DESC {
                    Format: Dxgi::Common::DXGI_FORMAT_R8G8_UNORM,
                    ViewDimension: Direct3D::D3D11_SRV_DIMENSION_TEXTURE2DARRAY,
                    Anonymous: D3D11::D3D11_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2DArray: D3D11::D3D11_TEX2D_ARRAY_SRV {
                            MostDetailedMip: 0,
                            MipLevels: tex_desc.MipLevels,
                            FirstArraySlice: array_slice,
                            ArraySize: 1,
                        },
                    },
                };
                let mut srv_uv = None;
                d3d11_device
                    .CreateShaderResourceView(
                        decoder_texture,
                        Some(&srv_desc_uv),
                        Some(&mut srv_uv),
                    )
                    .map_err(|e| {
                        eprintln!("[opengl] CreateShaderResourceView(UV) FAILED: {:?}", e);
                        Error::TextureShare
                    })?;
                let srv_uv = srv_uv.ok_or_else(|| {
                    eprintln!("[opengl] CreateShaderResourceView(UV) returned null SRV");
                    Error::TextureShare
                })?;
                eprintln!(
                    "[opengl] Y/UV SRVs created OK (array_slice={})",
                    array_slice
                );
                let pipeline = match build_plane_copy_pipeline(d3d11_device) {
                    Ok(p) => {
                        eprintln!("[opengl] build_plane_copy_pipeline OK");
                        p
                    }
                    Err(e) => {
                        eprintln!("[opengl] build_plane_copy_pipeline FAILED: {:?}", e);
                        return Err(e);
                    }
                };
                Ok(PlaneCopyCore {
                    device: d3d11_device.clone(),
                    context,
                    pipeline,
                    srv_y,
                    srv_uv,
                })
            }
        }
    }

    /// Persistent WGL-direct-interop ring. Replaces the per-frame
    /// `GL_EXT_memory_object_win32` re-copy with register-once + lock/unlock.
    #[cfg(feature = "experimental-wgl-interop")]
    struct GlInteropRing {
        core: PlaneCopyCore,
        wgl: wgl_nv_dx_interop::WglNvDxInterop,
        dx_device: *mut c_void,
        device: wgpu::Device,
        slots: Vec<GlInteropSlot>,
        generation: u32,
        /// Slot whose views were last locked and should be sampled by the draw.
        current_slot: Option<u8>,
        depth: layout::Depth,
        width: u32,
        height: u32,
        /// FFmpeg hwctx lock/unlock (serialize D3D11 immediate-context access).
        lock: Option<unsafe extern "C" fn(*mut c_void)>,
        unlock: Option<unsafe extern "C" fn(*mut c_void)>,
        lock_ctx: *mut c_void,
    }

    /// Active interop strategy. The cross-vendor `MemoryObject` ring is the
    /// production default. The `Wgl` variant is compiled in only with the
    /// `experimental-wgl-interop` Cargo feature and selected at runtime via
    /// `FFGPU_GL_INTEROP=wgl`.
    pub(super) enum InteropMode {
        #[cfg(feature = "experimental-wgl-interop")]
        Wgl(GlInteropRing),
        MemoryObject(MemoryObjectRing),
    }

    pub(super) struct D3D11GlImport {
        mode: Option<InteropMode>,
        depth: layout::Depth,
        width: u32,
        height: u32,
        color_space: ffmpeg_next::color::Space,
        d3d11_device: D3D11::ID3D11Device,
        #[allow(dead_code)] // held to keep the wgpu device alive for the import path
        device: wgpu::Device,
        decoder_texture: D3D11::ID3D11Texture2D,
        array_slice: u32,
        lock: Option<unsafe extern "C" fn(*mut c_void)>,
        unlock: Option<unsafe extern "C" fn(*mut c_void)>,
        lock_ctx: *mut c_void,
    }

    impl D3D11GlImport {
        pub(super) unsafe fn new(
            device: &wgpu::Device,
            decoder_texture: &D3D11::ID3D11Texture2D,
            array_slice: u32,
            width: u32,
            height: u32,
            depth: layout::Depth,
            color_space: ffmpeg_next::color::Space,
            d3d11_device: &D3D11::ID3D11Device,
            lock: Option<unsafe extern "C" fn(*mut c_void)>,
            unlock: Option<unsafe extern "C" fn(*mut c_void)>,
            lock_ctx: *mut c_void,
        ) -> Result<Self> {
            Ok(D3D11GlImport {
                mode: None,
                depth,
                width,
                height,
                color_space,
                d3d11_device: d3d11_device.clone(),
                device: device.clone(),
                decoder_texture: decoder_texture.clone(),
                array_slice,
                lock,
                unlock,
                lock_ctx,
            })
        }

        /// Build the interop mode on first import (the GL context is current
        /// here). Selects the cross-vendor `GL_EXT_memory_object_win32` ring
        /// when its capability gates pass. The WGL path is only attempted when
        /// the `experimental-wgl-interop` feature is on AND the user explicitly
        /// requested it via `FFGPU_GL_INTEROP=wgl`.
        unsafe fn init_mode(
            &mut self,
            device: &wgpu::Device,
            pipeline_cache: &mut PipelineCache,
        ) -> Result<()> {
            // Install the `gl` crate loader before any `gl::*` call (the loader is
            // process-global and idempotent). Without this, `gl::GenTextures` etc.
            // are `missing_fn_panic` stubs and OpenGL playback crashes on the
            // first decoded frame.
            ensure_gl_loaded(device);
            // Log D3D11 device flags early (useful for diagnosing interop bail).
            let dev_flags = self.d3d11_device.GetCreationFlags();
            eprintln!(
                "[opengl] D3D11 device flags: 0x{:08X}{}",
                dev_flags,
                if dev_flags & (D3D11::D3D11_CREATE_DEVICE_SINGLETHREADED.0 as u32) != 0 {
                    " [SINGLETHREADED]"
                } else {
                    ""
                }
            );
            // D3D11.1+ is required for video-resource ShaderResourceViews over
            // NV12 multi-plane views, and is also the runtime that exposes
            // ID3D11Device1. Feature level is reported separately and may
            // legitimately be 11_0 (0xB000) on a device whose runtime supports
            // 11.1 — the ID3D11Device1 availability is the runtime-level signal.
            let feature_level = self.d3d11_device.GetFeatureLevel();
            eprintln!("[opengl] D3D11 feature level: 0x{:04X}", feature_level.0);
            let has_device1 = self.d3d11_device.cast::<D3D11::ID3D11Device1>().is_ok();
            eprintln!("[opengl] ID3D11Device1 available: {}", has_device1);
            let pref = interop_preference();
            eprintln!("[opengl] interop_preference: {:?}", pref);

            // WGL is opt-in: only when the experimental Cargo feature is on AND
            // the user explicitly asked for it via FFGPU_GL_INTEROP=wgl. The
            // cross-vendor memory-object path is the production default.
            #[cfg(feature = "experimental-wgl-interop")]
            {
                if pref == InteropPreference::Wgl {
                    match GlInteropRing::new(
                        device,
                        pipeline_cache,
                        &self.d3d11_device,
                        &self.decoder_texture,
                        self.array_slice,
                        self.width,
                        self.height,
                        self.depth,
                        self.color_space,
                        self.lock,
                        self.unlock,
                        self.lock_ctx,
                    ) {
                        Ok(ring) => {
                            self.mode = Some(InteropMode::Wgl(ring));
                            eprintln!(
                                "[opengl] GL interop: WGL_NV_DX_interop2 ring active ({} slots)",
                                INTEROP_RING_SIZE
                            );
                            return Ok(());
                        }
                        Err(e) => {
                            eprintln!(
                                "[opengl] GL interop: WGL_NV_DX_interop2 unavailable ({:?}), \
                                 falling back to memory-object",
                                e
                            );
                        }
                    }
                }
            }

            // Production path: GL_EXT_memory_object_win32 + GL_EXT_win32_keyed_mutex.
            if pref != InteropPreference::Cpu {
                match MemoryObjectRing::new(
                    device,
                    pipeline_cache,
                    &self.d3d11_device,
                    &self.decoder_texture,
                    self.array_slice,
                    self.width,
                    self.height,
                    self.depth,
                    self.color_space,
                    self.lock,
                    self.unlock,
                    self.lock_ctx,
                ) {
                    Ok(ring) => {
                        self.mode = Some(InteropMode::MemoryObject(ring));
                        eprintln!(
                            "[opengl] GL interop: memory-object ring active ({} slots)",
                            INTEROP_RING_SIZE
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        eprintln!("[opengl] GL interop: memory-object ring failed ({:?})", e);
                    }
                }
            }
            Err(Error::TextureShare)
        }

        /// Run the import for the active mode. Returns a GL-interop ticket when
        /// the active ring locked a slot (the engine finishes it after submit);
        /// `None` is not currently produced (the memory-object ring always
        /// returns a ticket so the engine can insert the release fence).
        pub(super) unsafe fn import_frame(
            &mut self,
            device: &wgpu::Device,
            pipeline_cache: &mut PipelineCache,
            frame: NonNull<ff::AVFrame>,
        ) -> Result<Option<GlInteropTicket>> {
            if self.mode.is_none() {
                self.init_mode(device, pipeline_cache)?;
            }
            match self.mode.as_mut().unwrap() {
                #[cfg(feature = "experimental-wgl-interop")]
                InteropMode::Wgl(ring) => ring.import_frame(device, frame),
                InteropMode::MemoryObject(ring) => ring.import_frame(device, frame),
            }
        }

        pub(super) fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
            match &self.mode {
                #[cfg(feature = "experimental-wgl-interop")]
                Some(InteropMode::Wgl(ring)) => ring.plane_views(),
                Some(InteropMode::MemoryObject(ring)) => ring.plane_views(),
                None => None,
            }
        }

        pub(super) fn bind_group(&self) -> Option<&wgpu::BindGroup> {
            match &self.mode {
                Some(InteropMode::MemoryObject(ring)) => ring.bind_group(),
                #[cfg(feature = "experimental-wgl-interop")]
                Some(InteropMode::Wgl(_)) => None,
                None => None,
            }
        }

        pub(super) fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
            match &self.mode {
                Some(InteropMode::MemoryObject(ring)) => ring.layout_identity(),
                #[cfg(feature = "experimental-wgl-interop")]
                Some(InteropMode::Wgl(_)) => None,
                None => None,
            }
        }

        pub(super) fn finish_gl_frames(&mut self, tickets: &[GlInteropTicket]) -> Result<()> {
            match self.mode.as_mut() {
                Some(InteropMode::MemoryObject(ring)) => unsafe { ring.finish_gl_frames(tickets)? },
                #[cfg(feature = "experimental-wgl-interop")]
                Some(InteropMode::Wgl(ring)) => unsafe { ring.finish_gl_frames(tickets)? },
                None => {}
            }
            Ok(())
        }

        #[allow(dead_code)] // GL-interop cancel path reserved for the render-error handler
        pub(super) fn cancel_gl_frame(&mut self, ticket: GlInteropTicket) -> Result<()> {
            match self.mode.as_mut() {
                Some(InteropMode::MemoryObject(ring)) => unsafe { ring.cancel_gl_frame(ticket)? },
                #[cfg(feature = "experimental-wgl-interop")]
                Some(InteropMode::Wgl(ring)) => unsafe { ring.cancel_gl_frame(ticket)? },
                None => {}
            }
            Ok(())
        }
    }

    impl Drop for D3D11GlImport {
        fn drop(&mut self) {
            // The WGL ring holds resources that need explicit teardown (it
            // owns the WGL device association). The memory-object ring releases
            // its GL resources through the wgpu Texture's DropCallback on each
            // slot, so no teardown is required here.
            #[cfg(feature = "experimental-wgl-interop")]
            if let Some(InteropMode::Wgl(ring)) = self.mode.take() {
                unsafe { ring.teardown() };
            }
            let _ = self.mode.take();
        }
    }

    /// Persistent memory-object ring implementation.
    ///
    /// One-shot build:
    ///   1. Build the shared `PlaneCopyCore` (decoder SRV + plane-copy PSO).
    ///   2. Load `GL_EXT_memory_object` / `GL_EXT_memory_object_win32` /
    ///      `GL_EXT_win32_keyed_mutex` entry points.
    ///   3. Compare D3D11 adapter LUID with `GL_DEVICE_LUID_EXT`. Missing or
    ///      mismatched LUIDs reject the universal import before allocation.
    ///   4. For each slot:
    ///      - create the D3D11 R8/RG8 textures (`SHARED_NTHANDLE |
    ///        SHARED_KEYEDMUTEX`);
    ///      - mark the GL memory object as dedicated, then import the NT
    ///        handle with size = 0 (D3D11 image handles ignore the size hint
    ///        and the spec notes that 0 has broader compatibility);
    ///      - bind the GL texture storage to the imported memory object via
    ///        `glTexStorageMem2DEXT` (works on an OpenGL 3.3 baseline — no
    ///        DSA required);
    ///      - wrap the GL texture as a wgpu `Texture` (the DropCallback frees
    ///        the GL name and memory object on wgpu-side teardown);
    ///      - close the NT handle — GL does not take ownership of it.
    ///
    /// Per-frame sync (D3D11 side: COM `IDXGIKeyedMutex`):
    ///   - `AcquireSync(0)` (waits for key 0 to be released)
    ///   - render the plane-copy pass into the slot's R8/RG8 targets
    ///   - `ReleaseSync(1)` (signals key 1)
    ///
    /// Per-frame sync (GL side: `GL_EXT_win32_keyed_mutex`):
    ///   - `glAcquireKeyedMutexWin32EXT(mem, 1, timeout)` — enqueues a wait
    ///     on the GL command stream for the D3D11 producer to release key 1
    ///   - engine records the video draw and submits it via `queue.submit`
    ///   - `glReleaseKeyedMutexWin32EXT(mem, 0)` (after the engine's
    ///     `finish_all_gl_frames`) — releases the mutex back to key 0
    ///   - one `glFlush()` per submission, so the release leaves the client
    ///     command buffer promptly
    impl MemoryObjectRing {
        unsafe fn new(
            device: &wgpu::Device,
            pipeline_cache: &mut PipelineCache,
            d3d11_device: &D3D11::ID3D11Device,
            decoder_texture: &D3D11::ID3D11Texture2D,
            array_slice: u32,
            width: u32,
            height: u32,
            depth: layout::Depth,
            color_space: ffmpeg_next::color::Space,
            lock: Option<unsafe extern "C" fn(*mut c_void)>,
            unlock: Option<unsafe extern "C" fn(*mut c_void)>,
            lock_ctx: *mut c_void,
        ) -> Result<Self> {
            unsafe {
                // (1) Load the required extensions + read the LUIDs + log the
                // GL ICD routing. All raw `glXxx` calls (glGetString,
                // glGetIntegerv, the `load()` probes) are done under a brief
                // GL-context lock. The lock MUST be dropped before any wgpu
                // API call (`PlaneCopyCore::new`, `build_memory_object_slot`,
                // `pipeline_cache.bind_frame_textures`), because wgpu's GL
                // backend re-locks the same (non-reentrant) mutex internally;
                // holding it across those calls would deadlock and
                // wgpu-hal-29.0.4 panics with "Could not lock adapter
                // context".
                let ext_static: &'static win32_gl_ext::Win32GlExt;
                let d3d11_luid: Option<[u8; 8]>;
                let gl_luid: Option<[u8; 8]>;
                {
                    let hal = device
                        .as_hal::<wgpu::hal::gles::Api>()
                        .ok_or(Error::UnsupportedBackend)?;
                    let _gl_guard = hal.context().lock();
                    if GL_WIN32_EXT.get().is_none() {
                        let loaded = win32_gl_ext::load().ok_or_else(|| {
                            eprintln!(
                                "[opengl] required external-memory entry points are unavailable"
                            );
                            Error::TextureShare
                        })?;
                        let _ = GL_WIN32_EXT.set(loaded);
                    }
                    ext_static = GL_WIN32_EXT.get().ok_or(Error::TextureShare)?;
                    log_gl_icd_diagnostics(ext_static);
                    d3d11_luid = read_d3d11_adapter_luid(d3d11_device);
                    gl_luid = read_gl_device_luid(ext_static);
                }

                // (2) LUID compare — external memory must originate from the
                // same device or device set as the current GL context. The
                // Win32 extension exposes the LUID as eight unsigned bytes;
                // accepting a missing or mismatched value would make the
                // subsequent import undefined and can select the wrong GPU.
                match (d3d11_luid, gl_luid) {
                    (Some(d), Some(g)) if d == g => {
                        eprintln!(
                            "[opengl] memory-object: D3D11 LUID matches GL LUID {:02X?}",
                            g
                        );
                    }
                    (Some(d), Some(g)) => {
                        eprintln!(
                            "[opengl] memory-object: LUID mismatch (D3D11={:02X?}, GL={:02X?}) — refusing import",
                            d, g
                        );
                        return Err(Error::TextureShare);
                    }
                    (d, g) => {
                        eprintln!(
                            "[opengl] memory-object: adapter LUID unavailable (D3D11={:?}, GL={:?}) — refusing import",
                            d, g
                        );
                        return Err(Error::TextureShare);
                    }
                }

                // (4) Shared decoder-side state: SRV, immediate context, plane-copy PSO.
                let core = PlaneCopyCore::new(d3d11_device, decoder_texture, array_slice)?;

                // (5) Build N slots. Each slot owns:
                //   - its D3D11 R8/RG8 textures (per-slot)
                //   - one GL memory object + one GL texture name per plane
                //   - the cached wgpu wrapper (external ownership; the
                //     DropCallback frees the GL name + memory object)
                eprintln!(
                    "[opengl] memory-object: building {} slots ({}x{}, depth={:?})",
                    INTEROP_RING_SIZE, width, height, depth
                );
                let mut slots = Vec::with_capacity(INTEROP_RING_SIZE);
                for slot_id in 0..(INTEROP_RING_SIZE as u8) {
                    eprintln!("[opengl]   memory-object slot {} ...", slot_id);
                    let slot = build_memory_object_slot(
                        device,
                        d3d11_device,
                        width,
                        height,
                        depth,
                        slot_id,
                        ext_static,
                    )?;
                    eprintln!("[opengl]   memory-object slot {} OK", slot_id);
                    slots.push(slot);
                }

                // (6) Cache a single bind group. The engine reads this for the
                // `direct_yuv()` gate; the actual draw uses `plane_views()` to
                // build a fresh bind group from the current slot's views. We
                // build against the first slot's views — if the engine ever
                // uses this bind group for the copy_to_rgb fallback, it
                // samples the first slot's stale frames. The engine's
                // `direct_yuv` path never hits copy_to_rgb.
                let (y_view, uv_view) = match (&slots[0].y_view, &slots[0].uv_view) {
                    (Some(y), Some(uv)) => (y, uv),
                    _ => return Err(Error::TextureShare),
                };
                let bind_group = pipeline_cache.bind_frame_textures(
                    &layout::FrameDescriptor {
                        planes: layout::PlaneLayout::PackedYUV420([
                            y_view.clone(),
                            uv_view.clone(),
                        ]),
                        depth,
                    },
                    color_space,
                );
                let _ = pipeline_cache; // (used above; kept for symmetry with the WGL path)

                // (7) Build the layout identity used by the engine to select
                // the correct YUV→RGB conversion matrix.
                let (y_fmt, _uv_fmt) = match depth {
                    layout::Depth::D16 => (
                        wgpu::TextureFormat::R16Unorm,
                        wgpu::TextureFormat::Rg16Unorm,
                    ),
                    _ => (wgpu::TextureFormat::R8Unorm, wgpu::TextureFormat::Rg8Unorm),
                };
                let layout_identity = layout::FrameDescriptor {
                    planes: layout::PlaneLayout::PackedYUV420([
                        y_fmt,
                        wgpu::TextureFormat::Rg8Unorm,
                    ]),
                    depth,
                }
                .as_identity();

                Ok(MemoryObjectRing {
                    core,
                    ext: ext_static,
                    device: device.clone(),
                    slots,
                    generation: 1,
                    current_slot: None,
                    depth,
                    width,
                    height,
                    lock,
                    unlock,
                    lock_ctx,
                    bind_group: Some(bind_group),
                    layout_identity,
                })
            }
        }

        /// Run the D3D11 plane-copy + GL keyed-mutex acquire for one frame.
        /// Returns a ticket identifying the locked slot; the engine hands it
        /// back via `finish_gl_frames` (after `queue.submit`) so we can insert
        /// the matching `glReleaseKeyedMutexWin32EXT` and a single `glFlush`.
        pub(super) unsafe fn import_frame(
            &mut self,
            device: &wgpu::Device,
            frame: NonNull<ff::AVFrame>,
        ) -> Result<Option<GlInteropTicket>> {
            unsafe {
                self.reclaim_completed();
                let frame_ref = frame.as_ref();
                if frame_ref.data[0].is_null() {
                    return Err(Error::InvalidFrame);
                }

                // (a) Pick a free slot. Starvation drops the frame: the engine
                // keeps sampling the last locked slot instead of blocking.
                let slot_idx = match self
                    .slots
                    .iter()
                    .position(|s| s.state == MemoryObjectSlotState::Free)
                {
                    Some(i) => i,
                    None => {
                        eprintln!(
                            "[opengl] memory-object: ring starvation — dropping decoded frame"
                        );
                        return Ok(None);
                    }
                };

                // (b) D3D11 side: AcquireSync(0) → plane copy → ReleaseSync(1).
                if let (Some(lock_fn), ctx) = (self.lock, self.lock_ctx) {
                    lock_fn(ctx);
                }
                let y_km = self.slots[slot_idx]
                    .y_d3d11
                    .cast::<Dxgi::IDXGIKeyedMutex>()
                    .map_err(|_| Error::TextureShare)?;
                let uv_km = self.slots[slot_idx]
                    .uv_d3d11
                    .cast::<Dxgi::IDXGIKeyedMutex>()
                    .map_err(|_| Error::TextureShare)?;
                y_km.AcquireSync(0, u32::MAX)
                    .map_err(|_| Error::TextureShare)?;
                uv_km
                    .AcquireSync(0, u32::MAX)
                    .map_err(|_| Error::TextureShare)?;

                {
                    let ctx = &self.core.context;
                    ctx.ClearState();
                    ctx.IASetInputLayout(&self.core.pipeline.layout);
                    ctx.IASetPrimitiveTopology(Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    ctx.VSSetShader(Some(&self.core.pipeline.vs), None);
                    let samplers = [Some(self.core.pipeline.sampler.clone())];
                    let y_srvs = [Some(self.core.srv_y.clone())];
                    let uv_srvs = [Some(self.core.srv_uv.clone())];
                    let y_rtvs = [Some(self.slots[slot_idx].y_rtv.clone())];
                    let uv_rtvs = [Some(self.slots[slot_idx].uv_rtv.clone())];
                    ctx.PSSetSamplers(0, Some(&samplers));
                    ctx.PSSetShaderResources(0, Some(&y_srvs));
                    ctx.OMSetRenderTargets(Some(&y_rtvs), None);
                    let y_viewport = plane_copy_viewport(self.width, self.height);
                    ctx.RSSetViewports(Some(std::slice::from_ref(&y_viewport)));
                    ctx.PSSetShader(Some(&self.core.pipeline.ps_y), None);
                    ctx.Draw(3, 0);
                    ctx.OMSetRenderTargets(Some(&uv_rtvs), None);
                    let uv_viewport = plane_copy_viewport(self.width / 2, self.height / 2);
                    ctx.RSSetViewports(Some(std::slice::from_ref(&uv_viewport)));
                    ctx.PSSetShaderResources(0, Some(&uv_srvs));
                    ctx.PSSetShader(Some(&self.core.pipeline.ps_uv), None);
                    ctx.Draw(3, 0);
                    ctx.Flush();
                }

                y_km.ReleaseSync(1).map_err(|_| Error::TextureShare)?;
                uv_km.ReleaseSync(1).map_err(|_| Error::TextureShare)?;
                if let (Some(unlock_fn), ctx) = (self.unlock, self.lock_ctx) {
                    unlock_fn(ctx);
                }

                // (c) GL side: enqueue the keyed-mutex wait for key=1. This
                // does NOT block the calling thread; it inserts a wait on the
                // GL command stream. The actual stall happens when the next
                // GL command (the engine's video draw) tries to sample the
                // memory.
                let hal = device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();
                let acquire_ok_y = (self.ext.acquire_keyed_mutex)(
                    self.slots[slot_idx].y_mem,
                    1,
                    KEYED_MUTEX_TIMEOUT_MS,
                );
                let acquire_ok_uv = (self.ext.acquire_keyed_mutex)(
                    self.slots[slot_idx].uv_mem,
                    1,
                    KEYED_MUTEX_TIMEOUT_MS,
                );
                if acquire_ok_y == 0 || acquire_ok_uv == 0 {
                    eprintln!(
                        "[opengl] memory-object: glAcquireKeyedMutexWin32EXT failed (y={}, uv={})",
                        acquire_ok_y, acquire_ok_uv
                    );
                    // Try to release back to key=0 so the producer can move on.
                    let _ = (self.ext.release_keyed_mutex)(self.slots[slot_idx].y_mem, 0);
                    let _ = (self.ext.release_keyed_mutex)(self.slots[slot_idx].uv_mem, 0);
                    return Err(Error::TextureShare);
                }

                self.slots[slot_idx].state = MemoryObjectSlotState::Locked;
                self.current_slot = Some(self.slots[slot_idx].slot_id);
                let _ = device; // reserved for future per-frame work

                Ok(Some(GlInteropTicket {
                    generation: self.generation,
                    slot_id: self.slots[slot_idx].slot_id,
                }))
            }
        }

        /// Insert a GL completion fence for every submitted ticket's slot.
        /// The keyed mutex is released only after the fence signals; releasing
        /// immediately after `queue.submit` races the GL sampler and is
        /// rejected by `GL_EXT_win32_keyed_mutex`.
        pub(super) unsafe fn finish_gl_frames(
            &mut self,
            tickets: &[GlInteropTicket],
        ) -> Result<()> {
            if tickets.is_empty() {
                return Ok(());
            }
            unsafe {
                let hal = self
                    .device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();
                for t in tickets {
                    if t.generation != self.generation {
                        continue;
                    }
                    let Some(slot) = self.slots.iter_mut().find(|s| s.slot_id == t.slot_id) else {
                        continue;
                    };
                    if slot.state != MemoryObjectSlotState::Locked {
                        continue;
                    }
                    let fence = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
                    if fence.is_null() {
                        return Err(Error::Unknown);
                    }
                    slot.fence = Some(fence);
                    slot.state = MemoryObjectSlotState::Submitted;
                }
                gl::Flush();
            }
            Ok(())
        }

        /// Reclaim slots after the GL sampler has finished. This is called
        /// before the next D3D11 producer pass, so `AcquireSync(0)` never races
        /// the previous GL submission.
        unsafe fn reclaim_completed(&mut self) {
            unsafe {
                let hal = match self.device.as_hal::<wgpu::hal::gles::Api>() {
                    Some(hal) => hal,
                    None => return,
                };
                let _gl_guard = hal.context().lock();
                for slot in self.slots.iter_mut() {
                    let Some(fence) = slot.fence else {
                        continue;
                    };
                    if slot.state != MemoryObjectSlotState::Submitted {
                        continue;
                    }
                    let status = gl::ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT, 0);
                    if status != gl::ALREADY_SIGNALED && status != gl::CONDITION_SATISFIED {
                        continue;
                    }
                    let r_y = (self.ext.release_keyed_mutex)(slot.y_mem, 0);
                    let r_uv = (self.ext.release_keyed_mutex)(slot.uv_mem, 0);
                    if r_y != 0 && r_uv != 0 {
                        gl::DeleteSync(fence);
                        slot.fence = None;
                        slot.state = MemoryObjectSlotState::Free;
                    } else {
                        eprintln!(
                            "[opengl] memory-object: keyed-mutex release failed after fence (y={}, uv={})",
                            r_y, r_uv
                        );
                    }
                }
            }
        }

        /// Cancel a locked slot whose draw was never recorded. Enqueue the
        /// release back to key 0 so the producer can move on.
        #[allow(dead_code)] // GL-interop cancel path reserved for the render-error handler
        pub(super) unsafe fn cancel_gl_frame(&mut self, ticket: GlInteropTicket) -> Result<()> {
            unsafe {
                if ticket.generation != self.generation {
                    return Ok(());
                }
                let Some(slot) = self.slots.iter_mut().find(|s| s.slot_id == ticket.slot_id) else {
                    return Ok(());
                };
                if slot.state != MemoryObjectSlotState::Locked {
                    return Ok(());
                }
                let hal = self
                    .device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();
                let _ = (self.ext.release_keyed_mutex)(slot.y_mem, 0);
                let _ = (self.ext.release_keyed_mutex)(slot.uv_mem, 0);
                if let Some(fence) = slot.fence.take() {
                    gl::DeleteSync(fence);
                }
                slot.state = MemoryObjectSlotState::Free;
            }
            Ok(())
        }

        /// Per-plane `wgpu::TextureView`s (Y, then UV) for direct engine
        /// sampling. The engine builds a bind group from these views via
        /// `PipelineCache::bind_frame_textures`.
        pub(super) fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
            let slot_id = self.current_slot?;
            let slot = self.slots.iter().find(|s| s.slot_id == slot_id)?;
            Some(vec![
                slot.y_view.as_ref()?.clone(),
                slot.uv_view.as_ref()?.clone(),
            ])
        }

        /// Cached YUV bind group built from the first slot's views. Kept only
        /// so the engine's `direct_yuv()` gate (`bind_group().is_some()`)
        /// returns true; the actual draw uses `plane_views()`.
        pub(super) fn bind_group(&self) -> Option<&wgpu::BindGroup> {
            self.bind_group.as_ref()
        }

        pub(super) fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
            Some(self.layout_identity)
        }
    }

    /// Bounded timeout (in **milliseconds**) for `glAcquireKeyedMutexWin32EXT`.
    /// 100 ms is generous at 60 fps (16 ms/frame) and short enough to surface
    /// a stuck producer via the `Err(TextureShare)` path instead of freezing
    /// the render thread.
    const KEYED_MUTEX_TIMEOUT_MS: gl::types::GLuint64 = 100;

    /// Read the D3D11 device's adapter LUID via `IDXGIDevice::GetAdapter`.
    /// Returns the complete eight-byte Windows LUID in native byte order.
    unsafe fn read_d3d11_adapter_luid(d3d11_device: &D3D11::ID3D11Device) -> Option<[u8; 8]> {
        unsafe {
            let dxgi_device = d3d11_device.cast::<Dxgi::IDXGIDevice>().ok()?;
            let adapter = dxgi_device.GetAdapter().ok()?;
            let desc = adapter.GetDesc().ok()?;
            let mut bytes = [0u8; 8];
            bytes[..4].copy_from_slice(&desc.AdapterLuid.LowPart.to_ne_bytes());
            bytes[4..].copy_from_slice(&desc.AdapterLuid.HighPart.to_ne_bytes());
            Some(bytes)
        }
    }

    /// Read the active WGL context's device LUID via the Win32 external-memory
    /// extension. LUID is an eight-byte unsigned-byte query; using
    /// `GetIntegerv` here returns unrelated state and can manufacture a false
    /// adapter mismatch.
    unsafe fn read_gl_device_luid(ext: &win32_gl_ext::Win32GlExt) -> Option<[u8; 8]> {
        unsafe {
            let mut luid = [0u8; 8];
            (ext.get_unsigned_bytev)(GL_DEVICE_LUID_EXT, luid.as_mut_ptr());
            if gl::GetError() != gl::NO_ERROR || luid == [0; 8] {
                None
            } else {
                Some(luid)
            }
        }
    }

    /// Print one line of OpenGL ICD routing info so we can see exactly which
    /// vendor's driver the system picked. Useful when D3D11 and the GL
    /// context end up on different physical adapters (Optimus, MS Hybrid,
    /// Microsoft Basic Render Driver, Mesa swrast, etc.).
    unsafe fn log_gl_icd_diagnostics(ext: &win32_gl_ext::Win32GlExt) {
        unsafe {
            let vendor = gl_str(gl::VENDOR);
            let renderer = gl_str(gl::RENDERER);
            let version = gl_str(gl::VERSION);
            let luid = read_gl_device_luid(ext);
            let mut node_mask = 0i32;
            gl::GetIntegerv(GL_DEVICE_NODE_MASK_EXT, &mut node_mask as *mut _);
            eprintln!(
                "[opengl] GL ICD: vendor={} renderer={} version={} luid={:02X?} node_mask=0x{:x}",
                vendor, renderer, version, luid, node_mask as u32
            );
        }
    }

    unsafe fn check_gl_error(stage: &str) -> Result<()> {
        let error = gl::GetError();
        if error != gl::NO_ERROR {
            eprintln!("[opengl] GL error 0x{:04X} during {}", error, stage);
            Err(Error::TextureShare)
        } else {
            Ok(())
        }
    }

    /// `glGetString` wrapper that returns a printable `&str`. Null is treated
    /// as an empty string (some drivers can return NULL for an unknown
    /// pname).
    unsafe fn gl_str(pname: gl::types::GLenum) -> &'static str {
        unsafe {
            let raw = gl::GetString(pname);
            if raw.is_null() {
                return "<unavail>";
            }
            // SAFETY: GL strings are NUL-terminated and live for the
            // process lifetime (the GL context is current and the string
            // is owned by the ICD).
            let cstr = std::ffi::CStr::from_ptr(raw as *const i8);
            cstr.to_str().unwrap_or("<unavail>")
        }
    }

    /// Build one persistent memory-object ring slot.
    ///
    /// Creates the D3D11 R8/RG8 textures, marks the GL memory object as
    /// dedicated, imports the NT handle (size 0), binds the GL texture
    /// storage, wraps it as a wgpu `Texture` (with a DropCallback that frees
    /// the GL name and the memory object on wgpu-side teardown), and closes
    /// the NT handle.
    unsafe fn build_memory_object_slot(
        device: &wgpu::Device,
        d3d11_device: &D3D11::ID3D11Device,
        width: u32,
        height: u32,
        depth: layout::Depth,
        slot_id: u8,
        ext: &win32_gl_ext::Win32GlExt,
    ) -> Result<MemoryObjectSlot> {
        unsafe {
            // (A) D3D11 side — no GL lock needed.
            let textures = create_plane_copy_textures(d3d11_device, width, height, depth)?;
            // Move the textures / RTVs / handles out of the helper struct so
            // we can keep them alive for the rest of this function.
            let PlaneCopyTextures {
                y: y_d3d11,
                uv: uv_d3d11,
                y_rtv,
                uv_rtv,
                y_handle,
                uv_handle,
            } = textures;

            let (y_fmt, uv_fmt) = match depth {
                layout::Depth::D16 => (GL_R16, GL_RG16),
                _ => (GL_R8, GL_RG8),
            };
            let (y_w, y_h) = (width, height);
            let (uv_w, uv_h) = (width / 2, height / 2);

            // (B) GL side — take the context lock briefly for the raw
            // `glXxx` calls. The lock MUST be released before the
            // `wrap_external_gl_texture` and `create_view` calls below,
            // because wgpu's GL backend re-locks the same (non-reentrant)
            // mutex internally; holding it across those wgpu calls would
            // deadlock (`wgpu-hal-29.0.4/src/gles/wgl.rs:68` panics with
            // "Could not lock adapter context. This is most-likely a
            // deadlock.").
            let (y_mem, uv_mem, y_gl, uv_gl) = {
                let hal = device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();

                // (1) Allocate one GL memory object per plane and mark it
                // as a dedicated image. The Khronos spec requires the
                // dedicated flag for image handles (D3D11 image handles
                // identify a dedicated image allocation).
                let mut mems = [0u32, 0u32];
                (ext.create_memory_objects)(2, mems.as_mut_ptr());
                check_gl_error("CreateMemoryObjectsEXT")?;
                if mems[0] == 0 || mems[1] == 0 {
                    return Err(Error::TextureShare);
                }
                let true_val = [GL_TRUE as gl::types::GLint];
                (ext.memory_object_parameteriv)(
                    mems[0],
                    GL_DEDICATED_MEMORY_OBJECT_EXT,
                    true_val.as_ptr(),
                );
                check_gl_error("MemoryObjectParameterivEXT(Y)")?;
                (ext.memory_object_parameteriv)(
                    mems[1],
                    GL_DEDICATED_MEMORY_OBJECT_EXT,
                    true_val.as_ptr(),
                );
                check_gl_error("MemoryObjectParameterivEXT(UV)")?;

                // (2) Import each NT handle into its memory object. Size 0
                // has broader compatibility — the spec says D3D11 image
                // handles ignore the size and some drivers mis-validate
                // non-zero.
                (ext.import_memory_win32)(
                    mems[0],
                    0,
                    GL_HANDLE_TYPE_D3D11_IMAGE_EXT,
                    y_handle.0 as *mut std::ffi::c_void,
                );
                check_gl_error("ImportMemoryWin32HandleEXT(Y)")?;
                (ext.import_memory_win32)(
                    mems[1],
                    0,
                    GL_HANDLE_TYPE_D3D11_IMAGE_EXT,
                    uv_handle.0 as *mut std::ffi::c_void,
                );
                check_gl_error("ImportMemoryWin32HandleEXT(UV)")?;
                // GL does not take ownership of the NT handle; the
                // application remains responsible for closing it.
                let _ = windows::Win32::Foundation::CloseHandle(y_handle);
                let _ = windows::Win32::Foundation::CloseHandle(uv_handle);

                // (3) Generate the GL texture names and bind their storage
                // to the imported memory. We use the non-DSA
                // `glTexStorageMem2DEXT` (works on an OpenGL 3.3 baseline;
                // the `gl::TexStorage*` DSA variants only exist when the
                // driver exposes the right DSA entry point, which is not
                // guaranteed).
                let mut names = [0u32, 0u32];
                gl::GenTextures(2, names.as_mut_ptr());
                check_gl_error("GenTextures")?;
                if names[0] == 0 || names[1] == 0 {
                    return Err(Error::TextureShare);
                }
                gl::BindTexture(GL_TEXTURE_2D, names[0]);
                gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MIN_FILTER, GL_NEAREST as i32);
                gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MAG_FILTER, GL_NEAREST as i32);
                (ext.tex_storage_mem_2d)(
                    GL_TEXTURE_2D,
                    1,
                    y_fmt,
                    y_w as gl::types::GLsizei,
                    y_h as gl::types::GLsizei,
                    mems[0],
                    0,
                );
                check_gl_error("TexStorageMem2DEXT(Y)")?;
                gl::BindTexture(GL_TEXTURE_2D, names[1]);
                gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MIN_FILTER, GL_NEAREST as i32);
                gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MAG_FILTER, GL_NEAREST as i32);
                (ext.tex_storage_mem_2d)(
                    GL_TEXTURE_2D,
                    1,
                    uv_fmt,
                    uv_w as gl::types::GLsizei,
                    uv_h as gl::types::GLsizei,
                    mems[1],
                    0,
                );
                check_gl_error("TexStorageMem2DEXT(UV)")?;

                (mems[0], mems[1], names[0], names[1])
                // _gl_guard dropped here, BEFORE any wgpu API call
            };

            // (C) wgpu wrapping — no GL lock held by us; wgpu locks the
            // context itself as needed. Externally imported textures are
            // registered (not actually compiled), so this is the path that
            // does NOT panic on re-entry.
            let y_wgpu =
                wrap_external_gl_texture(device, y_gl, y_w, y_h, wgpu_y_format(depth), ext, y_mem)?;
            let y_view = y_wgpu.create_view(&wgpu::TextureViewDescriptor::default());
            let uv_wgpu = wrap_external_gl_texture(
                device,
                uv_gl,
                uv_w,
                uv_h,
                wgpu_uv_format(depth),
                ext,
                uv_mem,
            )?;
            let uv_view = uv_wgpu.create_view(&wgpu::TextureViewDescriptor::default());

            Ok(MemoryObjectSlot {
                slot_id,
                y_d3d11,
                uv_d3d11,
                y_rtv,
                uv_rtv,
                y_mem,
                uv_mem,
                y_gl,
                uv_gl,
                y_wgpu: Some(y_wgpu),
                uv_wgpu: Some(uv_wgpu),
                y_view: Some(y_view),
                uv_view: Some(uv_view),
                state: MemoryObjectSlotState::Free,
                fence: None,
            })
        }
    }

    /// Wrap a GL texture name backed by an external memory object as a wgpu
    /// `Texture`. The `DropCallback` releases the GL name and the GL memory
    /// object when wgpu drops the texture, so no separate teardown is
    /// required.
    unsafe fn wrap_external_gl_texture(
        device: &wgpu::Device,
        gl_name: gl::types::GLuint,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        ext: &win32_gl_ext::Win32GlExt,
        mem: gl::types::GLuint,
    ) -> Result<wgpu::Texture> {
        unsafe {
            let hal = device
                .as_hal::<wgpu::hal::gles::Api>()
                .ok_or(Error::UnsupportedBackend)?;
            // Capture the GL name and the memory object handle by value so
            // the DropCallback is `'static`. Cast through `usize` so the
            // raw pointer doesn't break `Send + Sync`.
            let gl_name_us = gl_name as usize;
            let mem_us = mem as usize;
            let delete_mem = ext.delete_memory_objects;
            let drop_cb: Option<wgpu::hal::DropCallback> = Some(Box::new(move || {
                let mut name = [gl_name_us as gl::types::GLuint];
                gl::DeleteTextures(1, name.as_mut_ptr());
                let mut mems = [mem_us as gl::types::GLuint];
                delete_mem(1, mems.as_mut_ptr());
            }));

            let hal_tex = hal.texture_from_raw(
                std::num::NonZeroU32::new(gl_name).ok_or(Error::Unknown)?,
                &wgpu::hal::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_DST,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                drop_cb,
            );
            Ok(device.create_texture_from_hal::<wgpu::hal::gles::Api>(
                hal_tex,
                &wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            ))
        }
    }

    /// `read_gl_device_luid` keeps the LUID compare as the authoritative gate.
    /// When `GL_DEVICE_LUID_EXT` is not exposed by the driver, the import is
    /// rejected — a cross-adapter mismatch is never silently accepted.

    #[cfg(feature = "experimental-wgl-interop")]
    impl GlInteropRing {
        unsafe fn new(
            device: &wgpu::Device,
            pipeline_cache: &mut PipelineCache,
            d3d11_device: &D3D11::ID3D11Device,
            decoder_texture: &D3D11::ID3D11Texture2D,
            array_slice: u32,
            width: u32,
            height: u32,
            depth: layout::Depth,
            color_space: ffmpeg_next::color::Space,
            lock: Option<unsafe extern "C" fn(*mut c_void)>,
            unlock: Option<unsafe extern "C" fn(*mut c_void)>,
            lock_ctx: *mut c_void,
        ) -> Result<Self> {
            unsafe {
                // (1) Lock the GL context so WGL functions are callable.
                let hal = device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();

                // (2) Diagnostic: adapter LUID, GL_RENDERER, WGL extension presence.
                //     Logged BEFORE the WGL load so the bail message below is the
                //     only line you need to read.
                log_diagnostic_adapter_info(d3d11_device);

                // (3) Load WGL_NV_DX_interop2 entry points.
                let wgl = wgl_nv_dx_interop::load().ok_or_else(|| {
                    eprintln!("[opengl] WGL_NV_DX_interop2: wgl_nv_dx_interop::load() returned None — no interop entry points available");
                    Error::TextureShare
                })?;

                // (4) Device validation — reject single-threaded D3D11 devices
                // (WGL requires a multithreaded device).
                let flags = d3d11_device.GetCreationFlags();
                if flags & (D3D11::D3D11_CREATE_DEVICE_SINGLETHREADED.0 as u32) != 0 {
                    eprintln!(
                        "[opengl] WGL interop rejected: D3D11 device is SINGLETHREADED (flags=0x{:08X})",
                        flags
                    );
                    return Err(Error::TextureShare);
                }

                // (5) Open the D3D11 device for WGL interop.
                let dx_device = (wgl.dx_open_device)(d3d11_device.as_raw() as *mut c_void);
                if dx_device.is_null() {
                    eprintln!(
                        "[opengl] wglDXOpenDeviceNV FAILED — device incompatibility or cross-GPU mismatch"
                    );
                    return Err(Error::TextureShare);
                }
                eprintln!("[opengl] wglDXOpenDeviceNV OK (handle={:p})", dx_device);

                // (5) Shared plane-copy core (pipeline + decoder Y/UV SRVs).
                // Keep WGL and the universal memory-object path on the same
                // validated NV12 setup. In particular, the decoder SRVs must
                // expose both planes; a single R8 SRV leaves the chroma plane
                // undefined and produces a solid-color frame.
                let core = PlaneCopyCore::new(d3d11_device, decoder_texture, array_slice)?;

                // (6) Build N persistent slots.
                eprintln!(
                    "[opengl] building {} ring slots ({}x{})",
                    INTEROP_RING_SIZE, width, height
                );
                let mut slots = Vec::with_capacity(INTEROP_RING_SIZE);
                for slot_id in 0..(INTEROP_RING_SIZE as u8) {
                    eprintln!("[opengl]   creating ring slot {}...", slot_id);
                    let slot = create_ring_slot(
                        &wgl,
                        dx_device,
                        device,
                        d3d11_device,
                        width,
                        height,
                        depth,
                        slot_id,
                    )?;
                    eprintln!("[opengl]   ring slot {} OK", slot_id);
                    slots.push(slot);
                }

                let _ = pipeline_cache; // unused on WGL path; kept for symmetry
                let _ = color_space;

                Ok(GlInteropRing {
                    core,
                    wgl,
                    dx_device,
                    device: device.clone(),
                    slots,
                    generation: 1,
                    current_slot: None,
                    depth,
                    width,
                    height,
                    lock,
                    unlock,
                    lock_ctx,
                })
            }
        }

        /// Reclaim any `Submitted` slot whose GL fence has signaled.
        unsafe fn reclaim(&mut self) {
            unsafe {
                let hal = match self.device.as_hal::<wgpu::hal::gles::Api>() {
                    Some(h) => h,
                    None => return,
                };
                let _gl_guard = hal.context().lock();
                for slot in self.slots.iter_mut() {
                    if let (WglSlotState::Submitted, Some(fence)) = (slot.state, slot.fence) {
                        let status = gl::ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT, 0);
                        if status == gl::ALREADY_SIGNALED || status == gl::CONDITION_SATISFIED {
                            gl::DeleteSync(fence);
                            slot.fence = None;
                            Self::unlock_slot(&self.wgl, self.dx_device, slot);
                        }
                    }
                }
            }
        }

        // Associated function (no `&self`) so it can run while a `self.slots`
        // element is already mutably borrowed by the caller.
        unsafe fn unlock_slot(
            wgl: &wgl_nv_dx_interop::WglNvDxInterop,
            dx_device: *mut c_void,
            slot: &mut GlInteropSlot,
        ) {
            unsafe {
                let names = [slot.y_gl, slot.uv_gl];
                let resources = [slot.y_wgl, slot.uv_wgl];
                let ok = (wgl.dx_unlock_objects)(dx_device, 2, names.as_ptr(), resources.as_ptr());
                if ok == 0 {
                    eprintln!("[opengl] wglDXUnlockObjectsNV failed during reclaim");
                }
                slot.state = WglSlotState::Free;
            }
        }

        pub(super) unsafe fn import_frame(
            &mut self,
            device: &wgpu::Device,
            frame: NonNull<ff::AVFrame>,
        ) -> Result<Option<GlInteropTicket>> {
            unsafe {
                eprintln!("[opengl] import_frame: enter");
                self.reclaim();
                eprintln!("[opengl] import_frame: reclaimed");

                // Pick a Free slot. On starvation, drop this frame (the engine
                // keeps sampling the last locked slot) rather than overwrite an
                // in-use one or block the renderer.
                let slot_idx = match self
                    .slots
                    .iter()
                    .position(|s| s.state == WglSlotState::Free)
                {
                    Some(i) => i,
                    None => {
                        eprintln!("[opengl] interop ring starvation: dropping decoded frame");
                        return Ok(None);
                    }
                };
                eprintln!("[opengl] import_frame: picked slot {}", slot_idx);

                let (y_fmt, uv_fmt) = match self.depth {
                    layout::Depth::D16 => (
                        Dxgi::Common::DXGI_FORMAT_R16_UNORM,
                        Dxgi::Common::DXGI_FORMAT_R16G16_UNORM,
                    ),
                    _ => (
                        Dxgi::Common::DXGI_FORMAT_R8_UNORM,
                        Dxgi::Common::DXGI_FORMAT_R8G8_UNORM,
                    ),
                };
                let _ = (y_fmt, uv_fmt);

                // (a) D3D11 plane-copy render into the slot's R8/RG8 targets.
                eprintln!(
                    "[opengl] import_frame: pre-lock lock={:?} ctx={:p}",
                    self.lock, self.lock_ctx
                );
                if let (Some(lock), ctx) = (self.lock, self.lock_ctx) {
                    unsafe {
                        lock(ctx);
                    }
                }
                eprintln!("[opengl] import_frame: locked");
                {
                    let ctx = &self.core.context;
                    ctx.ClearState();
                    eprintln!("[opengl] import_frame: ClearState OK");
                    ctx.IASetInputLayout(&self.core.pipeline.layout);
                    ctx.IASetPrimitiveTopology(Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    ctx.VSSetShader(Some(&self.core.pipeline.vs), None);
                    let samplers = [Some(self.core.pipeline.sampler.clone())];
                    let y_srvs = [Some(self.core.srv_y.clone())];
                    let uv_srvs = [Some(self.core.srv_uv.clone())];
                    let y_rtvs = [Some(self.slots[slot_idx].y_rtv.clone())];
                    let uv_rtvs = [Some(self.slots[slot_idx].uv_rtv.clone())];
                    ctx.PSSetSamplers(0, Some(&samplers));
                    ctx.PSSetShaderResources(0, Some(&y_srvs));
                    ctx.OMSetRenderTargets(Some(&y_rtvs), None);
                    let y_viewport = plane_copy_viewport(self.width, self.height);
                    ctx.RSSetViewports(Some(std::slice::from_ref(&y_viewport)));
                    ctx.PSSetShader(Some(&self.core.pipeline.ps_y), None);
                    eprintln!("[opengl] import_frame: pre-Draw-Y");
                    ctx.Draw(3, 0);
                    eprintln!("[opengl] import_frame: Draw-Y OK");
                    ctx.OMSetRenderTargets(Some(&uv_rtvs), None);
                    let uv_viewport = plane_copy_viewport(self.width / 2, self.height / 2);
                    ctx.RSSetViewports(Some(std::slice::from_ref(&uv_viewport)));
                    ctx.PSSetShaderResources(0, Some(&uv_srvs));
                    ctx.PSSetShader(Some(&self.core.pipeline.ps_uv), None);
                    eprintln!("[opengl] import_frame: pre-Draw-UV");
                    ctx.Draw(3, 0);
                    eprintln!("[opengl] import_frame: Draw-UV OK");
                    ctx.Flush();
                    eprintln!("[opengl] import_frame: Flush OK");
                }
                if let (Some(unlock), ctx) = (self.unlock, self.lock_ctx) {
                    unsafe {
                        unlock(ctx);
                    }
                }
                eprintln!("[opengl] import_frame: unlocked");

                // (b) Hand ownership to OpenGL.
                let hal = device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();
                eprintln!("[opengl] import_frame: GL guard acquired");
                let slot = &mut self.slots[slot_idx];
                let names = [slot.y_gl, slot.uv_gl];
                let resources = [slot.y_wgl, slot.uv_wgl];
                eprintln!(
                    "[opengl] import_frame: pre-dx_lock_objects names={:?} resources={:?}",
                    names, resources
                );
                let ok = (self.wgl.dx_lock_objects)(
                    self.dx_device,
                    2,
                    names.as_ptr(),
                    resources.as_ptr(),
                );
                eprintln!("[opengl] import_frame: dx_lock_objects returned {}", ok);
                if ok == 0 {
                    eprintln!("[opengl] wglDXLockObjectsNV failed");
                    return Err(Error::TextureShare);
                }
                slot.state = WglSlotState::GlLockedReady;
                self.current_slot = Some(slot.slot_id);

                Ok(Some(GlInteropTicket {
                    generation: self.generation,
                    slot_id: slot.slot_id,
                }))
            }
        }

        pub(super) fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
            let slot_id = self.current_slot?;
            let slot = self.slots.iter().find(|s| s.slot_id == slot_id)?;
            Some(vec![
                slot.y_view.as_ref()?.clone(),
                slot.uv_view.as_ref()?.clone(),
            ])
        }

        /// Insert ONE GL fence for every submitted ticket's slot, after the
        /// shared `queue.submit`. The fence is ordered after all video draws in
        /// that submission; we flush again because our fence is inserted after
        /// wgpu-hal's own flush.
        pub(super) unsafe fn finish_gl_frames(
            &mut self,
            tickets: &[GlInteropTicket],
        ) -> Result<()> {
            if tickets.is_empty() {
                return Ok(());
            }
            unsafe {
                let hal = self
                    .device
                    .as_hal::<wgpu::hal::gles::Api>()
                    .ok_or(Error::UnsupportedBackend)?;
                let _gl_guard = hal.context().lock();
                let fence = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
                gl::Flush();
                if fence.is_null() {
                    return Err(Error::Unknown);
                }
                for t in tickets {
                    if t.generation != self.generation {
                        continue;
                    }
                    if let Some(slot) = self.slots.iter_mut().find(|s| s.slot_id == t.slot_id) {
                        if slot.state == WglSlotState::GlLockedReady {
                            slot.state = WglSlotState::AwaitingSubmit;
                        }
                        slot.state = WglSlotState::Submitted;
                        slot.fence = Some(fence);
                    }
                }
            }
            Ok(())
        }

        pub(super) unsafe fn cancel_gl_frame(&mut self, ticket: GlInteropTicket) -> Result<()> {
            unsafe {
                if ticket.generation != self.generation {
                    return Ok(());
                }
                if let Some(slot) = self.slots.iter_mut().find(|s| s.slot_id == ticket.slot_id) {
                    if slot.state == WglSlotState::GlLockedReady
                        || slot.state == WglSlotState::AwaitingSubmit
                        || slot.state == WglSlotState::Submitted
                    {
                        if let Some(fence) = slot.fence {
                            gl::DeleteSync(fence);
                            slot.fence = None;
                        }
                        let hal = self
                            .device
                            .as_hal::<wgpu::hal::gles::Api>()
                            .ok_or(Error::UnsupportedBackend)?;
                        let _gl_guard = hal.context().lock();
                        Self::unlock_slot(&self.wgl, self.dx_device, slot);
                    }
                }
            }
            Ok(())
        }

        /// Tear down the ring. GL/WGL resource disposal is linked to wgpu's
        /// RAII `Texture` Drop via each slot's `DropCallback` (set in
        /// `wrap_gl_name`): dropping the slots unregisters the WGL objects and
        /// deletes the GL names automatically. We only release any pending GL
        /// fences and close the DX interop device association here.
        unsafe fn teardown(self) {
            unsafe {
                let hal = match self.device.as_hal::<wgpu::hal::gles::Api>() {
                    Some(h) => h,
                    None => return,
                };
                let _gl_guard = hal.context().lock();
                // Free any still-pending GL fences so they don't leak.
                for slot in self.slots.iter() {
                    if let Some(fence) = slot.fence {
                        gl::DeleteSync(fence);
                    }
                }
                // Dropping the slots fires each wgpu Texture's DropCallback,
                // which unregisters the WGL object and deletes the GL name.
                drop(self.slots);
                (self.wgl.dx_close_device)(self.dx_device);
            }
        }
    }

    /// Create one persistent interop slot: D3D11 R8/RG8 textures, GL names,
    /// WGL registrations, and cached wgpu wrappers (external ownership).
    #[cfg(feature = "experimental-wgl-interop")]
    unsafe fn create_ring_slot(
        wgl: &wgl_nv_dx_interop::WglNvDxInterop,
        dx_device: *mut c_void,
        device: &wgpu::Device,
        d3d11_device: &D3D11::ID3D11Device,
        width: u32,
        height: u32,
        depth: layout::Depth,
        slot_id: u8,
    ) -> Result<GlInteropSlot> {
        unsafe {
            let (y_fmt, uv_fmt) = match depth {
                layout::Depth::D16 => (
                    Dxgi::Common::DXGI_FORMAT_R16_UNORM,
                    Dxgi::Common::DXGI_FORMAT_R16G16_UNORM,
                ),
                _ => (
                    Dxgi::Common::DXGI_FORMAT_R8_UNORM,
                    Dxgi::Common::DXGI_FORMAT_R8G8_UNORM,
                ),
            };

            let shared_desc = D3D11::D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: y_fmt,
                SampleDesc: Dxgi::Common::DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11::D3D11_USAGE_DEFAULT,
                BindFlags: D3D11::D3D11_BIND_RENDER_TARGET.0 as u32
                    | D3D11::D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11::D3D11_RESOURCE_MISC_SHARED.0 as u32,
            };
            let shared_desc_uv = D3D11::D3D11_TEXTURE2D_DESC {
                Width: width / 2,
                Height: height / 2,
                Format: uv_fmt,
                ..shared_desc
            };

            let mut y_d3d11 = None;
            d3d11_device
                .CreateTexture2D(&shared_desc, None, Some(&mut y_d3d11))
                .map_err(|_| Error::TextureShare)?;
            let y_d3d11 = y_d3d11.ok_or(Error::TextureShare)?;
            let mut uv_d3d11 = None;
            d3d11_device
                .CreateTexture2D(&shared_desc_uv, None, Some(&mut uv_d3d11))
                .map_err(|_| Error::TextureShare)?;
            let uv_d3d11 = uv_d3d11.ok_or(Error::TextureShare)?;

            let mut y_rtv = None;
            d3d11_device
                .CreateRenderTargetView(&y_d3d11, None, Some(&mut y_rtv))
                .map_err(|_| Error::TextureShare)?;
            let y_rtv = y_rtv.ok_or(Error::TextureShare)?;
            let mut uv_rtv = None;
            d3d11_device
                .CreateRenderTargetView(&uv_d3d11, None, Some(&mut uv_rtv))
                .map_err(|_| Error::TextureShare)?;
            let uv_rtv = uv_rtv.ok_or(Error::TextureShare)?;

            // Register each D3D11 plane as a GL texture and wrap it as a wgpu
            // texture. `wrap_gl_name` is exception-safe (self-cleans on failure)
            // and its wgpu Texture Drop owns the GL/WGL disposal.
            let (y_wgpu, y_view, y_gl, y_wgl) = wrap_gl_name(
                device,
                y_d3d11.as_raw() as *mut c_void,
                width,
                height,
                wgpu_y_format(depth),
                *wgl,
                dx_device,
            )?;
            let (uv_wgpu, uv_view, uv_gl, uv_wgl) = wrap_gl_name(
                device,
                uv_d3d11.as_raw() as *mut c_void,
                width / 2,
                height / 2,
                wgpu_uv_format(depth),
                *wgl,
                dx_device,
            )?;

            Ok(GlInteropSlot {
                slot_id,
                y_d3d11,
                uv_d3d11,
                y_rtv,
                uv_rtv,
                y_gl,
                uv_gl,
                y_wgl,
                uv_wgl,
                y_wgpu: Some(y_wgpu),
                uv_wgpu: Some(uv_wgpu),
                y_view: Some(y_view),
                uv_view: Some(uv_view),
                state: WglSlotState::Free,
                fence: None,
            })
        }
    }

    fn wgpu_y_format(depth: layout::Depth) -> wgpu::TextureFormat {
        match depth {
            layout::Depth::D16 => wgpu::TextureFormat::R16Unorm,
            _ => wgpu::TextureFormat::R8Unorm,
        }
    }
    fn wgpu_uv_format(depth: layout::Depth) -> wgpu::TextureFormat {
        match depth {
            layout::Depth::D16 => wgpu::TextureFormat::Rg16Unorm,
            _ => wgpu::TextureFormat::Rg8Unorm,
        }
    }

    /// Register a D3D11 plane as a GL texture and wrap it as a wgpu texture +
    /// view. Exception-safe: on any failure it frees the GL name and
    /// unregisters the WGL object itself, so callers never leak. The wgpu
    /// `Texture`'s `Drop` callback owns the *final* unregister + delete, linking
    /// GL/WGL disposal to wgpu's RAII drop (wgpu-hal skips `glDeleteTextures`
    /// for a texture that carries a `DropCallback`).
    ///
    /// Returns the wgpu texture, its view, and the raw GL name + WGL registration
    /// handle — the latter two are kept by the slot for per-frame WGL lock/unlock.
    #[cfg(feature = "experimental-wgl-interop")]
    unsafe fn wrap_gl_name(
        device: &wgpu::Device,
        d3d11_resource: *mut c_void,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        wgl: wgl_nv_dx_interop::WglNvDxInterop,
        dx_device: *mut c_void,
    ) -> Result<(
        wgpu::Texture,
        wgpu::TextureView,
        gl::types::GLuint,
        *mut c_void,
    )> {
        unsafe {
            let mut gl_name = 0u32;
            gl::GenTextures(1, &mut gl_name);
            if gl_name == 0 {
                eprintln!(
                    "[opengl] wrap_gl_name: glGenTextures returned 0 ({}x{})",
                    width, height
                );
                return Err(Error::TextureShare);
            }
            let wgl_reg = (wgl.dx_register_object)(
                dx_device,
                d3d11_resource,
                gl_name,
                GL_TEXTURE_2D,
                WGL_ACCESS_READ_ONLY_NV,
            );
            if wgl_reg.is_null() {
                eprintln!(
                    "[opengl] wrap_gl_name: wglDXRegisterObjectNV FAILED (resource={:p}, gl_name={}, {}x{}, fmt={:?})",
                    d3d11_resource, gl_name, width, height, format
                );
                gl::DeleteTextures(1, &gl_name);
                return Err(Error::TextureShare);
            }

            // Stable NEAREST filtering for the aliased storage.
            gl::BindTexture(GL_TEXTURE_2D, gl_name);
            gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MIN_FILTER, GL_NEAREST as i32);
            gl::TexParameteri(GL_TEXTURE_2D, gl::TEXTURE_MAG_FILTER, GL_NEAREST as i32);

            // wgpu-hal skips glDeleteTextures when a DropCallback is set, firing
            // this instead on Texture drop — unregister + delete the GL name.
            let unregister = wgl.dx_unregister_object;
            // Opaque D3D11/WGL handles captured by the Send+Sync drop callback.
            // Cast through usize so the raw pointers don't break Send+Sync.
            let dx = dx_device as usize;
            let reg = wgl_reg as usize;
            let drop_cb: Option<wgpu::hal::DropCallback> = Some(Box::new(move || unsafe {
                unregister(dx as *mut c_void, reg as *mut c_void);
                gl::DeleteTextures(1, &gl_name);
            }));

            let hal = device
                .as_hal::<wgpu::hal::gles::Api>()
                .ok_or(Error::UnsupportedBackend)?;
            // `texture_from_raw` returns the hal texture directly (no Result) and
            // stores `drop_cb`; wgpu-hal fires it on Texture drop instead of
            // calling `glDeleteTextures` on the WGL-registered name.
            let hal_tex = hal.texture_from_raw(
                std::num::NonZeroU32::new(gl_name).ok_or(Error::Unknown)?,
                &wgpu::hal::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_DST,
                    memory_flags: wgpu::hal::MemoryFlags::empty(),
                    view_formats: vec![],
                },
                drop_cb,
            );
            let tex = device.create_texture_from_hal::<wgpu::hal::gles::Api>(
                hal_tex,
                &wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
            );
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            Ok((tex, view, gl_name, wgl_reg))
        }
    }

    /// Log the D3D11 adapter LUID and the active GL_RENDERER (diagnostic only).
    #[cfg(feature = "experimental-wgl-interop")]
    #[allow(unused_variables)]
    unsafe fn log_diagnostic_adapter_info(d3d11_device: &D3D11::ID3D11Device) {
        unsafe {
            if let Ok(dxgi) = d3d11_device.cast::<Dxgi::IDXGIDevice>() {
                if let Ok(adapter) = dxgi.GetAdapter() {
                    if let Ok(desc) = adapter.GetDesc() {
                        eprintln!(
                            "[opengl] WGL interop D3D11 adapter: {} (LUID: hi={} lo={})",
                            String::from_utf16_lossy(&desc.Description),
                            desc.AdapterLuid.HighPart,
                            desc.AdapterLuid.LowPart,
                        );
                    }
                }
            }
            let renderer = gl::GetString(gl::RENDERER);
            if !renderer.is_null() {
                eprintln!(
                    "[opengl] WGL interop GL_RENDERER: {}",
                    std::ffi::CStr::from_ptr(renderer as *const i8).to_string_lossy()
                );
            }
            // Log which WGL_NV_DX_interop2 entry points the driver exposes.
            wgl_nv_dx_interop::log_extension_presence();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn plane_copy_viewports_match_nv12_plane_extents() {
            let y = plane_copy_viewport(3840, 2160);
            assert_eq!((y.Width, y.Height), (3840.0, 2160.0));
            assert_eq!((y.MinDepth, y.MaxDepth), (0.0, 1.0));

            let uv = plane_copy_viewport(1920, 1080);
            assert_eq!((uv.Width, uv.Height), (1920.0, 1080.0));
            assert_eq!((uv.MinDepth, uv.MaxDepth), (0.0, 1.0));
        }

        #[test]
        fn plane_copy_shaders_keep_luma_and_interleaved_chroma_separate() {
            assert!(PLANE_COPY_PS_Y.contains(".r"));
            assert!(PLANE_COPY_PS_UV.contains("float2 c"));
            assert!(PLANE_COPY_PS_UV.contains("return float4(c"));
        }
    }
}

// ===========================================================================
// Public adapters
// ===========================================================================

/// Linux VA-API → GL zero-copy adapter (`DirectPlaneImport`).
#[cfg(target_os = "linux")]
pub struct OpenGlLinuxFrameAdapter {
    imported: Option<linux::VaapiEglImport>,
    #[allow(dead_code)]
    path: OpenGlInteropPath,
}

#[cfg(target_os = "linux")]
impl FrameAdapterBuilder for OpenGlLinuxFrameAdapter {
    unsafe fn new(_decoder: NonNull<ff::AVCodecContext>) -> Result<Self> {
        Ok(OpenGlLinuxFrameAdapter {
            imported: None,
            path: OpenGlInteropPath::DirectPlaneImport,
        })
    }

    fn supports_format(format: ff::AVPixelFormat) -> bool {
        format == ff::AVPixelFormat::AV_PIX_FMT_VAAPI
    }
}

#[cfg(target_os = "linux")]
impl FrameAdapter for OpenGlLinuxFrameAdapter {
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        _instance: &wgpu::Instance,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<Option<GlInteropTicket>> {
        unsafe {
            let frame_ref = frame.as_ref();
            if frame_ref.format != ff::AVPixelFormat::AV_PIX_FMT_VAAPI as i32 {
                return Err(Error::UnsupportedPixelFormat);
            }

            let imported = if let Some(imported) = self.imported.as_mut() {
                imported
            } else {
                self.imported
                    .insert(linux::VaapiEglImport::new(device, pipeline_cache, frame)?)
            };

            imported.attach(device)?;
            Ok(None)
        }
    }

    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.imported.as_ref().map(|i| &i.frame().bg0)
    }

    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.imported.as_ref().map(|i| i.frame().identity)
    }

    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.imported.as_ref().map(|i| {
            let f = i.frame();
            vec![
                f.y_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
                f.uv_texture
                    .create_view(&wgpu::TextureViewDescriptor::default()),
            ]
        })
    }

    fn name(&self) -> &'static str {
        "VA-API GL zero-copy (EGL DMA-BUF)"
    }
}

/// Windows D3D11VA → GL adapter (`GpuPlaneCopyThenImport`).
#[cfg(target_os = "windows")]
pub struct OpenGlWindowsFrameAdapter {
    imported: Option<win::D3D11GlImport>,
    d3d11_device: std::mem::ManuallyDrop<windows::Win32::Graphics::Direct3D11::ID3D11Device>,
    lock: Option<unsafe extern "C" fn(*mut c_void)>,
    unlock: Option<unsafe extern "C" fn(*mut c_void)>,
    lock_ctx: *mut c_void,
    #[allow(dead_code)]
    path: OpenGlInteropPath,
}

#[cfg(target_os = "windows")]
impl FrameAdapterBuilder for OpenGlWindowsFrameAdapter {
    unsafe fn new(decoder: NonNull<ff::AVCodecContext>) -> Result<Self> {
        unsafe {
            let hwctx = (*decoder.as_ptr()).hw_device_ctx;
            let device_ctx = (hwctx.as_ref().unwrap().data as *mut ff::AVHWDeviceContext)
                .as_ref()
                .unwrap();
            let d3d11_ctx = (device_ctx.hwctx as *mut super::d3d11va::AVD3D11VADeviceContext)
                .as_ref()
                .unwrap();
            let d3d11_device: std::mem::ManuallyDrop<
                windows::Win32::Graphics::Direct3D11::ID3D11Device,
            > = std::mem::ManuallyDrop::new(std::mem::transmute((*d3d11_ctx).device));

            // FFmpeg serializes D3D11 immediate-context access via these
            // callbacks. They are valid for the lifetime of the decoder.
            let lock = (*d3d11_ctx).lock;
            let unlock = (*d3d11_ctx).unlock;
            let lock_ctx = (*d3d11_ctx).lock_ctx;

            Ok(OpenGlWindowsFrameAdapter {
                imported: None,
                d3d11_device,
                lock: Some(lock),
                unlock: Some(unlock),
                lock_ctx,
                path: OpenGlInteropPath::GpuPlaneCopyThenImport,
            })
        }
    }

    fn supports_format(format: ff::AVPixelFormat) -> bool {
        format == ff::AVPixelFormat::AV_PIX_FMT_D3D11
    }
}

#[cfg(target_os = "windows")]
impl FrameAdapter for OpenGlWindowsFrameAdapter {
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        _instance: &wgpu::Instance,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<Option<GlInteropTicket>> {
        unsafe {
            let frame_ref = frame.as_ref();
            if frame_ref.format != ff::AVPixelFormat::AV_PIX_FMT_D3D11 as i32 {
                return Err(Error::UnsupportedPixelFormat);
            }
            if frame_ref.data[0].is_null() {
                return Err(Error::InvalidFrame);
            }

            let decoder_texture: windows::Win32::Graphics::Direct3D11::ID3D11Texture2D =
                std::mem::transmute(frame_ref.data[0]);
            let array_slice = frame_ref.data[1] as u32;

            let mut desc = windows::Win32::Graphics::Direct3D11::D3D11_TEXTURE2D_DESC::default();
            decoder_texture.GetDesc(&mut desc);

            if desc.Format != windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_NV12 {
                eprintln!(
                    "[opengl] native WGL interop supports NV12 only, got {:?}",
                    desc.Format
                );
                return Err(Error::UnsupportedPixelFormat);
            }
            let depth = layout::Depth::D8;

            let imported = if let Some(imported) = self.imported.as_mut() {
                imported
            } else {
                self.imported.insert(win::D3D11GlImport::new(
                    device,
                    &decoder_texture,
                    array_slice,
                    desc.Width,
                    desc.Height,
                    depth,
                    frame_ref.colorspace.into(),
                    &self.d3d11_device,
                    self.lock,
                    self.unlock,
                    self.lock_ctx,
                )?)
            };

            imported.import_frame(device, pipeline_cache, frame)
        }
    }

    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.imported.as_ref().and_then(|i| i.bind_group())
    }

    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.imported.as_ref().and_then(|i| i.layout_identity())
    }

    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.imported.as_ref().and_then(|i| i.plane_views())
    }

    fn finish_gl_frames(&mut self, tickets: &[GlInteropTicket]) -> Result<()> {
        if let Some(imported) = self.imported.as_mut() {
            imported.finish_gl_frames(tickets)?;
        }
        Ok(())
    }

    fn cancel_gl_frame(&mut self, ticket: GlInteropTicket) -> Result<()> {
        if let Some(imported) = self.imported.as_mut() {
            imported.cancel_gl_frame(ticket)?;
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        "D3D11VA GL (WGL_NV_DX_interop2 ring / GL_EXT_memory_object fallback)"
    }
}
