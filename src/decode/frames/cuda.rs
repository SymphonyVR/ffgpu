//! CUDA NVDEC frame adapter for Linux / WSL2.
//!
//! Imports NVIDIA hardware-decoded frames (AV_PIX_FMT_CUDA) using pinned host DMA
//! (cuMemAllocHost / cuMemcpy2DAsync) directly into wgpu textures for hardware YUV->RGB sampling.

#[cfg(target_os = "linux")]
use crate::{
    Error, Result,
    context::{layout, pipeline_cache::PipelineCache},
    cuda_interop::{
        CUDA_MEMCPY2D, CUDA_SUCCESS, CUmemorytype, CUstream, CudaDriver, get_cuda_driver,
    },
    decode::frames::{FrameAdapter, FrameAdapterBuilder, GlInteropTicket},
};
#[cfg(target_os = "linux")]
use ffmpeg_next::sys as ff;
#[cfg(target_os = "linux")]
use std::ffi::c_void;
#[cfg(target_os = "linux")]
use std::ptr::NonNull;

#[cfg(target_os = "linux")]
struct CudaPlanes {
    texture_y: wgpu::Texture,
    view_y: wgpu::TextureView,
    texture_uv: wgpu::Texture,
    view_uv: wgpu::TextureView,
    host_buf_y: *mut c_void,
    host_buf_y_size: usize,
    host_buf_uv: *mut c_void,
    host_buf_uv_size: usize,
    bg0: wgpu::BindGroup,
    layout: layout::FrameDescriptor<()>,
    driver: &'static CudaDriver,
}

#[cfg(target_os = "linux")]
unsafe impl Send for CudaPlanes {}
#[cfg(target_os = "linux")]
unsafe impl Sync for CudaPlanes {}

#[cfg(target_os = "linux")]
impl Drop for CudaPlanes {
    fn drop(&mut self) {
        unsafe {
            if !self.host_buf_y.is_null() {
                (self.driver.cu_mem_free_host)(self.host_buf_y);
            }
            if !self.host_buf_uv.is_null() {
                (self.driver.cu_mem_free_host)(self.host_buf_uv);
            }
        }
    }
}

#[cfg(target_os = "linux")]
impl CudaPlanes {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        is_p010: bool,
        colorspace: ff::AVColorSpace,
        driver: &'static CudaDriver,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<Self> {
        let (fmt_y, fmt_uv, depth) = if is_p010 {
            (
                wgpu::TextureFormat::R16Unorm,
                wgpu::TextureFormat::Rg16Unorm,
                layout::Depth::D16,
            )
        } else {
            (
                wgpu::TextureFormat::R8Unorm,
                wgpu::TextureFormat::Rg8Unorm,
                layout::Depth::D8,
            )
        };

        let texture_y = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CUDA NVDEC Plane Y"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt_y,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view_y = texture_y.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_uv = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CUDA NVDEC Plane UV"),
            size: wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt_uv,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view_uv = texture_uv.create_view(&wgpu::TextureViewDescriptor::default());

        let bytes_per_elem = if is_p010 { 2 } else { 1 };
        let host_buf_y_size = (width as usize) * (height as usize) * bytes_per_elem;
        let host_buf_uv_size = (width as usize) * ((height / 2) as usize) * bytes_per_elem;

        let mut host_buf_y: *mut c_void = std::ptr::null_mut();
        let mut host_buf_uv: *mut c_void = std::ptr::null_mut();

        let _ = unsafe { (driver.cu_ctx_set_current)(driver.primary_ctx) };

        let res = unsafe { (driver.cu_mem_alloc_host)(&mut host_buf_y, host_buf_y_size) };
        if res != CUDA_SUCCESS || host_buf_y.is_null() {
            eprintln!("[CUDA-NVDEC] cuMemAllocHost for Y failed: {:?}", res);
            return Err(Error::HardwareContext);
        }

        let res = unsafe { (driver.cu_mem_alloc_host)(&mut host_buf_uv, host_buf_uv_size) };
        if res != CUDA_SUCCESS || host_buf_uv.is_null() {
            eprintln!("[CUDA-NVDEC] cuMemAllocHost for UV failed: {:?}", res);
            unsafe {
                (driver.cu_mem_free_host)(host_buf_y);
            }
            return Err(Error::HardwareContext);
        }

        let plane_views = layout::PlaneLayout::PackedYUV420([
            view_y.clone(),
            view_uv.clone(),
        ]);

        let bg0 = pipeline_cache.bind_frame_textures(
            &layout::FrameDescriptor {
                planes: plane_views,
                depth,
            },
            colorspace.into(),
        );

        let layout = layout::FrameDescriptor {
            planes: layout::PlaneLayout::PackedYUV420([(), ()]),
            depth,
        };

        Ok(Self {
            texture_y,
            view_y,
            texture_uv,
            view_uv,
            host_buf_y,
            host_buf_y_size,
            host_buf_uv,
            host_buf_uv_size,
            bg0,
            layout,
            driver,
        })
    }
}

#[cfg(target_os = "linux")]
pub struct CudaZeroCopyFrameAdapter {
    _decoder: NonNull<ff::AVCodecContext>,
    driver: &'static CudaDriver,
    stream: CUstream,
    planes: Option<CudaPlanes>,
    width: u32,
    height: u32,
    is_p010: bool,
}

#[cfg(target_os = "linux")]
impl FrameAdapterBuilder for CudaZeroCopyFrameAdapter {
    unsafe fn new(decoder: NonNull<ff::AVCodecContext>) -> Result<Self> {
        let driver = get_cuda_driver().ok_or(Error::UnsupportedBackend)?;

        let _ = unsafe { (driver.cu_ctx_set_current)(driver.primary_ctx) };
        let mut stream: CUstream = std::ptr::null_mut();
        let res = unsafe { (driver.cu_stream_create)(&mut stream, 0) };
        if res != CUDA_SUCCESS || stream.is_null() {
            eprintln!("[CUDA-NVDEC] cuStreamCreate failed: {:?}", res);
            return Err(Error::HardwareContext);
        }

        let width = unsafe { (*decoder.as_ptr()).width as u32 };
        let height = unsafe { (*decoder.as_ptr()).height as u32 };

        Ok(Self {
            _decoder: decoder,
            driver,
            stream,
            planes: None,
            width,
            height,
            is_p010: false,
        })
    }

    fn supports_format(format: ff::AVPixelFormat) -> bool {
        format == ff::AVPixelFormat::AV_PIX_FMT_CUDA
    }
}

#[cfg(target_os = "linux")]
impl FrameAdapter for CudaZeroCopyFrameAdapter {
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        _instance: &wgpu::Instance,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _encoder: &mut wgpu::CommandEncoder,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<Option<GlInteropTicket>> {
        let frame_ref = unsafe { frame.as_ref() };

        let is_p010 = if !frame_ref.hw_frames_ctx.is_null() {
            unsafe {
                let hwctx = (*frame_ref.hw_frames_ctx).data as *mut ff::AVHWFramesContext;
                (*hwctx).sw_format == ff::AVPixelFormat::AV_PIX_FMT_P010LE
            }
        } else {
            false
        };

        let width = frame_ref.width as u32;
        let height = frame_ref.height as u32;

        if self.planes.is_none()
            || self.width != width
            || self.height != height
            || self.is_p010 != is_p010
        {
            // Drop old planes if any (frees host buffers)
            self.planes = None;

            let planes = CudaPlanes::new(
                device,
                width,
                height,
                is_p010,
                frame_ref.colorspace,
                self.driver,
                pipeline_cache,
            )?;

            self.planes = Some(planes);
            self.width = width;
            self.height = height;
            self.is_p010 = is_p010;
        }

        let planes = self.planes.as_ref().unwrap();
        let bytes_per_elem = if is_p010 { 2 } else { 1 };

        let _ = unsafe { (self.driver.cu_ctx_set_current)(self.driver.primary_ctx) };

        let copy_y = CUDA_MEMCPY2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: CUmemorytype::Device,
            src_host: std::ptr::null(),
            src_device: frame_ref.data[0] as u64,
            src_array: std::ptr::null_mut(),
            src_pitch: frame_ref.linesize[0] as usize,

            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: CUmemorytype::Host,
            dst_host: planes.host_buf_y,
            dst_device: 0,
            dst_array: std::ptr::null_mut(),
            dst_pitch: (width as usize) * bytes_per_elem,

            width_in_bytes: (width as usize) * bytes_per_elem,
            height: height as usize,
        };
        let res = unsafe { (self.driver.cu_memcpy_2d_async)(&copy_y, self.stream) };
        if res != CUDA_SUCCESS {
            eprintln!("[CUDA-NVDEC] cuMemcpy2DAsync for Y plane failed: {:?}", res);
            return Err(Error::HardwareContext);
        }

        let copy_uv = CUDA_MEMCPY2D {
            src_x_in_bytes: 0,
            src_y: 0,
            src_memory_type: CUmemorytype::Device,
            src_host: std::ptr::null(),
            src_device: frame_ref.data[1] as u64,
            src_array: std::ptr::null_mut(),
            src_pitch: frame_ref.linesize[1] as usize,

            dst_x_in_bytes: 0,
            dst_y: 0,
            dst_memory_type: CUmemorytype::Host,
            dst_host: planes.host_buf_uv,
            dst_device: 0,
            dst_array: std::ptr::null_mut(),
            dst_pitch: (width as usize) * bytes_per_elem,

            width_in_bytes: (width as usize) * bytes_per_elem,
            height: (height / 2) as usize,
        };
        let res = unsafe { (self.driver.cu_memcpy_2d_async)(&copy_uv, self.stream) };
        if res != CUDA_SUCCESS {
            eprintln!("[CUDA-NVDEC] cuMemcpy2DAsync for UV plane failed: {:?}", res);
            return Err(Error::HardwareContext);
        }

        let res = unsafe { (self.driver.cu_stream_synchronize)(self.stream) };
        if res != CUDA_SUCCESS {
            eprintln!("[CUDA-NVDEC] cuStreamSynchronize failed: {:?}", res);
            return Err(Error::HardwareContext);
        }

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &planes.texture_y,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            unsafe {
                std::slice::from_raw_parts(
                    planes.host_buf_y as *const u8,
                    planes.host_buf_y_size,
                )
            },
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((width * bytes_per_elem as u32) as _),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &planes.texture_uv,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            unsafe {
                std::slice::from_raw_parts(
                    planes.host_buf_uv as *const u8,
                    planes.host_buf_uv_size,
                )
            },
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some((width * bytes_per_elem as u32) as _),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: width / 2,
                height: height / 2,
                depth_or_array_layers: 1,
            },
        );

        Ok(None)
    }

    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.planes.as_ref().map(|p| p.layout)
    }

    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.planes.as_ref().map(|p| &p.bg0)
    }

    fn name(&self) -> &'static str {
        "CUDA NVDEC"
    }

    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.planes.as_ref().map(|p| {
            vec![
                p.view_y.clone(),
                p.view_uv.clone(),
            ]
        })
    }
}

#[cfg(target_os = "linux")]
impl Drop for CudaZeroCopyFrameAdapter {
    fn drop(&mut self) {
        self.planes = None;
        if !self.stream.is_null() {
            unsafe {
                (self.driver.cu_stream_destroy)(self.stream);
            }
        }
    }
}
