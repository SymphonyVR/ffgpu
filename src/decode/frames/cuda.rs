//! CUDA NVDEC frame adapter for Linux and Windows.
//!
//! Two-tier architecture:
//! - Tier 1: True Zero-Copy (VRAM) via Vulkan external memory (`cuImportExternalMemory`).
//!   Decoded frames stay strictly in GPU memory with zero host/PCIe round-trip.
//! - Tier 2: Pinned Host DMA fallback (`cuMemAllocHost` + `cuMemcpy2DAsync`).
//!   Used when external memory import is unsupported by the platform (e.g. WSL2).
//!   GPU DMA engine streams directly into page-locked host memory with zero CPU pixel processing.

#[cfg(any(target_os = "linux", target_os = "windows"))]
use crate::{
    Error, Result,
    context::{layout, pipeline_cache::PipelineCache},
    cuda_interop::{
        CUDA_MEMCPY2D, CUDA_SUCCESS, CUarray, CUexternalMemory, CUmipmappedArray,
        CUmemorytype, CUstream, CudaDriver, get_cuda_driver,
    },
    decode::frames::{FrameAdapter, FrameAdapterBuilder, GlInteropTicket},
};
#[cfg(target_os = "linux")]
use crate::cuda_interop::{
    CUDA_ARRAY3D_DESCRIPTOR, CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC, CUarray_format,
    CUexternalMemoryHandle, CUexternalMemoryHandleType,
};
#[cfg(any(target_os = "linux", target_os = "windows"))]
use ffmpeg_next::sys as ff;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::ffi::c_void;
#[cfg(any(target_os = "linux", target_os = "windows"))]
use std::ptr::NonNull;

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct ExportedDirectPlane {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
    ext_mem: CUexternalMemory,
    mipmapped_array: CUmipmappedArray,
    cu_array: CUarray,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct DirectPlanes {
    plane_y: ExportedDirectPlane,
    plane_uv: ExportedDirectPlane,
    bg0: wgpu::BindGroup,
    layout: layout::FrameDescriptor<()>,
    driver: &'static CudaDriver,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct HostDmaPlanes {
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

#[cfg(any(target_os = "linux", target_os = "windows"))]
enum CudaPlanes {
    Direct(DirectPlanes),
    HostDma(HostDmaPlanes),
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Send for CudaPlanes {}
#[cfg(any(target_os = "linux", target_os = "windows"))]
unsafe impl Sync for CudaPlanes {}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl Drop for CudaPlanes {
    fn drop(&mut self) {
        match self {
            CudaPlanes::Direct(d) => unsafe {
                if !d.plane_y.mipmapped_array.is_null() {
                    (d.driver.cu_mipmapped_array_destroy)(d.plane_y.mipmapped_array);
                }
                if !d.plane_y.ext_mem.is_null() {
                    (d.driver.cu_destroy_external_memory)(d.plane_y.ext_mem);
                }
                if !d.plane_uv.mipmapped_array.is_null() {
                    (d.driver.cu_mipmapped_array_destroy)(d.plane_uv.mipmapped_array);
                }
                if !d.plane_uv.ext_mem.is_null() {
                    (d.driver.cu_destroy_external_memory)(d.plane_uv.ext_mem);
                }
            },
            CudaPlanes::HostDma(h) => unsafe {
                if !h.host_buf_y.is_null() {
                    (h.driver.cu_mem_free_host)(h.host_buf_y);
                }
                if !h.host_buf_uv.is_null() {
                    (h.driver.cu_mem_free_host)(h.host_buf_uv);
                }
            },
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn create_exported_direct_plane(
    device: &wgpu::Device,
    adapter: &wgpu::Adapter,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
    driver: &CudaDriver,
) -> Result<ExportedDirectPlane> {
    unsafe {
        use ash::vk;

    let hal_device = device
        .as_hal::<wgpu::hal::vulkan::Api>()
        .ok_or(Error::UnsupportedBackend)?;
    let hal_adapter = adapter
        .as_hal::<wgpu::hal::vulkan::Api>()
        .ok_or(Error::UnsupportedBackend)?;

    let raw_device = hal_device.raw_device();
    let raw_instance = hal_device.shared_instance().raw_instance();
    let physical_device = hal_adapter.raw_physical_device();

    let vk_format = match format {
        wgpu::TextureFormat::R8Unorm => vk::Format::R8_UNORM,
        wgpu::TextureFormat::Rg8Unorm => vk::Format::R8G8_UNORM,
        wgpu::TextureFormat::R16Unorm => vk::Format::R16_UNORM,
        wgpu::TextureFormat::Rg16Unorm => vk::Format::R16G16_UNORM,
        _ => return Err(Error::UnsupportedPixelFormat),
    };

    let mut external_image_info = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk_format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external_image_info);

    let vk_image = raw_device
        .create_image(&image_info, None)
        .map_err(|_| Error::UnsupportedBackend)?;

    let mem_reqs = raw_device.get_image_memory_requirements(vk_image);
    let mem_props = raw_instance.get_physical_device_memory_properties(physical_device);

    let mem_type_index = mem_props
        .memory_types_as_slice()
        .iter()
        .enumerate()
        .find(|(i, mt)| {
            (mem_reqs.memory_type_bits & (1 << i)) != 0
                && mt.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .map(|(i, _)| i as u32)
        .ok_or_else(|| {
            raw_device.destroy_image(vk_image, None);
            Error::UnsupportedBackend
        })?;

    let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfoKHR::default()
        .image(vk_image);

    let mut export_info = vk::ExportMemoryAllocateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index)
        .push_next(&mut export_info)
        .push_next(&mut dedicated_alloc_info);

    let device_memory = match raw_device.allocate_memory(&alloc_info, None) {
        Ok(m) => m,
        Err(_) => {
            raw_device.destroy_image(vk_image, None);
            return Err(Error::UnsupportedBackend);
        }
    };

    if let Err(_) = raw_device.bind_image_memory(vk_image, device_memory, 0) {
        raw_device.free_memory(device_memory, None);
        raw_device.destroy_image(vk_image, None);
        return Err(Error::UnsupportedBackend);
    }

    let ext_fd_loader = ash::khr::external_memory_fd::Device::new(raw_instance, raw_device);
    let get_fd_info = vk::MemoryGetFdInfoKHR::default()
        .memory(device_memory)
        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);

    let fd = match ext_fd_loader.get_memory_fd(&get_fd_info) {
        Ok(fd) => fd,
        Err(_) => {
            raw_device.free_memory(device_memory, None);
            raw_device.destroy_image(vk_image, None);
            return Err(Error::UnsupportedBackend);
        }
    };

    let handle_desc = CUDA_EXTERNAL_MEMORY_HANDLE_DESC {
        handle_type: CUexternalMemoryHandleType::OpaqueFd,
        handle: CUexternalMemoryHandle { fd },
        size: mem_reqs.size,
        flags: 0,
        reserved: [0; 16],
    };

    let _ = (driver.cu_ctx_set_current)(driver.primary_ctx);
    let mut ext_mem: CUexternalMemory = std::ptr::null_mut();
    let cu_res = (driver.cu_import_external_memory)(&mut ext_mem, &handle_desc);
    libc::close(fd);

    if cu_res != CUDA_SUCCESS || ext_mem.is_null() {
        raw_device.free_memory(device_memory, None);
        raw_device.destroy_image(vk_image, None);
        return Err(Error::HardwareContext);
    }

    let (cuda_fmt, channels) = match format {
        wgpu::TextureFormat::R8Unorm => (CUarray_format::UnsignedInt8, 1),
        wgpu::TextureFormat::Rg8Unorm => (CUarray_format::UnsignedInt8, 2),
        wgpu::TextureFormat::R16Unorm => (CUarray_format::UnsignedInt16, 1),
        wgpu::TextureFormat::Rg16Unorm => (CUarray_format::UnsignedInt16, 2),
        _ => return Err(Error::UnsupportedPixelFormat),
    };

    let array_desc = CUDA_ARRAY3D_DESCRIPTOR {
        width: width as usize,
        height: height as usize,
        depth: 0,
        format: cuda_fmt,
        num_channels: channels,
        flags: 0,
    };

    let mip_desc = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC {
        offset: 0,
        array_desc,
        num_levels: 1,
        reserved: [0; 16],
    };

    let mut mipmapped_array: CUmipmappedArray = std::ptr::null_mut();
    let cu_res = (driver.cu_external_memory_get_mapped_mipmapped_array)(
        &mut mipmapped_array,
        ext_mem,
        &mip_desc,
    );

    if cu_res != CUDA_SUCCESS || mipmapped_array.is_null() {
        (driver.cu_destroy_external_memory)(ext_mem);
        raw_device.free_memory(device_memory, None);
        raw_device.destroy_image(vk_image, None);
        return Err(Error::HardwareContext);
    }

    let mut cu_array: CUarray = std::ptr::null_mut();
    let cu_res = (driver.cu_mipmapped_array_get_level)(&mut cu_array, mipmapped_array, 0);

    if cu_res != CUDA_SUCCESS || cu_array.is_null() {
        (driver.cu_mipmapped_array_destroy)(mipmapped_array);
        (driver.cu_destroy_external_memory)(ext_mem);
        raw_device.free_memory(device_memory, None);
        raw_device.destroy_image(vk_image, None);
        return Err(Error::HardwareContext);
    }

    let hal_texture = hal_device.texture_from_raw(
        vk_image,
        &wgpu::hal::TextureDescriptor {
            label: Some(label),
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
        None,
        wgpu::hal::vulkan::TextureMemory::Dedicated(device_memory),
    );

    let texture = device.create_texture_from_hal::<wgpu::hal::vulkan::Api>(
        hal_texture,
        &wgpu::TextureDescriptor {
            label: Some(label),
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

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        Ok(ExportedDirectPlane {
            _texture: texture,
            view,
            ext_mem,
            mipmapped_array,
            cu_array,
        })
    }
}

#[cfg(not(target_os = "linux"))]
unsafe fn create_exported_direct_plane(
    _device: &wgpu::Device,
    _adapter: &wgpu::Adapter,
    _width: u32,
    _height: u32,
    _format: wgpu::TextureFormat,
    _label: &str,
    _driver: &CudaDriver,
) -> Result<ExportedDirectPlane> {
    Err(Error::UnsupportedBackend)
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl CudaPlanes {
    fn new(
        device: &wgpu::Device,
        adapter: &wgpu::Adapter,
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

        // 1. Try Tier 1: True Zero-Copy (VRAM)
        let direct_res: Result<DirectPlanes> = (|| unsafe {
            let plane_y = create_exported_direct_plane(
                device,
                adapter,
                width,
                height,
                fmt_y,
                "CUDA Zero-Copy Plane Y",
                driver,
            )?;
            let plane_uv = create_exported_direct_plane(
                device,
                adapter,
                width / 2,
                height / 2,
                fmt_uv,
                "CUDA Zero-Copy Plane UV",
                driver,
            )?;
            let plane_views = layout::PlaneLayout::PackedYUV420([
                plane_y.view.clone(),
                plane_uv.view.clone(),
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
            Ok(DirectPlanes {
                plane_y,
                plane_uv,
                bg0,
                layout,
                driver,
            })
        })();

        if let Ok(direct_planes) = direct_res {
            eprintln!("[CUDA-NVDEC] Active mode: Tier 1 True Zero-Copy (GPU VRAM)");
            return Ok(CudaPlanes::Direct(direct_planes));
        }

        // 2. Fall back to Tier 2: Pinned Host DMA
        eprintln!("[CUDA-NVDEC] Tier 1 external memory unavailable; falling back to Tier 2 Pinned Host DMA");

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

        Ok(CudaPlanes::HostDma(HostDmaPlanes {
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
        }))
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub struct CudaZeroCopyFrameAdapter {
    _decoder: NonNull<ff::AVCodecContext>,
    driver: &'static CudaDriver,
    stream: CUstream,
    planes: Option<CudaPlanes>,
    width: u32,
    height: u32,
    is_p010: bool,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
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

#[cfg(any(target_os = "linux", target_os = "windows"))]
impl FrameAdapter for CudaZeroCopyFrameAdapter {
    unsafe fn import_frame(
        &mut self,
        frame: NonNull<ff::AVFrame>,
        _instance: &wgpu::Instance,
        adapter: &wgpu::Adapter,
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
            // Drop old planes if any
            self.planes = None;

            let planes = CudaPlanes::new(
                device,
                adapter,
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

        match planes {
            CudaPlanes::Direct(d) => {
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
                    dst_memory_type: CUmemorytype::Array,
                    dst_host: std::ptr::null_mut(),
                    dst_device: 0,
                    dst_array: d.plane_y.cu_array,
                    dst_pitch: 0,

                    width_in_bytes: (width as usize) * bytes_per_elem,
                    height: height as usize,
                };
                let res = unsafe { (self.driver.cu_memcpy_2d_async)(&copy_y, self.stream) };
                if res != CUDA_SUCCESS {
                    eprintln!("[CUDA-NVDEC] cuMemcpy2DAsync direct Y failed: {:?}", res);
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
                    dst_memory_type: CUmemorytype::Array,
                    dst_host: std::ptr::null_mut(),
                    dst_device: 0,
                    dst_array: d.plane_uv.cu_array,
                    dst_pitch: 0,

                    width_in_bytes: (width as usize) * bytes_per_elem,
                    height: (height / 2) as usize,
                };
                let res = unsafe { (self.driver.cu_memcpy_2d_async)(&copy_uv, self.stream) };
                if res != CUDA_SUCCESS {
                    eprintln!("[CUDA-NVDEC] cuMemcpy2DAsync direct UV failed: {:?}", res);
                    return Err(Error::HardwareContext);
                }

                let res = unsafe { (self.driver.cu_stream_synchronize)(self.stream) };
                if res != CUDA_SUCCESS {
                    eprintln!("[CUDA-NVDEC] cuStreamSynchronize direct failed: {:?}", res);
                    return Err(Error::HardwareContext);
                }
            }
            CudaPlanes::HostDma(h) => {
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
                    dst_host: h.host_buf_y,
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
                    dst_host: h.host_buf_uv,
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
                        texture: &h.texture_y,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    unsafe {
                        std::slice::from_raw_parts(
                            h.host_buf_y as *const u8,
                            h.host_buf_y_size,
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
                        texture: &h.texture_uv,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    unsafe {
                        std::slice::from_raw_parts(
                            h.host_buf_uv as *const u8,
                            h.host_buf_uv_size,
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
            }
        }

        Ok(None)
    }

    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.planes.as_ref().map(|p| match p {
            CudaPlanes::Direct(d) => d.layout,
            CudaPlanes::HostDma(h) => h.layout,
        })
    }

    fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.planes.as_ref().map(|p| match p {
            CudaPlanes::Direct(d) => &d.bg0,
            CudaPlanes::HostDma(h) => &h.bg0,
        })
    }

    fn name(&self) -> &'static str {
        match &self.planes {
            Some(CudaPlanes::Direct(_)) => "CUDA NVDEC (Zero-Copy VRAM)",
            Some(CudaPlanes::HostDma(_)) => "CUDA NVDEC (Pinned Host DMA)",
            None => "CUDA NVDEC",
        }
    }

    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.planes.as_ref().map(|p| match p {
            CudaPlanes::Direct(d) => vec![d.plane_y.view.clone(), d.plane_uv.view.clone()],
            CudaPlanes::HostDma(h) => vec![h.view_y.clone(), h.view_uv.clone()],
        })
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
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
