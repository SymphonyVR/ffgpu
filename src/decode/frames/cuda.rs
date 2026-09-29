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
struct DirectPlane {
    plane: ExportedDirectPlane,
    width_in_bytes: usize,
    height: usize,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct DirectPlanes {
    planes: Vec<DirectPlane>,
    bg0: wgpu::BindGroup,
    layout: layout::FrameDescriptor<()>,
    driver: &'static CudaDriver,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct HostDmaPlane {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    host_buf: *mut c_void,
    host_buf_size: usize,
    width_in_bytes: usize,
    height: usize,
    tex_width: u32,
    tex_height: u32,
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
struct HostDmaPlanes {
    planes: Vec<HostDmaPlane>,
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
                for p in &d.planes {
                    if !p.plane.mipmapped_array.is_null() {
                        (d.driver.cu_mipmapped_array_destroy)(p.plane.mipmapped_array);
                    }
                    if !p.plane.ext_mem.is_null() {
                        (d.driver.cu_destroy_external_memory)(p.plane.ext_mem);
                    }
                }
            },
            CudaPlanes::HostDma(h) => unsafe {
                for p in &h.planes {
                    if !p.host_buf.is_null() {
                        (h.driver.cu_mem_free_host)(p.host_buf);
                    }
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
        sw_format: ff::AVPixelFormat,
        colorspace: ff::AVColorSpace,
        driver: &'static CudaDriver,
        pipeline_cache: &mut PipelineCache,
    ) -> Result<Self> {
        let descriptor = layout::av_pixel_texture_format(sw_format)
            .ok_or(Error::UnsupportedPixelFormat)?;

        let bytes_per_channel = match descriptor.depth {
            layout::Depth::D8 => 1,
            layout::Depth::D10 | layout::Depth::D12 | layout::Depth::D16 => 2,
        };

        // 1. Try Tier 1: True Zero-Copy (VRAM)
        let direct_res: Result<DirectPlanes> = (|| unsafe {
            let mut direct_planes = Vec::new();
            let plane_views = match descriptor.planes {
                layout::PlaneLayout::PackedYUV420([fmt_y, fmt_uv]) => {
                    let py = create_exported_direct_plane(
                        device,
                        adapter,
                        width,
                        height,
                        fmt_y,
                        "CUDA Zero-Copy Plane Y",
                        driver,
                    )?;
                    let puv = create_exported_direct_plane(
                        device,
                        adapter,
                        width / 2,
                        height / 2,
                        fmt_uv,
                        "CUDA Zero-Copy Plane UV",
                        driver,
                    )?;
                    let views = layout::PlaneLayout::PackedYUV420([
                        py.view.clone(),
                        puv.view.clone(),
                    ]);
                    direct_planes.push(DirectPlane {
                        plane: py,
                        width_in_bytes: (width as usize) * bytes_per_channel,
                        height: height as usize,
                    });
                    direct_planes.push(DirectPlane {
                        plane: puv,
                        width_in_bytes: (width as usize) * bytes_per_channel,
                        height: (height / 2) as usize,
                    });
                    views
                }
                layout::PlaneLayout::YUV420([fmt_y, fmt_u, fmt_v]) => {
                    let py = create_exported_direct_plane(
                        device,
                        adapter,
                        width,
                        height,
                        fmt_y,
                        "CUDA Zero-Copy Plane Y",
                        driver,
                    )?;
                    let pu = create_exported_direct_plane(
                        device,
                        adapter,
                        width / 2,
                        height / 2,
                        fmt_u,
                        "CUDA Zero-Copy Plane U",
                        driver,
                    )?;
                    let pv = create_exported_direct_plane(
                        device,
                        adapter,
                        width / 2,
                        height / 2,
                        fmt_v,
                        "CUDA Zero-Copy Plane V",
                        driver,
                    )?;
                    let views = layout::PlaneLayout::YUV420([
                        py.view.clone(),
                        pu.view.clone(),
                        pv.view.clone(),
                    ]);
                    direct_planes.push(DirectPlane {
                        plane: py,
                        width_in_bytes: (width as usize) * bytes_per_channel,
                        height: height as usize,
                    });
                    direct_planes.push(DirectPlane {
                        plane: pu,
                        width_in_bytes: ((width / 2) as usize) * bytes_per_channel,
                        height: (height / 2) as usize,
                    });
                    direct_planes.push(DirectPlane {
                        plane: pv,
                        width_in_bytes: ((width / 2) as usize) * bytes_per_channel,
                        height: (height / 2) as usize,
                    });
                    views
                }
                layout::PlaneLayout::YUV444([fmt_y, fmt_u, fmt_v]) => {
                    let py = create_exported_direct_plane(
                        device,
                        adapter,
                        width,
                        height,
                        fmt_y,
                        "CUDA Zero-Copy Plane Y",
                        driver,
                    )?;
                    let pu = create_exported_direct_plane(
                        device,
                        adapter,
                        width,
                        height,
                        fmt_u,
                        "CUDA Zero-Copy Plane U",
                        driver,
                    )?;
                    let pv = create_exported_direct_plane(
                        device,
                        adapter,
                        width,
                        height,
                        fmt_v,
                        "CUDA Zero-Copy Plane V",
                        driver,
                    )?;
                    let views = layout::PlaneLayout::YUV444([
                        py.view.clone(),
                        pu.view.clone(),
                        pv.view.clone(),
                    ]);
                    direct_planes.push(DirectPlane {
                        plane: py,
                        width_in_bytes: (width as usize) * bytes_per_channel,
                        height: height as usize,
                    });
                    direct_planes.push(DirectPlane {
                        plane: pu,
                        width_in_bytes: (width as usize) * bytes_per_channel,
                        height: height as usize,
                    });
                    direct_planes.push(DirectPlane {
                        plane: pv,
                        width_in_bytes: (width as usize) * bytes_per_channel,
                        height: height as usize,
                    });
                    views
                }
                layout::PlaneLayout::RGB(_) => return Err(Error::UnsupportedPixelFormat),
            };

            let bg0 = pipeline_cache.bind_frame_textures(
                &layout::FrameDescriptor {
                    planes: plane_views,
                    depth: descriptor.depth,
                },
                colorspace.into(),
            );
            let layout = layout::FrameDescriptor {
                planes: descriptor.planes.as_identity(),
                depth: descriptor.depth,
            };
            Ok(DirectPlanes {
                planes: direct_planes,
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

        let mut host_planes = Vec::new();

        let create_plane = |tex_w: u32,
                            tex_h: u32,
                            fmt: wgpu::TextureFormat,
                            channels: usize,
                            label: &str|
         -> Result<HostDmaPlane> {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: tex_w,
                    height: tex_h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

            let width_in_bytes = (tex_w as usize) * channels * bytes_per_channel;
            let height_lines = tex_h as usize;
            let host_buf_size = width_in_bytes * height_lines;

            let mut host_buf: *mut c_void = std::ptr::null_mut();
            let _ = unsafe { (driver.cu_ctx_set_current)(driver.primary_ctx) };
            let res = unsafe { (driver.cu_mem_alloc_host)(&mut host_buf, host_buf_size) };
            if res != CUDA_SUCCESS || host_buf.is_null() {
                eprintln!("[CUDA-NVDEC] cuMemAllocHost for {} failed: {:?}", label, res);
                return Err(Error::HardwareContext);
            }

            Ok(HostDmaPlane {
                texture,
                view,
                host_buf,
                host_buf_size,
                width_in_bytes,
                height: height_lines,
                tex_width: tex_w,
                tex_height: tex_h,
            })
        };

        let plane_views = match descriptor.planes {
            layout::PlaneLayout::PackedYUV420([fmt_y, fmt_uv]) => {
                let py = match create_plane(width, height, fmt_y, 1, "CUDA NVDEC Plane Y") {
                    Ok(p) => p,
                    Err(e) => return Err(e),
                };
                let puv = match create_plane(
                    width / 2,
                    height / 2,
                    fmt_uv,
                    2,
                    "CUDA NVDEC Plane UV",
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        unsafe {
                            if !py.host_buf.is_null() {
                                (driver.cu_mem_free_host)(py.host_buf);
                            }
                        }
                        return Err(e);
                    }
                };
                let views = layout::PlaneLayout::PackedYUV420([
                    py.view.clone(),
                    puv.view.clone(),
                ]);
                host_planes.push(py);
                host_planes.push(puv);
                views
            }
            layout::PlaneLayout::YUV420([fmt_y, fmt_u, fmt_v]) => {
                let py = match create_plane(width, height, fmt_y, 1, "CUDA NVDEC Plane Y") {
                    Ok(p) => p,
                    Err(e) => return Err(e),
                };
                let pu = match create_plane(
                    width / 2,
                    height / 2,
                    fmt_u,
                    1,
                    "CUDA NVDEC Plane U",
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        unsafe {
                            if !py.host_buf.is_null() {
                                (driver.cu_mem_free_host)(py.host_buf);
                            }
                        }
                        return Err(e);
                    }
                };
                let pv = match create_plane(
                    width / 2,
                    height / 2,
                    fmt_v,
                    1,
                    "CUDA NVDEC Plane V",
                ) {
                    Ok(p) => p,
                    Err(e) => {
                        unsafe {
                            if !py.host_buf.is_null() {
                                (driver.cu_mem_free_host)(py.host_buf);
                            }
                            if !pu.host_buf.is_null() {
                                (driver.cu_mem_free_host)(pu.host_buf);
                            }
                        }
                        return Err(e);
                    }
                };
                let views = layout::PlaneLayout::YUV420([
                    py.view.clone(),
                    pu.view.clone(),
                    pv.view.clone(),
                ]);
                host_planes.push(py);
                host_planes.push(pu);
                host_planes.push(pv);
                views
            }
            layout::PlaneLayout::YUV444([fmt_y, fmt_u, fmt_v]) => {
                let py = match create_plane(width, height, fmt_y, 1, "CUDA NVDEC Plane Y") {
                    Ok(p) => p,
                    Err(e) => return Err(e),
                };
                let pu = match create_plane(width, height, fmt_u, 1, "CUDA NVDEC Plane U") {
                    Ok(p) => p,
                    Err(e) => {
                        unsafe {
                            if !py.host_buf.is_null() {
                                (driver.cu_mem_free_host)(py.host_buf);
                            }
                        }
                        return Err(e);
                    }
                };
                let pv = match create_plane(width, height, fmt_v, 1, "CUDA NVDEC Plane V") {
                    Ok(p) => p,
                    Err(e) => {
                        unsafe {
                            if !py.host_buf.is_null() {
                                (driver.cu_mem_free_host)(py.host_buf);
                            }
                            if !pu.host_buf.is_null() {
                                (driver.cu_mem_free_host)(pu.host_buf);
                            }
                        }
                        return Err(e);
                    }
                };
                let views = layout::PlaneLayout::YUV444([
                    py.view.clone(),
                    pu.view.clone(),
                    pv.view.clone(),
                ]);
                host_planes.push(py);
                host_planes.push(pu);
                host_planes.push(pv);
                views
            }
            layout::PlaneLayout::RGB(_) => return Err(Error::UnsupportedPixelFormat),
        };

        let bg0 = pipeline_cache.bind_frame_textures(
            &layout::FrameDescriptor {
                planes: plane_views,
                depth: descriptor.depth,
            },
            colorspace.into(),
        );

        let layout = layout::FrameDescriptor {
            planes: descriptor.planes.as_identity(),
            depth: descriptor.depth,
        };

        Ok(CudaPlanes::HostDma(HostDmaPlanes {
            planes: host_planes,
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
    sw_format: ff::AVPixelFormat,
    colorspace: ff::AVColorSpace,
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
            sw_format: ff::AVPixelFormat::AV_PIX_FMT_NONE,
            colorspace: ff::AVColorSpace::AVCOL_SPC_UNSPECIFIED,
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

        let sw_format = if !frame_ref.hw_frames_ctx.is_null() {
            unsafe {
                let hwctx = (*frame_ref.hw_frames_ctx).data as *mut ff::AVHWFramesContext;
                (*hwctx).sw_format
            }
        } else if unsafe { !(*self._decoder.as_ptr()).hw_frames_ctx.is_null() } {
            unsafe {
                let hwctx =
                    (*(*self._decoder.as_ptr()).hw_frames_ctx).data as *mut ff::AVHWFramesContext;
                (*hwctx).sw_format
            }
        } else {
            unsafe { (*self._decoder.as_ptr()).sw_pix_fmt }
        };

        let sw_format = if sw_format == ff::AVPixelFormat::AV_PIX_FMT_NONE {
            ff::AVPixelFormat::AV_PIX_FMT_NV12
        } else {
            sw_format
        };

        let width = frame_ref.width as u32;
        let height = frame_ref.height as u32;
        let colorspace = frame_ref.colorspace;

        if self.planes.is_none()
            || self.width != width
            || self.height != height
            || self.sw_format != sw_format
            || self.colorspace != colorspace
        {
            // Drop old planes if any
            self.planes = None;

            let planes = CudaPlanes::new(
                device,
                adapter,
                width,
                height,
                sw_format,
                colorspace,
                self.driver,
                pipeline_cache,
            )?;

            self.planes = Some(planes);
            self.width = width;
            self.height = height;
            self.sw_format = sw_format;
            self.colorspace = colorspace;
        }

        let planes = self.planes.as_ref().unwrap();

        let _ = unsafe { (self.driver.cu_ctx_set_current)(self.driver.primary_ctx) };

        match planes {
            CudaPlanes::Direct(d) => {
                for (i, p) in d.planes.iter().enumerate() {
                    let copy = CUDA_MEMCPY2D {
                        src_x_in_bytes: 0,
                        src_y: 0,
                        src_memory_type: CUmemorytype::Device,
                        src_host: std::ptr::null(),
                        src_device: frame_ref.data[i] as u64,
                        src_array: std::ptr::null_mut(),
                        src_pitch: frame_ref.linesize[i] as usize,

                        dst_x_in_bytes: 0,
                        dst_y: 0,
                        dst_memory_type: CUmemorytype::Array,
                        dst_host: std::ptr::null_mut(),
                        dst_device: 0,
                        dst_array: p.plane.cu_array,
                        dst_pitch: 0,

                        width_in_bytes: p.width_in_bytes,
                        height: p.height,
                    };
                    let res = unsafe { (self.driver.cu_memcpy_2d_async)(&copy, self.stream) };
                    if res != CUDA_SUCCESS {
                        eprintln!("[CUDA-NVDEC] cuMemcpy2DAsync direct plane {} failed: {:?}", i, res);
                        return Err(Error::HardwareContext);
                    }
                }

                let res = unsafe { (self.driver.cu_stream_synchronize)(self.stream) };
                if res != CUDA_SUCCESS {
                    eprintln!("[CUDA-NVDEC] cuStreamSynchronize direct failed: {:?}", res);
                    return Err(Error::HardwareContext);
                }
            }
            CudaPlanes::HostDma(h) => {
                for (i, p) in h.planes.iter().enumerate() {
                    let copy = CUDA_MEMCPY2D {
                        src_x_in_bytes: 0,
                        src_y: 0,
                        src_memory_type: CUmemorytype::Device,
                        src_host: std::ptr::null(),
                        src_device: frame_ref.data[i] as u64,
                        src_array: std::ptr::null_mut(),
                        src_pitch: frame_ref.linesize[i] as usize,

                        dst_x_in_bytes: 0,
                        dst_y: 0,
                        dst_memory_type: CUmemorytype::Host,
                        dst_host: p.host_buf,
                        dst_device: 0,
                        dst_array: std::ptr::null_mut(),
                        dst_pitch: p.width_in_bytes,

                        width_in_bytes: p.width_in_bytes,
                        height: p.height,
                    };
                    let res = unsafe { (self.driver.cu_memcpy_2d_async)(&copy, self.stream) };
                    if res != CUDA_SUCCESS {
                        eprintln!("[CUDA-NVDEC] cuMemcpy2DAsync for plane {} failed: {:?}", i, res);
                        return Err(Error::HardwareContext);
                    }
                }

                let res = unsafe { (self.driver.cu_stream_synchronize)(self.stream) };
                if res != CUDA_SUCCESS {
                    eprintln!("[CUDA-NVDEC] cuStreamSynchronize failed: {:?}", res);
                    return Err(Error::HardwareContext);
                }

                for p in &h.planes {
                    queue.write_texture(
                        wgpu::TexelCopyTextureInfo {
                            texture: &p.texture,
                            mip_level: 0,
                            origin: wgpu::Origin3d::ZERO,
                            aspect: wgpu::TextureAspect::All,
                        },
                        unsafe {
                            std::slice::from_raw_parts(
                                p.host_buf as *const u8,
                                p.host_buf_size,
                            )
                        },
                        wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(p.width_in_bytes as u32),
                            rows_per_image: None,
                        },
                        wgpu::Extent3d {
                            width: p.tex_width,
                            height: p.tex_height,
                            depth_or_array_layers: 1,
                        },
                    );
                }
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
            CudaPlanes::Direct(d) => d.planes.iter().map(|p| p.plane.view.clone()).collect(),
            CudaPlanes::HostDma(h) => h.planes.iter().map(|p| p.view.clone()).collect(),
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
