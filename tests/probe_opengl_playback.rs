//! GL-backend playback probe: verifies the decoded frames actually reach the
//! RGBA session texture with real content. The existing engine test only
//! counts decoded frames; this one reads the pixels back.
//!
//! Mimics the engine bridge: `update(encoder)` -> `queue.submit` ->
//! `finish_pending_gl_frames` -> `release_completed_frames`.

use ffgpu::Context;
use std::time::{Duration, Instant};

const COPY_WGSL: &str = r#"
@group(0) @binding(0) var t: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32(i % 2u) * 4.0 - 1.0, f32(i / 2u) * 4.0 - 1.0);
    return vec4<f32>(p, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(t, vec2<i32>(pos.xy), 0);
}
"#;

/// Render the session RGBA texture into a probe staging texture that carries
/// COPY_SRC, then read it back. The production texture omits COPY_SRC so a
/// direct texture→buffer copy is rejected by validation.
fn read_texture_mean(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
) -> (f64, f64) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("probe_copy"),
        source: wgpu::ShaderSource::Wgsl(COPY_WGSL.into()),
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("probe_copy"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview_mask: None,
        cache: None,
    });
    let staging = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("probe_staging"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let staging_view = staging.create_view(&Default::default());
    let src_view = texture.create_view(&Default::default());
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&src_view),
        }],
    });

    let row_bytes = ((width * 4 + 255) / 256) * 256;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("probe_readback"),
        size: (row_bytes * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("probe_readback_enc"),
    });
    {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("probe_copy_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &staging_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &bind, &[]);
        rpass.draw(0..3, 0..1);
    }
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &staging,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_bytes),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    let slice = buf.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv_timeout(Duration::from_secs(10))
        .expect("map_async callback")
        .expect("map_async ok");
    let mapped = slice.get_mapped_range();
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut n = 0u64;
    // luma of a sparse 32x16 subsample grid
    for y in (0..height).step_by(32usize.max(1)) {
        let row_start = (y * row_bytes) as usize;
        for x in (0..width).step_by(16usize.max(1)) {
            let i = row_start + (x * 4) as usize;
            if i + 2 >= mapped.len() {
                continue;
            }
            let (r, g, b) = (mapped[i] as f64, mapped[i + 1] as f64, mapped[i + 2] as f64);
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            sum += luma;
            sum_sq += luma * luma;
            n += 1;
        }
    }
    drop(mapped);
    buf.unmap();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = sum / n as f64;
    let var = (sum_sq / n as f64) - mean * mean;
    (mean, var.max(0.0).sqrt())
}

#[test]
fn probe_gl_playback_pixels_live() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL,
        ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
    });
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    })) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("[probe] no GL adapter available; skipping");
            return;
        }
    };
    eprintln!("[probe] GL adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: ffgpu::required_wgpu_device_features(&adapter),
        ..Default::default()
    }))
    .expect("GL device");

    let path = format!("{}/../../Test.mp4", env!("CARGO_MANIFEST_DIR"));
    let path = std::path::Path::new(&path);
    if !path.exists() {
        eprintln!("[probe] Test.mp4 missing at repo root; skipping");
        return;
    }
    let mut ctx = Context::new(&instance, &adapter, &device, &queue).expect("ffgpu context");
    let (mut video, _audio) = ctx.create_video(path).expect("open Test.mp4");
    video.set_looping(true);
    eprintln!(
        "[probe] video {}x{} {:.2}s",
        video.width(),
        video.height(),
        video.duration().as_secs_f64()
    );

    let width = video.width();
    let height = video.height();

    // Drive decode at REAL playback pace (engine-bridge style): each update
    // returns a wait hint; sleep for it. This is the cadence the engine
    // applies in production at 60 fps playback.
    let start = Instant::now();
    let mut decoded = 0u32;
    // Track texture-content progression to expose stuck frames.
    let mut sample_log: Vec<(f64, f64, f64)> = Vec::new();
    let mut last_sample_m: Option<f64> = None;
    while start.elapsed() < Duration::from_secs(30) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe_update"),
        });
        let (wait, d) = video.update(&mut encoder).expect("video.update");
        queue.submit(Some(encoder.finish()));
        let _ = video.finish_pending_gl_frames();
        let _ = video.release_completed_frames();
        if d {
            decoded += 1;
        }
        if start.elapsed().as_millis() % 500 < 16 {
            let pos = video.position().as_secs_f64();
            let (m, s) = read_texture_mean(&device, &queue, video.texture(), width, height);
            let delta = last_sample_m.map(|p| (m - p).abs()).unwrap_or(0.0);
            sample_log.push((start.elapsed().as_secs_f64(), m, delta));
            eprintln!(
                "[probe] t={:.2}s pos={pos:.2}s texture mean={m:.2} sd={s:.2} delta={delta:.2}",
                start.elapsed().as_secs_f64()
            );
            last_sample_m = Some(m);
        }
        std::thread::sleep(wait.min(Duration::from_millis(33)));
    }
    let frozen_windows = sample_log
        .windows(2)
        .filter(|w| w[1].2 < 0.1)
        .count();
    eprintln!(
        "[probe] {} samples, {} frozen-adjacent pairs",
        sample_log.len(),
        frozen_windows
    );
    eprintln!("[probe] decoded {} frames in {:?}", decoded, start.elapsed());
    assert!(decoded >= 1, "no frames decoded on GL path");

    let (mean, sd) = read_texture_mean(&device, &queue, video.texture(), width, height);
    eprintln!("[probe] first sample: mean={mean:.2} sd={sd:.2}");
    assert!(
        mean > 2.0,
        "GL video texture is black (mean={mean:.2}) — pixels never arrived"
    );
    assert!(
        sd > 1.0,
        "GL video texture is flat (sd={sd:.2}); mean={mean:.2}"
    );

    // Let playback advance, then confirm the content actually CHANGED (live).
    let mut changed = false;
    for tick in 0..3 {
        std::thread::sleep(Duration::from_millis(700));
        // Drive update() in a burst for ~1/4s of video time so A/V pacing
        // definitely delivers fresh frames, then re-sample.
        let t = Instant::now();
        let mut decoded_now = 0u32;
        while t.elapsed() < Duration::from_millis(250) {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("probe_update2"),
            });
            let (_w, d) = video.update(&mut encoder).expect("video.update");
            queue.submit(Some(encoder.finish()));
            let _ = video.finish_pending_gl_frames();
            let _ = video.release_completed_frames();
            if d {
                decoded_now += 1;
            } else {
                std::thread::sleep(Duration::from_millis(2));
            }
        }
        let (m2, s2) = read_texture_mean(&device, &queue, video.texture(), width, height);
        eprintln!(
            "[probe] later sample {tick}: mean={m2:.2} sd={s2:.2} decoded_now={decoded_now} pos={:.2}s",
            video.position().as_secs_f64()
        );
        if (m2 - mean).abs() > 0.5 {
            changed = true;
        }
    }
    assert!(
        changed,
        "GL video texture content frozen across 2s of playback"
    );
}
