from pathlib import Path


def replace_once(path, old, new):
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, got {count}")
    p.write_text(text.replace(old, new, 1))


replace_once(
    "src/decode/frames/opengl.rs",
    '''    extern "C" {
        fn eglGetCurrentDisplay() -> *mut c_void;
''',
    '''    unsafe extern "C" {
        fn eglGetCurrentDisplay() -> *mut c_void;
''',
)
replace_once(
    "src/decode/frames/opengl.rs",
    '''    extern "C" {
        fn resolve_egl_proc_address(name: *const std::os::raw::c_char) -> *mut c_void;
''',
    '''    unsafe extern "C" {
        fn resolve_egl_proc_address(name: *const std::os::raw::c_char) -> *mut c_void;
''',
)
replace_once(
    "src/decode/frames/vaapi.rs",
    '''use super::FrameAdapter;
use super::GlInteropTicket;
use crate::{
    Error,
    context::{layout, pipeline_cache::PipelineCache},
    decode::hw::FrameAdapterBuilder,
    error::Result,
};
''',
    '''use super::{FrameAdapter, FrameAdapterBuilder, GlInteropTicket};
use crate::{
    Error,
    context::{layout, pipeline_cache::PipelineCache},
    error::Result,
};
''',
)
replace_once(
    "src/decode/frames/vaapi.rs",
    '''    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.imported.as_ref().map(|imported| imported.identity)
    }

    fn name(&self) -> &'static str {
''',
    '''    fn layout_identity(&self) -> Option<layout::FrameDescriptor<()>> {
        self.imported.as_ref().map(|imported| imported.identity)
    }

    fn plane_views(&self) -> Option<Vec<wgpu::TextureView>> {
        self.imported.as_ref().map(|imported| {
            vec![
                imported.y_texture.create_view(&wgpu::TextureViewDescriptor::default()),
                imported.uv_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            ]
        })
    }

    fn name(&self) -> &'static str {
''',
)

print("pre-existing Linux compile regressions repaired")
