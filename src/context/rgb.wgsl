struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var plane_rgb: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let rgb2rgb = mat3x3<f32>($color_matrix);

    var c = textureLoad(plane_rgb, vec2<u32>(in.position.xy), 0).rgb * $scale;

    var rgb = clamp(c * rgb2rgb, vec3<f32>(0), vec3<f32>(1));

    return vec4<f32>(rgb, 1);
}
