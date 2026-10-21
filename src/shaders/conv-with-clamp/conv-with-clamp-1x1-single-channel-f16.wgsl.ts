/**
 * FP16 pointwise convolution for a single output channel, with FP32 accumulation.
 * Keeps packed FP16 storage and widens operands before multiplication.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_1X1_SINGLE_CHANNEL_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> t_x: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> t_weight: array<f16>;
@group(0) @binding(3) var<storage, read> t_bias: array<f16>;

struct Params {
  N: u32,
  IC: u32,
  H_in: u32,
  W_in: u32,
  OC: u32,
  H_out: u32,
  W_out: u32,
  Kh: u32,
  Kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dil_h: u32,
  dil_w: u32,
  has_bias: u32,
  numel: u32,
  groups: u32,
  ic_per_group: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  output_min: f32,
  output_max: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let plane = (params.H_out * params.W_out) >> 2u;
  let pos = gid.x;
  let batch = gid.y;
  if (pos >= plane) { return; }

  var acc = vec4<f32>(0.0);
  let base = batch * params.IC * plane + pos;
  for (var ic = 0u; ic < params.IC; ic = ic + 1u) {
    acc += vec4<f32>(t_x[base + ic * plane]) * f32(t_weight[ic]);
  }
  if (params.has_bias != 0u) { acc += vec4<f32>(f32(t_bias[0])); }
  t_out[batch * plane + pos] = vec4<f16>(clamp(acc, vec4<f32>(params.output_min), vec4<f32>(params.output_max)));
}
`;
