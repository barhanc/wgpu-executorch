/**
 * WGSL compute shader for fused depthwise 2D convolution with clamp in packed HWC4 layout (FP32).
 * 4 channels processed concurrently per load with 75% memory bandwidth reduction and fused clamp.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_HWC4_DW = /* wgsl */ `
struct Params {
  B: u32,
  C4: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  has_bias: u32,
  stride_x: u32,
  output_min: f32,
  output_max: f32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.stride_x + gid.x;
  let total_pixels = params.B * params.OH * params.OW;
  let total = total_pixels * params.C4;
  if (idx >= total) {
    return;
  }

  let c4 = idx % params.C4;
  let rem0 = idx / params.C4;
  let ow = rem0 % params.OW;
  let rem1 = rem0 / params.OW;
  let oh = rem1 % params.OH;
  let b = rem1 / params.OH;

  var acc = vec4<f32>(0.0);
  if (params.has_bias == 1u) {
    acc = bias[c4];
  }

  let in_b_base = b * (params.IH * params.IW) * params.C4;
  let wt_c_base = c4 * (params.KH * params.KW);

  for (var kh = 0u; kh < params.KH; kh++) {
    let ih = i32(oh * params.stride_h + kh * params.dilation_h) - i32(params.pad_h);
    if (ih < 0 || u32(ih) >= params.IH) {
      continue;
    }
    for (var kw = 0u; kw < params.KW; kw++) {
      let iw = i32(ow * params.stride_w + kw * params.dilation_w) - i32(params.pad_w);
      if (iw < 0 || u32(iw) >= params.IW) {
        continue;
      }
      let in_idx = in_b_base + (u32(ih) * params.IW + u32(iw)) * params.C4 + c4;
      let wt_idx = wt_c_base + kh * params.KW + kw;
      acc += x[in_idx] * weights[wt_idx];
    }
  }

  acc = clamp(acc, vec4<f32>(params.output_min), vec4<f32>(params.output_max));
  out[idx] = acc;
}
`;
