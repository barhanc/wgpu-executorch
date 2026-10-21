/**
 * WGSL compute shader for depthwise 2D convolution with clamp in packed HWC4 layout (Native FP16).
 * 4-pixel horizontal register tiling with 4 channels processed concurrently per load,
 * 75% memory bandwidth reduction on weights, fused clamp, and coalesced 128-bit memory transactions.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_HWC4_DW_F16 = /* wgsl */ `
enable f16;

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
  ow_blocks: u32,
  pad0: u32,
  output_min: f32,
  output_max: f32,
  pad1: u32,
  pad2: u32,
  pad3: u32,
  pad4: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.stride_x + gid.x;
  let total_work = params.B * params.OH * params.ow_blocks * params.C4;
  if (idx >= total_work) {
    return;
  }

  let c4 = idx % params.C4;
  let rem0 = idx / params.C4;
  let ow_block = rem0 % params.ow_blocks;
  let rem1 = rem0 / params.ow_blocks;
  let oh = rem1 % params.OH;
  let b = rem1 / params.OH;

  let ow0 = ow_block * 4u;
  let ow1 = ow0 + 1u;
  let ow2 = ow0 + 2u;
  let ow3 = ow0 + 3u;

  var acc0 = vec4<f16>(0.0h);
  var acc1 = vec4<f16>(0.0h);
  var acc2 = vec4<f16>(0.0h);
  var acc3 = vec4<f16>(0.0h);
  if (params.has_bias == 1u) {
    let b_val = bias[c4];
    acc0 = b_val;
    acc1 = b_val;
    acc2 = b_val;
    acc3 = b_val;
  }

  let in_b_base = b * (params.IH * params.IW) * params.C4;
  let wt_c_base = c4 * (params.KH * params.KW);
  let sW = i32(params.stride_w);

  for (var kh = 0u; kh < params.KH; kh++) {
    let ih = i32(oh * params.stride_h + kh * params.dilation_h) - i32(params.pad_h);
    if (ih < 0 || u32(ih) >= params.IH) {
      continue;
    }
    let in_row_base = in_b_base + u32(ih) * params.IW * params.C4 + c4;

    for (var kw = 0u; kw < params.KW; kw++) {
      let iw_base = i32(ow0 * params.stride_w + kw * params.dilation_w) - i32(params.pad_w);
      let iw0 = iw_base;
      let iw1 = iw_base + sW;
      let iw2 = iw_base + 2 * sW;
      let iw3 = iw_base + 3 * sW;

      let valid0 = ow0 < params.OW && iw0 >= 0 && u32(iw0) < params.IW;
      let valid1 = ow1 < params.OW && iw1 >= 0 && u32(iw1) < params.IW;
      let valid2 = ow2 < params.OW && iw2 >= 0 && u32(iw2) < params.IW;
      let valid3 = ow3 < params.OW && iw3 >= 0 && u32(iw3) < params.IW;

      if (!valid0 && !valid1 && !valid2 && !valid3) {
        continue;
      }

      let wt_idx = wt_c_base + kh * params.KW + kw;
      let wt_val = weights[wt_idx];

      if (valid0) {
        acc0 += x[in_row_base + u32(iw0) * params.C4] * wt_val;
      }
      if (valid1) {
        acc1 += x[in_row_base + u32(iw1) * params.C4] * wt_val;
      }
      if (valid2) {
        acc2 += x[in_row_base + u32(iw2) * params.C4] * wt_val;
      }
      if (valid3) {
        acc3 += x[in_row_base + u32(iw3) * params.C4] * wt_val;
      }
    }
  }

  acc0 = clamp(acc0, vec4<f16>(f16(params.output_min)), vec4<f16>(f16(params.output_max)));
  acc1 = clamp(acc1, vec4<f16>(f16(params.output_min)), vec4<f16>(f16(params.output_max)));
  acc2 = clamp(acc2, vec4<f16>(f16(params.output_min)), vec4<f16>(f16(params.output_max)));
  acc3 = clamp(acc3, vec4<f16>(f16(params.output_min)), vec4<f16>(f16(params.output_max)));

  let out_row_base = (b * params.OH + oh) * params.OW;
  if (ow0 < params.OW) { out[(out_row_base + ow0) * params.C4 + c4] = acc0; }
  if (ow1 < params.OW) { out[(out_row_base + ow1) * params.C4 + c4] = acc1; }
  if (ow2 < params.OW) { out[(out_row_base + ow2) * params.C4 + c4] = acc2; }
  if (ow3 < params.OW) { out[(out_row_base + ow3) * params.C4 + c4] = acc3; }
}
`;
