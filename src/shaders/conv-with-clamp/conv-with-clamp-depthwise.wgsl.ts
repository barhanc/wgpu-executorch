/**
 * WGSL compute shader for depthwise convolution with clamp bounds (up to 7x7 kernels).
 * Uses 2D spatial workgroup layout (16x16) with 2x2 register tiling per thread (32x32 per workgroup).
 * Weights and bias are cached in shared memory then private registers to avoid bank conflicts.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_DEPTHWISE = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_x: array<f32>;
@group(0) @binding(2) var<storage, read> t_weight: array<f32>;
@group(0) @binding(3) var<storage, read> t_bias: array<f32>;

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

var<workgroup> w_tile: array<f32, 49>;
var<workgroup> bias_val: f32;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let oc = wg_id.z % params.OC;
  let b = wg_id.z / params.OC;

  let tid = local_id.y * 16u + local_id.x;
  let khw = params.Kh * params.Kw;
  if (tid < khw) {
    w_tile[tid] = t_weight[oc * khw + tid];
  }
  if (tid == 0u) {
    if (params.has_bias != 0u) {
      bias_val = t_bias[oc];
    } else {
      bias_val = 0.0;
    }
  }
  workgroupBarrier();

  let ow0 = gid.x * 2u;
  let oh0 = gid.y * 2u;

  if (ow0 >= params.W_out || oh0 >= params.H_out || b >= params.N) {
    return;
  }

  var acc00 = bias_val;
  var acc01 = bias_val;
  var acc10 = bias_val;
  var acc11 = bias_val;

  let sH = i32(params.stride_h);
  let sW = i32(params.stride_w);
  let dH = i32(params.dil_h);
  let dW = i32(params.dil_w);
  let iH = i32(params.H_in);
  let iW = i32(params.W_in);

  let ih0_base = i32(oh0) * sH - i32(params.pad_h);
  let iw0_base = i32(ow0) * sW - i32(params.pad_w);
  let in_base = (b * params.IC + oc) * params.H_in;

  var w_reg: array<f32, 49>;
  for (var i = 0u; i < khw; i = i + 1u) {
    w_reg[i] = w_tile[i];
  }

  for (var kh = 0u; kh < params.Kh; kh = kh + 1u) {
    let ih0 = ih0_base + i32(kh) * dH;
    let ih1 = ih0 + sH;
    let valid_h0 = ih0 >= 0 && ih0 < iH;
    let valid_h1 = ih1 >= 0 && ih1 < iH;

    var row_offset0: u32 = 0u;
    if (valid_h0) {
      row_offset0 = (in_base + u32(ih0)) * params.W_in;
    }
    var row_offset1: u32 = 0u;
    if (valid_h1) {
      row_offset1 = (in_base + u32(ih1)) * params.W_in;
    }
    let w_offset = kh * params.Kw;

    for (var kw = 0u; kw < params.Kw; kw = kw + 1u) {
      let iw0 = iw0_base + i32(kw) * dW;
      let iw1 = iw0 + sW;
      let valid_w0 = iw0 >= 0 && iw0 < iW;
      let valid_w1 = iw1 >= 0 && iw1 < iW;
      let w_val = w_reg[w_offset + kw];

      if (valid_h0) {
        if (valid_w0) {
          acc00 = acc00 + t_x[row_offset0 + u32(iw0)] * w_val;
        }
        if (valid_w1) {
          acc01 = acc01 + t_x[row_offset0 + u32(iw1)] * w_val;
        }
      }
      if (valid_h1) {
        if (valid_w0) {
          acc10 = acc10 + t_x[row_offset1 + u32(iw0)] * w_val;
        }
        if (valid_w1) {
          acc11 = acc11 + t_x[row_offset1 + u32(iw1)] * w_val;
        }
      }
    }
  }

  acc00 = clamp(acc00, params.output_min, params.output_max);
  acc01 = clamp(acc01, params.output_min, params.output_max);
  acc10 = clamp(acc10, params.output_min, params.output_max);
  acc11 = clamp(acc11, params.output_min, params.output_max);

  let out_ch_base = (b * params.OC + oc) * params.H_out;
  let out_row0 = (out_ch_base + oh0) * params.W_out;
  t_out[out_row0 + ow0] = acc00;
  if (ow0 + 1u < params.W_out) {
    t_out[out_row0 + ow0 + 1u] = acc01;
  }
  if (oh0 + 1u < params.H_out) {
    let out_row1 = (out_ch_base + oh0 + 1u) * params.W_out;
    t_out[out_row1 + ow0] = acc10;
    if (ow0 + 1u < params.W_out) {
      t_out[out_row1 + ow0 + 1u] = acc11;
    }
  }
}
`;
