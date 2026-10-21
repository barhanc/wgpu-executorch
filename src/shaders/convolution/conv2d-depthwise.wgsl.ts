/**
 * WGSL compute shader for depthwise convolution (up to 7x7 kernels).
 * Uses 2D spatial workgroup layout (16x16) with 2x2 register tiling per thread (32x32 per workgroup).
 * Weights and bias are cached in shared memory then private registers to avoid bank conflicts.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_DEPTHWISE = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> w_tile: array<f32, 49>;
var<workgroup> bias_val: f32;

override fuse_silu: bool = false;

// Preserve the convolution output and sigmoid's storage precision at the fusion boundary.
fn conv_epilogue(value: f32) -> f32 {
  if (!fuse_silu) { return value; }
  let gate = f32(1.0) / (f32(1.0) + exp(-value));
  return value * gate;
}

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let oc = wg_id.z % params.OC;
  let b = wg_id.z / params.OC;

  let tid = local_id.y * 16u + local_id.x;
  let khw = params.KH * params.KW;
  if (tid < khw) {
    w_tile[tid] = weight[oc * khw + tid];
  }
  if (tid == 0u) {
    if (params.has_bias != 0u) {
      bias_val = bias[oc];
    } else {
      bias_val = 0.0;
    }
  }
  workgroupBarrier();

  let ow0 = gid.x * 2u;
  let oh0 = gid.y * 2u;

  if (ow0 >= params.OW || oh0 >= params.OH || b >= params.B) {
    return;
  }

  var acc00 = bias_val;
  var acc01 = bias_val;
  var acc10 = bias_val;
  var acc11 = bias_val;

  let sH = i32(params.sH);
  let sW = i32(params.sW);
  let dH = i32(params.dH);
  let dW = i32(params.dW);
  let iH = i32(params.IH);
  let iW = i32(params.IW);

  let ih0_base = i32(oh0) * sH - i32(params.pH);
  let iw0_base = i32(ow0) * sW - i32(params.pW);
  let in_base = (b * params.IC + oc) * params.IH;

  var w_reg: array<f32, 49>;
  for (var i = 0u; i < khw; i = i + 1u) {
    w_reg[i] = w_tile[i];
  }

  for (var kh = 0u; kh < params.KH; kh = kh + 1u) {
    let ih0 = ih0_base + i32(kh) * dH;
    let ih1 = ih0 + sH;
    let valid_h0 = ih0 >= 0 && ih0 < iH;
    let valid_h1 = ih1 >= 0 && ih1 < iH;

    var row_offset0: u32 = 0u;
    if (valid_h0) {
      row_offset0 = (in_base + u32(ih0)) * params.IW;
    }
    var row_offset1: u32 = 0u;
    if (valid_h1) {
      row_offset1 = (in_base + u32(ih1)) * params.IW;
    }
    let w_offset = kh * params.KW;

    for (var kw = 0u; kw < params.KW; kw = kw + 1u) {
      let iw0 = iw0_base + i32(kw) * dW;
      let iw1 = iw0 + sW;
      let valid_w0 = iw0 >= 0 && iw0 < iW;
      let valid_w1 = iw1 >= 0 && iw1 < iW;
      let w_val = w_reg[w_offset + kw];

      if (valid_h0) {
        if (valid_w0) {
          acc00 = acc00 + input[row_offset0 + u32(iw0)] * w_val;
        }
        if (valid_w1) {
          acc01 = acc01 + input[row_offset0 + u32(iw1)] * w_val;
        }
      }
      if (valid_h1) {
        if (valid_w0) {
          acc10 = acc10 + input[row_offset1 + u32(iw0)] * w_val;
        }
        if (valid_w1) {
          acc11 = acc11 + input[row_offset1 + u32(iw1)] * w_val;
        }
      }
    }
  }

  let out_ch_base = (b * params.OC + oc) * params.OH;
  let out_row0 = (out_ch_base + oh0) * params.OW;
  out[out_row0 + ow0] = conv_epilogue(acc00);
  if (ow0 + 1u < params.OW) {
    out[out_row0 + ow0 + 1u] = conv_epilogue(acc01);
  }
  if (oh0 + 1u < params.OH) {
    let out_row1 = (out_ch_base + oh0 + 1u) * params.OW;
    out[out_row1 + ow0] = conv_epilogue(acc10);
    if (ow0 + 1u < params.OW) {
      out[out_row1 + ow0 + 1u] = conv_epilogue(acc11);
    }
  }
}
`;
