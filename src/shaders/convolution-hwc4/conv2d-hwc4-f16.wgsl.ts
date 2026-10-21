/**
 * WGSL compute shader for general 2D convolution in packed HWC4 layout (Native FP16).
 * 4-pixel horizontal register tiling with 4x4 matrix-vector multiplication,
 * 75% memory bandwidth reduction on weights, and coalesced 128-bit memory transactions.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_HWC4_F16 = /* wgsl */ `
enable f16;

struct Params {
  B: u32,
  IC4: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  OC4: u32,
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
  pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f16>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.stride_x + gid.x;
  let total_work = params.B * params.OH * params.ow_blocks * params.OC4;
  if (idx >= total_work) {
    return;
  }

  let oc4 = idx % params.OC4;
  let rem0 = idx / params.OC4;
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
    let b_val = bias[oc4];
    acc0 = b_val;
    acc1 = b_val;
    acc2 = b_val;
    acc3 = b_val;
  }

  let kk = params.KH * params.KW;
  let wt_oc_base = oc4 * params.IC4 * kk * 4u;
  let in_b_base = b * (params.IH * params.IW) * params.IC4;
  let sW = i32(params.stride_w);

  for (var kh = 0u; kh < params.KH; kh++) {
    let ih = i32(oh * params.stride_h + kh * params.dilation_h) - i32(params.pad_h);
    if (ih < 0 || u32(ih) >= params.IH) {
      continue;
    }
    let in_row_base = in_b_base + u32(ih) * params.IW * params.IC4;

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

      let in_px0 = in_row_base + u32(iw0) * params.IC4;
      let in_px1 = in_row_base + u32(iw1) * params.IC4;
      let in_px2 = in_row_base + u32(iw2) * params.IC4;
      let in_px3 = in_row_base + u32(iw3) * params.IC4;

      let k_offset = (kh * params.KW + kw) * 4u;

      for (var ic4 = 0u; ic4 < params.IC4; ic4++) {
        let wt_base = wt_oc_base + (ic4 * kk * 4u) + k_offset;
        let w0 = weights[wt_base + 0u];
        let w1 = weights[wt_base + 1u];
        let w2 = weights[wt_base + 2u];
        let w3 = weights[wt_base + 3u];

        if (valid0) {
          let x0 = x[in_px0 + ic4];
          acc0 += w0 * x0.x + w1 * x0.y + w2 * x0.z + w3 * x0.w;
        }
        if (valid1) {
          let x1 = x[in_px1 + ic4];
          acc1 += w0 * x1.x + w1 * x1.y + w2 * x1.z + w3 * x1.w;
        }
        if (valid2) {
          let x2 = x[in_px2 + ic4];
          acc2 += w0 * x2.x + w1 * x2.y + w2 * x2.z + w3 * x2.w;
        }
        if (valid3) {
          let x3 = x[in_px3 + ic4];
          acc3 += w0 * x3.x + w1 * x3.y + w2 * x3.z + w3 * x3.w;
        }
      }
    }
  }

  let out_row_base = (b * params.OH + oh) * params.OW;
  if (ow0 < params.OW) { out[(out_row_base + ow0) * params.OC4 + oc4] = acc0; }
  if (ow1 < params.OW) { out[(out_row_base + ow1) * params.OC4 + oc4] = acc1; }
  if (ow2 < params.OW) { out[(out_row_base + ow2) * params.OC4 + oc4] = acc2; }
  if (ow3 < params.OW) { out[(out_row_base + ow3) * params.OC4 + oc4] = acc3; }
}
`;
