/**
 * WGSL compute shader for depthwise convolution (up to 7x7 kernels).
 * Uses 2D spatial workgroup layout (16x16) and workgroup-shared cached weights and bias.
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

  let ow = gid.x;
  let oh = gid.y;

  if (ow >= params.OW || oh >= params.OH || b >= params.B) {
    return;
  }

  var acc = bias_val;
  let ih0 = i32(oh) * i32(params.sH) - i32(params.pH);
  let iw0 = i32(ow) * i32(params.sW) - i32(params.pW);
  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_base = (b * params.IC + oc) * params.IH;

  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    let ih = ih0 + i32(kh) * i32(params.dH);
    if (ih >= 0 && ih < iH) {
      let row_offset = (in_base + u32(ih)) * params.IW;
      let w_offset = kh * params.KW;
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = iw0 + i32(kw) * i32(params.dW);
        if (iw >= 0 && iw < iW) {
          acc = acc + input[row_offset + u32(iw)] * w_tile[w_offset + kw];
        }
      }
    }
  }

  let out_idx = ((b * params.OC + oc) * params.OH + oh) * params.OW + ow;
  out[out_idx] = acc;
}
`;
