/**
 * WGSL compute shader for depthwise 3x3 convolution in FP16 with shared-memory halo tile caching.
 * Loads a 34x34 halo tile cooperatively across 256 threads, eliminating redundant global memory reads.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_DEPTHWISE_3X3_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> out: array<f16>;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> in_tile: array<array<f16, 34>, 34>;
var<workgroup> w_tile: array<f16, 9>;
var<workgroup> bias_val: f16;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let oc = wg_id.z % params.OC;
  let b = wg_id.z / params.OC;

  if (tid < 9u) {
    w_tile[tid] = weight[oc * 9u + tid];
  }
  if (tid == 0u) {
    if (params.has_bias != 0u) {
      bias_val = bias[oc];
    } else {
      bias_val = 0.0h;
    }
  }

  let in_hw = params.IH * params.IW;
  let in_w = params.IW;
  let in_base = (b * params.IC + oc) * in_hw;
  let iH = i32(params.IH);
  let iW = i32(params.IW);

  let tile_ih_start = i32(wg_id.y * 32u) - 1;
  let tile_iw_start = i32(wg_id.x * 32u) - 1;

  // Cooperatively load 34x34 halo tile (1156 f16s) into shared memory
  for (var idx = tid; idx < 1156u; idx = idx + 256u) {
    let tr = idx / 34u;
    let tc = idx % 34u;
    let ih = tile_ih_start + i32(tr);
    let iw = tile_iw_start + i32(tc);
    if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
      in_tile[tr][tc] = input[in_base + u32(ih) * in_w + u32(iw)];
    } else {
      in_tile[tr][tc] = 0.0h;
    }
  }
  workgroupBarrier();

  let ow0 = wg_id.x * 32u + local_id.x * 2u;
  let oh0 = wg_id.y * 32u + local_id.y * 2u;

  if (ow0 >= params.OW || oh0 >= params.OH || b >= params.B) {
    return;
  }

  var acc00 = bias_val;
  var acc01 = bias_val;
  var acc10 = bias_val;
  var acc11 = bias_val;

  let tr = local_id.y * 2u;
  let tc = local_id.x * 2u;

  let w0 = w_tile[0]; let w1 = w_tile[1]; let w2 = w_tile[2];
  let w3 = w_tile[3]; let w4 = w_tile[4]; let w5 = w_tile[5];
  let w6 = w_tile[6]; let w7 = w_tile[7]; let w8 = w_tile[8];

  // Row 0
  let r0_c0 = in_tile[tr + 0u][tc + 0u];
  let r0_c1 = in_tile[tr + 0u][tc + 1u];
  let r0_c2 = in_tile[tr + 0u][tc + 2u];
  let r0_c3 = in_tile[tr + 0u][tc + 3u];

  acc00 = acc00 + r0_c0 * w0 + r0_c1 * w1 + r0_c2 * w2;
  acc01 = acc01 + r0_c1 * w0 + r0_c2 * w1 + r0_c3 * w2;

  // Row 1
  let r1_c0 = in_tile[tr + 1u][tc + 0u];
  let r1_c1 = in_tile[tr + 1u][tc + 1u];
  let r1_c2 = in_tile[tr + 1u][tc + 2u];
  let r1_c3 = in_tile[tr + 1u][tc + 3u];

  acc00 = acc00 + r1_c0 * w3 + r1_c1 * w4 + r1_c2 * w5;
  acc01 = acc01 + r1_c1 * w3 + r1_c2 * w4 + r1_c3 * w5;
  acc10 = acc10 + r1_c0 * w0 + r1_c1 * w1 + r1_c2 * w2;
  acc11 = acc11 + r1_c1 * w0 + r1_c2 * w1 + r1_c3 * w2;

  // Row 2
  let r2_c0 = in_tile[tr + 2u][tc + 0u];
  let r2_c1 = in_tile[tr + 2u][tc + 1u];
  let r2_c2 = in_tile[tr + 2u][tc + 2u];
  let r2_c3 = in_tile[tr + 2u][tc + 3u];

  acc00 = acc00 + r2_c0 * w6 + r2_c1 * w7 + r2_c2 * w8;
  acc01 = acc01 + r2_c1 * w6 + r2_c2 * w7 + r2_c3 * w8;
  acc10 = acc10 + r2_c0 * w3 + r2_c1 * w4 + r2_c2 * w5;
  acc11 = acc11 + r2_c1 * w3 + r2_c2 * w4 + r2_c3 * w5;

  // Row 3
  let r3_c0 = in_tile[tr + 3u][tc + 0u];
  let r3_c1 = in_tile[tr + 3u][tc + 1u];
  let r3_c2 = in_tile[tr + 3u][tc + 2u];
  let r3_c3 = in_tile[tr + 3u][tc + 3u];

  acc10 = acc10 + r3_c0 * w6 + r3_c1 * w7 + r3_c2 * w8;
  acc11 = acc11 + r3_c1 * w6 + r3_c2 * w7 + r3_c3 * w8;

  let out_hw = params.OH * params.OW;
  let out_w = params.OW;
  let out_base = (b * params.OC + oc) * out_hw;

  let out_idx00 = out_base + oh0 * out_w + ow0;
  out[out_idx00] = acc00;

  if (ow0 + 1u < params.OW) {
    out[out_idx00 + 1u] = acc01;
  }
  if (oh0 + 1u < params.OH) {
    let out_idx10 = out_base + (oh0 + 1u) * out_w + ow0;
    out[out_idx10] = acc10;
    if (ow0 + 1u < params.OW) {
      out[out_idx10 + 1u] = acc11;
    }
  }
}
`;
