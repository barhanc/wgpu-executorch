/**
 * WGSL compute shader for 32x64 register-tiled 2D convolution GEMM in FP16 (4x4 per thread across 128 threads).
 * Optimized for small channel counts (OC <= 32) to maximize wave occupancy and eliminate thread divergence.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_TILE32_F16 = /* wgsl */ `
enable f16;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read_write> out: array<f16>;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> a_sub: array<array<vec4<f16>, 4>, 32>;
var<workgroup> b_sub: array<array<vec4<f16>, 16>, 16>;

@compute @workgroup_size(16, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let tile_row0 = wg_id.y * 32u;
  let tile_col0 = wg_id.x * 64u;
  let M = params.OC;
  let N = params.B * params.OH * params.OW;
  let K = params.IC * params.KH * params.KW;
  let K_vecs = K >> 2u;
  let khw = params.KH * params.KW;
  let ohw = params.OH * params.OW;

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_hw = params.IH * params.IW;
  let in_w = params.IW;
  let sH = i32(params.sH);
  let sW = i32(params.sW);
  let pH = i32(params.pH);
  let pW = i32(params.pW);
  let dH = i32(params.dH);
  let dW = i32(params.dW);

  var acc0 = vec4<f16>(0.0h);
  var acc1 = vec4<f16>(0.0h);
  var acc2 = vec4<f16>(0.0h);
  var acc3 = vec4<f16>(0.0h);

  let a_r = tid >> 2u;      // 0..31
  let a_c = tid & 3u;       // 0..3
  let a_m = tile_row0 + a_r;
  let a_valid = a_m < M;

  let b_r0 = tid >> 4u;      // 0..7
  let b_c4_0 = tid & 15u;    // 0..15
  let b_r1 = b_r0 + 8u;      // 8..15
  let b_c4_1 = b_c4_0;

  let n_base0 = tile_col0 + (b_c4_0 << 2u);

  var col_b0: array<u32, 4>;
  var col_ih_base0: array<i32, 4>;
  var col_iw_base0: array<i32, 4>;
  var col_valid0: array<bool, 4>;
  for (var kk = 0u; kk < 4u; kk = kk + 1u) {
    let n = n_base0 + kk;
    if (n < N) {
      col_valid0[kk] = true;
      let b_val = n / ohw;
      let sp = n % ohw;
      col_b0[kk] = b_val;
      let oh = sp / params.OW;
      let ow = sp % params.OW;
      col_ih_base0[kk] = i32(oh) * sH - pH;
      col_iw_base0[kk] = i32(ow) * sW - pW;
    } else {
      col_valid0[kk] = false;
      col_b0[kk] = 0u;
      col_ih_base0[kk] = 0;
      col_iw_base0[kk] = 0;
    }
  }

  let num_tiles = (K + 15u) >> 4u;
  for (var t = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t << 2u;
    let k_start = t << 4u;

    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = weight[a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f16>(0.0h);
    }

    let bk0 = k_start + b_r0;
    if (bk0 < K) {
      let ic0 = bk0 / khw;
      let rem0 = bk0 % khw;
      let kh0 = i32(rem0 / params.KW) * dH;
      let kw0 = i32(rem0 % params.KW) * dW;

      var b_vec0 = vec4<f16>(0.0h);
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        if (col_valid0[kk]) {
          let ih0 = col_ih_base0[kk] + kh0;
          let iw0 = col_iw_base0[kk] + kw0;
          if (ih0 >= 0 && ih0 < iH && iw0 >= 0 && iw0 < iW) {
            let in_idx0 = (col_b0[kk] * params.IC + ic0) * in_hw + u32(ih0) * in_w + u32(iw0);
            b_vec0[kk] = input[in_idx0];
          }
        }
      }
      b_sub[b_r0][b_c4_0] = b_vec0;
    } else {
      b_sub[b_r0][b_c4_0] = vec4<f16>(0.0h);
    }

    let bk1 = k_start + b_r1;
    if (bk1 < K) {
      let ic1 = bk1 / khw;
      let rem1 = bk1 % khw;
      let kh1 = i32(rem1 / params.KW) * dH;
      let kw1 = i32(rem1 % params.KW) * dW;

      var b_vec1 = vec4<f16>(0.0h);
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        if (col_valid0[kk]) {
          let ih1 = col_ih_base0[kk] + kh1;
          let iw1 = col_iw_base0[kk] + kw1;
          if (ih1 >= 0 && ih1 < iH && iw1 >= 0 && iw1 < iW) {
            let in_idx1 = (col_b0[kk] * params.IC + ic1) * in_hw + u32(ih1) * in_w + u32(iw1);
            b_vec1[kk] = input[in_idx1];
          }
        }
      }
      b_sub[b_r1][b_c4_1] = b_vec1;
    } else {
      b_sub[b_r1][b_c4_1] = vec4<f16>(0.0h);
    }

    workgroupBarrier();

    let thread_r = local_id.y << 2u;
    let thread_c4 = local_id.x;

    for (var k4 = 0u; k4 < 4u; k4 = k4 + 1u) {
      let b0 = b_sub[k4 * 4u + 0u][thread_c4];
      let b1 = b_sub[k4 * 4u + 1u][thread_c4];
      let b2 = b_sub[k4 * 4u + 2u][thread_c4];
      let b3 = b_sub[k4 * 4u + 3u][thread_c4];

      let a0 = a_sub[thread_r + 0u][k4];
      let a1 = a_sub[thread_r + 1u][k4];
      let a2 = a_sub[thread_r + 2u][k4];
      let a3 = a_sub[thread_r + 3u][k4];

      acc0 = acc0 + b0 * a0.x + b1 * a0.y + b2 * a0.z + b3 * a0.w;
      acc1 = acc1 + b0 * a1.x + b1 * a1.y + b2 * a1.z + b3 * a1.w;
      acc2 = acc2 + b0 * a2.x + b1 * a2.y + b2 * a2.z + b3 * a2.w;
      acc3 = acc3 + b0 * a3.x + b1 * a3.y + b2 * a3.z + b3 * a3.w;
    }
    workgroupBarrier();
  }

  let thread_r = local_id.y << 2u;
  let out_r0 = tile_row0 + thread_r;
  let out_c_base = tile_col0 + (local_id.x << 2u);

  var bias_val = vec4<f16>(0.0h);
  if (params.has_bias != 0u) {
    if (out_r0 + 0u < M) { bias_val.x = bias[out_r0 + 0u]; }
    if (out_r0 + 1u < M) { bias_val.y = bias[out_r0 + 1u]; }
    if (out_r0 + 2u < M) { bias_val.z = bias[out_r0 + 2u]; }
    if (out_r0 + 3u < M) { bias_val.w = bias[out_r0 + 3u]; }
  }

  acc0 = acc0 + vec4<f16>(bias_val.x);
  acc1 = acc1 + vec4<f16>(bias_val.y);
  acc2 = acc2 + vec4<f16>(bias_val.z);
  acc3 = acc3 + vec4<f16>(bias_val.w);

  for (var kk = 0u; kk < 4u; kk = kk + 1u) {
    let out_c = out_c_base + kk;
    if (out_c < N) {
      if (out_r0 + 0u < M) { out[(out_r0 + 0u) * N + out_c] = acc0[kk]; }
      if (out_r0 + 1u < M) { out[(out_r0 + 1u) * N + out_c] = acc1[kk]; }
      if (out_r0 + 2u < M) { out[(out_r0 + 2u) * N + out_c] = acc2[kk]; }
      if (out_r0 + 3u < M) { out[(out_r0 + 3u) * N + out_c] = acc3[kk]; }
    }
  }
}
`;
