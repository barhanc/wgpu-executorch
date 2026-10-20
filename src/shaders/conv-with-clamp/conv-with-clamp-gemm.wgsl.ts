/**
 * WGSL compute shader for fused convolution with clamp bounds as
 * implicit-im2col tiled GEMM (groups == 1).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
export const SHADER_CONV_WITH_CLAMP_GEMM = /* wgsl */ `
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

// Tiled GEMM over implicit im2col (groups == 1): M=OC, N=N*H_out*W_out,
// K=IC*Kh*Kw. A=weight [OC,K] (contiguous OIHW); B=input sampled on the fly,
// out-of-range -> 0.0 implements padding. bias per-ROW (OC). Output written
// NCHW with the clamp bounds applied.
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

fn gK() -> u32 { return params.IC * params.Kh * params.Kw; }
fn gN() -> u32 { return params.N * params.H_out * params.W_out; }

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>) {
  let M = params.OC;
  let N = gN();
  let K = gK();
  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  var acc: array<vec4<f32>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    acc[ir] = vec4<f32>(0.0);
  }

  let ohw = params.H_out * params.W_out;
  let khw = params.Kh * params.Kw;
  let iH = i32(params.H_in);
  let iW = i32(params.W_in);
  let sH = i32(params.stride_h);
  let sW = i32(params.stride_w);
  let pH = i32(params.pad_h);
  let pW = i32(params.pad_w);
  let dH = i32(params.dil_h);
  let dW = i32(params.dil_w);

  // Precompute spatial coords for this thread's 4 columns in B (invariant across K tiles)
  var col_b: array<u32, 4>;
  var col_b_ic: array<u32, 4>;
  var col_valid: array<bool, 4>;
  var col_oh_arr: array<i32, 4>;
  var col_ow_arr: array<i32, 4>;
  var col_ih_base: array<i32, 4>;
  var col_iw_base: array<i32, 4>;
  for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
    let n = tile_col0 + local_id.x * RPT + kk;
    if (n < N) {
      col_valid[kk] = true;
      let b_val = n / ohw;
      let sp = n % ohw;
      col_b[kk] = b_val;
      col_b_ic[kk] = b_val * params.IC;
      let oh_val = i32(sp / params.W_out);
      let ow_val = i32(sp % params.W_out);
      col_oh_arr[kk] = oh_val;
      col_ow_arr[kk] = ow_val;
      col_ih_base[kk] = oh_val * sH - pH;
      col_iw_base[kk] = ow_val * sW - pW;
    } else {
      col_valid[kk] = false;
      col_b[kk] = 0u;
      col_b_ic[kk] = 0u;
      col_oh_arr[kk] = 0;
      col_ow_arr[kk] = 0;
      col_ih_base[kk] = 0;
      col_iw_base[kk] = 0;
    }
  }

  let in_hw = params.H_in * params.W_in;
  let in_w = params.W_in;
  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      let m = tile_row0 + arow;
      let kk_idx = k_start + arow;

      var k_valid = false;
      var k_ic = 0u;
      var kh_dH = 0;
      var kw_dW = 0;
      if (kk_idx < K) {
        k_valid = true;
        k_ic = kk_idx / khw;
        let rem = kk_idx % khw;
        kh_dH = i32(rem / params.Kw) * dH;
        kw_dW = i32(rem % params.Kw) * dW;
      }

      let m_valid = m < M;
      let w_row_base = m * K + k_start + local_id.x * RPT;

      var b_row: vec4<f32>;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        let k_col = k_start + col;
        if (m_valid && k_col < K) {
          a_sub[arow][col] = t_weight[w_row_base + kk];
        } else {
          a_sub[arow][col] = 0.0;
        }

        if (k_valid && col_valid[kk]) {
          let ih = col_ih_base[kk] + kh_dH;
          let iw = col_iw_base[kk] + kw_dW;
          if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
            let ch_idx = col_b_ic[kk] + k_ic;
            b_row[kk] = t_x[ch_idx * in_hw + u32(ih) * in_w + u32(iw)];
          } else {
            b_row[kk] = 0.0;
          }
        } else {
          b_row[kk] = 0.0;
        }
      }
      b_sub[arow][local_id.x] = b_row;
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let b_vec = b_sub[k][local_id.x];
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        acc[ir] = acc[ir] + aval * b_vec;
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    let m = tile_row0 + tile_row + ir;
    if (m < M) {
      var bias_val = 0.0;
      if (params.has_bias != 0u) {
        bias_val = t_bias[m];
      }
      let v = clamp(acc[ir] + vec4<f32>(bias_val), vec4<f32>(params.output_min), vec4<f32>(params.output_max));
      for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
        let n = tile_col0 + tile_col + ic;
        if (n < N) {
          let b = col_b[ic];
          let oh = u32(col_oh_arr[ic]);
          let ow = u32(col_ow_arr[ic]);
          t_out[((b * params.OC + m) * params.H_out + oh) * params.W_out + ow] = v[ic];
        }
      }
    }
  }
}
`;
