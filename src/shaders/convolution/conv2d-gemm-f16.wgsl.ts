/**
 * WGSL compute shader for float16 2D convolution as implicit-im2col tiled GEMM.
 * Requires WebGPU shader-f16 feature.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_GEMM_F16 = /* wgsl */ `
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

const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f16, 32>, 32>;
var<workgroup> b_sub: array<array<vec4<f16>, 8>, 32>;

fn gK() -> u32 { return params.IC * params.KH * params.KW; }
fn gN() -> u32 { return params.B * params.OH * params.OW; }

override fuse_silu: bool = false;

// Preserve the convolution output and sigmoid's storage precision at the fusion boundary.
fn conv_epilogue(value: f16) -> f16 {
  if (!fuse_silu) { return value; }
  let stored = f16(quantizeToF16(f32(value)));
  let gate = f16(quantizeToF16(f32(f16(1.0) / (f16(1.0) + exp(-stored)))));
  return stored * gate;
}

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

  var acc: array<vec4<f16>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    acc[ir] = vec4<f16>(0.0h);
  }

  let ohw = params.OH * params.OW;
  let khw = params.KH * params.KW;
  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let sH = i32(params.sH);
  let sW = i32(params.sW);
  let pH = i32(params.pH);
  let pW = i32(params.pW);
  let dH = i32(params.dH);
  let dW = i32(params.dW);

  var col_b: array<u32, 4>;
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
      let oh_val = i32(sp / params.OW);
      let ow_val = i32(sp % params.OW);
      col_oh_arr[kk] = oh_val;
      col_ow_arr[kk] = ow_val;
      col_ih_base[kk] = oh_val * sH - pH;
      col_iw_base[kk] = ow_val * sW - pW;
    } else {
      col_valid[kk] = false;
      col_b[kk] = 0u;
      col_oh_arr[kk] = 0;
      col_ow_arr[kk] = 0;
      col_ih_base[kk] = 0;
      col_iw_base[kk] = 0;
    }
  }

  let in_hw = params.IH * params.IW;
  let in_w = params.IW;
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
        kh_dH = i32(rem / params.KW) * dH;
        kw_dW = i32(rem % params.KW) * dW;
      }

      let m_valid = m < M;
      let w_row_base = m * K + k_start + local_id.x * RPT;

      var b_row: vec4<f16>;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        let k_col = k_start + col;
        if (m_valid && k_col < K) {
          a_sub[arow][col] = weight[w_row_base + kk];
        } else {
          a_sub[arow][col] = 0.0h;
        }

        if (k_valid && col_valid[kk]) {
          let ih = col_ih_base[kk] + kh_dH;
          let iw = col_iw_base[kk] + kw_dW;
          if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
            let ch_idx = col_b[kk] * params.IC + k_ic;
            b_row[kk] = input[ch_idx * in_hw + u32(ih) * in_w + u32(iw)];
          } else {
            b_row[kk] = 0.0h;
          }
        } else {
          b_row[kk] = 0.0h;
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
      var bias_val = 0.0h;
      if (params.has_bias != 0u) {
        bias_val = bias[m];
      }
      let v = acc[ir] + vec4<f16>(bias_val);
      for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
        let n = tile_col0 + tile_col + ic;
        if (n < N) {
          let b = col_b[ic];
          let oh = u32(col_oh_arr[ic]);
          let ow = u32(col_ow_arr[ic]);
          out[((b * params.OC + m) * params.OH + oh) * params.OW + ow] = conv_epilogue(v[ic]);
        }
      }
    }
  }
}
`;
