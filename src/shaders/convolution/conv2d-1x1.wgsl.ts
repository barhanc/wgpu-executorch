/**
 * WGSL compute shader for direct 2D 1x1 convolution (pointwise GEMM).
 * Avoids im2col overhead since 1x1 conv with stride 1, padding 0 is a direct matrix multiply.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_1X1 = /* wgsl */ `
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

const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let M = params.OC;
  let spatial = params.OH * params.OW;
  let N_total = params.B * spatial;
  let K = params.IC;

  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;

    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        let m = tile_row0 + arow;
        let k = k_start + col;
        if (m < M && k < K) {
          a_sub[arow][col] = weight[m * K + k];
        } else {
          a_sub[arow][col] = 0.0;
        }

        let b_k = k_start + arow;
        let n = tile_col0 + col;
        if (b_k < K && n < N_total) {
          let b_idx = n / spatial;
          let sp = n % spatial;
          b_sub[arow][col] = input[(b_idx * params.IC + b_k) * spatial + sp];
        } else {
          b_sub[arow][col] = 0.0;
        }
      }
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      let b0 = b_sub[k][tile_col + 0u];
      let b1 = b_sub[k][tile_col + 1u];
      let b2 = b_sub[k][tile_col + 2u];
      let b3 = b_sub[k][tile_col + 3u];

      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        acc[ir][0] = acc[ir][0] + aval * b0;
        acc[ir][1] = acc[ir][1] + aval * b1;
        acc[ir][2] = acc[ir][2] + aval * b2;
        acc[ir][3] = acc[ir][3] + aval * b3;
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    let m = tile_row0 + tile_row + ir;
    if (m < M) {
      var bias_val: f32 = 0.0;
      if (params.has_bias != 0u) {
        bias_val = bias[m];
      }
      for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
        let n = tile_col0 + tile_col + ic;
        if (n < N_total) {
          let val = acc[ir][ic] + bias_val;
          let b_idx = n / spatial;
          let sp = n % spatial;
          out[(b_idx * params.OC + m) * spatial + sp] = val;
        }
      }
    }
  }
}
`;
