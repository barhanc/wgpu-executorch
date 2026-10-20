import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.convolution.default';

/**
 * WGSL compute shader for direct 2D convolution (non-transposed).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Direct 2D convolution (non-transposed), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. Supports general stride/padding/dilation and
// groups. input [B,IC,IH,IW]; weight [OC, IC/groups, KH, KW]; bias [OC] (gated).
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
    let ic = ic0 + icg;
    let in_c_base = (b * params.IC + ic) * params.IH; // *IW added per-row below
    let w_c_base = (oc * icpg + icg) * params.KH; // *KW added per-row below
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= iH) {
        continue;
      }
      let in_row = (in_c_base + u32(ih)) * params.IW;
      let w_row = (w_c_base + kh) * params.KW;
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= iW) {
          continue;
        }
        acc = acc + input[in_row + u32(iw)] * weight[w_row + kw];
      }
    }
  }
  out[i] = acc;
}
`;

/**
 * WGSL compute shader for direct 2D convolution as implicit-im2col tiled GEMM.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV2D_GEMM = /* wgsl */ `
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

// Standard (groups==1) conv2d as an implicit-im2col tiled GEMM, reusing the
// linear_fp32_tiled skeleton (TILE=32, RPT=4, shared-mem tiles). M=OC, N=B*OH*OW,
// K=IC*KH*KW. A=weight [OC,K] (contiguous OIHW); B=input sampled on the fly (im2col),
// out-of-range -> 0.0 implements padding. bias is per-ROW (OC). Output written NCHW.
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

fn gK() -> u32 { return params.IC * params.KH * params.KW; }
fn gN() -> u32 { return params.B * params.OH * params.OW; }

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

  // Precompute spatial coords for this thread's 4 columns in B (invariant across K tiles)
  var col_b: array<u32, 4>;
  var col_valid: array<bool, 4>;
  var col_oh_arr: array<i32, 4>;
  var col_ow_arr: array<i32, 4>;
  for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
    let n = tile_col0 + local_id.x * RPT + kk;
    if (n < N) {
      col_valid[kk] = true;
      let b_val = n / ohw;
      let sp = n % ohw;
      col_b[kk] = b_val;
      col_oh_arr[kk] = i32(sp / params.OW);
      col_ow_arr[kk] = i32(sp % params.OW);
    } else {
      col_valid[kk] = false;
      col_b[kk] = 0u;
      col_oh_arr[kk] = 0;
      col_ow_arr[kk] = 0;
    }
  }

  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      let m = tile_row0 + arow;
      let kk_idx = k_start + arow;

      var k_valid = false;
      var k_ic = 0u;
      var k_kh = 0;
      var k_kw = 0;
      if (kk_idx < K) {
        k_valid = true;
        k_ic = kk_idx / khw;
        let rem = kk_idx % khw;
        k_kh = i32(rem / params.KW);
        k_kw = i32(rem % params.KW);
      }

      var b_row: vec4<f32>;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        let k_col = k_start + col;
        if (m < M && k_col < K) {
          a_sub[arow][col] = weight[m * K + k_col];
        } else {
          a_sub[arow][col] = 0.0;
        }

        if (k_valid && col_valid[kk]) {
          let ih = col_oh_arr[kk] * sH - pH + k_kh * dH;
          let iw = col_ow_arr[kk] * sW - pW + k_kw * dW;
          if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
            b_row[kk] = input[((col_b[kk] * params.IC + k_ic) * params.IH + u32(ih)) * params.IW + u32(iw)];
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
        bias_val = bias[m];
      }
      let v = acc[ir] + vec4<f32>(bias_val);
      for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
        let n = tile_col0 + tile_col + ic;
        if (n < N) {
          let b = col_b[ic];
          let oh = u32(col_oh_arr[ic]);
          let ow = u32(col_ow_arr[ic]);
          out[((b * params.OC + m) * params.OH + oh) * params.OW + ow] = v[ic];
        }
      }
    }
  }
}
`;

/**
 * WGSL compute shader for direct 2D 1x1 convolution (pointwise GEMM).
 * Avoids im2col overhead since 1x1 conv with stride 1, padding 0 is a direct matrix multiply.
 */
const SHADER_CONV2D_1X1 = /* wgsl */ `
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

/**
 * WGSL compute shader for direct 2D 1x1 convolution (vec4 vectorized GEMM).
 * Processes 128-bit vec4 memory transactions along channels (K) and spatial dimensions (N).
 */
const SHADER_CONV2D_1X1_VEC4 = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
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
const TILE4: u32 = 8u;

var<workgroup> a_sub: array<array<vec4<f32>, 8>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let M = params.OC;
  let spatial = params.OH * params.OW;
  let sp_vecs = spatial / 4u;
  let N_vecs = params.B * sp_vecs;
  let K = params.IC;
  let K_vecs = K / 4u;

  let tile_row0 = wg_id.y * TILE;
  let tile_col0_4 = wg_id.x * TILE4;
  let tile_row = local_id.y * RPT;
  let tile_col4 = local_id.x;

  var acc: array<vec4<f32>, 4>;
  acc[0] = vec4<f32>(0.0);
  acc[1] = vec4<f32>(0.0);
  acc[2] = vec4<f32>(0.0);
  acc[3] = vec4<f32>(0.0);

  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * TILE4;

    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      let m = tile_row0 + arow;
      let k4 = k4_start + local_id.x;
      if (m < M && k4 < K_vecs) {
        a_sub[arow][local_id.x] = weight[m * K_vecs + k4];
      } else {
        a_sub[arow][local_id.x] = vec4<f32>(0.0);
      }

      let b_k = t * TILE + arow;
      let n4 = tile_col0_4 + local_id.x;
      if (b_k < K && n4 < N_vecs) {
        let b_idx = n4 / sp_vecs;
        let sp4 = n4 % sp_vecs;
        b_sub[arow][local_id.x] = input[(b_idx * params.IC + b_k) * sp_vecs + sp4];
      } else {
        b_sub[arow][local_id.x] = vec4<f32>(0.0);
      }
    }
    workgroupBarrier();

    for (var k4: u32 = 0u; k4 < TILE4; k4 = k4 + 1u) {
      let b0 = b_sub[k4 * 4u + 0u][tile_col4];
      let b1 = b_sub[k4 * 4u + 1u][tile_col4];
      let b2 = b_sub[k4 * 4u + 2u][tile_col4];
      let b3 = b_sub[k4 * 4u + 3u][tile_col4];

      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let ac = a_sub[tile_row + ir][k4];
        acc[ir] = acc[ir] + b0 * ac.x + b1 * ac.y + b2 * ac.z + b3 * ac.w;
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    let m = tile_row0 + tile_row + ir;
    let n4 = tile_col0_4 + tile_col4;
    if (m < M && n4 < N_vecs) {
      var val = acc[ir];
      if (params.has_bias != 0u) {
        val = val + vec4<f32>(bias[m]);
      }
      let b_idx = n4 / sp_vecs;
      let sp4 = n4 % sp_vecs;
      out[(b_idx * params.OC + m) * sp_vecs + sp4] = val;
    }
  }
}
`;

/**
 * WGSL compute shader for 3x3 depthwise convolution.
 * Uses 2D spatial workgroup layout (16x16) and workgroup-shared cached 3x3 weights and bias.
 */
const SHADER_CONV2D_DEPTHWISE3X3 = /* wgsl */ `
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

var<workgroup> w_tile: array<f32, 9>;
var<workgroup> bias_val: f32;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let oc = wg_id.z % params.OC;
  let b = wg_id.z / params.OC;

  if (local_id.x < 3u && local_id.y < 3u) {
    let k_idx = local_id.y * 3u + local_id.x;
    w_tile[k_idx] = weight[oc * 9u + k_idx];
  }
  if (local_id.x == 0u && local_id.y == 0u) {
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

  for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
    let ih = ih0 + i32(kh) * i32(params.dH);
    if (ih >= 0 && ih < iH) {
      let row_offset = (in_base + u32(ih)) * params.IW;
      let w_offset = kh * 3u;
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
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

/**
 * WGSL compute shader for transposed 2D convolution (gather form).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_TRANSPOSE2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Transposed 2D convolution (gather form), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. weight layout = torch convT [IC, OC/groups,
// KH, KW] (NOT flipped). For each kernel tap (kh,kw): an input row ih
// contributes iff (oh + pH - kh*dH) is divisible by sH and ih in range (the
// scatter-inversion). CPU-derisked vs torch.conv_transpose2d to fp64 round-off
// (/tmp/convtr_derisk.py), incl. non-square spatial + non-square kernel.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;
  let oc_in_g = oc % ocpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    let num_h = i32(oh) + i32(params.pH) - i32(kh) * i32(params.dH);
    if (num_h % i32(params.sH) != 0) {
      continue;
    }
    let ih = num_h / i32(params.sH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
      let num_w = i32(ow) + i32(params.pW) - i32(kw) * i32(params.dW);
      if (num_w % i32(params.sW) != 0) {
        continue;
      }
      let iw = num_w / i32(params.sW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
        let ic = ic0 + icg;
        let in_idx =
        ((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw);
        // weight [IC, OC/groups, KH, KW]: index (ic, oc_in_g, kh, kw)
        let w_idx =
        ((ic * ocpg + oc_in_g) * params.KH + kh) * params.KW + kw;
        acc = acc + input[in_idx] * weight[w_idx];
      }
    }
  }
  out[i] = acc;
}
`;

/**
 * Positional argument tuple for `aten::convolution.default`:
 * `[in, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out]`
 */
export type ConvolutionArgs = readonly [
  inTensor: Tensor,
  weight: Tensor,
  bias: Tensor | undefined,
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  transposed: boolean,
  outputPadding: readonly number[],
  groups: number,
  outTensor: Tensor,
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `aten::convolution.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConvolutionArgs): void {
  const [
    inTensor,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    outputPadding,
    groups,
    outTensor,
  ] = args;

  if (inTensor.dtype !== 'float32' || weight.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (inTensor.shape.length !== 4 || weight.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input, weight, and output`);
  }

  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');
  const [opH, opW] = parseHw(outputPadding, 'output_padding');

  const [b, ic, ih, iw] = inTensor.shape as [number, number, number, number];
  const hasBias = bias !== undefined;

  let oc: number;
  let kh: number;
  let kw: number;

  if (transposed) {
    // weight layout for transposed conv: [IC, OC / groups, KH, KW]
    kh = weight.shape[2];
    kw = weight.shape[3];
    const ocpg = weight.shape[1];
    oc = ocpg * groups;

    if (weight.shape[0] !== ic) {
      throw new Error(`${name}: weight dim0 !== IC (${weight.shape[0]} != ${ic})`);
    }
    if (opH >= sH || opW >= sW) {
      throw new Error(`${name}: output_padding >= stride`);
    }

    const expectedOh = (ih - 1) * sH - 2 * pH + dH * (kh - 1) + opH + 1;
    const expectedOw = (iw - 1) * sW - 2 * pW + dW * (kw - 1) + opW + 1;
    if (outTensor.shape[2] !== expectedOh || outTensor.shape[3] !== expectedOw) {
      throw new Error(`${name}: expected output spatial [${expectedOh}, ${expectedOw}]`);
    }
  } else {
    // weight layout for direct conv: [OC, IC / groups, KH, KW]
    oc = weight.shape[0];
    kh = weight.shape[2];
    kw = weight.shape[3];

    if (outputPadding.some((v) => v !== 0)) {
      throw new Error(`${name}: non-zero output_padding unsupported for non-transposed conv`);
    }
    if (weight.shape[1] !== ic / groups) {
      throw new Error(`${name}: weight in-ch (${weight.shape[1]}) != IC/groups (${ic / groups})`);
    }

    const ohNum = ih + 2 * pH - dH * (kh - 1) - 1;
    const owNum = iw + 2 * pW - dW * (kw - 1) - 1;
    if (ohNum < 0 || owNum < 0) {
      throw new Error(`${name}: invalid geometry (kernel larger than input)`);
    }
    const expectedOh = Math.floor(ohNum / sH) + 1;
    const expectedOw = Math.floor(owNum / sW) + 1;
    if (outTensor.shape[2] !== expectedOh || outTensor.shape[3] !== expectedOw) {
      throw new Error(`${name}: expected output spatial [${expectedOh}, ${expectedOw}]`);
    }
  }

  const [, , oh, ow] = outTensor.shape as [number, number, number, number];

  const device = ctx.device;
  const wgSize = 256;

  // ConvParams (80 B): input/output geometry, kernel, stride, pad, dilation, flags
  const buffer = new ArrayBuffer(80);
  // prettier-ignore
  new Uint32Array(buffer).set([
    b, ic, ih, iw,
    oc, oh, ow,
    kh, kw,
    sH, sW,
    pH, pW,
    dH, dW,
    groups, hasBias ? 1 : 0,
  ]);

  const uniformBuffer = ctx.uniformBuffer(buffer);
  const biasBuffer = hasBias ? bias.buffer : weight.buffer;

  // Variant selection: 1x1 convolutions use the pointwise GEMM kernel (no im2col),
  // with 128-bit vec4 memory optimization when dimensions are divisible by 4.
  // General dense convolutions use implicit-im2col tiled GEMM, grouped/depthwise
  // convs through the direct kernel, and transposed convs through the gather kernel.
  const is1x1 =
    !transposed &&
    kh === 1 &&
    kw === 1 &&
    sH === 1 &&
    sW === 1 &&
    pH === 0 &&
    pW === 0 &&
    dH === 1 &&
    dW === 1;
  const is1x1Vec4 = is1x1 && ic % 4 === 0 && (oh * ow) % 4 === 0 && oc % 4 === 0;

  const isDepthwise3x3 =
    !transposed && groups === oc && ic === oc && weight.shape[1] === 1 && kh === 3 && kw === 3;

  let code = SHADER_CONV2D;
  if (transposed) {
    code = SHADER_CONV_TRANSPOSE2D;
  } else if (isDepthwise3x3) {
    code = SHADER_CONV2D_DEPTHWISE3X3;
  } else if (groups === 1) {
    if (is1x1Vec4) {
      code = SHADER_CONV2D_1X1_VEC4;
    } else if (is1x1) {
      code = SHADER_CONV2D_1X1;
    } else {
      code = SHADER_CONV2D_GEMM;
    }
  }

  let workgroupCountX: number;
  let workgroupCountY: number;
  let workgroupCountZ = 1;
  let constants: Record<string, number>;

  if (isDepthwise3x3) {
    // Spatial 16x16 workgroup per (batch, oc) channel slice
    workgroupCountX = Math.ceil(ow / 16);
    workgroupCountY = Math.ceil(oh / 16);
    workgroupCountZ = b * oc;
    constants = {};
  } else if (transposed || groups !== 1) {
    // Direct / transposed kernels: one thread per output element, 2D spill
    const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
    workgroupCountX = Math.min(totalWorkgroups, 65535);
    workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    constants = { wg_size: wgSize, stride_x: workgroupCountX * wgSize };
  } else {
    // GEMM kernel: fixed 8x8 workgroup covering a 32x32 output tile;
    // grid over (N = B*OH*OW cols, M = OC rows)
    const tile = 32;
    workgroupCountX = Math.ceil((b * oh * ow) / tile);
    workgroupCountY = Math.ceil(oc / tile);
    if (workgroupCountX > 65535 || workgroupCountY > 65535) {
      throw new Error(`${name}: GEMM tile grid exceeds dispatch limit`);
    }
    constants = {};
  }

  const bundle = createComputeBundle(
    device,
    code,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: weight.buffer },
      { binding: 3, buffer: biasBuffer },
      { binding: 4, buffer: uniformBuffer },
    ],
    constants
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
    workgroupCountZ,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::convolution.default`.
 *
 * Implements 2D dense and grouped convolutions as well as 2D transposed
 * convolutions. Dense convolutions are evaluated as implicit-im2col tiled GEMM
 * (1.1-2.4x over the direct kernel in the ExecuTorch reference), grouped and
 * depthwise convolutions via the direct kernel.
 */
export const convolution: Shader<ConvolutionArgs, typeof name> = {
  name,
  code: {
    conv2d: SHADER_CONV2D,
    conv2dGemm: SHADER_CONV2D_GEMM,
    conv2d1x1: SHADER_CONV2D_1X1,
    conv2d1x1Vec4: SHADER_CONV2D_1X1_VEC4,
    conv2dDepthwise3x3: SHADER_CONV2D_DEPTHWISE3X3,
    convTranspose2d: SHADER_CONV_TRANSPOSE2D,
  },
  recordIn,
};
