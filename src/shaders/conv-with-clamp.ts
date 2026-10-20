import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'et_vk.conv_with_clamp.default';

/**
 * WGSL compute shader for fused convolution with clamp bounds.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/conv_with_clamp/conv_with_clamp.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/conv_with_clamp/conv_with_clamp.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_WITH_CLAMP = /* wgsl */ `
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

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.numel) {
    return;
  }
  // Unravel the NCHW output index -> (n, oc, oh, ow).
  let ow = idx % params.W_out;
  var r = idx / params.W_out;
  let oh = r % params.H_out;
  r = r / params.H_out;
  let oc = r % params.OC;
  let n = r / params.OC;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = t_bias[oc];
  }
  // Grouped conv: output channel oc belongs to group g and dots only that
  // group's ic_per_group input channels. weight is [OC, IC/groups, Kh, Kw];
  // groups==1 (ic_per_group==IC, g==0) is the general dense case.
  let oc_per_group = params.OC / params.groups;
  let g = oc / oc_per_group;
  let ic_base = g * params.ic_per_group;
  for (var ic_local: u32 = 0u; ic_local < params.ic_per_group; ic_local = ic_local + 1u) {
    let ic = ic_base + ic_local;
    for (var kh: u32 = 0u; kh < params.Kh; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.stride_h) - i32(params.pad_h) +
      i32(kh) * i32(params.dil_h);
      if (ih < 0 || ih >= i32(params.H_in)) {
        continue;
      }
      let in_row = ((n * params.IC + ic) * params.H_in + u32(ih)) * params.W_in;
      let w_row =
      ((oc * params.ic_per_group + ic_local) * params.Kh + kh) * params.Kw;
      for (var kw: u32 = 0u; kw < params.Kw; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.stride_w) - i32(params.pad_w) +
        i32(kw) * i32(params.dil_w);
        if (iw < 0 || iw >= i32(params.W_in)) {
          continue;
        }
        acc = acc + t_x[in_row + u32(iw)] * t_weight[w_row + kw];
      }
    }
  }
  t_out[idx] = clamp(acc, params.output_min, params.output_max);
}
`;

/**
 * WGSL compute shader for fused convolution with clamp bounds as
 * implicit-im2col tiled GEMM (groups == 1).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_WITH_CLAMP_GEMM = /* wgsl */ `
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
var<workgroup> b_sub: array<array<f32, 32>, 32>;

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

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
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
      col_oh_arr[kk] = i32(sp / params.W_out);
      col_ow_arr[kk] = i32(sp % params.W_out);
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
        k_kh = i32(rem / params.Kw);
        k_kw = i32(rem % params.Kw);
      }

      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        let k_col = k_start + col;
        if (m < M && k_col < K) {
          a_sub[arow][col] = t_weight[m * K + k_col];
        } else {
          a_sub[arow][col] = 0.0;
        }

        if (k_valid && col_valid[kk]) {
          let ih = col_oh_arr[kk] * sH - pH + k_kh * dH;
          let iw = col_ow_arr[kk] * sW - pW + k_kw * dW;
          if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
            b_sub[arow][col] = t_x[((col_b[kk] * params.IC + k_ic) * params.H_in + u32(ih)) * params.W_in + u32(iw)];
          } else {
            b_sub[arow][col] = 0.0;
          }
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
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let m = tile_row0 + tile_row + ir; // oc
      let n = tile_col0 + tile_col + ic; // spatial index
      if (m < M && n < N) {
        var val = acc[ir][ic];
        if (params.has_bias != 0u) {
          val = val + t_bias[m];
        }
        let ohw = params.H_out * params.W_out;
        let b = n / ohw;
        let sp = n % ohw;
        let oh = sp / params.W_out;
        let ow = sp % params.W_out;
        t_out[((b * params.OC + m) * params.H_out + oh) * params.W_out + ow] =
        clamp(val, params.output_min, params.output_max);
      }
    }
  }
}
`;

/**
 * WGSL compute shader for fused 1x1 convolution with clamp bounds (pointwise GEMM).
 * Avoids im2col overhead since 1x1 conv with stride 1, padding 0 is a direct matrix multiply.
 */
const SHADER_CONV_WITH_CLAMP_1X1 = /* wgsl */ `
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
  let spatial = params.H_out * params.W_out;
  let N_total = params.N * spatial;
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
          a_sub[arow][col] = t_weight[m * K + k];
        } else {
          a_sub[arow][col] = 0.0;
        }

        let b_k = k_start + arow;
        let n = tile_col0 + col;
        if (b_k < K && n < N_total) {
          let b_idx = n / spatial;
          let sp = n % spatial;
          b_sub[arow][col] = t_x[(b_idx * params.IC + b_k) * spatial + sp];
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
        bias_val = t_bias[m];
      }
      for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
        let n = tile_col0 + tile_col + ic;
        if (n < N_total) {
          let val = acc[ir][ic] + bias_val;
          let b_idx = n / spatial;
          let sp = n % spatial;
          t_out[(b_idx * params.OC + m) * spatial + sp] = clamp(val, params.output_min, params.output_max);
        }
      }
    }
  }
}
`;

/**
 * WGSL compute shader for fused 1x1 convolution with clamp bounds (vec4 vectorized GEMM).
 * Processes 128-bit vec4 memory transactions along channels (K) and spatial dimensions (N).
 */
const SHADER_CONV_WITH_CLAMP_1X1_VEC4 = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> t_out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> t_x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> t_weight: array<vec4<f32>>;
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
  let spatial = params.H_out * params.W_out;
  let sp_vecs = spatial / 4u;
  let N_vecs = params.N * sp_vecs;
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
        a_sub[arow][local_id.x] = t_weight[m * K_vecs + k4];
      } else {
        a_sub[arow][local_id.x] = vec4<f32>(0.0);
      }

      let b_k = t * TILE + arow;
      let n4 = tile_col0_4 + local_id.x;
      if (b_k < K && n4 < N_vecs) {
        let b_idx = n4 / sp_vecs;
        let sp4 = n4 % sp_vecs;
        b_sub[arow][local_id.x] = t_x[(b_idx * params.IC + b_k) * sp_vecs + sp4];
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
        val = val + vec4<f32>(t_bias[m]);
      }
      val = clamp(val, vec4<f32>(params.output_min), vec4<f32>(params.output_max));
      let b_idx = n4 / sp_vecs;
      let sp4 = n4 % sp_vecs;
      t_out[(b_idx * params.OC + m) * sp_vecs + sp4] = val;
    }
  }
}
`;

/**
 * WGSL compute shader for fused 3x3 depthwise convolution with clamp bounds.
 * Uses 2D spatial workgroup layout (16x16) and workgroup-shared cached 3x3 weights and bias.
 */
const SHADER_CONV_WITH_CLAMP_DEPTHWISE3X3 = /* wgsl */ `
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
    w_tile[k_idx] = t_weight[oc * 9u + k_idx];
  }
  if (local_id.x == 0u && local_id.y == 0u) {
    if (params.has_bias != 0u) {
      bias_val = t_bias[oc];
    } else {
      bias_val = 0.0;
    }
  }
  workgroupBarrier();

  let ow = gid.x;
  let oh = gid.y;

  if (ow >= params.W_out || oh >= params.H_out || b >= params.N) {
    return;
  }

  var acc = bias_val;
  let ih0 = i32(oh) * i32(params.stride_h) - i32(params.pad_h);
  let iw0 = i32(ow) * i32(params.stride_w) - i32(params.pad_w);
  let iH = i32(params.H_in);
  let iW = i32(params.W_in);
  let in_base = (b * params.IC + oc) * params.H_in;

  for (var kh: u32 = 0u; kh < 3u; kh = kh + 1u) {
    let ih = ih0 + i32(kh) * i32(params.dil_h);
    if (ih >= 0 && ih < iH) {
      let row_offset = (in_base + u32(ih)) * params.W_in;
      let w_offset = kh * 3u;
      for (var kw: u32 = 0u; kw < 3u; kw = kw + 1u) {
        let iw = iw0 + i32(kw) * i32(params.dil_w);
        if (iw >= 0 && iw < iW) {
          acc = acc + t_x[row_offset + u32(iw)] * w_tile[w_offset + kw];
        }
      }
    }
  }

  let out_idx = ((b * params.OC + oc) * params.H_out + oh) * params.W_out + ow;
  t_out[out_idx] = clamp(acc, params.output_min, params.output_max);
}
`;

/**
 * Positional argument tuple for `et_vk::conv_with_clamp.default`:
 * `[in, weight, bias, stride, padding, dilation, transposed, output_padding, groups, min, max, out]`
 */
export type ConvWithClampArgs = readonly [
  inTensor: Tensor,
  weight: Tensor,
  bias: Tensor | undefined,
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  transposed: boolean,
  outputPadding: readonly number[],
  groups: number,
  outputMin: number | undefined,
  outputMax: number | undefined,
  outTensor: Tensor,
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `et_vk::conv_with_clamp.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConvWithClampArgs): void {
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
    outputMin,
    outputMax,
    outTensor,
  ] = args;

  if (inTensor.dtype !== 'float32' || weight.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (inTensor.shape.length !== 4 || weight.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input, weight, and output`);
  }
  if (transposed) {
    throw new Error(`${name}: transposed conv is unsupported in conv_with_clamp`);
  }
  if (groups < 1) {
    throw new Error(`${name}: groups must be >= 1`);
  }

  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');

  if (sH <= 0 || sW <= 0) {
    throw new Error(`${name}: stride must be positive`);
  }

  const [n, ic, hIn, wIn] = inTensor.shape as [number, number, number, number];
  const oc = weight.shape[0];
  const kh = weight.shape[2];
  const kw = weight.shape[3];

  if (outputPadding.some((v) => v !== 0)) {
    throw new Error(`${name}: non-zero output_padding unsupported for non-transposed conv`);
  }

  const icPerGroup = weight.shape[1];
  if (ic % groups !== 0 || oc % groups !== 0 || icPerGroup * groups !== ic) {
    throw new Error(`${name}: bad shape (IC/groups mismatch)`);
  }

  const hEff = hIn + 2 * pH - dH * (kh - 1) - 1;
  const wEff = wIn + 2 * pW - dW * (kw - 1) - 1;
  if (hEff < 0 || wEff < 0) {
    throw new Error(`${name}: invalid geometry (kernel larger than input)`);
  }

  const expectedHout = Math.floor(hEff / sH) + 1;
  const expectedWout = Math.floor(wEff / sW) + 1;
  if (outTensor.shape[2] !== expectedHout || outTensor.shape[3] !== expectedWout) {
    throw new Error(`${name}: output dims inconsistent with conv2d formula`);
  }

  const hasBias = bias !== undefined;
  if (hasBias && (bias.buffer === null || bias.shape[0] !== oc)) {
    throw new Error(`${name}: bias must be fp32 [OC]`);
  }

  const minVal = outputMin !== undefined ? outputMin : -Infinity;
  const maxVal = outputMax !== undefined ? outputMax : Infinity;

  const numel = outTensor.numel;
  const device = ctx.device;
  const wgSize = 64;

  const buffer = new ArrayBuffer(96);
  const u32View = new Uint32Array(buffer);
  const f32View = new Float32Array(buffer);

  // ConvWithClampParams (96 B): geometry, kernel, stride, pad, dilation, flags
  // prettier-ignore
  u32View.set([
    n, ic, hIn, wIn,
    oc, expectedHout, expectedWout,
    kh, kw,
    sH, sW,
    pH, pW,
    dH, dW,
    hasBias ? 1 : 0, numel,
    groups, icPerGroup,
    0, 0, 0,
  ]);
  f32View[22] = minVal; // output_min
  f32View[23] = maxVal; // output_max

  const uniformBuffer = ctx.uniformBuffer(buffer);
  const biasBuffer = hasBias ? bias.buffer : weight.buffer;

  // Variant selection: 1x1 convolutions use the pointwise GEMM kernel (no im2col),
  // with 128-bit vec4 memory optimization when dimensions are divisible by 4.
  // General dense convolutions use implicit-im2col tiled GEMM, and
  // grouped/depthwise convs use the direct kernel.
  const is1x1 =
    kh === 1 && kw === 1 && sH === 1 && sW === 1 && pH === 0 && pW === 0 && dH === 1 && dW === 1;
  const is1x1Vec4 =
    is1x1 && ic % 4 === 0 && (expectedHout * expectedWout) % 4 === 0 && oc % 4 === 0;

  const isDepthwise3x3 = groups === oc && ic === oc && icPerGroup === 1 && kh === 3 && kw === 3;

  let code = SHADER_CONV_WITH_CLAMP;
  if (isDepthwise3x3) {
    code = SHADER_CONV_WITH_CLAMP_DEPTHWISE3X3;
  } else if (groups === 1) {
    if (is1x1Vec4) {
      code = SHADER_CONV_WITH_CLAMP_1X1_VEC4;
    } else if (is1x1) {
      code = SHADER_CONV_WITH_CLAMP_1X1;
    } else {
      code = SHADER_CONV_WITH_CLAMP_GEMM;
    }
  }

  let workgroupCountX: number;
  let workgroupCountY: number;
  let workgroupCountZ = 1;
  let constants: Record<string, number>;

  if (isDepthwise3x3) {
    // Spatial 16x16 workgroup per (batch, oc) channel slice
    workgroupCountX = Math.ceil(expectedWout / 16);
    workgroupCountY = Math.ceil(expectedHout / 16);
    workgroupCountZ = n * oc;
    constants = {};
  } else if (groups !== 1) {
    // Direct kernel: one thread per output element, 2D spill
    const totalWorkgroups = Math.ceil(numel / wgSize);
    workgroupCountX = Math.min(totalWorkgroups, 65535);
    workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    constants = { wg_size: wgSize };
  } else {
    // GEMM kernel: fixed 8x8 workgroup covering a 32x32 output tile;
    // grid over (N = batch*H_out*W_out cols, M = OC rows)
    const tile = 32;
    workgroupCountX = Math.ceil((n * expectedHout * expectedWout) / tile);
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
 * WebGPU compute kernel for ExecuTorch `et_vk::conv_with_clamp.default`.
 *
 * Implements 2D direct and grouped convolutions with clamping. Dense
 * convolutions are evaluated as implicit-im2col tiled GEMM, grouped and
 * depthwise convolutions via the direct kernel.
 */
export const convWithClamp: Shader<ConvWithClampArgs, typeof name> = {
  name,
  code: {
    convWithClamp: SHADER_CONV_WITH_CLAMP,
    convWithClampGemm: SHADER_CONV_WITH_CLAMP_GEMM,
    convWithClamp1x1: SHADER_CONV_WITH_CLAMP_1X1,
    convWithClamp1x1Vec4: SHADER_CONV_WITH_CLAMP_1X1_VEC4,
    convWithClampDepthwise3x3: SHADER_CONV_WITH_CLAMP_DEPTHWISE3X3,
  },
  recordIn,
};
