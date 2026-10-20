/**
 * WGSL compute shader for 64x64 register-tiled 1x1 convolution with clamp bounds (N = 1).
 * Features 2x higher arithmetic intensity, computing 8x8 elements per thread (64 elements).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_1X1_64X64 = /* wgsl */ `
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

var<workgroup> a_sub: array<array<vec4<f32>, 4>, 64>;
var<workgroup> b_sub: array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let M = params.OC;
  let N = params.H_out * params.W_out;
  let K = params.IC;
  let K_vecs = K / 4u;
  let N_vecs = N / 4u;

  let tile_row0 = wg_id.y * 64u;
  let tile_col0_4 = wg_id.x * 16u;
  let tid = local_id.y * 8u + local_id.x;

  var acc0: array<vec4<f32>, 8>;
  var acc1: array<vec4<f32>, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    acc0[i] = vec4<f32>(0.0);
    acc1[i] = vec4<f32>(0.0);
  }

  let num_tiles = (K + 15u) / 16u;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * 4u;
    let a_row = tid;
    let a_m = tile_row0 + a_row;
    for (var col = 0u; col < 4u; col = col + 1u) {
      if (a_m < M && (k4_start + col) < K_vecs) {
        a_sub[a_row][col] = t_weight[a_m * K_vecs + k4_start + col];
      } else {
        a_sub[a_row][col] = vec4<f32>(0.0);
      }
    }

    let b_r = tid / 4u;
    let b_c_base = (tid % 4u) * 4u;
    let b_k = t * 16u + b_r;
    for (var i = 0u; i < 4u; i = i + 1u) {
      let b_c = b_c_base + i;
      let b_n4 = tile_col0_4 + b_c;
      if (b_k < K && b_n4 < N_vecs) {
        b_sub[b_r][b_c] = t_x[b_k * N_vecs + b_n4];
      } else {
        b_sub[b_r][b_c] = vec4<f32>(0.0);
      }
    }
    workgroupBarrier();

    let thread_row = local_id.y * 8u;
    let thread_col4 = local_id.x * 2u;

    for (var k4: u32 = 0u; k4 < 4u; k4 = k4 + 1u) {
      let b0_0 = b_sub[k4 * 4u + 0u][thread_col4];
      let b0_1 = b_sub[k4 * 4u + 0u][thread_col4 + 1u];

      let b1_0 = b_sub[k4 * 4u + 1u][thread_col4];
      let b1_1 = b_sub[k4 * 4u + 1u][thread_col4 + 1u];

      let b2_0 = b_sub[k4 * 4u + 2u][thread_col4];
      let b2_1 = b_sub[k4 * 4u + 2u][thread_col4 + 1u];

      let b3_0 = b_sub[k4 * 4u + 3u][thread_col4];
      let b3_1 = b_sub[k4 * 4u + 3u][thread_col4 + 1u];

      for (var ir = 0u; ir < 8u; ir = ir + 1u) {
        let ac = a_sub[thread_row + ir][k4];
        acc0[ir] = acc0[ir] + b0_0 * ac.x + b1_0 * ac.y + b2_0 * ac.z + b3_0 * ac.w;
        acc1[ir] = acc1[ir] + b0_1 * ac.x + b1_1 * ac.y + b2_1 * ac.z + b3_1 * ac.w;
      }
    }
    workgroupBarrier();
  }

  let thread_row = local_id.y * 8u;
  let col4 = tile_col0_4 + local_id.x * 2u;
  for (var ir = 0u; ir < 8u; ir = ir + 1u) {
    let r = tile_row0 + thread_row + ir;
    if (r < M) {
      var bias_val = vec4<f32>(0.0);
      if (params.has_bias != 0u) {
        bias_val = vec4<f32>(t_bias[r]);
      }
      let min_v = vec4<f32>(params.output_min);
      let max_v = vec4<f32>(params.output_max);
      if (col4 < N_vecs) {
        t_out[r * N_vecs + col4] = clamp(acc0[ir] + bias_val, min_v, max_v);
      }
      if (col4 + 1u < N_vecs) {
        t_out[r * N_vecs + col4 + 1u] = clamp(acc1[ir] + bias_val, min_v, max_v);
      }
    }
  }
}
`;
