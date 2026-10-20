/**
 * WGSL compute shader for 64x64 register-tiled 2D convolution with clamp bounds.
 *
 * Evaluates dense convolution out = clamp(conv2d(input, weight, bias), min, max)
 * using 2D register tiling (8x8 elements per thread across 64 threads per workgroup).
 * Each workgroup computes a 64x64 output block (64 output channels x 64 spatial elements)
 * over K = IC * Kh * Kw, reading weight in 128-bit vec4 transactions.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_TILE64 = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> t_out: array<f32>;
@group(0) @binding(1) var<storage, read> t_x: array<f32>;
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

// a_sub: 64 output channels x 16 K-elements (4 vec4s)
var<workgroup> a_sub: array<array<vec4<f32>, 4>, 64>;
// b_sub: 16 K-elements x 64 spatial elements (16 vec4s)
var<workgroup> b_sub: array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let tile_row0 = wg_id.y * 64u;
  let tile_col0 = wg_id.x * 64u;
  let M = params.OC;
  let N = params.N * params.H_out * params.W_out;
  let K = params.IC * params.Kh * params.Kw;
  let K_vecs = K / 4u;
  let khw = params.Kh * params.Kw;
  let ohw = params.H_out * params.W_out;

  let iH = i32(params.H_in);
  let iW = i32(params.W_in);
  let in_hw = params.H_in * params.W_in;
  let in_w = params.W_in;
  let sH = i32(params.stride_h);
  let sW = i32(params.stride_w);
  let pH = i32(params.pad_h);
  let pW = i32(params.pad_w);
  let dH = i32(params.dil_h);
  let dW = i32(params.dil_w);

  var acc0: array<vec4<f32>, 8>;
  var acc1: array<vec4<f32>, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    acc0[i] = vec4<f32>(0.0);
    acc1[i] = vec4<f32>(0.0);
  }

  let num_tiles = (K + 15u) / 16u;
  for (var t = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * 16u;
    let k4_start = t * 4u;

    // 1. Load a_sub (weights): each thread tid (0..63) loads row tid (4 vec4s)
    let a_m = tile_row0 + tid;
    let a_valid = a_m < M;
    let a_base = a_m * K_vecs + k4_start;
    for (var col = 0u; col < 4u; col = col + 1u) {
      if (a_valid && (k4_start + col) < K_vecs) {
        a_sub[tid][col] = t_weight[a_base + col];
      } else {
        a_sub[tid][col] = vec4<f32>(0.0);
      }
    }

    // 2. Load b_sub (implicit im2col input): 16 rows x 16 vec4s = 256 vec4s total
    // 64 threads -> 4 vec4s per thread
    let b_vec_idx = tid << 2u;
    for (var i = 0u; i < 4u; i = i + 1u) {
      let flat_idx = b_vec_idx + i;
      let b_r = flat_idx >> 4u;
      let b_c4 = flat_idx & 15u;
      let kk_idx = k_start + b_r;
      let n_base = tile_col0 + (b_c4 << 2u);

      var b_row = vec4<f32>(0.0);
      if (kk_idx < K) {
        let k_ic = kk_idx / khw;
        let rem = kk_idx % khw;
        let kh = i32(rem / params.Kw) * dH;
        let kw = i32(rem % params.Kw) * dW;

        for (var kk = 0u; kk < 4u; kk = kk + 1u) {
          let n = n_base + kk;
          if (n < N) {
            let b_val = n / ohw;
            let sp = n % ohw;
            let oh = sp / params.W_out;
            let ow = sp % params.W_out;
            let ih = i32(oh) * sH - pH + kh;
            let iw = i32(ow) * sW - pW + kw;
            if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
              let ch_idx = b_val * params.IC + k_ic;
              b_row[kk] = t_x[ch_idx * in_hw + u32(ih) * in_w + u32(iw)];
            }
          }
        }
      }
      b_sub[b_r][b_c4] = b_row;
    }

    workgroupBarrier();

    // 3. Compute phase: 8x8 register accumulation per thread
    let thread_row = local_id.y * 8u;
    let thread_col4 = local_id.x * 2u;

    for (var k4 = 0u; k4 < 4u; k4 = k4 + 1u) {
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

  // 4. Write output with clamp bounds
  let thread_row = local_id.y * 8u;
  let thread_col4 = local_id.x * 2u;
  let clamp_min = vec4<f32>(params.output_min);
  let clamp_max = vec4<f32>(params.output_max);

  for (var ir = 0u; ir < 8u; ir = ir + 1u) {
    let m = tile_row0 + thread_row + ir;
    if (m < M) {
      var b_val = 0.0;
      if (params.has_bias != 0u) {
        b_val = t_bias[m];
      }
      let v0 = clamp(acc0[ir] + vec4<f32>(b_val), clamp_min, clamp_max);
      let v1 = clamp(acc1[ir] + vec4<f32>(b_val), clamp_min, clamp_max);

      let n_base0 = tile_col0 + thread_col4 * 4u;
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let n = n_base0 + kk;
        if (n < N) {
          let b_val = n / ohw;
          let sp = n % ohw;
          let oh = sp / params.W_out;
          let ow = sp % params.W_out;
          t_out[((b_val * params.OC + m) * params.H_out + oh) * params.W_out + ow] = v0[kk];
        }
      }

      let n_base1 = n_base0 + 4u;
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let n = n_base1 + kk;
        if (n < N) {
          let b_val = n / ohw;
          let sp = n % ohw;
          let oh = sp / params.W_out;
          let ow = sp % params.W_out;
          t_out[((b_val * params.OC + m) * params.H_out + oh) * params.W_out + ow] = v1[kk];
        }
      }
    }
  }
}
`;
