/**
 * WGSL compute shader for 64x64 register-tiled 2D convolution with clamp bounds in float16 (4x4 per thread across 256 threads).
 * High occupancy, coalesced 128-bit memory transactions, minimal register pressure for all GPUs.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_TILE64_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> t_out: array<f16>;
@group(0) @binding(1) var<storage, read> t_x: array<f16>;
@group(0) @binding(2) var<storage, read> t_weight: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> t_bias: array<f16>;

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

// a_sub: 64 output channels x 16 K elements (4 vec4s) = 256 vec4s (2 KB)
var<workgroup> a_sub: array<array<vec4<f16>, 4>, 64>;
// b_sub: 16 K elements x 64 spatial elements (16 vec4s) = 256 vec4s (2 KB)
var<workgroup> b_sub: array<array<vec4<f16>, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
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
  let K_vecs = K >> 2u;
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

  var acc0 = vec4<f16>(0.0h);
  var acc1 = vec4<f16>(0.0h);
  var acc2 = vec4<f16>(0.0h);
  var acc3 = vec4<f16>(0.0h);

  // Cooperative load coordinates for 256 threads (1 vec4 weight + 1 vec4 input per thread)
  let a_r = tid >> 2u;      // 0..63
  let a_c = tid & 3u;       // 0..3
  let a_m = tile_row0 + a_r;
  let a_valid = a_m < M;

  let b_r = tid >> 4u;      // 0..15 (row in K tile)
  let b_c4 = tid & 15u;     // 0..15 (col in spatial tile)
  let n_base = tile_col0 + (b_c4 << 2u);

  let num_tiles = (K + 15u) >> 4u;
  for (var t = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t << 4u;
    let k4_start = t << 2u;

    // 1. Coalesced 128-bit load of weights (256 threads load 256 vec4s)
    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = t_weight[a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f16>(0.0h);
    }

    // 2. Coalesced load of input (256 threads load 256 vec4s)
    let kk_idx = k_start + b_r;
    var b_row = vec4<f16>(0.0h);
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
    workgroupBarrier();

    // 3. Compute phase: 4x4 register accumulation per thread
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

  // 4. Output write: 4 rows x 4 cols per thread with clamp
  let thread_r = local_id.y << 2u;
  let out_c_base = tile_col0 + (local_id.x << 2u);
  let min_v = f16(params.output_min);
  let max_v = f16(params.output_max);

  for (var ir = 0u; ir < 4u; ir = ir + 1u) {
    let m = tile_row0 + thread_r + ir;
    if (m < M) {
      var bias_val = 0.0h;
      if (params.has_bias != 0u) {
        bias_val = t_bias[m];
      }
      var acc_v = acc0;
      if (ir == 1u) { acc_v = acc1; }
      else if (ir == 2u) { acc_v = acc2; }
      else if (ir == 3u) { acc_v = acc3; }
      acc_v = acc_v + vec4<f16>(bias_val);

      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let n = out_c_base + kk;
        if (n < N) {
          let b_val = n / ohw;
          let sp = n % ohw;
          let oh = sp / params.W_out;
          let ow = sp % params.W_out;
          let val = clamp(acc_v[kk], min_v, max_v);
          t_out[((b_val * params.OC + m) * params.H_out + oh) * params.W_out + ow] = val;
        }
      }
    }
  }
}
`;
