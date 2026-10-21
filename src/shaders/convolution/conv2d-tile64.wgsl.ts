/**
 * WGSL compute shader for 64x64 register-tiled 2D convolution GEMM (4x4 per thread across 256 threads).
 * High occupancy, coalesced 128-bit memory transactions, minimal register pressure for all GPUs.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_TILE64 = /* wgsl */ `
struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

// Geometry constants let every backend simplify division in the gather loop.
override kernel_h: u32 = 0u;
override kernel_w: u32 = 0u;

// Zero preserves uniform-driven execution for callers using the raw template.
fn kernel_height() -> u32 {
  return select(params.KH, kernel_h, kernel_h != 0u);
}
fn kernel_width() -> u32 {
  return select(params.KW, kernel_w, kernel_w != 0u);
}

// a_sub: 64 output channels x 16 K elements (4 vec4s) = 256 vec4s (4 KB)
var<workgroup> a_sub: array<array<vec4<f32>, 4>, 64>;
// b_sub: 16 K elements x 64 spatial elements (16 vec4s) = 256 vec4s (4 KB)
var<workgroup> b_sub: array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let tile_row0 = wg_id.y * 64u;
  let tile_col0 = wg_id.x * 64u;
  let M = params.OC;
  let N = params.B * params.OH * params.OW;
  let K = params.IC * kernel_height() * kernel_width();
  let K_vecs = K >> 2u;
  let khw = kernel_height() * kernel_width();
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

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  // Cooperative load coordinates for 256 threads (1 vec4 weight + 1 vec4 input per thread)
  let a_r = tid >> 2u;      // 0..63
  let a_c = tid & 3u;       // 0..3
  let a_m = tile_row0 + a_r;
  let a_valid = a_m < M;

  let b_r = tid >> 4u;      // 0..15 (row in K tile)
  let b_c4 = tid & 15u;     // 0..15 (col in spatial tile)
  let n_base = tile_col0 + (b_c4 << 2u);

  // Precompute invariant spatial coordinates for this thread's 4 input columns once
  var col_b: array<u32, 4>;
  var col_ih_base: array<i32, 4>;
  var col_iw_base: array<i32, 4>;
  var col_valid: array<bool, 4>;
  for (var kk = 0u; kk < 4u; kk = kk + 1u) {
    let n = n_base + kk;
    if (n < N) {
      col_valid[kk] = true;
      let b_val = n / ohw;
      let sp = n % ohw;
      col_b[kk] = b_val;
      let oh = sp / params.OW;
      let ow = sp % params.OW;
      col_ih_base[kk] = i32(oh) * sH - pH;
      col_iw_base[kk] = i32(ow) * sW - pW;
    } else {
      col_valid[kk] = false;
      col_b[kk] = 0u;
      col_ih_base[kk] = 0;
      col_iw_base[kk] = 0;
    }
  }

  // For 3x3 kernels, validity is invariant across input channels. Interior
  // columns get all nine bits; borders and spatial tails get only valid taps.
  var tap_mask: array<u32, 4>;
  if (kernel_height() == 3u && kernel_width() == 3u) {
    for (var col = 0u; col < 4u; col++) {
      var mask = 0u;
      if (col_valid[col]) {
        if (col_ih_base[col] >= 0 && col_iw_base[col] >= 0 &&
            col_ih_base[col] + 2 * dH < i32(params.IH) &&
            col_iw_base[col] + 2 * dW < i32(params.IW)) {
          mask = 0x1ffu;
        } else {
          for (var ky = 0u; ky < 3u; ky++) {
            let iy = col_ih_base[col] + i32(ky) * dH;
            for (var kx = 0u; kx < 3u; kx++) {
              let ix = col_iw_base[col] + i32(kx) * dW;
              if (iy >= 0 && iy < i32(params.IH) && ix >= 0 && ix < i32(params.IW)) {
                mask = mask | (1u << (ky * 3u + kx));
              }
            }
          }
        }
      }
      tap_mask[col] = mask;
    }
  }

  let num_tiles = (K + 15u) >> 4u;
  for (var t = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t << 4u;
    let k4_start = t << 2u;

    // 1. Coalesced 128-bit load of weights (256 threads load 256 vec4s)
    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = weight[a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f32>(0.0);
    }

    // 2. Coalesced load of input using precomputed spatial bases
    let kk_idx = k_start + b_r;
    var b_row = vec4<f32>(0.0);
    if (kk_idx < K) {
      let k_ic = kk_idx / khw;
      let rem = kk_idx % khw;
      let kh = i32(rem / kernel_width()) * dH;
      let kw = i32(rem % kernel_width()) * dW;

      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let ih = col_ih_base[kk] + kh;
        let iw = col_iw_base[kk] + kw;
        if ((kernel_height() == 3u && kernel_width() == 3u && (tap_mask[kk] & (1u << rem)) != 0u) ||
            ((kernel_height() != 3u || kernel_width() != 3u) && col_valid[kk] && ih >= 0 && ih < iH && iw >= 0 && iw < iW)) {
          let ch_idx = col_b[kk] * params.IC + k_ic;
          b_row[kk] = input[ch_idx * in_hw + u32(ih) * in_w + u32(iw)];
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

  // 4. Output write: 4 rows x 4 cols per thread
  let thread_r = local_id.y << 2u;
  let out_c_base = tile_col0 + (local_id.x << 2u);

  for (var ir = 0u; ir < 4u; ir = ir + 1u) {
    let m = tile_row0 + thread_r + ir;
    if (m < M) {
      var bias_val = 0.0;
      if (params.has_bias != 0u) {
        bias_val = bias[m];
      }
      var acc_v = acc0;
      if (ir == 1u) { acc_v = acc1; }
      else if (ir == 2u) { acc_v = acc2; }
      else if (ir == 3u) { acc_v = acc3; }
      acc_v = acc_v + vec4<f32>(bias_val);

      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let n = out_c_base + kk;
        if (n < N) {
          // Reuse the batch decoded for input gathering instead of dividing again.
          let out_base = col_b[kk] * (M - 1u) * ohw + n;
          out[m * ohw + out_base] = acc_v[kk];
        }
      }
    }
  }
}
`;
