/**
 * WGSL compute shader for 32x64 register-tiled 2D convolution with clamped activation in FP16 (4x4 per thread across 128 threads).
 * Optimized for small channel counts (OC <= 32) to maximize wave occupancy and eliminate thread divergence.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_TILE32_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> t_out: array<f16>;
@group(0) @binding(1) var<storage, read> t_in: array<f16>;
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

// Geometry constants let every backend simplify division in the gather loop.
override kernel_h: u32 = 0u;
override kernel_w: u32 = 0u;

// Zero preserves uniform-driven execution for callers using the raw template.
fn kernel_height() -> u32 {
  return select(params.Kh, kernel_h, kernel_h != 0u);
}
fn kernel_width() -> u32 {
  return select(params.Kw, kernel_w, kernel_w != 0u);
}

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
  let N = params.N * params.H_out * params.W_out;
  let K = params.IC * kernel_height() * kernel_width();
  let K_vecs = K >> 2u;
  let khw = kernel_height() * kernel_width();
  let ohw = params.H_out * params.W_out;

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
      let oh = sp / params.W_out;
      let ow = sp % params.W_out;
      col_ih_base0[kk] = i32(oh) * sH - pH;
      col_iw_base0[kk] = i32(ow) * sW - pW;
    } else {
      col_valid0[kk] = false;
      col_b0[kk] = 0u;
      col_ih_base0[kk] = 0;
      col_iw_base0[kk] = 0;
    }
  }

  // For 3x3 kernels, validity is invariant across input channels. Interior
  // columns get all nine bits; borders and spatial tails get only valid taps.
  var tap_mask: array<u32, 4>;
  if (kernel_height() == 3u && kernel_width() == 3u) {
    for (var col = 0u; col < 4u; col++) {
      var mask = 0u;
      if (col_valid0[col]) {
        if (col_ih_base0[col] >= 0 && col_iw_base0[col] >= 0 &&
            col_ih_base0[col] + 2 * dH < i32(params.H_in) &&
            col_iw_base0[col] + 2 * dW < i32(params.W_in)) {
          mask = 0x1ffu;
        } else {
          for (var ky = 0u; ky < 3u; ky++) {
            let iy = col_ih_base0[col] + i32(ky) * dH;
            for (var kx = 0u; kx < 3u; kx++) {
              let ix = col_iw_base0[col] + i32(kx) * dW;
              if (iy >= 0 && iy < i32(params.H_in) && ix >= 0 && ix < i32(params.W_in)) {
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
    let k4_start = t << 2u;
    let k_start = t << 4u;

    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = t_weight[a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f16>(0.0h);
    }

    let bk0 = k_start + b_r0;
    if (bk0 < K) {
      let ic0 = bk0 / khw;
      let rem0 = bk0 % khw;
      let kh0 = i32(rem0 / kernel_width()) * dH;
      let kw0 = i32(rem0 % kernel_width()) * dW;

      var b_row0 = vec4<f16>(0.0h);
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let ih0 = col_ih_base0[kk] + kh0;
        let iw0 = col_iw_base0[kk] + kw0;
        if ((kernel_height() == 3u && kernel_width() == 3u && (tap_mask[kk] & (1u << rem0)) != 0u) ||
            ((kernel_height() != 3u || kernel_width() != 3u) && col_valid0[kk] && ih0 >= 0 && ih0 < i32(params.H_in) && iw0 >= 0 && iw0 < i32(params.W_in))) {
          let in_idx0 = (col_b0[kk] * params.IC + ic0) * in_hw + u32(ih0) * in_w + u32(iw0);
          b_row0[kk] = t_in[in_idx0];
        }
      }
      b_sub[b_r0][b_c4_0] = b_row0;
    } else {
      b_sub[b_r0][b_c4_0] = vec4<f16>(0.0h);
    }

    let bk1 = k_start + b_r1;
    if (bk1 < K) {
      let ic1 = bk1 / khw;
      let rem1 = bk1 % khw;
      let kh1 = i32(rem1 / kernel_width()) * dH;
      let kw1 = i32(rem1 % kernel_width()) * dW;

      var b_row1 = vec4<f16>(0.0h);
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let ih1 = col_ih_base0[kk] + kh1;
        let iw1 = col_iw_base0[kk] + kw1;
        if ((kernel_height() == 3u && kernel_width() == 3u && (tap_mask[kk] & (1u << rem1)) != 0u) ||
            ((kernel_height() != 3u || kernel_width() != 3u) && col_valid0[kk] && ih1 >= 0 && ih1 < i32(params.H_in) && iw1 >= 0 && iw1 < i32(params.W_in))) {
          let in_idx1 = (col_b0[kk] * params.IC + ic1) * in_hw + u32(ih1) * in_w + u32(iw1);
          b_row1[kk] = t_in[in_idx1];
        }
      }
      b_sub[b_r1][b_c4_1] = b_row1;
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
          // Reuse the batch decoded for input gathering instead of dividing again.
          let out_base = col_b0[kk] * (M - 1u) * ohw + n;
          let val = clamp(acc_v[kk], min_v, max_v);
          t_out[m * ohw + out_base] = val;
        }
      }
    }
  }
}
`;
