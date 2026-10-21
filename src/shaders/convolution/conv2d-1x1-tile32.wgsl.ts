/**
 * WGSL compute shader for 32x64 register-tiled 1x1 convolution (B = 1, 4x4 per thread across 128 threads).
 * Optimized for small channel counts (OC <= 32) to maximize wave occupancy and eliminate thread divergence.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_1X1_TILE32 = /* wgsl */ `
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

// 32 rows x 16 K elements (4 vec4s) = 128 vec4s (2 KB)
var<workgroup> a_sub: array<array<vec4<f32>, 4>, 32>;
// 16 K rows x 64 N elements (16 vec4s) = 256 vec4s (4 KB)
var<workgroup> b_sub: array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(16, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let M = params.OC;
  let N = params.OH * params.OW;
  let K = params.IC;
  let K_vecs = K >> 2u;
  let N_vecs = N >> 2u;

  let tile_row0 = wg_id.y * 32u;
  let tile_col0_4 = wg_id.x * 16u;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  // Load A coordinates for 128 threads (each thread loads 1 vec4 of weight)
  let a_r = tid >> 2u;      // 0..31
  let a_c = tid & 3u;       // 0..3
  let a_m = tile_row0 + a_r;
  let a_valid = a_m < M;

  // Load B coordinates for 128 threads (each thread loads 2 vec4s of input)
  let b_r0 = tid >> 4u;     // 0..7
  let b_c0 = tid & 15u;     // 0..15
  let b_n4_0 = tile_col0_4 + b_c0;
  let b_c0_valid = b_n4_0 < N_vecs;

  let b_r1 = b_r0 + 8u;     // 8..15
  let b_c1 = b_c0;
  let b_n4_1 = b_n4_0;
  let b_c1_valid = b_c0_valid;

  let num_tiles = (K + 15u) >> 4u;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t << 2u;
    let k_start = t << 4u;

    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = weight[a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f32>(0.0);
    }

    let bk0 = k_start + b_r0;
    if (bk0 < K && b_c0_valid) {
      b_sub[b_r0][b_c0] = input[bk0 * N_vecs + b_n4_0];
    } else {
      b_sub[b_r0][b_c0] = vec4<f32>(0.0);
    }

    let bk1 = k_start + b_r1;
    if (bk1 < K && b_c1_valid) {
      b_sub[b_r1][b_c1] = input[bk1 * N_vecs + b_n4_1];
    } else {
      b_sub[b_r1][b_c1] = vec4<f32>(0.0);
    }

    workgroupBarrier();

    let thread_r = local_id.y << 2u; // 0..28 (step 4)
    let thread_c4 = local_id.x;      // 0..15

    for (var k4: u32 = 0u; k4 < 4u; k4 = k4 + 1u) {
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
  let out_r0 = tile_row0 + thread_r;
  let out_c4 = tile_col0_4 + local_id.x;

  if (out_c4 < N_vecs) {
    var b_val = vec4<f32>(0.0);
    if (params.has_bias != 0u) {
      if (out_r0 + 0u < M) { b_val.x = bias[out_r0 + 0u]; }
      if (out_r0 + 1u < M) { b_val.y = bias[out_r0 + 1u]; }
      if (out_r0 + 2u < M) { b_val.z = bias[out_r0 + 2u]; }
      if (out_r0 + 3u < M) { b_val.w = bias[out_r0 + 3u]; }
    }

    if (out_r0 + 0u < M) { out[(out_r0 + 0u) * N_vecs + out_c4] = acc0 + vec4<f32>(b_val.x); }
    if (out_r0 + 1u < M) { out[(out_r0 + 1u) * N_vecs + out_c4] = acc1 + vec4<f32>(b_val.y); }
    if (out_r0 + 2u < M) { out[(out_r0 + 2u) * N_vecs + out_c4] = acc2 + vec4<f32>(b_val.z); }
    if (out_r0 + 3u < M) { out[(out_r0 + 3u) * N_vecs + out_c4] = acc3 + vec4<f32>(b_val.w); }
  }
}
`;
