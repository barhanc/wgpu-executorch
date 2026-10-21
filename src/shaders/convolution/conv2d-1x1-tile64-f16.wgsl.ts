/**
 * WGSL compute shader for 64x64 register-tiled 1x1 convolution in float16 (B = 1, 4x4 per thread across 256 threads).
 * High occupancy, coalesced 128-bit memory transactions, minimal register pressure for all GPUs.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_1X1_64X64_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> input: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// 64 rows x 16 K elements (4 vec4s) = 256 vec4s (2 KB)
var<workgroup> a_sub: array<array<vec4<f16>, 4>, 64>;
// 16 K rows x 64 N elements (16 vec4s) = 256 vec4s (2 KB)
var<workgroup> b_sub: array<array<vec4<f16>, 16>, 16>;

override fuse_silu: bool = false;

// Preserve the convolution output and sigmoid's storage precision at the fusion boundary.
fn conv_epilogue(value: vec4<f16>) -> vec4<f16> {
  if (!fuse_silu) { return value; }
  let stored = vec4<f16>(quantizeToF16(vec4<f32>(value)));
  let gate = vec4<f16>(quantizeToF16(vec4<f32>(vec4<f16>(1.0) / (vec4<f16>(1.0) + exp(-stored)))));
  return stored * gate;
}

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let M = params.OC;
  let N = params.OH * params.OW;
  let K = params.IC;
  let K_vecs = K >> 2u;
  let N_vecs = N >> 2u;

  let tile_row0 = wg_id.y * 64u;
  let tile_col0_4 = wg_id.x * 16u;
  let tid = local_id.y * 16u + local_id.x; // 0..255

  var acc0 = vec4<f16>(0.0h);
  var acc1 = vec4<f16>(0.0h);
  var acc2 = vec4<f16>(0.0h);
  var acc3 = vec4<f16>(0.0h);

  // Load coordinates for 256 threads (each thread loads 1 vec4 of A/weight and 1 vec4 of B/input)
  let a_r = tid >> 2u;      // 0..63
  let a_c = tid & 3u;       // 0..3
  let a_m = tile_row0 + a_r;
  let a_valid = a_m < M;

  let b_r = tid >> 4u;      // 0..15
  let b_c = tid & 15u;      // 0..15
  let b_n4 = tile_col0_4 + b_c;
  let b_c_valid = b_n4 < N_vecs;

  let num_tiles = (K + 15u) >> 4u;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t << 2u;
    let k_start = t << 4u;

    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = weight[a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f16>(0.0h);
    }

    let bk = k_start + b_r;
    if (bk < K && b_c_valid) {
      b_sub[b_r][b_c] = input[bk * N_vecs + b_n4];
    } else {
      b_sub[b_r][b_c] = vec4<f16>(0.0h);
    }
    workgroupBarrier();

    let thread_r = local_id.y << 2u; // 0..60 (step 4)
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
    if (out_r0 + 0u < M) {
      var v0 = acc0;
      if (params.has_bias != 0u) { v0 = v0 + vec4<f16>(bias[out_r0 + 0u]); }
      out[(out_r0 + 0u) * N_vecs + out_c4] = conv_epilogue(v0);
    }
    if (out_r0 + 1u < M) {
      var v1 = acc1;
      if (params.has_bias != 0u) { v1 = v1 + vec4<f16>(bias[out_r0 + 1u]); }
      out[(out_r0 + 1u) * N_vecs + out_c4] = conv_epilogue(v1);
    }
    if (out_r0 + 2u < M) {
      var v2 = acc2;
      if (params.has_bias != 0u) { v2 = v2 + vec4<f16>(bias[out_r0 + 2u]); }
      out[(out_r0 + 2u) * N_vecs + out_c4] = conv_epilogue(v2);
    }
    if (out_r0 + 3u < M) {
      var v3 = acc3;
      if (params.has_bias != 0u) { v3 = v3 + vec4<f16>(bias[out_r0 + 3u]); }
      out[(out_r0 + 3u) * N_vecs + out_c4] = conv_epilogue(v3);
    }
  }
}
`;
