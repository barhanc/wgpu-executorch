/**
 * WGSL compute shader for 64x64 register-tiled batched matrix multiplication (4x4 per thread across 256 threads).
 * High occupancy, coalesced 128-bit memory transactions, minimal register pressure for all GPUs.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_BMM_64X64 = /* wgsl */ `
struct Params {
  B: u32,
  M: u32,
  N: u32,
  K: u32,
};

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

// 64 rows x 16 K elements (4 vec4s) = 256 vec4s (4 KB)
var<workgroup> a_sub: array<array<vec4<f32>, 4>, 64>;
// 16 K rows x 64 N elements (16 vec4s) = 256 vec4s (4 KB)
var<workgroup> b_sub: array<array<vec4<f32>, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let bn = wg_id.z;
  let tile_row0 = wg_id.y * 64u;
  let tile_col0_4 = wg_id.x * 16u;
  let tid = local_id.y * 16u + local_id.x; // 0..255

  let K_vecs = params.K / 4u;
  let N_vecs = params.N / 4u;

  let a_base4 = bn * params.M * K_vecs;
  let b_base4 = bn * params.K * N_vecs;
  let out_base4 = bn * params.M * N_vecs;

  var acc0 = vec4<f32>(0.0);
  var acc1 = vec4<f32>(0.0);
  var acc2 = vec4<f32>(0.0);
  var acc3 = vec4<f32>(0.0);

  // Load coordinates for 256 threads (each thread loads 1 vec4 of A and 1 vec4 of B)
  let a_r = tid >> 2u;      // 0..63
  let a_c = tid & 3u;       // 0..3
  let a_m = tile_row0 + a_r;
  let a_valid = a_m < params.M;

  let b_r = tid >> 4u;      // 0..15
  let b_c = tid & 15u;      // 0..15
  let b_n4 = tile_col0_4 + b_c;
  let b_c_valid = b_n4 < N_vecs;

  let num_tiles = (params.K + 15u) >> 4u;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t << 2u;
    let k_start = t << 4u;

    if (a_valid && (k4_start + a_c) < K_vecs) {
      a_sub[a_r][a_c] = a[a_base4 + a_m * K_vecs + k4_start + a_c];
    } else {
      a_sub[a_r][a_c] = vec4<f32>(0.0);
    }

    let bk = k_start + b_r;
    if (bk < params.K && b_c_valid) {
      b_sub[b_r][b_c] = b[b_base4 + bk * N_vecs + b_n4];
    } else {
      b_sub[b_r][b_c] = vec4<f32>(0.0);
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
    if (out_r0 + 0u < params.M) {
      out[out_base4 + (out_r0 + 0u) * N_vecs + out_c4] = acc0;
    }
    if (out_r0 + 1u < params.M) {
      out[out_base4 + (out_r0 + 1u) * N_vecs + out_c4] = acc1;
    }
    if (out_r0 + 2u < params.M) {
      out[out_base4 + (out_r0 + 2u) * N_vecs + out_c4] = acc2;
    }
    if (out_r0 + 3u < params.M) {
      out[out_base4 + (out_r0 + 3u) * N_vecs + out_c4] = acc3;
    }
  }
}
`;
