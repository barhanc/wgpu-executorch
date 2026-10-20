/**
 * WGSL compute shader for 64x64 register-tiled float16 batched matrix multiplication.
 * Computes 8x8 elements per thread with 128-bit memory vectors and 3D workgroup grid.
 * Requires WebGPU shader-f16 feature.
 */
export const SHADER_BMM_64X64_F16 = /* wgsl */ `
enable f16;

struct Params {
  B: u32,
  M: u32,
  N: u32,
  K: u32,
};

@group(0) @binding(0) var<storage, read> a: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> a_sub: array<array<vec4<f16>, 4>, 64>;
var<workgroup> b_sub: array<array<vec4<f16>, 16>, 16>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let bn = wg_id.z;
  let tile_row0 = wg_id.y * 64u;
  let tile_col0_4 = wg_id.x * 16u;
  let tid = local_id.y * 8u + local_id.x;

  let K_vecs = params.K / 4u;
  let N_vecs = params.N / 4u;

  let a_base4 = bn * params.M * K_vecs;
  let b_base4 = bn * params.K * N_vecs;
  let out_base4 = bn * params.M * N_vecs;

  var acc0: array<vec4<f16>, 8>;
  var acc1: array<vec4<f16>, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    acc0[i] = vec4<f16>(0.0h);
    acc1[i] = vec4<f16>(0.0h);
  }

  let num_tiles = (params.K + 15u) / 16u;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t << 2u;
    let a_row = tid;
    let a_m = tile_row0 + a_row;
    let a_valid = a_m < params.M;
    let a_base = a_base4 + a_m * K_vecs + k4_start;
    for (var col = 0u; col < 4u; col = col + 1u) {
      if (a_valid && (k4_start + col) < K_vecs) {
        a_sub[a_row][col] = a[a_base + col];
      } else {
        a_sub[a_row][col] = vec4<f16>(0.0h);
      }
    }

    let b_r = tid >> 2u;
    let b_c_base = (tid & 3u) << 2u;
    let b_k = (t << 4u) + b_r;
    let b_k_valid = b_k < params.K;
    let b_base = b_base4 + b_k * N_vecs + tile_col0_4 + b_c_base;
    for (var i = 0u; i < 4u; i = i + 1u) {
      let b_c = b_c_base + i;
      let b_n4 = tile_col0_4 + b_c;
      if (b_k_valid && b_n4 < N_vecs) {
        b_sub[b_r][b_c] = b[b_base + i];
      } else {
        b_sub[b_r][b_c] = vec4<f16>(0.0h);
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
    if (r < params.M) {
      if (col4 < N_vecs) {
        out[out_base4 + r * N_vecs + col4] = acc0[ir];
      }
      if (col4 + 1u < N_vecs) {
        out[out_base4 + r * N_vecs + col4 + 1u] = acc1[ir];
      }
    }
  }
}
`;
