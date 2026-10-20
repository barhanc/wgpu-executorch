/**
 * WGSL compute shader for fused 1x1 convolution with clamp bounds (vec4 vectorized GEMM).
 * Processes 128-bit vec4 memory transactions along channels (K) and spatial dimensions (N).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV_WITH_CLAMP_1X1_VEC4 = /* wgsl */ `
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
