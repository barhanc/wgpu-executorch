/**
 * WGSL compute shader for direct 2D 1x1 convolution (vec4 vectorized GEMM).
 * Processes 128-bit vec4 memory transactions along channels (K) and spatial dimensions (N).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_1X1_VEC4 = /* wgsl */ `
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

const TILE: u32 = 32u;
const RPT: u32 = 4u;
const TILE4: u32 = 8u;

var<workgroup> a_sub: array<array<vec4<f32>, 8>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

override fuse_silu: bool = false;

// Preserve the convolution output and sigmoid's storage precision at the fusion boundary.
fn conv_epilogue(value: vec4<f32>) -> vec4<f32> {
  if (!fuse_silu) { return value; }
  let gate = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-value));
  return value * gate;
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let M = params.OC;
  let spatial = params.OH * params.OW;
  let sp_vecs = spatial / 4u;
  let N_vecs = params.B * sp_vecs;
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
        a_sub[arow][local_id.x] = weight[m * K_vecs + k4];
      } else {
        a_sub[arow][local_id.x] = vec4<f32>(0.0);
      }

      let b_k = t * TILE + arow;
      let n4 = tile_col0_4 + local_id.x;
      if (b_k < K && n4 < N_vecs) {
        let b_idx = n4 / sp_vecs;
        let sp4 = n4 % sp_vecs;
        b_sub[arow][local_id.x] = input[(b_idx * params.IC + b_k) * sp_vecs + sp4];
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
        val = val + vec4<f32>(bias[m]);
      }
      let b_idx = n4 / sp_vecs;
      let sp4 = n4 % sp_vecs;
      out[(b_idx * params.OC + m) * sp_vecs + sp4] = conv_epilogue(val);
    }
  }
}
`;
