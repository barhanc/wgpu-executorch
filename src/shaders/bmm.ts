import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.bmm.default';

/**
 * WGSL compute shader for batched matrix multiplication (tiled).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/bmm/bmm_tiled.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/bmm/bmm_tiled.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
struct Params {
  B: u32,
  M: u32,
  N: u32,
  K: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn read_a(base: u32, row: u32, col: u32) -> f32 {
  if (row < params.M && col < params.K) {
    return a[base + row * params.K + col];
  }
  return 0.0;
}

fn read_b(base: u32, krow: u32, col: u32) -> f32 {
  if (krow < params.K && col < params.N) {
    return b[base + krow * params.N + col];
  }
  return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>) {
  let tiles_per_m = (params.M + TILE - 1u) / TILE;
  let bn = wg_id.y / tiles_per_m;
  let row_tile = wg_id.y % tiles_per_m;
  let tile_row0 = row_tile * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  let a_base = bn * params.M * params.K;
  let b_base = bn * params.K * params.N;
  let out_base = bn * params.M * params.N;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        a_sub[arow][col] = read_a(a_base, tile_row0 + arow, k_start + col);
        b_sub[arow][col] = read_b(b_base, k_start + arow, tile_col0 + col);
      }
    }
    workgroupBarrier();

    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
          acc[ir][ic] = acc[ir][ic] + aval * b_sub[k][tile_col + ic];
        }
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let r = tile_row0 + tile_row + ir;
      let c = tile_col0 + tile_col + ic;
      if (r < params.M && c < params.N) {
        out[out_base + r * params.N + c] = acc[ir][ic];
      }
    }
  }
}
`;

/**
 * Positional argument tuple for `aten::bmm.default`:
 * `[a, b, out]`
 */
export type BmmArgs = readonly [a: Tensor, b: Tensor, out: Tensor];

/**
 * Internal dispatch builder for `aten::bmm.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: BmmArgs): void {
  const [a, b, out] = args;

  if (a.dtype !== 'float32' || b.dtype !== 'float32' || out.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (a.shape.length !== 3 || b.shape.length !== 3 || out.shape.length !== 3) {
    throw new Error(`${name}: inputs and output must be 3D`);
  }

  const batch = a.shape[0];
  const m = a.shape[1];
  const k = a.shape[2];
  const n = b.shape[2];
  if (b.shape[0] !== batch || b.shape[1] !== k) {
    throw new Error(`${name}: batch/K mismatch between a and b`);
  }
  if (out.shape[0] !== batch || out.shape[1] !== m || out.shape[2] !== n) {
    throw new Error(`${name}: output shape [${out.shape}] != [${batch}, ${m}, ${n}]`);
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([batch, m, n, k]));

  const bundle = createComputeBundle(ctx.device, SHADER, [
    { binding: 0, buffer: a.buffer },
    { binding: 1, buffer: b.buffer },
    { binding: 2, buffer: out.buffer },
    { binding: 3, buffer: paramsBuffer },
  ]);

  const tile = 32;
  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX: Math.ceil(n / tile),
    workgroupCountY: batch * Math.ceil(m / tile),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::bmm.default`.
 *
 * Batched matrix multiplication via shared-memory 32x32 tiling —
 * each workgroup computes a 32x32 output tile of one batch.
 */
export const bmm: Shader<BmmArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
