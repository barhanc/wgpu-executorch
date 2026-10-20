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
      let b0 = b_sub[k][tile_col + 0u];
      let b1 = b_sub[k][tile_col + 1u];
      let b2 = b_sub[k][tile_col + 2u];
      let b3 = b_sub[k][tile_col + 3u];
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        acc[ir][0] = acc[ir][0] + aval * b0;
        acc[ir][1] = acc[ir][1] + aval * b1;
        acc[ir][2] = acc[ir][2] + aval * b2;
        acc[ir][3] = acc[ir][3] + aval * b3;
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
 * WGSL compute shader for vectorized 128-bit memory-bandwidth optimized batched
 * matrix multiplication. Operates on vec4<f32> for 4x wider memory transactions
 * when K and N are multiples of 4.
 */
const SHADER_VEC4 = /* wgsl */ `
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

const TILE: u32 = 32u;
const RPT: u32 = 4u;
const TILE4: u32 = 8u;

var<workgroup> a_sub: array<array<vec4<f32>, 8>, 32>;
var<workgroup> b_sub: array<array<vec4<f32>, 8>, 32>;

fn read_a4(base4: u32, row: u32, k4: u32) -> vec4<f32> {
  if (row < params.M && k4 * 4u < params.K) {
    return a[base4 + row * (params.K / 4u) + k4];
  }
  return vec4<f32>(0.0);
}

fn read_b4(base4: u32, krow: u32, n4: u32) -> vec4<f32> {
  if (krow < params.K && n4 * 4u < params.N) {
    return b[base4 + krow * (params.N / 4u) + n4];
  }
  return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let bn = wg_id.z;
  let tile_row0 = wg_id.y * TILE;
  let tile_col0_4 = wg_id.x * TILE4;
  let tile_row = local_id.y * RPT;
  let tile_col4 = local_id.x;

  let a_base4 = bn * params.M * (params.K / 4u);
  let b_base4 = bn * params.K * (params.N / 4u);
  let out_base4 = bn * params.M * (params.N / 4u);

  var acc: array<vec4<f32>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    acc[ir] = vec4<f32>(0.0);
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * TILE4;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      a_sub[arow][local_id.x] = read_a4(a_base4, tile_row0 + arow, k4_start + local_id.x);
      b_sub[arow][local_id.x] = read_b4(b_base4, t * TILE + arow, tile_col0_4 + local_id.x);
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
    let r = tile_row0 + tile_row + ir;
    let c4 = tile_col0_4 + tile_col4;
    if (r < params.M && c4 * 4u < params.N) {
      out[out_base4 + r * (params.N / 4u) + c4] = acc[ir];
    }
  }
}
`;

/**
 * WGSL compute shader for 64x64 register-tiled batched matrix multiplication.
 * Computes 8x8 elements per thread with 128-bit memory vectors and 3D workgroup grid.
 */
const SHADER_64X64 = /* wgsl */ `
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

var<workgroup> a_sub: array<array<vec4<f32>, 4>, 64>;
var<workgroup> b_sub: array<array<vec4<f32>, 16>, 16>;

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

  var acc0: array<vec4<f32>, 8>;
  var acc1: array<vec4<f32>, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    acc0[i] = vec4<f32>(0.0);
    acc1[i] = vec4<f32>(0.0);
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
        a_sub[a_row][col] = vec4<f32>(0.0);
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
        b_sub[b_r][b_c] = vec4<f32>(0.0);
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

  // Use 64x64 register-tiled GEMM when dimensions are sufficiently large and divisible by 4,
  // falling back to 32x32 vec4 or 32x32 scalar tiled paths.
  const useVec4 = k % 4 === 0 && n % 4 === 0;
  const use64x64 = useVec4 && m >= 64 && n >= 64 && k >= 16;

  let code = SHADER;
  let workgroupCountX = Math.ceil(n / 32);
  let workgroupCountY = batch * Math.ceil(m / 32);
  let workgroupCountZ = 1;

  if (use64x64) {
    code = SHADER_64X64;
    workgroupCountX = Math.ceil(n / 64);
    workgroupCountY = Math.ceil(m / 64);
    workgroupCountZ = batch;
  } else if (useVec4) {
    code = SHADER_VEC4;
    workgroupCountX = Math.ceil(n / 32);
    workgroupCountY = Math.ceil(m / 32);
    workgroupCountZ = batch;
  }

  const bundle = createComputeBundle(ctx.device, code, [
    { binding: 0, buffer: a.buffer },
    { binding: 1, buffer: b.buffer },
    { binding: 2, buffer: out.buffer },
    { binding: 3, buffer: paramsBuffer },
  ]);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
    workgroupCountZ,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::bmm.default`.
 *
 * Batched matrix multiplication via shared-memory tiling —
 * Automatically selects 64x64 register-tiled or 32x32 vectorized vec4<f32> path
 * when dimensions permit.
 */
export const bmm: Shader<BmmArgs, typeof name> = {
  name,
  code: { tiled: SHADER, vec4: SHADER_VEC4, tiled64x64: SHADER_64X64 },
  recordIn,
};
