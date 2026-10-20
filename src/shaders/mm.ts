import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.mm.default';

/**
 * WGSL compute shader for matrix multiplication with shared-memory tiling
 * (32x32 tile, 4x4 per thread).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/mm/mm_tiled.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/mm/mm_tiled.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_TILED = /* wgsl */ `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn read_a(row: u32, col: u32) -> f32 {
  if (row < params.M && col < params.K) {
    return a[row * params.K + col];
  }
  return 0.0;
}

fn read_b(row: u32, col: u32) -> f32 {
  if (row < params.K && col < params.N) {
    return b[row * params.N + col];
  }
  return 0.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

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
        a_sub[arow][col] = read_a(tile_row0 + arow, k_start + col);
        b_sub[arow][col] = read_b(k_start + arow, tile_col0 + col);
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
        out[r * params.N + c] = acc[ir][ic];
      }
    }
  }
}
`;

/**
 * WGSL compute shader for vectorized 128-bit memory-bandwidth optimized matrix
 * multiplication. Operates on vec4<f32> for 4x wider memory transactions when K
 * and N are multiples of 4.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/mm/mm_vec4.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/mm/mm_vec4.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_VEC4 = /* wgsl */ `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad_: u32,
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

fn read_a4(row: u32, k4: u32) -> vec4<f32> {
  if (row < params.M && k4 * 4u < params.K) {
    return a[row * (params.K / 4u) + k4];
  }
  return vec4<f32>(0.0);
}

fn read_b4(krow: u32, n4: u32) -> vec4<f32> {
  if (krow < params.K && n4 * 4u < params.N) {
    return b[krow * (params.N / 4u) + n4];
  }
  return vec4<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let tile_row0 = wg_id.y * TILE;
  let tile_col0_4 = wg_id.x * TILE4;
  let tile_row = local_id.y * RPT;
  let tile_col4 = local_id.x;

  var acc: array<vec4<f32>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    acc[ir] = vec4<f32>(0.0);
  }

  let num_tiles = (params.K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * TILE4;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      a_sub[arow][local_id.x] = read_a4(tile_row0 + arow, k4_start + local_id.x);
      b_sub[arow][local_id.x] = read_b4(t * TILE + arow, tile_col0_4 + local_id.x);
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
      out[r * (params.N / 4u) + c4] = acc[ir];
    }
  }
}
`;

/**
 * WGSL compute shader for 64x64 tiled GEMM with 2D register tiling (8x8 per thread).
 * Features 2x higher arithmetic intensity and memory reuse than the 32x32 kernel.
 */
const SHADER_64X64 = /* wgsl */ `
struct Params {
  M: u32,
  N: u32,
  K: u32,
  pad_: u32,
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
  let tile_row0 = wg_id.y * 64u;
  let tile_col0_4 = wg_id.x * 16u;
  let tid = local_id.y * 8u + local_id.x;
  let K_vecs = params.K / 4u;
  let N_vecs = params.N / 4u;

  var acc0: array<vec4<f32>, 8>;
  var acc1: array<vec4<f32>, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    acc0[i] = vec4<f32>(0.0);
    acc1[i] = vec4<f32>(0.0);
  }

  let num_tiles = (params.K + 15u) / 16u;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k4_start = t * 4u;
    let a_row = tid;
    let a_m = tile_row0 + a_row;
    for (var col = 0u; col < 4u; col = col + 1u) {
      if (a_m < params.M && (k4_start + col) < K_vecs) {
        a_sub[a_row][col] = a[a_m * K_vecs + k4_start + col];
      } else {
        a_sub[a_row][col] = vec4<f32>(0.0);
      }
    }

    for (var i = 0u; i < 4u; i = i + 1u) {
      let b_idx = tid * 4u + i;
      let b_r = b_idx / 16u;
      let b_c = b_idx % 16u;
      let b_k = t * 16u + b_r;
      let b_n4 = tile_col0_4 + b_c;
      if (b_k < params.K && b_n4 < N_vecs) {
        b_sub[b_r][b_c] = b[b_k * N_vecs + b_n4];
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
        out[r * N_vecs + col4] = acc0[ir];
      }
      if (col4 + 1u < N_vecs) {
        out[r * N_vecs + col4 + 1u] = acc1[ir];
      }
    }
  }
}
`;

/**
 * Output tile dimension per workgroup (32x32).
 */
const TILE = 32;

/**
 * Positional argument tuple for `aten::mm.default`:
 * `[in1, in2, out]`
 */
export type MmArgs = readonly [in1: Tensor, in2: Tensor, out: Tensor];

/**
 * Internal dispatch builder for `aten::mm.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, [in1, in2, out]: MmArgs): void {
  if (in1.dtype !== 'float32' || in2.dtype !== 'float32' || out.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (in1.shape.length !== 2 || in2.shape.length !== 2 || out.shape.length !== 2) {
    throw new Error(`${name}: Tensors must be 2D matrices`);
  }

  const [m, k1] = in1.shape as [number, number];
  const [k2, n] = in2.shape as [number, number];
  const [outM, outN] = out.shape as [number, number];

  if (k1 !== k2) {
    throw new Error(`${name}: Matrix inner dimensions must match (${k1} !== ${k2})`);
  }
  if (outM !== m || outN !== n) {
    throw new Error(`${name}: Output shape [${outM}, ${outN}] !== expected [${m}, ${n}]`);
  }

  const device = ctx.device;
  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([m, n, k1, 0]));

  // Use 64x64 register-tiled GEMM when dimensions are sufficiently large and divisible by 4,
  // falling back to 32x32 vec4 or 32x32 scalar tiled paths.
  const useVec4 = k1 % 4 === 0 && n % 4 === 0;
  const use64x64 = useVec4 && m >= 64 && n >= 64 && k1 >= 16;

  let code = SHADER_TILED;
  let workgroupCountX = Math.ceil(n / TILE);
  let workgroupCountY = Math.ceil(m / TILE);

  if (use64x64) {
    code = SHADER_64X64;
    workgroupCountX = Math.ceil(n / 64);
    workgroupCountY = Math.ceil(m / 64);
  } else if (useVec4) {
    code = SHADER_VEC4;
  }

  const bundle = createComputeBundle(device, code, [
    { binding: 0, buffer: in1.buffer },
    { binding: 1, buffer: in2.buffer },
    { binding: 2, buffer: out.buffer },
    { binding: 3, buffer: paramsBuffer },
  ]);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::mm.default`.
 *
 * Computes matrix multiplication `out = A @ B` using a shared-memory tiled GEMM.
 * Automatically selects 64x64 register-tiled or 32x32 128-bit vectorized `vec4<f32>` paths
 * when dimensions permit, falling back to the standard scalar tiled path.
 */
export const mm: Shader<MmArgs, typeof name> = {
  name,
  code: { tiled: SHADER_TILED, vec4: SHADER_VEC4, tiled64x64: SHADER_64X64 },
  recordIn,
};
