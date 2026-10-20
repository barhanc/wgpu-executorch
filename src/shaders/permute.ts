import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { encodeTensorMeta, TENSOR_META_WGSL } from './utils/meta';

const name = 'aten.permute_copy.default';

/**
 * WGSL compute shader for permuting tensor dimensions.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/permute/permute.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/permute/permute.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

${TENSOR_META_WGSL}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  perm: array<vec4<u32>, 2>,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
  let out_bufi = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (out_bufi >= out_meta.numel) {
    return;
  }

  // Gather: out coord d -> in coord perm[d] (Vulkan permute_buffer.glsl).
  var rem = out_bufi;
  var in_bufi: u32 = 0u;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    let coord = rem / out_meta.strides[d >> 2u][d & 3u];
    rem = rem % out_meta.strides[d >> 2u][d & 3u];
    let p = params.perm[d >> 2u][d & 3u];
    in_bufi = in_bufi + coord * in_meta.strides[p >> 2u][p & 3u];
  }
  output[out_bufi] = input[in_bufi];
}
`;

/**
 * High-performance 2D matrix transpose using shared-memory tile (16x16 with 17-stride to prevent bank conflicts).
 * Ensures fully coalesced reads and fully coalesced writes without integer coordinate divisions.
 */
const SHADER_TRANSPOSE_2D = /* wgsl */ `
struct Transpose2dParams {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Transpose2dParams;

var<workgroup> tile: array<array<f32, 17>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let x = wg_id.x * 16u + local_id.x;
  let y = wg_id.y * 16u + local_id.y;

  if (x < params.cols && y < params.rows) {
    tile[local_id.y][local_id.x] = input[y * params.cols + x];
  }
  workgroupBarrier();

  let out_x = wg_id.y * 16u + local_id.x;
  let out_y = wg_id.x * 16u + local_id.y;

  if (out_x < params.rows && out_y < params.cols) {
    output[out_y * params.rows + out_x] = tile[local_id.x][local_id.y];
  }
}
`;

/**
 * Positional argument tuple for `aten::permute_copy.default`:
 * `[in, dims, out]`
 */
export type PermuteCopyArgs = readonly [
  inTensor: Tensor,
  dims: readonly number[],
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::permute_copy.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: PermuteCopyArgs): void {
  const [inTensor, dims, outTensor] = args;

  if (inTensor.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  const rank = inTensor.shape.length;
  if (outTensor.shape.length !== rank) {
    throw new Error(`${name}: output rank != input rank`);
  }
  if (dims.length !== rank) {
    throw new Error(`${name}: dims count (${dims.length}) != input rank (${rank})`);
  }

  const perm = new Uint32Array(8);
  const seen = new Set<number>();
  for (let d = 0; d < rank; d++) {
    let p = dims[d];
    if (p < 0) {
      p += rank;
    }
    if (p < 0 || p >= rank || seen.has(p)) {
      throw new Error(`${name}: invalid permute dims ${dims}`);
    }
    seen.add(p);
    perm[d] = p;
  }
  for (let d = rank; d < 8; d++) {
    perm[d] = d; // identity padding for unused slots
  }
  for (let d = 0; d < rank; d++) {
    if (outTensor.shape[d] !== inTensor.shape[perm[d]]) {
      throw new Error(`${name}: output shape does not match permuted input shape`);
    }
  }

  // Fast path: 2D matrix transpose
  if (rank === 2 && perm[0] === 1 && perm[1] === 0) {
    const rows = inTensor.shape[0];
    const cols = inTensor.shape[1];
    const paramsBuffer = ctx.uniformBuffer(new Uint32Array([rows, cols]));

    const bundle = createComputeBundle(ctx.device, SHADER_TRANSPOSE_2D, [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
    ]);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(cols / 16),
      workgroupCountY: Math.ceil(rows / 16),
    });
    return;
  }

  const wgSize = 64;
  const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(outTensor));
  const inMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(inTensor));
  const paramsBuffer = ctx.uniformBuffer(perm);

  const bundle = createComputeBundle(
    ctx.device,
    SHADER,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: outMetaBuffer },
      { binding: 3, buffer: inMetaBuffer },
      { binding: 4, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX: Math.min(totalWorkgroups, 65535),
    workgroupCountY: Math.ceil(totalWorkgroups / 65535),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::permute_copy.default`.
 *
 * Copies a tensor with its dimensions reordered — a gather where each output
 * coordinate maps to the permuted input coordinate.
 */
export const permuteCopy: Shader<PermuteCopyArgs, typeof name> = {
  name,
  code: { default: SHADER, transpose2d: SHADER_TRANSPOSE_2D },
  recordIn,
};
