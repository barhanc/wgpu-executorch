import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { TENSOR_META_WGSL } from './utils/meta';

const name = 'aten.split_with_sizes_copy.default';

/**
 * WGSL compute shader for slicing along one dimension.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/slice/slice.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/slice/slice.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

${TENSOR_META_WGSL}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  dim: u32,
  start: u32,
  step: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_bufi = gid.x;
  if (out_bufi >= out_meta.numel) {
    return;
  }

  // Gather: out_bufi -> in_bufi, sliced dim coord = start + coord*step.
  var rem = out_bufi;
  var in_bufi: u32 = 0u;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    let coord = rem / out_meta.strides[d >> 2u][d & 3u];
    rem = rem % out_meta.strides[d >> 2u][d & 3u];
    var in_coord = coord;
    if (d == params.dim) {
      in_coord = params.start + coord * params.step;
    }
    in_bufi = in_bufi + in_coord * in_meta.strides[d >> 2u][d & 3u];
  }
  output[out_bufi] = input[in_bufi];
}
`;

/**
 * Fast block-gather shader for contiguous tensor slicing/splitting.
 * Eliminates coordinate-unraveling integer divisions across all dimensions.
 */
const SHADER_BLOCK = /* wgsl */ `
struct BlockParams {
  c_in: u32,
  c_out: u32,
  inner_size: u32,
  start: u32,
  total_elements: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: BlockParams;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.total_elements) {
    return;
  }
  let inner_idx = idx % params.inner_size;
  let slice_idx = idx / params.inner_size;
  let c = slice_idx % params.c_out;
  let b = slice_idx / params.c_out;

  let in_idx = (b * params.c_in + params.start + c) * params.inner_size + inner_idx;
  output[idx] = input[in_idx];
}
`;

/**
 * 128-bit vectorized block-gather shader for contiguous tensor slicing/splitting
 * when inner_size is divisible by 4.
 */
const SHADER_BLOCK_VEC4 = /* wgsl */ `
struct BlockParams {
  c_in: u32,
  c_out: u32,
  inner_size: u32,
  start: u32,
  total_elements: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: BlockParams;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.total_elements) {
    return;
  }
  let inner_idx = idx % params.inner_size;
  let slice_idx = idx / params.inner_size;
  let c = slice_idx % params.c_out;
  let b = slice_idx / params.c_out;

  let in_idx = (b * params.c_in + params.start + c) * params.inner_size + inner_idx;
  output[idx] = input[in_idx];
}
`;

/**
 * Positional argument tuple for `aten::split_with_sizes_copy.default`:
 * `[in, split_sizes, dim, outs]`
 */
export type SplitWithSizesArgs = readonly [
  inTensor: Tensor,
  splitSizes: readonly number[],
  dim: number,
  outs: readonly Tensor[],
];

/**
 * Internal dispatch builder for `aten::split_with_sizes_copy.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: SplitWithSizesArgs): void {
  const [inTensor, splitSizes, dimArg, outs] = args;

  const ndim = inTensor.shape.length;
  let dim = dimArg;
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw new Error(`${name}: dim out of range`);
  }
  if (outs.length !== splitSizes.length) {
    throw new Error(`${name}: outputs (${outs.length}) != sizes (${splitSizes.length})`);
  }

  let sizesSum = 0;
  for (const s of splitSizes) {
    if (s < 0) {
      throw new Error(`${name}: negative split size`);
    }
    sizesSum += s;
  }
  if (sizesSum !== inTensor.shape[dim]) {
    throw new Error(`${name}: sizes must sum to the split dim extent`);
  }

  let innerSize = 1;
  for (let d = dim + 1; d < ndim; d++) {
    innerSize *= inTensor.shape[d];
  }

  const wgSize = 256;
  const cIn = inTensor.shape[dim];
  let start = 0;

  for (let i = 0; i < outs.length; i++) {
    const out = outs[i];
    if (inTensor.dtype !== 'float32' || out.dtype !== 'float32') {
      throw new Error(`${name}: Only float32 tensors are currently supported`);
    }
    if (out.shape[dim] !== splitSizes[i]) {
      throw new Error(`${name}: output ${i} dim ${dim} != split size`);
    }

    const cOut = splitSizes[i];
    const isVec4 = innerSize % 4 === 0;
    const effectiveInner = isVec4 ? innerSize / 4 : innerSize;
    const totalElements = isVec4 ? out.numel / 4 : out.numel;

    const paramsBuffer = ctx.uniformBuffer(
      new Uint32Array([cIn, cOut, effectiveInner, start, totalElements, 0, 0, 0])
    );
    start += cOut;

    const bundle = createComputeBundle(
      ctx.device,
      isVec4 ? SHADER_BLOCK_VEC4 : SHADER_BLOCK,
      [
        { binding: 0, buffer: inTensor.buffer },
        { binding: 1, buffer: out.buffer },
        { binding: 2, buffer: paramsBuffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { wg_size: wgSize }
    );

    const totalWorkgroups = Math.ceil(totalElements / wgSize);
    const workgroupCountX = Math.min(totalWorkgroups, 65535);
    const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX,
      workgroupCountY,
    });
  }
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::split_with_sizes_copy.default`.
 *
 * Splits a tensor into contiguous chunks along one dimension using fast block-gather
 * and 128-bit vectorization.
 */
export const splitWithSizesCopy: Shader<SplitWithSizesArgs, typeof name> = {
  name,
  code: { default: SHADER, block: SHADER_BLOCK, blockVec4: SHADER_BLOCK_VEC4 },
  recordIn,
};
