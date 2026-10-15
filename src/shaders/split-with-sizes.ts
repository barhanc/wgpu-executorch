import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { encodeTensorMeta, TENSOR_META_WGSL } from './utils/meta';
import { floatKind, scalarCode } from './utils/scalar';

const name = 'aten.split_with_sizes_copy.default';

/**
 * WGSL compute shader for slicing along one dimension.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/slice/slice.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/slice/slice.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
alias scalar_t = f32;

@group(0) @binding(0) var<storage, read> input: array<scalar_t>;
@group(0) @binding(1) var<storage, read_write> output: array<scalar_t>;

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

const CODE = scalarCode(SHADER);

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

  const kind = floatKind(name, [inTensor, ...outs]);

  const wgSize = 64;
  const inMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(inTensor));

  let start = 0;
  for (let i = 0; i < outs.length; i++) {
    const out = outs[i];
    if (out.shape[dim] !== splitSizes[i]) {
      throw new Error(`${name}: output ${i} dim ${dim} != split size`);
    }

    const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(out));
    const paramsBuffer = ctx.uniformBuffer(new Uint32Array([dim, start, 1, 0]));
    start += splitSizes[i];

    const bundle = createComputeBundle(
      ctx.device,
      CODE[kind],
      [
        { binding: 0, buffer: inTensor.buffer },
        { binding: 1, buffer: out.buffer },
        { binding: 2, buffer: outMetaBuffer },
        { binding: 3, buffer: inMetaBuffer },
        { binding: 4, buffer: paramsBuffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { wg_size: wgSize }
    );

    const workgroupCount = Math.ceil(out.numel / wgSize);
    if (workgroupCount > 65535) {
      throw new Error(`${name}: output exceeds dispatch limit`);
    }
    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: workgroupCount,
      workgroupCountY: 1,
    });
  }
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::split_with_sizes_copy.default`.
 *
 * Splits a tensor into contiguous chunks along one dimension — one slice
 * gather dispatch per output tensor.
 */
export const splitWithSizesCopy: Shader<SplitWithSizesArgs, typeof name> = {
  name,
  code: CODE,
  recordIn,
};
