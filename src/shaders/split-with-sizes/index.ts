import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_SPLIT_WITH_SIZES } from './split-with-sizes.wgsl';
import { SHADER_SPLIT_WITH_SIZES_BLOCK } from './split-with-sizes-block.wgsl';
import { SHADER_SPLIT_WITH_SIZES_BLOCK_VEC4 } from './split-with-sizes-block-vec4.wgsl';
import { SHADER_SPLIT_WITH_SIZES_F16 } from './split-with-sizes-f16.wgsl';
import { SHADER_SPLIT_WITH_SIZES_BLOCK_F16 } from './split-with-sizes-block-f16.wgsl';
import { SHADER_SPLIT_WITH_SIZES_BLOCK_VEC4_F16 } from './split-with-sizes-block-vec4-f16.wgsl';

const name = 'aten.split_with_sizes_copy.default';

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

  const isF16 = inTensor.dtype === 'float16';
  const device = ctx.device;
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
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
    if (
      (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
      out.dtype !== inTensor.dtype
    ) {
      throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
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

    let code = isVec4 ? SHADER_SPLIT_WITH_SIZES_BLOCK_VEC4 : SHADER_SPLIT_WITH_SIZES_BLOCK;
    if (isF16) {
      code = isVec4 ? SHADER_SPLIT_WITH_SIZES_BLOCK_VEC4_F16 : SHADER_SPLIT_WITH_SIZES_BLOCK_F16;
    }

    const bundle = createComputeBundle(
      device,
      code,
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
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const splitWithSizesCopy: Shader<SplitWithSizesArgs, typeof name> = {
  name,
  code: {
    default: SHADER_SPLIT_WITH_SIZES,
    block: SHADER_SPLIT_WITH_SIZES_BLOCK,
    blockVec4: SHADER_SPLIT_WITH_SIZES_BLOCK_VEC4,
    defaultF16: SHADER_SPLIT_WITH_SIZES_F16,
    blockF16: SHADER_SPLIT_WITH_SIZES_BLOCK_F16,
    blockVec4F16: SHADER_SPLIT_WITH_SIZES_BLOCK_VEC4_F16,
  },
  recordIn,
};
