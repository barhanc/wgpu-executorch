import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_CAT } from './cat.wgsl';
import { SHADER_CAT_BLOCK } from './cat-block.wgsl';
import { SHADER_CAT_BLOCK_VEC4 } from './cat-block-vec4.wgsl';
import { SHADER_CAT_F16 } from './cat-f16.wgsl';
import { SHADER_CAT_BLOCK_F16 } from './cat-block-f16.wgsl';
import { SHADER_CAT_BLOCK_VEC4_F16 } from './cat-block-vec4-f16.wgsl';

const name = 'aten.cat.default';

/**
 * Positional argument tuple for `aten::cat.default`:
 * `[tensors, dim, out]`
 */
export type CatArgs = readonly [tensors: readonly Tensor[], dim: number, out: Tensor];

/**
 * Internal dispatch builder for `aten::cat.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: CatArgs): void {
  const [tensors, dimArg, out] = args;

  if (tensors.length === 0) {
    throw new Error(`${name}: empty input list`);
  }

  const ndim = out.shape.length;
  let dim = dimArg;
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw new Error(`${name}: dim out of range`);
  }

  const isF16 = out.dtype === 'float16';
  const device = ctx.device;
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  let innerSize = 1;
  for (let d = dim + 1; d < ndim; d++) {
    innerSize *= out.shape[d];
  }

  let offK = 0;
  let concatSum = 0;
  const wgSize = 256;

  for (const inp of tensors) {
    if ((inp.dtype !== 'float32' && inp.dtype !== 'float16') || inp.dtype !== out.dtype) {
      throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
    }
    if (inp.shape.length !== ndim) {
      throw new Error(`${name}: input rank != output rank`);
    }
    for (let d = 0; d < ndim; d++) {
      if (d !== dim && inp.shape[d] !== out.shape[d]) {
        throw new Error(`${name}: non-concat dim ${d} size mismatch`);
      }
    }
    concatSum += inp.shape[dim];

    const cIn = inp.shape[dim];
    const cOut = out.shape[dim];
    const isVec4 = innerSize % 4 === 0;
    const effectiveInner = isVec4 ? innerSize / 4 : innerSize;
    const totalElements = isVec4 ? inp.numel / 4 : inp.numel;

    const paramsBuffer = ctx.uniformBuffer(
      new Uint32Array([cIn, cOut, effectiveInner, offK, totalElements, 0, 0, 0])
    );
    offK += cIn;

    let code = isVec4 ? SHADER_CAT_BLOCK_VEC4 : SHADER_CAT_BLOCK;
    if (isF16) {
      code = isVec4 ? SHADER_CAT_BLOCK_VEC4_F16 : SHADER_CAT_BLOCK_F16;
    }

    const bundle = createComputeBundle(
      device,
      code,
      [
        { binding: 0, buffer: inp.buffer },
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

  if (concatSum !== out.shape[dim]) {
    throw new Error(`${name}: concat dim sizes do not sum to output`);
  }
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::cat.default`.
 *
 * Concatenates a list of tensors along one dimension. Uses fast block-scatter
 * and 128-bit vectorization for contiguous slices.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const cat: Shader<CatArgs, typeof name> = {
  name,
  code: {
    default: SHADER_CAT,
    block: SHADER_CAT_BLOCK,
    blockVec4: SHADER_CAT_BLOCK_VEC4,
    defaultF16: SHADER_CAT_F16,
    blockF16: SHADER_CAT_BLOCK_F16,
    blockVec4F16: SHADER_CAT_BLOCK_VEC4_F16,
  },
  recordIn,
};
