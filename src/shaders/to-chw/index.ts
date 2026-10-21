import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_TO_CHW } from './to-chw.wgsl';
import { SHADER_TO_CHW_F16 } from './to-chw-f16.wgsl';

const name = 'et_vk.to_chw.default';

/**
 * Positional argument tuple for `et_vk::to_chw.default`:
 * `[in, out]`
 */
export type ToChwArgs = readonly [inTensor: Tensor, outTensor: Tensor];

/**
 * Internal dispatch builder for `et_vk::to_chw.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ToChwArgs): void {
  const [inTensor, outTensor] = args;

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  if (outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensor for output`);
  }

  const [b, c, h, w] = outTensor.shape as [number, number, number, number];
  const c4 = Math.ceil(c / 4);
  const total = b * h * w * c4;

  const wgSize = 256;
  const totalWorkgroups = Math.ceil(total / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
  const strideX = workgroupCountX * wgSize;

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([b, c, h, w, c4, strideX, 0, 0]));

  const bundle = createComputeBundle(device, isF16 ? SHADER_TO_CHW_F16 : SHADER_TO_CHW, [
    { binding: 0, buffer: inTensor.buffer },
    { binding: 1, buffer: outTensor.buffer },
    { binding: 2, buffer: paramsBuffer },
  ]);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch layout conversion `et_vk::to_chw.default`.
 *
 * Converts packed channels-last activations [B, H, W, C/4] (vec4) back to planar NCHW [B, C, H, W].
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const toChw: Shader<ToChwArgs, typeof name> = {
  name,
  code: {
    default: SHADER_TO_CHW,
    defaultF16: SHADER_TO_CHW_F16,
  },
  recordIn,
};
