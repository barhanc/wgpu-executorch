import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_TO_HWC4 } from './to-hwc4.wgsl';
import { SHADER_TO_HWC4_F16 } from './to-hwc4-f16.wgsl';

const name = 'et_vk.to_hwc4.default';

/**
 * Positional argument tuple for `et_vk::to_hwc4.default`:
 * `[in, out]`
 */
export type ToHwc4Args = readonly [inTensor: Tensor, outTensor: Tensor];

/**
 * Internal dispatch builder for `et_vk::to_hwc4.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ToHwc4Args): void {
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

  if (inTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensor for input`);
  }

  const [b, c, h, w] = inTensor.shape as [number, number, number, number];
  const c4 = Math.ceil(c / 4);
  const total = b * h * w * c4;

  const wgSize = 256;
  const totalWorkgroups = Math.ceil(total / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
  const strideX = workgroupCountX * wgSize;

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([b, c, h, w, c4, strideX, 0, 0]));

  const bundle = createComputeBundle(device, isF16 ? SHADER_TO_HWC4_F16 : SHADER_TO_HWC4, [
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
 * WebGPU compute kernel for ExecuTorch layout conversion `et_vk::to_hwc4.default`.
 *
 * Converts planar NCHW activations [B, C, H, W] to packed channels-last [B, H, W, C/4] (vec4).
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const toHwc4: Shader<ToHwc4Args, typeof name> = {
  name,
  code: {
    default: SHADER_TO_HWC4,
    defaultF16: SHADER_TO_HWC4_F16,
  },
  recordIn,
};
