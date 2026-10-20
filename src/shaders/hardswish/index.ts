import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_HARDSWISH } from './hardswish.wgsl';
import { SHADER_HARDSWISH_VEC4 } from './hardswish-vec4.wgsl';
import { SHADER_HARDSWISH_F16 } from './hardswish-f16.wgsl';
import { SHADER_HARDSWISH_VEC4_F16 } from './hardswish-vec4-f16.wgsl';

const name = 'aten.hardswish.default';

/**
 * Positional argument tuple for `aten::hardswish.default`:
 * `[inTensor, outTensor]`
 */
export type HardswishArgs = readonly [inTensor: Tensor, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::hardswish.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, [inTensor, outTensor]: HardswishArgs): void {
  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (inTensor.numel !== outTensor.numel) {
    throw new Error(`${name}: input numel != output numel`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const wgSize = 256;
  const isVec4 = outTensor.numel % 4 === 0;
  const elements = isVec4 ? outTensor.numel / 4 : outTensor.numel;

  let code = isVec4 ? SHADER_HARDSWISH_VEC4 : SHADER_HARDSWISH;
  if (isF16) {
    code = isVec4 ? SHADER_HARDSWISH_VEC4_F16 : SHADER_HARDSWISH_F16;
  }

  const totalWorkgroups = Math.ceil(elements / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const buffer = new ArrayBuffer(16);
  new Uint32Array(buffer)[0] = elements;
  const uniformBuffer = ctx.uniformBuffer(buffer);

  const bundle = createComputeBundle(
    device,
    code,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: uniformBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::hardswish.default`.
 *
 * Computes elementwise `out = x * clamp(x + 3, 0, 6) / 6`.
 * Automatically vectorizes using 128-bit vec4<f32> or vec4<f16> when tensor size is divisible by 4.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const hardswish: Shader<HardswishArgs, typeof name> = {
  name,
  code: {
    scalar: SHADER_HARDSWISH,
    vec4: SHADER_HARDSWISH_VEC4,
    scalarF16: SHADER_HARDSWISH_F16,
    vec4F16: SHADER_HARDSWISH_VEC4_F16,
  },
  recordIn,
};
