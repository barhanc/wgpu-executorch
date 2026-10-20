import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_SIGMOID } from './sigmoid.wgsl';
import { SHADER_SIGMOID_VEC4 } from './sigmoid-vec4.wgsl';
import { SHADER_SIGMOID_F16 } from './sigmoid-f16.wgsl';
import { SHADER_SIGMOID_VEC4_F16 } from './sigmoid-vec4-f16.wgsl';

const name = 'aten.sigmoid.default';

/**
 * Positional argument tuple for `aten::sigmoid.default`:
 * `[inTensor, outTensor]`
 */
export type SigmoidArgs = readonly [inTensor: Tensor, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::sigmoid.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, [inTensor, outTensor]: SigmoidArgs): void {
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
  const numel = outTensor.numel;
  const useVec4 = numel % 4 === 0;
  const effectiveNumel = useVec4 ? numel / 4 : numel;
  let code = useVec4 ? SHADER_SIGMOID_VEC4 : SHADER_SIGMOID;
  if (isF16) {
    code = useVec4 ? SHADER_SIGMOID_VEC4_F16 : SHADER_SIGMOID_F16;
  }

  const totalWorkgroups = Math.ceil(effectiveNumel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const buffer = new ArrayBuffer(16);
  new Uint32Array(buffer)[0] = effectiveNumel;
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
 * WebGPU compute kernel for ExecuTorch `aten::sigmoid.default`.
 *
 * Element-wise sigmoid activation: 1 / (1 + exp(-x)). Automatically
 * vectorizes using 128-bit vec4<f32> or vec4<f16> when tensor size is divisible by 4.
 */
export const sigmoid: Shader<SigmoidArgs, typeof name> = {
  name,
  code: {
    scalar: SHADER_SIGMOID,
    vec4: SHADER_SIGMOID_VEC4,
    scalarF16: SHADER_SIGMOID_F16,
    vec4F16: SHADER_SIGMOID_VEC4_F16,
  },
  recordIn,
};
