import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_RELU } from './relu.wgsl';
import { SHADER_RELU_VEC4 } from './relu-vec4.wgsl';
import { SHADER_RELU_F16 } from './relu-f16.wgsl';
import { SHADER_RELU_VEC4_F16 } from './relu-vec4-f16.wgsl';

const name = 'aten.relu.default';

/**
 * Positional argument tuple for `aten::relu.default`:
 * `[inTensor, outTensor]`
 */
export type ReluArgs = readonly [inTensor: Tensor, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::relu.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, [inTensor, outTensor]: ReluArgs): void {
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
  let code = useVec4 ? SHADER_RELU_VEC4 : SHADER_RELU;
  if (isF16) {
    code = useVec4 ? SHADER_RELU_VEC4_F16 : SHADER_RELU_F16;
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
 * WebGPU compute kernel for ExecuTorch `aten::relu.default`.
 *
 * Element-wise rectified linear unit: max(x, 0). Automatically vectorizes
 * using 128-bit vec4<f32> or vec4<f16> when tensor size is divisible by 4.
 */
export const relu: Shader<ReluArgs, typeof name> = {
  name,
  code: {
    scalar: SHADER_RELU,
    vec4: SHADER_RELU_VEC4,
    scalarF16: SHADER_RELU_F16,
    vec4F16: SHADER_RELU_VEC4_F16,
  },
  recordIn,
};
