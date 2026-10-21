import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_UPSAMPLE_BILINEAR2D_HWC4 } from './upsample-bilinear2d-hwc4.wgsl';
import { SHADER_UPSAMPLE_BILINEAR2D_HWC4_F16 } from './upsample-bilinear2d-hwc4-f16.wgsl';

const name = 'aten.upsample_bilinear2d_hwc4.vec';

/**
 * Positional argument tuple for `aten::upsample_bilinear2d_hwc4.vec`:
 * `[in, output_size, align_corners, scales, out]`
 */
export type UpsampleBilinear2dHwc4Args = readonly [
  inTensor: Tensor,
  outputSize: readonly number[],
  alignCorners: boolean,
  scales: readonly number[] | undefined,
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::upsample_bilinear2d_hwc4.vec`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: UpsampleBilinear2dHwc4Args): void {
  const inTensor = args[0];
  const alignCorners = args[2];
  const outTensor = args[4];

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (inTensor.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D in/out tensors`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const [b, ih, iw, c] = inTensor.shape as [number, number, number, number];
  const [, oh, ow, outC] = outTensor.shape as [number, number, number, number];

  if (outC !== c) {
    throw new Error(`${name}: Channel mismatch`);
  }

  const c4 = Math.ceil(c / 4);
  const total = b * oh * ow * c4;
  const wgSize = 256;
  const totalWorkgroups = Math.ceil(total / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
  const strideX = workgroupCountX * wgSize;

  const buffer = new ArrayBuffer(32);
  const u32View = new Uint32Array(buffer);
  // prettier-ignore
  u32View.set([
    b, c4,
    ih, iw,
    oh, ow,
    alignCorners ? 1 : 0,
    strideX,
  ]);

  const uniformBuffer = ctx.uniformBuffer(buffer);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_UPSAMPLE_BILINEAR2D_HWC4_F16 : SHADER_UPSAMPLE_BILINEAR2D_HWC4,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: uniformBuffer },
    ]
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for 2D bilinear upsampling in packed HWC4 layout.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const upsampleBilinear2dHwc4: Shader<UpsampleBilinear2dHwc4Args, typeof name> = {
  name,
  code: {
    default: SHADER_UPSAMPLE_BILINEAR2D_HWC4,
    defaultF16: SHADER_UPSAMPLE_BILINEAR2D_HWC4_F16,
  },
  recordIn,
};
