import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_UPSAMPLE_NEAREST2D } from './upsample-nearest-2d.wgsl';
import { SHADER_UPSAMPLE_NEAREST2D_SPATIAL } from './upsample-nearest-2d-spatial.wgsl';
import { SHADER_UPSAMPLE_NEAREST2D_F16 } from './upsample-nearest-2d-f16.wgsl';
import { SHADER_UPSAMPLE_NEAREST2D_SPATIAL_F16 } from './upsample-nearest-2d-spatial-f16.wgsl';

const name = 'aten.upsample_nearest2d.vec';

/**
 * Positional argument tuple for `aten::upsample_nearest2d.vec`:
 * `[in, output_size, scales, out]`
 */
export type UpsampleNearest2dArgs = readonly [
  inTensor: Tensor,
  outputSize: undefined,
  scales: readonly number[],
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::upsample_nearest2d.vec`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: UpsampleNearest2dArgs): void {
  const [inTensor, , , outTensor] = args;

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (inTensor.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: input and output must be 4D NCHW`);
  }
  const [b, c, ih, iw] = inTensor.shape as [number, number, number, number];
  const [ob, oc, oh, ow] = outTensor.shape as [number, number, number, number];
  if (ob !== b || oc !== c) {
    throw new Error(`${name}: output batch/channels must match input`);
  }
  if (oh < ih || ow < iw) {
    throw new Error(`${name}: output spatial dims must be >= input spatial dims`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([b, c, ih, iw, oh, ow, 0, 0]));

  // Use 2D spatial grid when dimensions fit in workgroup limits
  if (b * c <= 65535 && Math.ceil(ow / 16) <= 65535 && Math.ceil(oh / 16) <= 65535) {
    const bundle = createComputeBundle(
      device,
      isF16 ? SHADER_UPSAMPLE_NEAREST2D_SPATIAL_F16 : SHADER_UPSAMPLE_NEAREST2D_SPATIAL,
      [
        { binding: 0, buffer: outTensor.buffer },
        { binding: 1, buffer: inTensor.buffer },
        { binding: 2, buffer: paramsBuffer },
      ]
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(ow / 16),
      workgroupCountY: Math.ceil(oh / 16),
      workgroupCountZ: b * c,
    });
    return;
  }

  const wgSize = 256;
  const total = b * c * oh * ow;
  const workgroupCountX = Math.min(Math.ceil(total / wgSize), 65535);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_UPSAMPLE_NEAREST2D_F16 : SHADER_UPSAMPLE_NEAREST2D,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize, stride_x: workgroupCountX * wgSize }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY: Math.ceil(total / wgSize / 65535),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::upsample_nearest2d.vec`.
 *
 * Nearest-neighbour 2D upsampling — each output texel copies the input texel
 * at `floor(o * I / O)` (ATen convention).
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const upsampleNearest2d: Shader<UpsampleNearest2dArgs, typeof name> = {
  name,
  code: {
    default: SHADER_UPSAMPLE_NEAREST2D,
    spatial: SHADER_UPSAMPLE_NEAREST2D_SPATIAL,
    defaultF16: SHADER_UPSAMPLE_NEAREST2D_F16,
    spatialF16: SHADER_UPSAMPLE_NEAREST2D_SPATIAL_F16,
  },
  recordIn,
};
