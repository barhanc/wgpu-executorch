import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_UPSAMPLE_BILINEAR2D } from './upsample-bilinear2d.wgsl';
import { SHADER_UPSAMPLE_BILINEAR2D_SPATIAL } from './upsample-bilinear2d-spatial.wgsl';
import { SHADER_UPSAMPLE_BILINEAR2D_F16 } from './upsample-bilinear2d-f16.wgsl';
import { SHADER_UPSAMPLE_BILINEAR2D_SPATIAL_F16 } from './upsample-bilinear2d-spatial-f16.wgsl';

const name = 'aten.upsample_bilinear2d.vec';

/**
 * Positional argument tuple for `aten::upsample_bilinear2d.vec`:
 * `[in, output_size, align_corners, scales, out]`
 */
export type UpsampleBilinear2dArgs = readonly [
  inTensor: Tensor,
  outputSize: readonly number[],
  alignCorners: boolean,
  scales: readonly number[] | undefined,
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::upsample_bilinear2d.vec`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: UpsampleBilinear2dArgs): void {
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

  const [n, c, ih, iw] = inTensor.shape as [number, number, number, number];
  const [outN, outC, oh, ow] = outTensor.shape as [number, number, number, number];

  if (outN !== n || outC !== c) {
    throw new Error(`${name}: N/C mismatch`);
  }
  if (ih === 0 || iw === 0 || oh === 0 || ow === 0) {
    throw new Error(`${name}: zero spatial dim`);
  }

  const outNumel = outTensor.numel;
  const buffer = new ArrayBuffer(32);
  const u32View = new Uint32Array(buffer);
  u32View.set([n, c, ih, iw, oh, ow, alignCorners ? 1 : 0, 0]);

  const uniformBuffer = ctx.uniformBuffer(buffer);

  // Use 2D spatial grid when dimensions fit in workgroup limits
  if (n * c <= 65535 && Math.ceil(ow / 16) <= 65535 && Math.ceil(oh / 16) <= 65535) {
    const bundle = createComputeBundle(
      device,
      isF16 ? SHADER_UPSAMPLE_BILINEAR2D_SPATIAL_F16 : SHADER_UPSAMPLE_BILINEAR2D_SPATIAL,
      [
        { binding: 0, buffer: outTensor.buffer },
        { binding: 1, buffer: inTensor.buffer },
        { binding: 2, buffer: uniformBuffer },
      ]
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(ow / 16),
      workgroupCountY: Math.ceil(oh / 16),
      workgroupCountZ: n * c,
    });
    return;
  }

  const wgSize = 256;
  const totalWorkgroups = Math.ceil(outNumel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
  const strideX = workgroupCountX * wgSize;

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_UPSAMPLE_BILINEAR2D_F16 : SHADER_UPSAMPLE_BILINEAR2D,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: uniformBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { stride_x: strideX, wg_size: wgSize }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::upsample_bilinear2d.vec`.
 *
 * Implements bilinear 2D upsampling for NCHW float32 and float16 tensors.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const upsampleBilinear2d: Shader<UpsampleBilinear2dArgs, typeof name> = {
  name,
  code: {
    default: SHADER_UPSAMPLE_BILINEAR2D,
    spatial: SHADER_UPSAMPLE_BILINEAR2D_SPATIAL,
    defaultF16: SHADER_UPSAMPLE_BILINEAR2D_F16,
    spatialF16: SHADER_UPSAMPLE_BILINEAR2D_SPATIAL_F16,
  },
  recordIn,
};
