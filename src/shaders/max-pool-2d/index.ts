import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_MAX_POOL2D } from './max-pool-2d.wgsl';
import { SHADER_MAX_POOL2D_SPATIAL } from './max-pool-2d-spatial.wgsl';
import { SHADER_MAX_POOL2D_F16 } from './max-pool-2d-f16.wgsl';
import { SHADER_MAX_POOL2D_SPATIAL_F16 } from './max-pool-2d-spatial-f16.wgsl';

const name = 'aten.max_pool2d_with_indices.default';

/**
 * Positional argument tuple for `aten::max_pool2d_with_indices.default`:
 * `[in, kernel_size, stride, padding, dilation, ceil_mode, outs]`
 * where `outs` is `[values, indices]`.
 */
export type MaxPool2dArgs = readonly [
  inTensor: Tensor,
  kernelSize: readonly number[],
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  ceilMode: boolean,
  outs: readonly Tensor[],
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `aten::max_pool2d_with_indices.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: MaxPool2dArgs): void {
  const [inTensor, kernelSize, stride, padding, dilation, ceilMode, outs] = args;

  if (outs.length !== 2) {
    throw new Error(`${name}: expected 2 outputs [values, indices], got ${outs.length}`);
  }
  const [outVals, outIdx] = outs;

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outVals.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: values tensors must have matching float32 or float16 dtype`);
  }
  if (outIdx.dtype !== 'int32') {
    throw new Error(`${name}: indices output must be int32`);
  }
  if (inTensor.shape.length !== 4) {
    throw new Error(`${name}: input must be 4D NCHW`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const [b, c, ih, iw] = inTensor.shape as [number, number, number, number];
  const [kH, kW] = parseHw(kernelSize, 'kernel_size');
  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');

  const ohNum = ih + 2 * pH - (dH * (kH - 1) + 1);
  const owNum = iw + 2 * pW - (dW * (kW - 1) + 1);
  const oh = Math.max(0, (ceilMode ? Math.ceil(ohNum / sH) : Math.floor(ohNum / sH)) + 1);
  const ow = Math.max(0, (ceilMode ? Math.ceil(owNum / sW) : Math.floor(owNum / sW)) + 1);

  if (
    outVals.shape.length !== 4 ||
    outVals.shape[0] !== b ||
    outVals.shape[1] !== c ||
    outVals.shape[2] !== oh ||
    outVals.shape[3] !== ow
  ) {
    throw new Error(`${name}: values output shape ${outVals.shape} != [${b}, ${c}, ${oh}, ${ow}]`);
  }
  if (
    outIdx.shape.length !== 4 ||
    outIdx.shape[0] !== b ||
    outIdx.shape[1] !== c ||
    outIdx.shape[2] !== oh ||
    outIdx.shape[3] !== ow
  ) {
    throw new Error(`${name}: indices output shape ${outIdx.shape} != [${b}, ${c}, ${oh}, ${ow}]`);
  }

  // PoolParams: input/output geometry, kernel, stride, pad, dilation
  // prettier-ignore
  const paramsBuffer = ctx.uniformBuffer(
    new Uint32Array([
      b, c, ih, iw,
      oh, ow,
      kH, kW,
      sH, sW,
      pH, pW,
      dH, dW,
      0, 0,
    ])
  );

  // Use 2D spatial grid when workgroup dimensions fit within limits (eliminates unraveling divisions)
  if (b * c <= 65535 && Math.ceil(ow / 16) <= 65535 && Math.ceil(oh / 16) <= 65535) {
    const bundle = createComputeBundle(
      device,
      isF16 ? SHADER_MAX_POOL2D_SPATIAL_F16 : SHADER_MAX_POOL2D_SPATIAL,
      [
        { binding: 0, buffer: outVals.buffer },
        { binding: 1, buffer: inTensor.buffer },
        { binding: 2, buffer: paramsBuffer },
        { binding: 3, buffer: outIdx.buffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { write_indices: 1 }
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
    isF16 ? SHADER_MAX_POOL2D_F16 : SHADER_MAX_POOL2D,
    [
      { binding: 0, buffer: outVals.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
      { binding: 3, buffer: outIdx.buffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize, stride_x: workgroupCountX * wgSize, write_indices: 1 }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY: Math.ceil(total / wgSize / 65535),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::max_pool2d_with_indices.default`.
 *
 * Computes max pooling over a 2D NCHW input — windowed gather max per output
 * element, plus flat in-window argmax indices as an int32 tensor.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const maxPool2d: Shader<MaxPool2dArgs, typeof name> = {
  name,
  code: {
    default: SHADER_MAX_POOL2D,
    spatial: SHADER_MAX_POOL2D_SPATIAL,
    defaultF16: SHADER_MAX_POOL2D_F16,
    spatialF16: SHADER_MAX_POOL2D_SPATIAL_F16,
  },
  recordIn,
};
