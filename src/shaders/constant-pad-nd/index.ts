import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_CONSTANT_PAD_ND } from './constant-pad-nd.wgsl';
import { SHADER_CONSTANT_PAD_ND_SPATIAL } from './constant-pad-nd-spatial.wgsl';
import { SHADER_CONSTANT_PAD_ND_F16 } from './constant-pad-nd-f16.wgsl';
import { SHADER_CONSTANT_PAD_ND_SPATIAL_F16 } from './constant-pad-nd-spatial-f16.wgsl';

const name = 'aten.constant_pad_nd.default';

/**
 * Positional argument tuple for `aten::constant_pad_nd.default`:
 * - `[inTensor, pad, value, outTensor]`
 * - `[inTensor, pad, outTensor]` (value defaults to 0.0)
 */
export type ConstantPadNdArgs =
  | readonly [inTensor: Tensor, pad: readonly number[], outTensor: Tensor]
  | readonly [inTensor: Tensor, pad: readonly number[], value: number, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::constant_pad_nd.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConstantPadNdArgs): void {
  const [inTensor, pad] = args;
  const [value, outTensor] = args.length === 4 ? [args[2], args[3]] : [0.0, args[2]];

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }

  const isF16 = inTensor.dtype === 'float16';
  const device = ctx.device;
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const nd = inTensor.shape.length;
  if (nd === 0 || nd > 4) {
    throw new Error(`${name}: rank must be 1..4 (got ${nd})`);
  }
  if (outTensor.shape.length !== nd) {
    throw new Error(`${name}: in/out rank mismatch (${nd} !== ${outTensor.shape.length})`);
  }
  if (pad.length % 2 !== 0) {
    throw new Error(`${name}: pad must be even-length`);
  }

  const npad = pad.length / 2;
  if (npad > nd) {
    throw new Error(`${name}: pad longer than rank`);
  }

  // Per-dim left/right pad (pad list is reversed-dim, from the LAST dim)
  const left = [0, 0, 0, 0];
  const right = [0, 0, 0, 0];
  for (let k = 0; k < npad; k++) {
    const d = nd - 1 - k;
    left[d] = pad[2 * k];
    right[d] = pad[2 * k + 1];
  }

  for (let d = 0; d < nd; d++) {
    if (left[d] < 0 || right[d] < 0) {
      throw new Error(`${name}: negative pad (cropping) not supported`);
    }
    const expected = inTensor.shape[d] + left[d] + right[d];
    if (outTensor.shape[d] !== expected) {
      throw new Error(`${name}: output dim ${d} mismatch (${expected} != ${outTensor.shape[d]})`);
    }
  }

  // Right-align dims into 4D tuples: leading unused dims = 1 (pad = 0)
  const padLeft = (arr: readonly number[], fill: number): number[] => [
    ...Array<number>(4 - arr.length).fill(fill),
    ...arr,
  ];

  const left4 = padLeft(left.slice(0, nd), 0);
  const right4 = padLeft(right.slice(0, nd), 0);
  const outDims = padLeft(outTensor.shape, 1);
  const inDims = padLeft(inTensor.shape, 1);

  const buffer = new ArrayBuffer(64);
  new Uint32Array(buffer).set([...outDims, ...inDims, ...left4, outTensor.numel]);
  new Float32Array(buffer, 52, 1)[0] = value;

  const uniformBuffer = ctx.uniformBuffer(buffer);

  const isSpatialOnly = left4[0] === 0 && left4[1] === 0 && right4[0] === 0 && right4[1] === 0;
  const sliceCount = outDims[0] * outDims[1];

  if (isSpatialOnly && sliceCount <= 65535) {
    const bundle = createComputeBundle(
      device,
      isF16 ? SHADER_CONSTANT_PAD_ND_SPATIAL_F16 : SHADER_CONSTANT_PAD_ND_SPATIAL,
      [
        { binding: 0, buffer: outTensor.buffer },
        { binding: 1, buffer: inTensor.buffer },
        { binding: 2, buffer: uniformBuffer },
      ]
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(outDims[3] / 16),
      workgroupCountY: Math.ceil(outDims[2] / 16),
      workgroupCountZ: sliceCount,
    });
    return;
  }

  const wgSize = 256;
  const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_CONSTANT_PAD_ND_F16 : SHADER_CONSTANT_PAD_ND,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
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
 * WebGPU compute kernel for ExecuTorch `aten::constant_pad_nd.default`.
 *
 * Copies input tensor into output tensor with constant padding on up to 4D tensors.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const constantPadNd: Shader<ConstantPadNdArgs, typeof name> = {
  name,
  code: {
    default: SHADER_CONSTANT_PAD_ND,
    spatial: SHADER_CONSTANT_PAD_ND_SPATIAL,
    defaultF16: SHADER_CONSTANT_PAD_ND_F16,
    spatialF16: SHADER_CONSTANT_PAD_ND_SPATIAL_F16,
  },
  recordIn,
};
