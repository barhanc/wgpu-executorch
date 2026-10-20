import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_MEAN_DIM } from './mean-dim.wgsl';
import { SHADER_MEAN_DIM_F16 } from './mean-dim-f16.wgsl';

const name = 'aten.mean.dim';

/**
 * Positional argument tuple for `aten::mean.dim`:
 * `[in, dim, keepdim, dtype, out]`
 */
export type MeanDimArgs = readonly [
  inTensor: Tensor,
  dim: readonly number[] | undefined,
  keepdim: boolean,
  dtype: unknown,
  outTensor: Tensor,
];

/**
 * Normalizes and validates contiguous reduced dims, folding into [outer, r, inner].
 *
 * @param shape Input tensor shape.
 * @param dimList Dimensions to reduce over.
 * @returns Flattened outer, r, and inner sizes.
 */
function decomposeDims(
  shape: readonly number[],
  dimList: readonly number[]
): { outer: number; r: number; inner: number } {
  const ndim = shape.length;
  if (ndim === 0 || dimList.length === 0) {
    throw new Error(`${name}: dim out of range`);
  }

  const normalized = dimList.map((d) => (d < 0 ? d + ndim : d));
  for (const d of normalized) {
    if (d < 0 || d >= ndim) {
      throw new Error(`${name}: dim out of range`);
    }
  }

  normalized.sort((a, b) => a - b);
  for (let i = 1; i < normalized.length; i++) {
    if (normalized[i] === normalized[i - 1]) {
      throw new Error(`${name}: duplicate reduced dim`);
    }
    if (normalized[i] !== normalized[i - 1] + 1) {
      throw new Error(`${name}: only contiguous reduced dims supported`);
    }
  }

  const firstRd = normalized[0];
  const lastRd = normalized[normalized.length - 1];

  let outer = 1;
  let r = 1;
  let inner = 1;

  for (let d = 0; d < firstRd; d++) {
    outer *= shape[d];
  }
  for (let d = firstRd; d <= lastRd; d++) {
    r *= shape[d];
  }
  for (let d = lastRd + 1; d < ndim; d++) {
    inner *= shape[d];
  }

  return { outer, r, inner };
}

/**
 * Internal dispatch builder for `aten::mean.dim`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: MeanDimArgs): void {
  const inTensor = args[0];
  const dim = args[1];
  const outTensor = args[4];

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (inTensor.shape.length === 0) {
    throw new Error(`${name}: scalar input unsupported`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const reduceDims = dim?.length ? dim : [...inTensor.shape.keys()];

  const { outer, r, inner } = decomposeDims(inTensor.shape, reduceDims);
  if (outer === 0 || r === 0 || inner === 0) {
    throw new Error(`${name}: zero-sized reduction`);
  }

  const outputs = outer * inner;
  if (outTensor.numel !== outputs) {
    throw new Error(`${name}: output numel mismatch`);
  }

  const wgSize = 256;
  const workgroupCountX = Math.min(outputs, 65535);
  const workgroupCountY = Math.ceil(outputs / 65535);

  const buffer = new ArrayBuffer(16);
  const u32View = new Uint32Array(buffer);
  u32View[0] = outer;
  u32View[1] = r;
  u32View[2] = inner;
  u32View[3] = 1; // is_mean = 1

  const uniformBuffer = ctx.uniformBuffer(buffer);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_MEAN_DIM_F16 : SHADER_MEAN_DIM,
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
 * WebGPU compute kernel for ExecuTorch `aten::mean.dim`.
 *
 * Implements reduction across contiguous tensor dimensions with cooperative workgroups.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const meanDim: Shader<MeanDimArgs, typeof name> = {
  name,
  code: { default: SHADER_MEAN_DIM, defaultF16: SHADER_MEAN_DIM_F16 },
  recordIn,
};
