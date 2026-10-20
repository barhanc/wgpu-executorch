import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_SOFTMAX } from './softmax.wgsl';
import { SHADER_SOFTMAX_LAST_DIM } from './softmax-last-dim.wgsl';
import { SHADER_SOFTMAX_LAST_DIM_VEC4 } from './softmax-last-dim-vec4.wgsl';
import { SHADER_SOFTMAX_F16 } from './softmax-f16.wgsl';
import { SHADER_SOFTMAX_LAST_DIM_F16 } from './softmax-last-dim-f16.wgsl';
import { SHADER_SOFTMAX_LAST_DIM_VEC4_F16 } from './softmax-last-dim-vec4-f16.wgsl';

const name = 'aten._softmax.default';

/**
 * Positional argument tuple for `aten::_softmax.default`:
 * `[in, dim, half_to_float, out]`
 */
export type SoftmaxArgs = readonly [
  inTensor: Tensor,
  dim: number,
  halfToFloat: boolean,
  outTensor: Tensor,
];

/**
 * Decomposes dims into [outer, R, inner] for a reduction along `dim`.
 *
 * @param dims The tensor shape.
 * @param dim The reduction dimension (may be negative).
 * @param opName Operator name for error messages.
 * @returns Tuple of [outer, r, inner] extents.
 */
function decompose(
  dims: readonly number[],
  dim: number,
  opName: string
): [outer: number, r: number, inner: number] {
  const ndim = dims.length;
  const d = dim < 0 ? dim + ndim : dim;
  if (ndim === 0 || d < 0 || d >= ndim) {
    throw new Error(`${opName}: dim out of range`);
  }
  let outer = 1;
  let inner = 1;
  for (let i = 0; i < d; i++) {
    outer *= dims[i];
  }
  for (let i = d + 1; i < ndim; i++) {
    inner *= dims[i];
  }
  return [outer, dims[d], inner];
}

/**
 * Internal dispatch builder for `aten::_softmax.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: SoftmaxArgs): void {
  const [inTensor, dim, , outTensor] = args;

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (inTensor.shape.length === 0) {
    throw new Error(`${name}: scalar input unsupported`);
  }
  if (inTensor.numel !== outTensor.numel) {
    throw new Error(`${name}: input/output numel mismatch`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const [outer, r, inner] = decompose(inTensor.shape, dim, name);
  if (outer === 0 || r === 0 || inner === 0) {
    throw new Error(`${name}: zero-sized reduction`);
  }

  const wgSize = 256;

  let code = isF16 ? SHADER_SOFTMAX_F16 : SHADER_SOFTMAX;
  let lines = outer * inner;
  let paramsArr = [outer, r, inner, 0];

  if (inner === 1) {
    if (r % 4 === 0) {
      code = isF16 ? SHADER_SOFTMAX_LAST_DIM_VEC4_F16 : SHADER_SOFTMAX_LAST_DIM_VEC4;
      paramsArr = [outer, r / 4, 1, 0];
      lines = outer;
    } else {
      code = isF16 ? SHADER_SOFTMAX_LAST_DIM_F16 : SHADER_SOFTMAX_LAST_DIM;
      paramsArr = [outer, r, 1, 0];
      lines = outer;
    }
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array(paramsArr));

  const bundle = createComputeBundle(
    device,
    code,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  const totalWorkgroups = Math.ceil(lines / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::_softmax.default`.
 *
 * Computes softmax along one dimension with a numerically stable online
 * (flash-style) max/sum pass per reduction line.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const softmax: Shader<SoftmaxArgs, typeof name> = {
  name,
  code: {
    default: SHADER_SOFTMAX,
    lastDim: SHADER_SOFTMAX_LAST_DIM,
    lastDimVec4: SHADER_SOFTMAX_LAST_DIM_VEC4,
    defaultF16: SHADER_SOFTMAX_F16,
    lastDimF16: SHADER_SOFTMAX_LAST_DIM_F16,
    lastDimVec4F16: SHADER_SOFTMAX_LAST_DIM_VEC4_F16,
  },
  recordIn,
};
