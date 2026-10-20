import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';

import { recordFlatCopy } from '../utils/flat-copy';

const name = 'aten.squeeze_copy.dims';

/**
 * Positional argument tuple for `aten::squeeze_copy.dims`:
 * `[in, dims, out]`
 */
export type SqueezeCopyArgs = readonly [
  inTensor: Tensor,
  dims: readonly number[],
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::squeeze_copy.dims`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: SqueezeCopyArgs): void {
  const [inTensor, , outTensor] = args;
  recordFlatCopy(ctx, inTensor, outTensor, name);
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::squeeze_copy.dims`.
 *
 * Squeeze = numel-preserving flat copy.
 * Supports any dtype (float32, float16, etc.) via buffer copy.
 */
export const squeezeCopy: Shader<SqueezeCopyArgs, typeof name> = {
  name,
  code: '/* flat byte copy — no kernel dispatch */',
  recordIn,
};
