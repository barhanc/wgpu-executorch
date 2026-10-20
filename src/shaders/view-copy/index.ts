import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';

import { recordFlatCopy } from '../utils/flat-copy';

const name = 'aten.view_copy.default';

/**
 * Positional argument tuple for `aten::view_copy.default`:
 * `[in, size, out]`
 */
export type ViewCopyArgs = readonly [inTensor: Tensor, size: readonly number[], outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::view_copy.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ViewCopyArgs): void {
  const [inTensor, , outTensor] = args;
  recordFlatCopy(ctx, inTensor, outTensor, name);
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::view_copy.default`.
 *
 * Contiguous reshape = flat byte copy.
 * Supports any dtype (float32, float16, etc.) via buffer copy.
 */
export const viewCopy: Shader<ViewCopyArgs, typeof name> = {
  name,
  code: '/* flat byte copy — no kernel dispatch */',
  recordIn,
};
