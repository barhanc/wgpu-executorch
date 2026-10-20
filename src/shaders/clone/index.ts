import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';

import { recordFlatCopy } from '../utils/flat-copy';

const name = 'aten.clone.default';

/**
 * Positional argument tuple for `aten::clone.default`:
 * `[in, memory_format, out]`
 */
export type CloneArgs = readonly [inTensor: Tensor, memoryFormat: undefined, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::clone.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: CloneArgs): void {
  const [inTensor, , outTensor] = args;
  recordFlatCopy(ctx, inTensor, outTensor, name);
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::clone.default`.
 *
 * Clone = flat byte copy (survives the serializer's redundant-op elimination).
 * Supports any dtype (float32, float16, etc.) via buffer copy.
 */
export const clone: Shader<CloneArgs, typeof name> = {
  name,
  code: '/* flat byte copy — no kernel dispatch */',
  recordIn,
};
