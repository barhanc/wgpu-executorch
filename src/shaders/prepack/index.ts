import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';

import { recordFlatCopy } from '../utils/flat-copy';

const name = 'et_vk.prepack.default';

/**
 * Positional argument tuple for `et_vk::prepack.default`:
 * `[src, out]`
 */
export type PrepackArgs = readonly [src: Tensor, out: Tensor];

/**
 * Internal dispatch builder for `et_vk::prepack.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: PrepackArgs): void {
  const [src, out] = args;
  recordFlatCopy(ctx, src, out, name);
}

/**
 * WebGPU compute kernel for ExecuTorch `et_vk::prepack.default`.
 *
 * Materializes a constant into the prepack-output buffer — a single
 * write-once flat copy (no layout transformation on the WebGPU path).
 * Supports any dtype (float32, float16, etc.) via buffer copy.
 */
export const prepack: Shader<PrepackArgs, typeof name> = {
  name,
  code: '/* flat byte copy — no kernel dispatch */',
  recordIn,
};
