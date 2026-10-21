import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';

import { recordFlatCopy } from './utils/flat-copy';

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
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/prepack/Prepack.cpp)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/prepack/Prepack.cpp
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
export const prepack: Shader<PrepackArgs, typeof name> = {
  name,
  code: '/* flat byte copy — no kernel dispatch */',
  recordIn,
};
