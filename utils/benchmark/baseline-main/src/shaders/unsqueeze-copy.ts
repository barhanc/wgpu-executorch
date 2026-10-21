import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';

import { recordFlatCopy } from './utils/flat-copy';

const name = 'aten.unsqueeze_copy.default';

/**
 * Positional argument tuple for `aten::unsqueeze_copy.default`:
 * `[in, dim, out]`
 */
export type UnsqueezeCopyArgs = readonly [inTensor: Tensor, dim: number, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::unsqueeze_copy.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: UnsqueezeCopyArgs): void {
  const [inTensor, , outTensor] = args;
  recordFlatCopy(ctx, inTensor, outTensor, name);
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::unsqueeze_copy.default`.
 *
 * Unsqueeze = numel-preserving flat copy (the output shape is fixed ahead of
 * time, like `view_copy`).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/unsqueeze/Unsqueeze.cpp)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/unsqueeze/Unsqueeze.cpp
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
export const unsqueezeCopy: Shader<UnsqueezeCopyArgs, typeof name> = {
  name,
  code: '/* flat byte copy — no kernel dispatch */',
  recordIn,
};
