import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';

/**
 * Records a flat elementwise copy between two same-numel tensors — the
 * contiguous-reshape semantics of view/clone-style operators.
 *
 * Aliased in/out buffers (the same `GPUBuffer`) are already in place: WebGPU
 * rejects overlapping copyBufferToBuffer, so the copy is skipped entirely.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/view_copy/ViewCopy.cpp)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/view_copy/ViewCopy.cpp
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 *
 * @param ctx The execution context to record the copy into.
 * @param inTensor The source tensor.
 * @param outTensor The destination tensor (same numel and dtype).
 * @param opName Operator name for error messages.
 */
export function recordFlatCopy(
  ctx: WgpuExecutionContext,
  inTensor: Tensor,
  outTensor: Tensor,
  opName: string
): void {
  if (inTensor.dtype !== outTensor.dtype) {
    throw new Error(`${opName}: dtype mismatch (${inTensor.dtype} != ${outTensor.dtype})`);
  }
  if (inTensor.numel !== outTensor.numel) {
    throw new Error(`${opName}: numel mismatch (${inTensor.numel} != ${outTensor.numel})`);
  }
  if (inTensor.buffer === outTensor.buffer) {
    return; // aliased in/out already in place
  }
  ctx.copy(inTensor, outTensor);
}
