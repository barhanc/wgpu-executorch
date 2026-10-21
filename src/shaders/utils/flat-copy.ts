import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_FLAT_COPY } from './flat-copy.wgsl';
import { SHADER_FLAT_COPY_VEC4 } from './flat-copy-vec4.wgsl';

/**
 * Records a flat elementwise copy between two same-numel tensors — the
 * contiguous-reshape semantics of view/clone-style operators.
 *
 * Uses an in-pass compute shader dispatch (128-bit vec4<u32> memory streaming)
 * so that memory moves do not break the active compute pass into multiple passes.
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

  const totalU32 = Math.ceil(inTensor.byteLength / 4);
  const useVec4 = totalU32 % 4 === 0;
  const count = useVec4 ? totalU32 / 4 : totalU32;
  const code = useVec4 ? SHADER_FLAT_COPY_VEC4 : SHADER_FLAT_COPY;
  const wgSize = 256;

  const buffer = new ArrayBuffer(16);
  new Uint32Array(buffer)[0] = count;
  const uniformBuffer = ctx.uniformBuffer(buffer);

  const bundle = createComputeBundle(
    ctx.device,
    code,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: uniformBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  const totalWorkgroups = Math.ceil(count / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}
