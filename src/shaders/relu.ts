import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { floatKind, scalarCode } from './utils/scalar';

const name = 'aten.relu.default';

/**
 * WGSL compute shader for ReLU activation.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/relu/relu.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/relu/relu.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
alias scalar_t = f32;

@group(0) @binding(0) var<storage, read> input: array<scalar_t>;
@group(0) @binding(1) var<storage, read_write> output: array<scalar_t>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = scalar_t(max(f32(input[idx]), 0.0));
}
`;

const CODE = scalarCode(SHADER);

/**
 * Positional argument tuple for `aten::relu.default`:
 * `[inTensor, outTensor]`
 */
export type ReluArgs = readonly [inTensor: Tensor, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::relu.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, [inTensor, outTensor]: ReluArgs): void {
  const kind = floatKind(name, [inTensor, outTensor]);
  if (inTensor.numel !== outTensor.numel) {
    throw new Error(`${name}: input numel != output numel`);
  }

  const device = ctx.device;
  const wgSize = 256;
  const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const buffer = new ArrayBuffer(16);
  new Uint32Array(buffer)[0] = outTensor.numel;
  const uniformBuffer = ctx.uniformBuffer(buffer);

  const bundle = createComputeBundle(
    device,
    CODE[kind],
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
 * WebGPU compute kernel for ExecuTorch `aten::relu.default`.
 *
 * Element-wise rectified linear unit: max(x, 0).
 */
export const relu: Shader<ReluArgs, typeof name> = {
  name,
  code: CODE,
  recordIn,
};
