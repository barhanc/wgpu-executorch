import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { isBroadcastable } from './utils/broadcasting';
import { encodeTensorMeta, TENSOR_META_WGSL } from './utils/meta';

const name = 'aten.add.Tensor';

/**
 * WGSL compute shader for binary addition with broadcast support.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/add/binary_add.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/add/binary_add.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

${TENSOR_META_WGSL}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in1_meta: TensorMeta;
@group(0) @binding(5) var<uniform> in2_meta: TensorMeta;

override wg_size: u32 = 256u;
override alpha: f32 = 1.0;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= out_meta.numel) {
    return;
  }

  var same = true;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    if (
      in1_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u] ||
      in2_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u]
    ) {
      same = false;
    }
  }
  if (same) {
    output[idx] = input1[idx] + alpha * input2[idx];
    return;
  }

  var rem = idx;
  var l1: u32 = 0u;
  var l2: u32 = 0u;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    let coord = rem / out_meta.strides[d >> 2u][d & 3u];
    rem = rem % out_meta.strides[d >> 2u][d & 3u];
    l1 = l1 + min(coord, in1_meta.sizes[d >> 2u][d & 3u] - 1u) * in1_meta.strides[d >> 2u][d & 3u];
    l2 = l2 + min(coord, in2_meta.sizes[d >> 2u][d & 3u] - 1u) * in2_meta.strides[d >> 2u][d & 3u];
  }
  output[idx] = input1[l1] + alpha * input2[l2];
}
`;

/**
 * Positional argument tuple for `aten::add.Tensor`:
 * Standard ExecuTorch schema: `[in1, in2, alpha, out]`
 * Or 3-arg variant: `[in1, in2, out]`
 */
export type AddArgs =
  | readonly [in1: Tensor, in2: Tensor, out: Tensor]
  | readonly [in1: Tensor, in2: Tensor, alpha: number, out: Tensor];

/**
 * Internal dispatch builder for `aten::add.Tensor`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: AddArgs): void {
  const [in1, in2] = args;
  const [alpha, out] = args.length === 4 ? [args[2], args[3]] : [1.0, args[2]];

  if (in1.dtype !== 'float32' || in2.dtype !== 'float32' || out.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (!isBroadcastable(in1.shape, out.shape) || !isBroadcastable(in2.shape, out.shape)) {
    throw new Error(`${name}: Input shapes are not broadcastable to output shape`);
  }

  const outRank = out.shape.length;
  const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(out));
  const in1MetaBuffer = ctx.uniformBuffer(encodeTensorMeta(in1, outRank));
  const in2MetaBuffer = ctx.uniformBuffer(encodeTensorMeta(in2, outRank));

  const device = ctx.device;
  const wgSize = 256;

  const totalWorkgroups = Math.ceil(out.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const bundle = createComputeBundle(
    device,
    SHADER,
    [
      { binding: 0, buffer: in1.buffer },
      { binding: 1, buffer: in2.buffer },
      { binding: 2, buffer: out.buffer },
      { binding: 3, buffer: outMetaBuffer },
      { binding: 4, buffer: in1MetaBuffer },
      { binding: 5, buffer: in2MetaBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize, alpha }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::add.Tensor`.
 *
 * Computes elementwise `output = input1 + alpha * input2` with NumPy/PyTorch-style
 * broadcasting (right-aligned, dimensions equal or 1, up to rank 8). Features an
 * automatic fast path when input and output shapes match.
 */
export const add: Shader<AddArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
