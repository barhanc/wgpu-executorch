import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten._softmax.default';

/**
 * WGSL compute shader for softmax along one dimension.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/softmax/softmax.wgsl)
 * (the build-time $if LOG variant is resolved to the non-LOG path)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/softmax/softmax.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
struct Params {
  outer_: u32,
  r_: u32,
  inner_: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let lines = params.outer_ * params.inner_;
  if (t >= lines) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;
  // Online (flash-style) softmax; -3.4e38 seeds max (Tint overflows -FLT_MAX).
  var mx: f32 = -3.4e38;
  var ssum: f32 = 0.0;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let v = inp[base + r * params.inner_];
    if (v > mx) {
      ssum = ssum * exp(mx - v);
      mx = v;
    }
    ssum = ssum + exp(v - mx);
  }
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let idx = base + r * params.inner_;
    out[idx] = exp(inp[idx] - mx) / ssum;
  }
}
`;

/**
 * Positional argument tuple for `aten::_softmax.default`:
 * `[in, dim, half_to_float, out]`
 */
export type SoftmaxArgs = readonly [
  inTensor: Tensor,
  dim: number,
  halfToFloat: boolean,
  outTensor: Tensor,
];

/**
 * Decomposes dims into [outer, R, inner] for a reduction along `dim`.
 *
 * @param dims The tensor shape.
 * @param dim The reduction dimension (may be negative).
 * @param opName Operator name for error messages.
 * @returns Tuple of [outer, r, inner] extents.
 */
function decompose(
  dims: readonly number[],
  dim: number,
  opName: string
): [outer: number, r: number, inner: number] {
  const ndim = dims.length;
  const d = dim < 0 ? dim + ndim : dim;
  if (ndim === 0 || d < 0 || d >= ndim) {
    throw new Error(`${opName}: dim out of range`);
  }
  let outer = 1;
  let inner = 1;
  for (let i = 0; i < d; i++) {
    outer *= dims[i];
  }
  for (let i = d + 1; i < ndim; i++) {
    inner *= dims[i];
  }
  return [outer, dims[d], inner];
}

/**
 * Internal dispatch builder for `aten::_softmax.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: SoftmaxArgs): void {
  const [inTensor, dim, , outTensor] = args;

  if (inTensor.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (inTensor.shape.length === 0) {
    throw new Error(`${name}: scalar input unsupported`);
  }
  if (inTensor.numel !== outTensor.numel) {
    throw new Error(`${name}: input/output numel mismatch`);
  }

  const [outer, r, inner] = decompose(inTensor.shape, dim, name);
  if (outer === 0 || r === 0 || inner === 0) {
    throw new Error(`${name}: zero-sized reduction`);
  }

  const wgSize = 256;
  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([outer, r, inner, 0]));

  const bundle = createComputeBundle(
    ctx.device,
    SHADER,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  const lines = outer * inner;
  const workgroupCount = Math.ceil(lines / wgSize);
  if (workgroupCount > 65535) {
    throw new Error(`${name}: reduction exceeds dispatch limit`);
  }
  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX: workgroupCount,
    workgroupCountY: 1,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::_softmax.default`.
 *
 * Computes softmax along one dimension with a numerically stable online
 * (flash-style) max/sum pass per reduction line.
 */
export const softmax: Shader<SoftmaxArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
