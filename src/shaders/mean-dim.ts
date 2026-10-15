import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { floatKind, scalarCode } from './utils/scalar';

const name = 'aten.mean.dim';

/**
 * WGSL compute shader for dimension reduction (mean / sum).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/reduce/reduce.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/reduce/reduce.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
alias scalar_t = f32;

struct Params {
  outer_: u32,
  r_: u32,
  inner_: u32,
  is_mean: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<scalar_t>;
@group(0) @binding(1) var<storage, read_write> out: array<scalar_t>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

// Cooperative shared-memory reduction, one workgroup per output element: each
// thread sums a strided slice of the reduced dim into a shared partial, then
// thread 0 folds the partials. Same one-workgroup-per-row shared-memory shape as
// Vulkan's reduce_per_row_buffer.glsl. Fixed 256 upper bound >= any clamped
// wg_size; only [0, wg_size) is used.
var<workgroup> partials: array<f32, 256>;

@compute @workgroup_size(wg_size)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // One workgroup per output; 2D-fold lifts the 65535 grid cap. \`t\` is uniform
  // across the workgroup, so the early return keeps the barrier in uniform flow.
  let t = wid.x + wid.y * num_workgroups.x;
  let outs = params.outer_ * params.inner_;
  if (t >= outs) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;

  var acc: f32 = 0.0;
  var k: u32 = lid.x;
  while (k < params.r_) {
    acc = acc + f32(inp[base + k * params.inner_]);
    k = k + wg_size;
  }
  partials[lid.x] = acc;
  workgroupBarrier();

  if (lid.x == 0u) {
    var s: f32 = partials[0];
    for (var i: u32 = 1u; i < wg_size; i = i + 1u) {
      s = s + partials[i];
    }
    if (params.is_mean == 1u) {
      s = s / f32(params.r_);
    }
    out[t] = scalar_t(s);
  }
}
`;

const CODE = scalarCode(SHADER);

/**
 * Positional argument tuple for `aten::mean.dim`:
 * `[in, dim, keepdim, dtype, out]`
 */
export type MeanDimArgs = readonly [
  inTensor: Tensor,
  dim: readonly number[] | undefined,
  keepdim: boolean,
  dtype: unknown,
  outTensor: Tensor,
];

/**
 * Normalizes and validates contiguous reduced dims, folding into [outer, r, inner].
 *
 * @param shape Input tensor shape.
 * @param dimList Dimensions to reduce over.
 * @returns Flattened outer, r, and inner sizes.
 */
function decomposeDims(
  shape: readonly number[],
  dimList: readonly number[]
): { outer: number; r: number; inner: number } {
  const ndim = shape.length;
  if (ndim === 0 || dimList.length === 0) {
    throw new Error(`${name}: dim out of range`);
  }

  const normalized = dimList.map((d) => (d < 0 ? d + ndim : d));
  for (const d of normalized) {
    if (d < 0 || d >= ndim) {
      throw new Error(`${name}: dim out of range`);
    }
  }

  normalized.sort((a, b) => a - b);
  for (let i = 1; i < normalized.length; i++) {
    if (normalized[i] === normalized[i - 1]) {
      throw new Error(`${name}: duplicate reduced dim`);
    }
    if (normalized[i] !== normalized[i - 1] + 1) {
      throw new Error(`${name}: only contiguous reduced dims supported`);
    }
  }

  const firstRd = normalized[0];
  const lastRd = normalized[normalized.length - 1];

  let outer = 1;
  let r = 1;
  let inner = 1;

  for (let d = 0; d < firstRd; d++) {
    outer *= shape[d];
  }
  for (let d = firstRd; d <= lastRd; d++) {
    r *= shape[d];
  }
  for (let d = lastRd + 1; d < ndim; d++) {
    inner *= shape[d];
  }

  return { outer, r, inner };
}

/**
 * Internal dispatch builder for `aten::mean.dim`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: MeanDimArgs): void {
  const inTensor = args[0];
  const dim = args[1];
  const outTensor = args[4];

  const kind = floatKind(name, [inTensor, outTensor]);
  if (inTensor.shape.length === 0) {
    throw new Error(`${name}: scalar input unsupported`);
  }

  const reduceDims = dim?.length ? dim : [...inTensor.shape.keys()];

  const { outer, r, inner } = decomposeDims(inTensor.shape, reduceDims);
  if (outer === 0 || r === 0 || inner === 0) {
    throw new Error(`${name}: zero-sized reduction`);
  }

  const outputs = outer * inner;
  if (outTensor.numel !== outputs) {
    throw new Error(`${name}: output numel mismatch`);
  }

  const device = ctx.device;
  const wgSize = 256;
  const workgroupCountX = Math.min(outputs, 65535);
  const workgroupCountY = Math.ceil(outputs / 65535);

  const buffer = new ArrayBuffer(16);
  const u32View = new Uint32Array(buffer);
  u32View[0] = outer;
  u32View[1] = r;
  u32View[2] = inner;
  u32View[3] = 1; // is_mean = 1

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
 * WebGPU compute kernel for ExecuTorch `aten::mean.dim`.
 *
 * Implements reduction across contiguous tensor dimensions with cooperative workgroups.
 */
export const meanDim: Shader<MeanDimArgs, typeof name> = {
  name,
  code: CODE,
  recordIn,
};
