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
  let inv_ssum = 1.0 / ssum;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let idx = base + r * params.inner_;
    out[idx] = exp(inp[idx] - mx) * inv_ssum;
  }
}
`;

/**
 * WGSL compute shader for contiguous last-dimension softmax (inner_ == 1).
 * Eliminates per-thread integer divisions and modulos.
 */
const SHADER_LAST_DIM = /* wgsl */ `
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
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let row = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (row >= params.outer_) {
    return;
  }
  let base = row * params.r_;
  var mx: f32 = -3.4e38;
  var ssum: f32 = 0.0;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let v = inp[base + r];
    if (v > mx) {
      ssum = ssum * exp(mx - v);
      mx = v;
    }
    ssum = ssum + exp(v - mx);
  }
  let inv_ssum = 1.0 / ssum;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let idx = base + r;
    out[idx] = exp(inp[idx] - mx) * inv_ssum;
  }
}
`;

/**
 * WGSL compute shader for vectorized 128-bit contiguous last-dimension softmax.
 */
const SHADER_LAST_DIM_VEC4 = /* wgsl */ `
struct Params {
  outer_: u32,
  r_vecs: u32,
  inner_: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let row = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (row >= params.outer_) {
    return;
  }
  let base = row * params.r_vecs;
  var mx: f32 = -3.4e38;
  var ssum: f32 = 0.0;
  for (var r: u32 = 0u; r < params.r_vecs; r = r + 1u) {
    let v = inp[base + r];
    let row_max = max(max(v.x, v.y), max(v.z, v.w));
    if (row_max > mx) {
      ssum = ssum * exp(mx - row_max);
      mx = row_max;
    }
    let e = exp(v - vec4<f32>(mx));
    ssum = ssum + e.x + e.y + e.z + e.w;
  }
  let inv_ssum = 1.0 / ssum;
  for (var r: u32 = 0u; r < params.r_vecs; r = r + 1u) {
    let idx = base + r;
    out[idx] = exp(inp[idx] - vec4<f32>(mx)) * inv_ssum;
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

  let code = SHADER;
  let lines = outer * inner;
  let paramsArr = [outer, r, inner, 0];

  if (inner === 1) {
    if (r % 4 === 0) {
      code = SHADER_LAST_DIM_VEC4;
      paramsArr = [outer, r / 4, 1, 0];
      lines = outer;
    } else {
      code = SHADER_LAST_DIM;
      paramsArr = [outer, r, 1, 0];
      lines = outer;
    }
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array(paramsArr));

  const bundle = createComputeBundle(
    ctx.device,
    code,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  const totalWorkgroups = Math.ceil(lines / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
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
  code: { default: SHADER, lastDim: SHADER_LAST_DIM, lastDimVec4: SHADER_LAST_DIM_VEC4 },
  recordIn,
};
