import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { isBroadcastable } from './utils/broadcasting';
import { encodeTensorMeta, TENSOR_META_WGSL } from './utils/meta';

const name = 'aten.sub.Tensor';

/**
 * WGSL compute shader for binary subtraction with broadcast support.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/binary_op/binary_sub_wgsl.h)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/binary_op/binary_sub_wgsl.h
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

override wg_size: u32 = 64u;
override alpha: f32 = 1.0;

fn op(a: f32, b: f32) -> f32 {
  return a - alpha * b;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= out_meta.numel) {
    return;
  }

  var same = true;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    if (in1_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u] ||
      in2_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u]) {
      same = false;
    }
  }
  if (same) {
    output[idx] = op(input1[idx], input2[idx]);
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
  output[idx] = op(input1[l1], input2[l2]);
}
`;

const SHADER_ELEMENTWISE_VEC4 = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input1: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> input2: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256u;
override alpha: f32 = 1.0;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = input1[idx] - alpha * input2[idx];
}
`;

const SHADER_ELEMENTWISE = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input1: array<f32>;
@group(0) @binding(1) var<storage, read> input2: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256u;
override alpha: f32 = 1.0;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = input1[idx] - alpha * input2[idx];
}
`;

const SHADER_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> input1: array<f16>;
@group(0) @binding(1) var<storage, read> input2: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

${TENSOR_META_WGSL}
@group(0) @binding(3) var<uniform> out_meta: TensorMeta;
@group(0) @binding(4) var<uniform> in1_meta: TensorMeta;
@group(0) @binding(5) var<uniform> in2_meta: TensorMeta;

override wg_size: u32 = 64u;
override alpha: f32 = 1.0;

fn op_f16(a: f16, b: f16) -> f16 {
  return a - f16(alpha) * b;
}

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= out_meta.numel) {
    return;
  }

  var same = true;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    if (in1_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u] ||
      in2_meta.sizes[d >> 2u][d & 3u] != out_meta.sizes[d >> 2u][d & 3u]) {
      same = false;
    }
  }
  if (same) {
    output[idx] = op_f16(input1[idx], input2[idx]);
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
  output[idx] = op_f16(input1[l1], input2[l2]);
}
`;

const SHADER_ELEMENTWISE_VEC4_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> input1: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read> input2: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f16>>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256u;
override alpha: f32 = 1.0;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = input1[idx] - f16(alpha) * input2[idx];
}
`;

const SHADER_ELEMENTWISE_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> input1: array<f16>;
@group(0) @binding(1) var<storage, read> input2: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256u;
override alpha: f32 = 1.0;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = input1[idx] - f16(alpha) * input2[idx];
}
`;

/**
 * Positional argument tuple for `aten::sub.Tensor`:
 * Standard ExecuTorch schema: `[in1, in2, alpha, out]`
 * Or 3-arg variant: `[in1, in2, out]`
 */
export type SubArgs =
  | readonly [in1: Tensor, in2: Tensor, out: Tensor]
  | readonly [in1: Tensor, in2: Tensor, alpha: number, out: Tensor];

/**
 * Internal dispatch builder for `aten::sub.Tensor`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: SubArgs): void {
  const [in1, in2] = args;
  const [alpha, out] = args.length === 4 ? [args[2], args[3]] : [1.0, args[2]];

  if (
    (in1.dtype !== 'float32' && in1.dtype !== 'float16') ||
    in2.dtype !== in1.dtype ||
    out.dtype !== in1.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (!isBroadcastable(in1.shape, out.shape) || !isBroadcastable(in2.shape, out.shape)) {
    throw new Error(`${name}: Input shapes are not broadcastable to output shape`);
  }

  const device = ctx.device;
  const isF16 = in1.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const sameShape =
    in1.shape.length === out.shape.length &&
    in2.shape.length === out.shape.length &&
    in1.shape.every((s, i) => s === out.shape[i]) &&
    in2.shape.every((s, i) => s === out.shape[i]);

  const wgSize = 256;

  if (sameShape) {
    const isVec4 = out.numel % 4 === 0;
    let code = isVec4 ? SHADER_ELEMENTWISE_VEC4 : SHADER_ELEMENTWISE;
    if (isF16) {
      code = isVec4 ? SHADER_ELEMENTWISE_VEC4_F16 : SHADER_ELEMENTWISE_F16;
    }
    const elements = isVec4 ? out.numel / 4 : out.numel;
    const totalWorkgroups = Math.ceil(elements / wgSize);
    const workgroupCountX = Math.min(totalWorkgroups, 65535);
    const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    const paramsBuffer = ctx.uniformBuffer(new Uint32Array([elements]));

    const bundle = createComputeBundle(
      device,
      code,
      [
        { binding: 0, buffer: in1.buffer },
        { binding: 1, buffer: in2.buffer },
        { binding: 2, buffer: out.buffer },
        { binding: 3, buffer: paramsBuffer },
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
    return;
  }

  const outRank = out.shape.length;
  const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(out));
  const in1MetaBuffer = ctx.uniformBuffer(encodeTensorMeta(in1, outRank));
  const in2MetaBuffer = ctx.uniformBuffer(encodeTensorMeta(in2, outRank));

  const totalWorkgroups = Math.ceil(out.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_F16 : SHADER,
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
 * WebGPU compute kernel for ExecuTorch `aten::sub.Tensor`.
 *
 * Computes elementwise `output = input1 - alpha * input2` with NumPy/PyTorch-style
 * broadcasting (right-aligned, dimensions equal or 1, up to rank 8). Features an
 * automatic fast path when input and output shapes match.
 */
export const sub: Shader<SubArgs, typeof name> = {
  name,
  code: {
    default: SHADER,
    elementwise: SHADER_ELEMENTWISE,
    elementwiseVec4: SHADER_ELEMENTWISE_VEC4,
    defaultF16: SHADER_F16,
    elementwiseF16: SHADER_ELEMENTWISE_F16,
    elementwiseVec4F16: SHADER_ELEMENTWISE_VEC4_F16,
  },
  recordIn,
};
