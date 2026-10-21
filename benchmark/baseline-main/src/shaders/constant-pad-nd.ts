import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.constant_pad_nd.default';

/**
 * WGSL compute shader for constant padding on up to 4D tensors.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/constant_pad_nd/constant_pad_nd.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/constant_pad_nd/constant_pad_nd.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

// Up to 4D. The handler right-aligns dims into [4] (leading entries = 1, left/
// right pad = 0 for unpadded/leading dims), so the shader is rank-agnostic and
// always iterates 4 dims. in_dims[d] = input extent, left[d] = that dim's
// left-pad, out_dims[d] = in_dims[d] + left[d] + right[d].
struct Params {
  out_dims: vec4<u32>,
  in_dims: vec4<u32>,
  left: vec4<u32>,
  out_numel: u32,
  value: f32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

// constant_pad_nd, gather form, NCHW row-major fp32. One thread per OUTPUT
// element: decode its 4D coords, subtract each dim's left-pad to get the input
// coord; if ALL input coords are in-bounds -> copy inp[flat_in], else write
// \`value\`. Pure copy/fill -> bit-exact. (CPU-derisked == torch at 0.)
@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let i = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (i >= params.out_numel) {
    return;
  }

  // decode out coords (last dim fastest)
  var rem = i;
  let o3 = rem % params.out_dims.w;
  rem = rem / params.out_dims.w;
  let o2 = rem % params.out_dims.z;
  rem = rem / params.out_dims.z;
  let o1 = rem % params.out_dims.y;
  rem = rem / params.out_dims.y;
  let o0 = rem % params.out_dims.x;

  // subtract left pad -> input coord (wrapping subtract; check via < in_dim on
  // the unsigned result catches negatives because they wrap to huge values)
  let c0 = o0 - params.left.x;
  let c1 = o1 - params.left.y;
  let c2 = o2 - params.left.z;
  let c3 = o3 - params.left.w;

  let in0 = o0 >= params.left.x && c0 < params.in_dims.x;
  let in1 = o1 >= params.left.y && c1 < params.in_dims.y;
  let in2 = o2 >= params.left.z && c2 < params.in_dims.z;
  let in3 = o3 >= params.left.w && c3 < params.in_dims.w;

  if (in0 && in1 && in2 && in3) {
    let in_idx =
        ((c0 * params.in_dims.y + c1) * params.in_dims.z + c2) * params.in_dims.w
        + c3;
    out[i] = inp[in_idx];
  } else {
    out[i] = params.value;
  }
}
`;

/**
 * Positional argument tuple for `aten::constant_pad_nd.default`:
 * - `[inTensor, pad, value, outTensor]`
 * - `[inTensor, pad, outTensor]` (value defaults to 0.0)
 */
export type ConstantPadNdArgs =
  | readonly [inTensor: Tensor, pad: readonly number[], outTensor: Tensor]
  | readonly [inTensor: Tensor, pad: readonly number[], value: number, outTensor: Tensor];

/**
 * Internal dispatch builder for `aten::constant_pad_nd.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConstantPadNdArgs): void {
  const [inTensor, pad] = args;
  const [value, outTensor] = args.length === 4 ? [args[2], args[3]] : [0.0, args[2]];

  if (inTensor.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }

  const nd = inTensor.shape.length;
  if (nd === 0 || nd > 4) {
    throw new Error(`${name}: rank must be 1..4 (got ${nd})`);
  }
  if (outTensor.shape.length !== nd) {
    throw new Error(`${name}: in/out rank mismatch (${nd} !== ${outTensor.shape.length})`);
  }
  if (pad.length % 2 !== 0) {
    throw new Error(`${name}: pad must be even-length`);
  }

  const npad = pad.length / 2;
  if (npad > nd) {
    throw new Error(`${name}: pad longer than rank`);
  }

  // Per-dim left/right pad (pad list is reversed-dim, from the LAST dim)
  const left = [0, 0, 0, 0];
  const right = [0, 0, 0, 0];
  for (let k = 0; k < npad; k++) {
    const d = nd - 1 - k;
    left[d] = pad[2 * k];
    right[d] = pad[2 * k + 1];
  }

  for (let d = 0; d < nd; d++) {
    if (left[d] < 0 || right[d] < 0) {
      throw new Error(`${name}: negative pad (cropping) not supported`);
    }
    const expected = inTensor.shape[d] + left[d] + right[d];
    if (outTensor.shape[d] !== expected) {
      throw new Error(`${name}: output dim ${d} mismatch (${expected} != ${outTensor.shape[d]})`);
    }
  }

  const device = ctx.device;
  const wgSize = 256;
  const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  // Right-align dims into 4D tuples: leading unused dims = 1 (pad = 0)
  const padLeft = (arr: readonly number[], fill: number): number[] => [
    ...Array<number>(4 - arr.length).fill(fill),
    ...arr,
  ];

  const buffer = new ArrayBuffer(64);
  new Uint32Array(buffer).set([
    ...padLeft(outTensor.shape, 1),
    ...padLeft(inTensor.shape, 1),
    ...padLeft(left.slice(0, nd), 0),
    outTensor.numel,
  ]);
  new Float32Array(buffer, 52, 1)[0] = value;

  const uniformBuffer = ctx.uniformBuffer(buffer);

  const bundle = createComputeBundle(
    device,
    SHADER,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
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
 * WebGPU compute kernel for ExecuTorch `aten::constant_pad_nd.default`.
 *
 * Copies input tensor into output tensor with constant padding on up to 4D tensors.
 */
export const constantPadNd: Shader<ConstantPadNdArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
