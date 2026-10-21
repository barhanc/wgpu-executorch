import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.max_pool2d_with_indices.default';

/**
 * WGSL compute shader for 2D max pooling (values + optional indices).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/max_pool2d/max_pool2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/max_pool2d/max_pool2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  kH: u32,
  kW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<storage, read_write> out_vals: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> out_idx: array<i32>;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill
override write_indices: u32 = 0u; // 1 = also write out_idx (compile-time gate, mirrors Vulkan)

// max_pool2d (values [+ optional indices]), NCHW row-major, fp32. One thread per
// output element (n, c, oh, ow); gather the window, take the max. General
// stride/pad/dilation. Argmax is ALWAYS tracked (mirrors Vulkan Pool.cpp, which
// computes indices unconditionally and only gates the final write) so the
// values-only and with-indices paths share one kernel; write_indices==0 skips
// only the out_idx store, not the tracking, keeping both paths bit-identical on
// out_vals. Indices are the flat (ih*IW+iw) spatial-plane offset (matches
// torch.nn.functional.max_pool2d(return_indices=True)'s documented convention,
// NOT an absolute offset into the full NCHW tensor). Pad cells are skipped
// (init -inf); out_idx is left at its bound (dummy, when unused) value.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.C * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let c = (i / (params.OW * params.OH)) % params.C;
  let n = i / (params.OW * params.OH * params.C);

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_c_base = (n * params.C + c) * params.IH; // * IW added per-row below

  var best: f32 = -3.4e38; // large finite negative max-init (Dawn/Tint rejects -3.40282347e38: > f32 max); matches q4gsw_dq8ca convention
  var best_idx: i32 = 0;
  for (var kh: u32 = 0u; kh < params.kH; kh = kh + 1u) {
    let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    let in_row = (in_c_base + u32(ih)) * params.IW;
    for (var kw: u32 = 0u; kw < params.kW; kw = kw + 1u) {
      let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      let v = inp[in_row + u32(iw)];
      if (v > best) {
        best = v;
        best_idx = ih * iW + iw;
      }
    }
  }
  out_vals[i] = best;
  if (write_indices != 0u) {
    out_idx[i] = best_idx;
  }
}
`;

/**
 * Positional argument tuple for `aten::max_pool2d_with_indices.default`:
 * `[in, kernel_size, stride, padding, dilation, ceil_mode, outs]`
 * where `outs` is `[values, indices]`.
 */
export type MaxPool2dArgs = readonly [
  inTensor: Tensor,
  kernelSize: readonly number[],
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  ceilMode: boolean,
  outs: readonly Tensor[],
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `aten::max_pool2d_with_indices.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: MaxPool2dArgs): void {
  const [inTensor, kernelSize, stride, padding, dilation, ceilMode, outs] = args;

  if (outs.length !== 2) {
    throw new Error(`${name}: expected 2 outputs [values, indices], got ${outs.length}`);
  }
  const [outVals, outIdx] = outs;

  if (inTensor.dtype !== 'float32' || outVals.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 value tensors are currently supported`);
  }
  if (outIdx.dtype !== 'int32') {
    throw new Error(`${name}: indices output must be int32`);
  }
  if (inTensor.shape.length !== 4) {
    throw new Error(`${name}: input must be 4D NCHW`);
  }

  const [b, c, ih, iw] = inTensor.shape as [number, number, number, number];
  const [kH, kW] = parseHw(kernelSize, 'kernel_size');
  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');

  const ohNum = ih + 2 * pH - (dH * (kH - 1) + 1);
  const owNum = iw + 2 * pW - (dW * (kW - 1) + 1);
  const oh = Math.max(0, (ceilMode ? Math.ceil(ohNum / sH) : Math.floor(ohNum / sH)) + 1);
  const ow = Math.max(0, (ceilMode ? Math.ceil(owNum / sW) : Math.floor(owNum / sW)) + 1);

  if (
    outVals.shape.length !== 4 ||
    outVals.shape[0] !== b ||
    outVals.shape[1] !== c ||
    outVals.shape[2] !== oh ||
    outVals.shape[3] !== ow
  ) {
    throw new Error(`${name}: values output shape ${outVals.shape} != [${b}, ${c}, ${oh}, ${ow}]`);
  }
  if (
    outIdx.shape.length !== 4 ||
    outIdx.shape[0] !== b ||
    outIdx.shape[1] !== c ||
    outIdx.shape[2] !== oh ||
    outIdx.shape[3] !== ow
  ) {
    throw new Error(`${name}: indices output shape ${outIdx.shape} != [${b}, ${c}, ${oh}, ${ow}]`);
  }

  // PoolParams: input/output geometry, kernel, stride, pad, dilation
  // prettier-ignore
  const paramsBuffer = ctx.uniformBuffer(
    new Uint32Array([
      b, c, ih, iw,
      oh, ow,
      kH, kW,
      sH, sW,
      pH, pW,
      dH, dW,
      0, 0,
    ])
  );

  const wgSize = 256;
  const total = b * c * oh * ow;
  const workgroupCountX = Math.min(Math.ceil(total / wgSize), 65535);

  const bundle = createComputeBundle(
    ctx.device,
    SHADER,
    [
      { binding: 0, buffer: outVals.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
      { binding: 3, buffer: outIdx.buffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize, stride_x: workgroupCountX * wgSize, write_indices: 1 }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY: Math.ceil(total / wgSize / 65535),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::max_pool2d_with_indices.default`.
 *
 * Computes max pooling over a 2D NCHW input — windowed gather max per output
 * element, plus flat in-window argmax indices as an int32 tensor.
 */
export const maxPool2d: Shader<MaxPool2dArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
