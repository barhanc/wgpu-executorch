import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.upsample_bilinear2d.vec';

/**
 * WGSL compute shader for bilinear 2D upsampling.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/upsample_bilinear2d/upsample_bilinear2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/upsample_bilinear2d/upsample_bilinear2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  align_corners: u32,
  _p0: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Bilinear NCHW fp32 upsample; src-index matches ATen upsample_bilinear2d.
fn src_index(dst: u32, insz: u32, outsz: u32, align: u32) -> f32 {
  if (align == 1u) {
    if (outsz <= 1u) {
      return 0.0;
    }
    return f32(dst) * f32(insz - 1u) / f32(outsz - 1u);
  }
  return (f32(dst) + 0.5) * f32(insz) / f32(outsz) - 0.5;
}

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

  let sh = src_index(oh, params.IH, params.OH, params.align_corners);
  let sw = src_index(ow, params.IW, params.OW, params.align_corners);

  let h0 = i32(floor(sh));
  let w0 = i32(floor(sw));
  let lh = sh - f32(h0);
  let lw = sw - f32(w0);

  let ih_max = i32(params.IH) - 1;
  let iw_max = i32(params.IW) - 1;
  let h0c = u32(clamp(h0, 0, ih_max));
  let h1c = u32(clamp(h0 + 1, 0, ih_max));
  let w0c = u32(clamp(w0, 0, iw_max));
  let w1c = u32(clamp(w0 + 1, 0, iw_max));

  let base = (n * params.C + c) * params.IH;
  let r0 = (base + h0c) * params.IW;
  let r1 = (base + h1c) * params.IW;
  let v00 = inp[r0 + w0c];
  let v01 = inp[r0 + w1c];
  let v10 = inp[r1 + w0c];
  let v11 = inp[r1 + w1c];

  let top = v00 + (v01 - v00) * lw;
  let bot = v10 + (v11 - v10) * lw;
  out[i] = top + (bot - top) * lh;
}
`;

/**
 * Positional argument tuple for `aten::upsample_bilinear2d.vec`:
 * `[in, output_size, align_corners, scales, out]`
 */
export type UpsampleBilinear2dArgs = readonly [
  inTensor: Tensor,
  outputSize: readonly number[],
  alignCorners: boolean,
  scales: readonly number[] | undefined,
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::upsample_bilinear2d.vec`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: UpsampleBilinear2dArgs): void {
  const inTensor = args[0];
  const alignCorners = args[2];
  const outTensor = args[4];

  if (inTensor.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (inTensor.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D in/out tensors`);
  }

  const [n, c, ih, iw] = inTensor.shape as [number, number, number, number];
  const [outN, outC, oh, ow] = outTensor.shape as [number, number, number, number];

  if (outN !== n || outC !== c) {
    throw new Error(`${name}: N/C mismatch`);
  }
  if (ih === 0 || iw === 0 || oh === 0 || ow === 0) {
    throw new Error(`${name}: zero spatial dim`);
  }

  const outNumel = outTensor.numel;
  const device = ctx.device;
  const wgSize = 256;
  const totalWorkgroups = Math.ceil(outNumel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
  const strideX = workgroupCountX * wgSize;

  const buffer = new ArrayBuffer(32);
  const u32View = new Uint32Array(buffer);
  u32View.set([n, c, ih, iw, oh, ow, alignCorners ? 1 : 0, 0]);

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
    { stride_x: strideX, wg_size: wgSize }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::upsample_bilinear2d.vec`.
 *
 * Implements bilinear 2D upsampling for NCHW float32 tensors.
 */
export const upsampleBilinear2d: Shader<UpsampleBilinear2dArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
