import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { floatKind, scalarCode } from './utils/scalar';

const name = 'aten.upsample_nearest2d.vec';

/**
 * WGSL compute shader for nearest-neighbour 2D upsampling.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/upsample_nearest2d/upsample_nearest2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/upsample_nearest2d/upsample_nearest2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
alias scalar_t = f32;

@group(0) @binding(0) var<storage, read_write> out: array<scalar_t>;
@group(0) @binding(1) var<storage, read> inp: array<scalar_t>;

struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Nearest-neighbour 2D upsample, NCHW row-major, fp32. One thread per output
// element (n, c, oh, ow). oh -> floor(oh*IH/OH), matching real ATen's
// nearest_neighbor_compute_source_index (UpSample.h) — intentionally NOT
// mirroring Vulkan's upsample_2d.glsl, which uses the half-pixel-center
// formula correct for bilinear but not for non-exact nearest (see diff summary).
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

  let ih = (oh * params.IH) / params.OH;
  let iw = (ow * params.IW) / params.OW;
  let in_idx = ((n * params.C + c) * params.IH + ih) * params.IW + iw;
  out[i] = inp[in_idx];
}
`;

const CODE = scalarCode(SHADER);

/**
 * Positional argument tuple for `aten::upsample_nearest2d.vec`:
 * `[in, output_size, scales, out]`
 */
export type UpsampleNearest2dArgs = readonly [
  inTensor: Tensor,
  outputSize: undefined,
  scales: readonly number[],
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::upsample_nearest2d.vec`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: UpsampleNearest2dArgs): void {
  const [inTensor, , , outTensor] = args;

  const kind = floatKind(name, [inTensor, outTensor]);
  if (inTensor.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: input and output must be 4D NCHW`);
  }
  const [b, c, ih, iw] = inTensor.shape as [number, number, number, number];
  const [ob, oc, oh, ow] = outTensor.shape as [number, number, number, number];
  if (ob !== b || oc !== c) {
    throw new Error(`${name}: output batch/channels must match input`);
  }
  if (oh < ih || ow < iw) {
    throw new Error(`${name}: output spatial dims must be >= input spatial dims`);
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([b, c, ih, iw, oh, ow, 0, 0]));

  const wgSize = 256;
  const total = b * c * oh * ow;
  const workgroupCountX = Math.min(Math.ceil(total / wgSize), 65535);

  const bundle = createComputeBundle(
    ctx.device,
    CODE[kind],
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize, stride_x: workgroupCountX * wgSize }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY: Math.ceil(total / wgSize / 65535),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::upsample_nearest2d.vec`.
 *
 * Nearest-neighbour 2D upsampling — each output texel copies the input texel
 * at `floor(o * I / O)` (ATen convention).
 */
export const upsampleNearest2d: Shader<UpsampleNearest2dArgs, typeof name> = {
  name,
  code: CODE,
  recordIn,
};
