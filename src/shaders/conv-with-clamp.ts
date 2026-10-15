import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { floatKind, scalarCode, variantCode } from './utils/scalar';

const name = 'et_vk.conv_with_clamp.default';

/**
 * WGSL compute shader for fused convolution with clamp bounds.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/conv_with_clamp/conv_with_clamp.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/conv_with_clamp/conv_with_clamp.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_WITH_CLAMP = /* wgsl */ `
alias scalar_t = f32;

@group(0) @binding(0) var<storage, read_write> t_out: array<scalar_t>;
@group(0) @binding(1) var<storage, read> t_x: array<scalar_t>;
@group(0) @binding(2) var<storage, read> t_weight: array<scalar_t>;
@group(0) @binding(3) var<storage, read> t_bias: array<scalar_t>;

struct Params {
  N: u32,
  IC: u32,
  H_in: u32,
  W_in: u32,
  OC: u32,
  H_out: u32,
  W_out: u32,
  Kh: u32,
  Kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dil_h: u32,
  dil_w: u32,
  has_bias: u32,
  numel: u32,
  groups: u32,
  ic_per_group: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  output_min: f32,
  output_max: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  // 2D-folded flat index (lifts the 65535 1D-dispatch cap for large numel).
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.numel) {
    return;
  }
  // Unravel the NCHW output index -> (n, oc, oh, ow).
  let ow = idx % params.W_out;
  var r = idx / params.W_out;
  let oh = r % params.H_out;
  r = r / params.H_out;
  let oc = r % params.OC;
  let n = r / params.OC;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = f32(t_bias[oc]);
  }
  // Grouped conv: output channel oc belongs to group g and dots only that
  // group's ic_per_group input channels. weight is [OC, IC/groups, Kh, Kw];
  // groups==1 (ic_per_group==IC, g==0) is the general dense case.
  let oc_per_group = params.OC / params.groups;
  let g = oc / oc_per_group;
  let ic_base = g * params.ic_per_group;
  for (var ic_local: u32 = 0u; ic_local < params.ic_per_group; ic_local = ic_local + 1u) {
    let ic = ic_base + ic_local;
    for (var kh: u32 = 0u; kh < params.Kh; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.stride_h) - i32(params.pad_h) +
      i32(kh) * i32(params.dil_h);
      if (ih < 0 || ih >= i32(params.H_in)) {
        continue;
      }
      let in_row = ((n * params.IC + ic) * params.H_in + u32(ih)) * params.W_in;
      let w_row =
      ((oc * params.ic_per_group + ic_local) * params.Kh + kh) * params.Kw;
      for (var kw: u32 = 0u; kw < params.Kw; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.stride_w) - i32(params.pad_w) +
        i32(kw) * i32(params.dil_w);
        if (iw < 0 || iw >= i32(params.W_in)) {
          continue;
        }
        acc = acc + f32(t_x[in_row + u32(iw)]) * f32(t_weight[w_row + kw]);
      }
    }
  }
  t_out[idx] = scalar_t(clamp(acc, params.output_min, params.output_max));
}
`;

/**
 * WGSL compute shader for fused convolution with clamp bounds as
 * implicit-im2col tiled GEMM (groups == 1).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_WITH_CLAMP_GEMM = /* wgsl */ `
alias scalar_t = f32;

@group(0) @binding(0) var<storage, read_write> t_out: array<scalar_t>;
@group(0) @binding(1) var<storage, read> t_x: array<scalar_t>;
@group(0) @binding(2) var<storage, read> t_weight: array<scalar_t>;
@group(0) @binding(3) var<storage, read> t_bias: array<scalar_t>;

struct Params {
  N: u32,
  IC: u32,
  H_in: u32,
  W_in: u32,
  OC: u32,
  H_out: u32,
  W_out: u32,
  Kh: u32,
  Kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dil_h: u32,
  dil_w: u32,
  has_bias: u32,
  numel: u32,
  groups: u32,
  ic_per_group: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  output_min: f32,
  output_max: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Tiled GEMM over implicit im2col (groups == 1): M=OC, N=N*H_out*W_out,
// K=IC*Kh*Kw. A=weight [OC,K] (contiguous OIHW); B=input sampled on the fly,
// out-of-range -> 0.0 implements padding. bias per-ROW (OC). Output written
// NCHW with the clamp bounds applied.
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn gK() -> u32 { return params.IC * params.Kh * params.Kw; }
fn gN() -> u32 { return params.N * params.H_out * params.W_out; }

fn read_a(m: u32, kk: u32) -> f32 { // weight[OC, IC*Kh*Kw] row-major
  if (m < params.OC && kk < gK()) {
    return f32(t_weight[m * gK() + kk]);
  }
  return 0.0;
}

fn read_b(kk: u32, n: u32) -> f32 { // im2col sample of input[N,IC,H_in,W_in]
  if (kk >= gK() || n >= gN()) {
    return 0.0;
  }
  let ohw = params.H_out * params.W_out;
  let b = n / ohw;
  let sp = n % ohw;
  let oh = sp / params.W_out;
  let ow = sp % params.W_out;
  let khw = params.Kh * params.Kw;
  let ic = kk / khw;
  let r = kk % khw;
  let kh = r / params.Kw;
  let kw = r % params.Kw;
  let ih = i32(oh) * i32(params.stride_h) - i32(params.pad_h) + i32(kh) * i32(params.dil_h);
  let iw = i32(ow) * i32(params.stride_w) - i32(params.pad_w) + i32(kw) * i32(params.dil_w);
  if (ih < 0 || ih >= i32(params.H_in) || iw < 0 || iw >= i32(params.W_in)) {
    return 0.0; // padding
  }
  return f32(t_x[((b * params.IC + ic) * params.H_in + u32(ih)) * params.W_in + u32(iw)]);
}

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>) {
  let M = params.OC;
  let N = gN();
  let K = gK();
  let tile_row0 = wg_id.y * TILE;
  let tile_col0 = wg_id.x * TILE;
  let tile_row = local_id.y * RPT;
  let tile_col = local_id.x * RPT;

  var acc: array<array<f32, 4>, 4>;
  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      acc[ir][ic] = 0.0;
    }
  }

  let num_tiles = (K + TILE - 1u) / TILE;
  for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * TILE;
    for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
      let arow = local_id.y * RPT + ir;
      for (var kk: u32 = 0u; kk < RPT; kk = kk + 1u) {
        let col = local_id.x * RPT + kk;
        a_sub[arow][col] = read_a(tile_row0 + arow, k_start + col);
        b_sub[arow][col] = read_b(k_start + arow, tile_col0 + col);
      }
    }
    workgroupBarrier();
    for (var k: u32 = 0u; k < TILE; k = k + 1u) {
      for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
        let aval = a_sub[tile_row + ir][k];
        for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
          acc[ir][ic] = acc[ir][ic] + aval * b_sub[k][tile_col + ic];
        }
      }
    }
    workgroupBarrier();
  }

  for (var ir: u32 = 0u; ir < RPT; ir = ir + 1u) {
    for (var ic: u32 = 0u; ic < RPT; ic = ic + 1u) {
      let m = tile_row0 + tile_row + ir; // oc
      let n = tile_col0 + tile_col + ic; // spatial index
      if (m < M && n < N) {
        var val = acc[ir][ic];
        if (params.has_bias != 0u) {
          val = val + f32(t_bias[m]);
        }
        let ohw = params.H_out * params.W_out;
        let b = n / ohw;
        let sp = n % ohw;
        let oh = sp / params.W_out;
        let ow = sp % params.W_out;
        t_out[((b * params.OC + m) * params.H_out + oh) * params.W_out + ow] =
        scalar_t(clamp(val, params.output_min, params.output_max));
      }
    }
  }
}
`;

const CODE_CONV_WITH_CLAMP = scalarCode(SHADER_CONV_WITH_CLAMP);
const CODE_CONV_WITH_CLAMP_GEMM = scalarCode(SHADER_CONV_WITH_CLAMP_GEMM);

/**
 * Positional argument tuple for `et_vk::conv_with_clamp.default`:
 * `[in, weight, bias, stride, padding, dilation, transposed, output_padding, groups, min, max, out]`
 */
export type ConvWithClampArgs = readonly [
  inTensor: Tensor,
  weight: Tensor,
  bias: Tensor | undefined,
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  transposed: boolean,
  outputPadding: readonly number[],
  groups: number,
  outputMin: number | undefined,
  outputMax: number | undefined,
  outTensor: Tensor,
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `et_vk::conv_with_clamp.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConvWithClampArgs): void {
  const [
    inTensor,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    outputPadding,
    groups,
    outputMin,
    outputMax,
    outTensor,
  ] = args;

  const kind = floatKind(name, [inTensor, weight, bias, outTensor]);
  if (inTensor.shape.length !== 4 || weight.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input, weight, and output`);
  }
  if (transposed) {
    throw new Error(`${name}: transposed conv is unsupported in conv_with_clamp`);
  }
  if (groups < 1) {
    throw new Error(`${name}: groups must be >= 1`);
  }

  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');

  if (sH <= 0 || sW <= 0) {
    throw new Error(`${name}: stride must be positive`);
  }

  const [n, ic, hIn, wIn] = inTensor.shape as [number, number, number, number];
  const oc = weight.shape[0];
  const kh = weight.shape[2];
  const kw = weight.shape[3];

  if (outputPadding.some((v) => v !== 0)) {
    throw new Error(`${name}: non-zero output_padding unsupported for non-transposed conv`);
  }

  const icPerGroup = weight.shape[1];
  if (ic % groups !== 0 || oc % groups !== 0 || icPerGroup * groups !== ic) {
    throw new Error(`${name}: bad shape (IC/groups mismatch)`);
  }

  const hEff = hIn + 2 * pH - dH * (kh - 1) - 1;
  const wEff = wIn + 2 * pW - dW * (kw - 1) - 1;
  if (hEff < 0 || wEff < 0) {
    throw new Error(`${name}: invalid geometry (kernel larger than input)`);
  }

  const expectedHout = Math.floor(hEff / sH) + 1;
  const expectedWout = Math.floor(wEff / sW) + 1;
  if (outTensor.shape[2] !== expectedHout || outTensor.shape[3] !== expectedWout) {
    throw new Error(`${name}: output dims inconsistent with conv2d formula`);
  }

  const hasBias = bias !== undefined;
  if (hasBias && (bias.buffer === null || bias.shape[0] !== oc)) {
    throw new Error(`${name}: bias must be fp32 [OC]`);
  }

  const minVal = outputMin !== undefined ? outputMin : -Infinity;
  const maxVal = outputMax !== undefined ? outputMax : Infinity;

  const numel = outTensor.numel;
  const device = ctx.device;
  const wgSize = 64;

  const buffer = new ArrayBuffer(96);
  const u32View = new Uint32Array(buffer);
  const f32View = new Float32Array(buffer);

  // ConvWithClampParams (96 B): geometry, kernel, stride, pad, dilation, flags
  // prettier-ignore
  u32View.set([
    n, ic, hIn, wIn,
    oc, expectedHout, expectedWout,
    kh, kw,
    sH, sW,
    pH, pW,
    dH, dW,
    hasBias ? 1 : 0, numel,
    groups, icPerGroup,
    0, 0, 0,
  ]);
  f32View[22] = minVal; // output_min
  f32View[23] = maxVal; // output_max

  const uniformBuffer = ctx.uniformBuffer(buffer);
  const biasBuffer = hasBias ? bias.buffer : weight.buffer;

  // Variant selection mirrors the ExecuTorch WebGPU runtime: groups==1 goes
  // through the implicit-im2col tiled GEMM, grouped/depthwise convs through
  // the direct kernel.
  const code = groups === 1 ? CODE_CONV_WITH_CLAMP_GEMM[kind] : CODE_CONV_WITH_CLAMP[kind];

  let workgroupCountX: number;
  let workgroupCountY: number;
  let constants: Record<string, number>;

  if (groups !== 1) {
    // Direct kernel: one thread per output element, 2D spill
    const totalWorkgroups = Math.ceil(numel / wgSize);
    workgroupCountX = Math.min(totalWorkgroups, 65535);
    workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    constants = { wg_size: wgSize };
  } else {
    // GEMM kernel: fixed 8x8 workgroup covering a 32x32 output tile;
    // grid over (N = batch*H_out*W_out cols, M = OC rows)
    const tile = 32;
    workgroupCountX = Math.ceil((n * expectedHout * expectedWout) / tile);
    workgroupCountY = Math.ceil(oc / tile);
    if (workgroupCountX > 65535 || workgroupCountY > 65535) {
      throw new Error(`${name}: GEMM tile grid exceeds dispatch limit`);
    }
    constants = {};
  }

  const bundle = createComputeBundle(
    device,
    code,
    [
      { binding: 0, buffer: outTensor.buffer },
      { binding: 1, buffer: inTensor.buffer },
      { binding: 2, buffer: weight.buffer },
      { binding: 3, buffer: biasBuffer },
      { binding: 4, buffer: uniformBuffer },
    ],
    constants
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `et_vk::conv_with_clamp.default`.
 *
 * Implements 2D direct and grouped convolutions with clamping. Dense
 * convolutions are evaluated as implicit-im2col tiled GEMM, grouped and
 * depthwise convolutions via the direct kernel.
 */
export const convWithClamp: Shader<ConvWithClampArgs, typeof name> = {
  name,
  code: variantCode({
    convWithClamp: CODE_CONV_WITH_CLAMP,
    convWithClampGemm: CODE_CONV_WITH_CLAMP_GEMM,
  }),
  recordIn,
};
