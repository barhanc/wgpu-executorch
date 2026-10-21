import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

const name = 'aten.convolution.default';

/**
 * WGSL compute shader for direct 2D convolution (non-transposed).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Direct 2D convolution (non-transposed), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. Supports general stride/padding/dilation and
// groups. input [B,IC,IH,IW]; weight [OC, IC/groups, KH, KW]; bias [OC] (gated).
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
    let ic = ic0 + icg;
    let in_c_base = (b * params.IC + ic) * params.IH; // *IW added per-row below
    let w_c_base = (oc * icpg + icg) * params.KH; // *KW added per-row below
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= iH) {
        continue;
      }
      let in_row = (in_c_base + u32(ih)) * params.IW;
      let w_row = (w_c_base + kh) * params.KW;
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= iW) {
          continue;
        }
        acc = acc + input[in_row + u32(iw)] * weight[w_row + kw];
      }
    }
  }
  out[i] = acc;
}
`;

/**
 * WGSL compute shader for direct 2D convolution as implicit-im2col tiled GEMM.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv2d_gemm.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV2D_GEMM = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

// Standard (groups==1) conv2d as an implicit-im2col tiled GEMM, reusing the
// linear_fp32_tiled skeleton (TILE=32, RPT=4, shared-mem tiles). M=OC, N=B*OH*OW,
// K=IC*KH*KW. A=weight [OC,K] (contiguous OIHW); B=input sampled on the fly (im2col),
// out-of-range -> 0.0 implements padding. bias is per-ROW (OC). Output written NCHW.
const TILE: u32 = 32u;
const RPT: u32 = 4u;

var<workgroup> a_sub: array<array<f32, 32>, 32>;
var<workgroup> b_sub: array<array<f32, 32>, 32>;

fn gK() -> u32 { return params.IC * params.KH * params.KW; }
fn gN() -> u32 { return params.B * params.OH * params.OW; }

fn read_a(m: u32, kk: u32) -> f32 { // weight[OC, IC*KH*KW] row-major
  if (m < params.OC && kk < gK()) {
    return weight[m * gK() + kk];
  }
  return 0.0;
}

fn read_b(kk: u32, n: u32) -> f32 { // im2col sample of input[B,IC,IH,IW]
  if (kk >= gK() || n >= gN()) {
    return 0.0;
  }
  let ohw = params.OH * params.OW;
  let b = n / ohw;
  let sp = n % ohw;
  let oh = sp / params.OW;
  let ow = sp % params.OW;
  let khw = params.KH * params.KW;
  let ic = kk / khw;
  let r = kk % khw;
  let kh = r / params.KW;
  let kw = r % params.KW;
  let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
  let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
  if (ih < 0 || ih >= i32(params.IH) || iw < 0 || iw >= i32(params.IW)) {
    return 0.0; // padding
  }
  return input[((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw)];
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
          val = val + bias[m];
        }
        let ohw = params.OH * params.OW;
        let b = n / ohw;
        let sp = n % ohw;
        let oh = sp / params.OW;
        let ow = sp % params.OW;
        out[((b * params.OC + m) * params.OH + oh) * params.OW + ow] = val;
      }
    }
  }
}
`;

/**
 * WGSL compute shader for transposed 2D convolution (gather form).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER_CONV_TRANSPOSE2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

// Transposed 2D convolution (gather form), NCHW row-major, fp32. ONE thread per
// (b, oc, oh, ow) output element. weight layout = torch convT [IC, OC/groups,
// KH, KW] (NOT flipped). For each kernel tap (kh,kw): an input row ih
// contributes iff (oh + pH - kh*dH) is divisible by sH and ih in range (the
// scatter-inversion). CPU-derisked vs torch.conv_transpose2d to fp64 round-off
// (/tmp/convtr_derisk.py), incl. non-square spatial + non-square kernel.
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;
  let oc_in_g = oc % ocpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    let num_h = i32(oh) + i32(params.pH) - i32(kh) * i32(params.dH);
    if (num_h % i32(params.sH) != 0) {
      continue;
    }
    let ih = num_h / i32(params.sH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
      let num_w = i32(ow) + i32(params.pW) - i32(kw) * i32(params.dW);
      if (num_w % i32(params.sW) != 0) {
        continue;
      }
      let iw = num_w / i32(params.sW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
        let ic = ic0 + icg;
        let in_idx =
        ((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw);
        // weight [IC, OC/groups, KH, KW]: index (ic, oc_in_g, kh, kw)
        let w_idx =
        ((ic * ocpg + oc_in_g) * params.KH + kh) * params.KW + kw;
        acc = acc + input[in_idx] * weight[w_idx];
      }
    }
  }
  out[i] = acc;
}
`;

/**
 * Positional argument tuple for `aten::convolution.default`:
 * `[in, weight, bias, stride, padding, dilation, transposed, output_padding, groups, out]`
 */
export type ConvolutionArgs = readonly [
  inTensor: Tensor,
  weight: Tensor,
  bias: Tensor | undefined,
  stride: readonly number[],
  padding: readonly number[],
  dilation: readonly number[],
  transposed: boolean,
  outputPadding: readonly number[],
  groups: number,
  outTensor: Tensor,
];

function parseHw(v: readonly number[], argName: string): [number, number] {
  if (v.length === 1) return [v[0], v[0]];
  if (v.length === 2) return [v[0], v[1]];
  throw new Error(`${name}: ${argName} must be 1 or 2 elements`);
}

/**
 * Internal dispatch builder for `aten::convolution.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConvolutionArgs): void {
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
    outTensor,
  ] = args;

  if (inTensor.dtype !== 'float32' || weight.dtype !== 'float32' || outTensor.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
  }
  if (inTensor.shape.length !== 4 || weight.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input, weight, and output`);
  }

  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');
  const [opH, opW] = parseHw(outputPadding, 'output_padding');

  const [b, ic, ih, iw] = inTensor.shape as [number, number, number, number];
  const hasBias = bias !== undefined;

  let oc: number;
  let kh: number;
  let kw: number;

  if (transposed) {
    // weight layout for transposed conv: [IC, OC / groups, KH, KW]
    kh = weight.shape[2];
    kw = weight.shape[3];
    const ocpg = weight.shape[1];
    oc = ocpg * groups;

    if (weight.shape[0] !== ic) {
      throw new Error(`${name}: weight dim0 !== IC (${weight.shape[0]} != ${ic})`);
    }
    if (opH >= sH || opW >= sW) {
      throw new Error(`${name}: output_padding >= stride`);
    }

    const expectedOh = (ih - 1) * sH - 2 * pH + dH * (kh - 1) + opH + 1;
    const expectedOw = (iw - 1) * sW - 2 * pW + dW * (kw - 1) + opW + 1;
    if (outTensor.shape[2] !== expectedOh || outTensor.shape[3] !== expectedOw) {
      throw new Error(`${name}: expected output spatial [${expectedOh}, ${expectedOw}]`);
    }
  } else {
    // weight layout for direct conv: [OC, IC / groups, KH, KW]
    oc = weight.shape[0];
    kh = weight.shape[2];
    kw = weight.shape[3];

    if (outputPadding.some((v) => v !== 0)) {
      throw new Error(`${name}: non-zero output_padding unsupported for non-transposed conv`);
    }
    if (weight.shape[1] !== ic / groups) {
      throw new Error(`${name}: weight in-ch (${weight.shape[1]}) != IC/groups (${ic / groups})`);
    }

    const ohNum = ih + 2 * pH - dH * (kh - 1) - 1;
    const owNum = iw + 2 * pW - dW * (kw - 1) - 1;
    if (ohNum < 0 || owNum < 0) {
      throw new Error(`${name}: invalid geometry (kernel larger than input)`);
    }
    const expectedOh = Math.floor(ohNum / sH) + 1;
    const expectedOw = Math.floor(owNum / sW) + 1;
    if (outTensor.shape[2] !== expectedOh || outTensor.shape[3] !== expectedOw) {
      throw new Error(`${name}: expected output spatial [${expectedOh}, ${expectedOw}]`);
    }
  }

  const [, , oh, ow] = outTensor.shape as [number, number, number, number];

  const device = ctx.device;
  const wgSize = 256;

  // ConvParams (80 B): input/output geometry, kernel, stride, pad, dilation, flags
  const buffer = new ArrayBuffer(80);
  // prettier-ignore
  new Uint32Array(buffer).set([
    b, ic, ih, iw,
    oc, oh, ow,
    kh, kw,
    sH, sW,
    pH, pW,
    dH, dW,
    groups, hasBias ? 1 : 0,
  ]);

  const uniformBuffer = ctx.uniformBuffer(buffer);
  const biasBuffer = hasBias ? bias.buffer : weight.buffer;

  // Variant selection mirrors the ExecuTorch WebGPU runtime: groups==1 goes
  // through the implicit-im2col tiled GEMM, grouped/depthwise convs through the
  // direct kernel, transposed convs through the gather kernel.
  let code = SHADER_CONV2D;
  if (transposed) {
    code = SHADER_CONV_TRANSPOSE2D;
  } else if (groups === 1) {
    code = SHADER_CONV2D_GEMM;
  }

  let workgroupCountX: number;
  let workgroupCountY: number;
  let constants: Record<string, number>;

  if (transposed || groups !== 1) {
    // Direct / transposed kernels: one thread per output element, 2D spill
    const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
    workgroupCountX = Math.min(totalWorkgroups, 65535);
    workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    constants = { wg_size: wgSize, stride_x: workgroupCountX * wgSize };
  } else {
    // GEMM kernel: fixed 8x8 workgroup covering a 32x32 output tile;
    // grid over (N = B*OH*OW cols, M = OC rows)
    const tile = 32;
    workgroupCountX = Math.ceil((b * oh * ow) / tile);
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
 * WebGPU compute kernel for ExecuTorch `aten::convolution.default`.
 *
 * Implements 2D dense and grouped convolutions as well as 2D transposed
 * convolutions. Dense convolutions are evaluated as implicit-im2col tiled GEMM
 * (1.1-2.4x over the direct kernel in the ExecuTorch reference), grouped and
 * depthwise convolutions via the direct kernel.
 */
export const convolution: Shader<ConvolutionArgs, typeof name> = {
  name,
  code: {
    conv2d: SHADER_CONV2D,
    conv2dGemm: SHADER_CONV2D_GEMM,
    convTranspose2d: SHADER_CONV_TRANSPOSE2D,
  },
  recordIn,
};
