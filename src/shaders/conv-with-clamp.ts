import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { SHADER_CONV_WITH_CLAMP } from './conv-with-clamp/conv-with-clamp.wgsl';
import { SHADER_CONV_WITH_CLAMP_GEMM } from './conv-with-clamp/conv-with-clamp-gemm.wgsl';
import { SHADER_CONV_WITH_CLAMP_1X1 } from './conv-with-clamp/conv-with-clamp-1x1.wgsl';
import { SHADER_CONV_WITH_CLAMP_1X1_VEC4 } from './conv-with-clamp/conv-with-clamp-1x1-vec4.wgsl';
import { SHADER_CONV_WITH_CLAMP_1X1_64X64 } from './conv-with-clamp/conv-with-clamp-1x1-tile64.wgsl';
import { SHADER_CONV_WITH_CLAMP_TILE64 } from './conv-with-clamp/conv-with-clamp-tile64.wgsl';
import { SHADER_CONV_WITH_CLAMP_DEPTHWISE } from './conv-with-clamp/conv-with-clamp-depthwise.wgsl';
import { SHADER_CONV_WITH_CLAMP_F16 } from './conv-with-clamp/conv-with-clamp-f16.wgsl';
import { SHADER_CONV_WITH_CLAMP_GEMM_F16 } from './conv-with-clamp/conv-with-clamp-gemm-f16.wgsl';
import { SHADER_CONV_WITH_CLAMP_TILE64_F16 } from './conv-with-clamp/conv-with-clamp-tile64-f16.wgsl';
import { SHADER_CONV_WITH_CLAMP_DEPTHWISE_F16 } from './conv-with-clamp/conv-with-clamp-depthwise-f16.wgsl';

const name = 'et_vk.conv_with_clamp.default';

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

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    weight.dtype !== inTensor.dtype ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (inTensor.shape.length !== 4 || weight.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input, weight, and output`);
  }
  if (bias !== undefined && bias.dtype !== inTensor.dtype) {
    throw new Error(`${name}: bias dtype must match input dtype`);
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
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }
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

  // Variant selection: 1x1 convolutions use the pointwise GEMM kernel (no im2col),
  // with 128-bit vec4 memory optimization when dimensions are divisible by 4.
  // General dense convolutions use implicit-im2col tiled GEMM, and
  // grouped/depthwise convs use the direct kernel.
  const is1x1 =
    kh === 1 && kw === 1 && sH === 1 && sW === 1 && pH === 0 && pW === 0 && dH === 1 && dW === 1;
  const is1x1Vec4 =
    is1x1 && ic % 4 === 0 && (expectedHout * expectedWout) % 4 === 0 && oc % 4 === 0;
  const is1x1Tile64 =
    is1x1Vec4 && n === 1 && oc >= 64 && expectedHout * expectedWout >= 64 && ic >= 16;

  const isDenseTile64 =
    groups === 1 && oc >= 64 && n * expectedHout * expectedWout >= 64 && (ic * kh * kw) % 4 === 0;

  const isDepthwise =
    groups === oc && ic === oc && icPerGroup === 1 && kh * kw <= 49 && n * oc <= 65535;

  let code = isF16 ? SHADER_CONV_WITH_CLAMP_F16 : SHADER_CONV_WITH_CLAMP;
  if (isDepthwise) {
    code = isF16 ? SHADER_CONV_WITH_CLAMP_DEPTHWISE_F16 : SHADER_CONV_WITH_CLAMP_DEPTHWISE;
  } else if (groups === 1) {
    if (isF16) {
      if (isDenseTile64) {
        code = SHADER_CONV_WITH_CLAMP_TILE64_F16;
      } else {
        code = SHADER_CONV_WITH_CLAMP_GEMM_F16;
      }
    } else {
      if (is1x1Tile64) {
        code = SHADER_CONV_WITH_CLAMP_1X1_64X64;
      } else if (is1x1Vec4) {
        code = SHADER_CONV_WITH_CLAMP_1X1_VEC4;
      } else if (is1x1) {
        code = SHADER_CONV_WITH_CLAMP_1X1;
      } else if (isDenseTile64) {
        code = SHADER_CONV_WITH_CLAMP_TILE64;
      } else {
        code = SHADER_CONV_WITH_CLAMP_GEMM;
      }
    }
  }

  let workgroupCountX: number;
  let workgroupCountY: number;
  let workgroupCountZ = 1;
  let constants: Record<string, number>;

  if (isDepthwise) {
    // Spatial 16x16 workgroup per (batch, oc) channel slice
    workgroupCountX = Math.ceil(expectedWout / 16);
    workgroupCountY = Math.ceil(expectedHout / 16);
    workgroupCountZ = n * oc;
    constants = {};
  } else if (groups !== 1) {
    // Direct kernel: one thread per output element, 2D spill
    const totalWorkgroups = Math.ceil(numel / wgSize);
    workgroupCountX = Math.min(totalWorkgroups, 65535);
    workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    constants = { wg_size: wgSize };
  } else if (!isF16 && is1x1Tile64) {
    // 64x64 register-tiled GEMM for pointwise 1x1 conv with clamp
    workgroupCountX = Math.ceil((expectedHout * expectedWout) / 64);
    workgroupCountY = Math.ceil(oc / 64);
    constants = {};
  } else if (isDenseTile64) {
    // 64x64 register-tiled GEMM for general dense convolution with clamp
    workgroupCountX = Math.ceil((n * expectedHout * expectedWout) / 64);
    workgroupCountY = Math.ceil(oc / 64);
    constants = {};
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
    workgroupCountZ,
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
  code: {
    convWithClamp: SHADER_CONV_WITH_CLAMP,
    convWithClampGemm: SHADER_CONV_WITH_CLAMP_GEMM,
    convWithClamp1x1: SHADER_CONV_WITH_CLAMP_1X1,
    convWithClamp1x1Vec4: SHADER_CONV_WITH_CLAMP_1X1_VEC4,
    convWithClamp1x1Tile64: SHADER_CONV_WITH_CLAMP_1X1_64X64,
    convWithClampTile64: SHADER_CONV_WITH_CLAMP_TILE64,
    convWithClampDepthwise3x3: SHADER_CONV_WITH_CLAMP_DEPTHWISE,
    convWithClampF16: SHADER_CONV_WITH_CLAMP_F16,
    convWithClampGemmF16: SHADER_CONV_WITH_CLAMP_GEMM_F16,
    convWithClampTile64F16: SHADER_CONV_WITH_CLAMP_TILE64_F16,
    convWithClampDepthwiseF16: SHADER_CONV_WITH_CLAMP_DEPTHWISE_F16,
  },
  recordIn,
};
