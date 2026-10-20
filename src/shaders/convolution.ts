import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { SHADER_CONV2D } from './convolution/conv2d.wgsl';
import { SHADER_CONV2D_GEMM } from './convolution/conv2d-gemm.wgsl';
import { SHADER_CONV2D_1X1 } from './convolution/conv2d-1x1.wgsl';
import { SHADER_CONV2D_1X1_VEC4 } from './convolution/conv2d-1x1-vec4.wgsl';
import { SHADER_CONV2D_1X1_64X64 } from './convolution/conv2d-1x1-tile64.wgsl';
import { SHADER_CONV2D_TILE64 } from './convolution/conv2d-tile64.wgsl';
import { SHADER_CONV2D_DEPTHWISE } from './convolution/conv2d-depthwise.wgsl';
import { SHADER_CONV_TRANSPOSE2D } from './convolution/conv-transpose2d.wgsl';
import { SHADER_CONV2D_F16 } from './convolution/conv2d-f16.wgsl';
import { SHADER_CONV2D_GEMM_F16 } from './convolution/conv2d-gemm-f16.wgsl';
import { SHADER_CONV2D_TILE64_F16 } from './convolution/conv2d-tile64-f16.wgsl';
import { SHADER_CONV2D_DEPTHWISE_F16 } from './convolution/conv2d-depthwise-f16.wgsl';

const name = 'aten.convolution.default';

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

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    weight.dtype !== inTensor.dtype ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (bias !== undefined && bias.dtype !== inTensor.dtype) {
    throw new Error(`${name}: bias dtype must match input dtype`);
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
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }
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

  // Variant selection: 1x1 convolutions use the pointwise GEMM kernel (no im2col),
  // with 128-bit vec4 memory optimization when dimensions are divisible by 4.
  // General dense convolutions use implicit-im2col tiled GEMM, grouped/depthwise
  // convs through the direct kernel, and transposed convs through the gather kernel.
  const is1x1 =
    !transposed &&
    kh === 1 &&
    kw === 1 &&
    sH === 1 &&
    sW === 1 &&
    pH === 0 &&
    pW === 0 &&
    dH === 1 &&
    dW === 1;
  const is1x1Vec4 = is1x1 && ic % 4 === 0 && (oh * ow) % 4 === 0 && oc % 4 === 0;
  const is1x1Tile64 = is1x1Vec4 && b === 1 && oc >= 64 && oh * ow >= 64 && ic >= 16;
  const isDenseTile64 =
    !transposed && groups === 1 && oc >= 64 && b * oh * ow >= 64 && (ic * kh * kw) % 4 === 0;

  const isDepthwise =
    !transposed &&
    groups === oc &&
    ic === oc &&
    weight.shape[1] === 1 &&
    kh * kw <= 49 &&
    b * oc <= 65535;

  let code = isF16 ? SHADER_CONV2D_F16 : SHADER_CONV2D;
  if (transposed) {
    if (isF16) {
      throw new Error(`${name}: float16 transposed conv not supported`);
    }
    code = SHADER_CONV_TRANSPOSE2D;
  } else if (isDepthwise) {
    code = isF16 ? SHADER_CONV2D_DEPTHWISE_F16 : SHADER_CONV2D_DEPTHWISE;
  } else if (groups === 1) {
    if (isF16) {
      if (isDenseTile64) {
        code = SHADER_CONV2D_TILE64_F16;
      } else {
        code = SHADER_CONV2D_GEMM_F16;
      }
    } else {
      if (is1x1Tile64) {
        code = SHADER_CONV2D_1X1_64X64;
      } else if (is1x1Vec4) {
        code = SHADER_CONV2D_1X1_VEC4;
      } else if (is1x1) {
        code = SHADER_CONV2D_1X1;
      } else if (isDenseTile64) {
        code = SHADER_CONV2D_TILE64;
      } else {
        code = SHADER_CONV2D_GEMM;
      }
    }
  }

  let workgroupCountX: number;
  let workgroupCountY: number;
  let workgroupCountZ = 1;
  let constants: Record<string, number>;

  if (isDepthwise) {
    // Spatial 16x16 workgroup per (batch, oc) channel slice
    workgroupCountX = Math.ceil(ow / 16);
    workgroupCountY = Math.ceil(oh / 16);
    workgroupCountZ = b * oc;
    constants = {};
  } else if (transposed || groups !== 1) {
    // Direct / transposed kernels: one thread per output element, 2D spill
    const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
    workgroupCountX = Math.min(totalWorkgroups, 65535);
    workgroupCountY = Math.ceil(totalWorkgroups / 65535);
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    constants = { wg_size: wgSize, stride_x: workgroupCountX * wgSize };
  } else if (!isF16 && is1x1Tile64) {
    // 64x64 register-tiled GEMM for pointwise 1x1 conv
    workgroupCountX = Math.ceil((oh * ow) / 64);
    workgroupCountY = Math.ceil(oc / 64);
    constants = {};
  } else if (isDenseTile64) {
    // 64x64 register-tiled GEMM for general dense convolution
    workgroupCountX = Math.ceil((b * oh * ow) / 64);
    workgroupCountY = Math.ceil(oc / 64);
    constants = {};
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
    workgroupCountZ,
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
    conv2d1x1: SHADER_CONV2D_1X1,
    conv2d1x1Vec4: SHADER_CONV2D_1X1_VEC4,
    conv2d1x1Tile64: SHADER_CONV2D_1X1_64X64,
    conv2dTile64: SHADER_CONV2D_TILE64,
    conv2dDepthwise3x3: SHADER_CONV2D_DEPTHWISE,
    convTranspose2d: SHADER_CONV_TRANSPOSE2D,
    conv2dF16: SHADER_CONV2D_F16,
    conv2dGemmF16: SHADER_CONV2D_GEMM_F16,
    conv2dTile64F16: SHADER_CONV2D_TILE64_F16,
    conv2dDepthwiseF16: SHADER_CONV2D_DEPTHWISE_F16,
  },
  recordIn,
};
