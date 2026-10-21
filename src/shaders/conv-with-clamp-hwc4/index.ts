import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_CONV_WITH_CLAMP_HWC4_1X1 } from './conv-with-clamp-hwc4-1x1.wgsl';
import { SHADER_CONV_WITH_CLAMP_HWC4_1X1_F16 } from './conv-with-clamp-hwc4-1x1-f16.wgsl';
import { SHADER_CONV_WITH_CLAMP_HWC4_DW } from './conv-with-clamp-hwc4-dw.wgsl';
import { SHADER_CONV_WITH_CLAMP_HWC4_DW_F16 } from './conv-with-clamp-hwc4-dw-f16.wgsl';
import { SHADER_CONV_WITH_CLAMP_HWC4 } from './conv-with-clamp-hwc4.wgsl';
import { SHADER_CONV_WITH_CLAMP_HWC4_F16 } from './conv-with-clamp-hwc4-f16.wgsl';

const name = 'et_vk.conv_with_clamp_hwc4.default';

/**
 * Positional argument tuple for `et_vk::conv_with_clamp_hwc4.default`:
 * `[in, weight, bias, stride, padding, dilation, transposed, output_padding, groups, min, max, out]`
 */
export type ConvWithClampHwc4Args = readonly [
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
 * Internal dispatch builder for `et_vk::conv_with_clamp_hwc4.default`.
 *
 * @param ctx The execution context.
 * @param args Positional arguments for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: ConvWithClampHwc4Args): void {
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

  if (outputPadding.some((v) => v !== 0)) {
    throw new Error(`${name}: non-zero output_padding unsupported in HWC4`);
  }

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
  if (inTensor.shape.length !== 4 || outTensor.shape.length !== 4) {
    throw new Error(`${name}: expected 4D tensors for input and output`);
  }
  if (transposed) {
    throw new Error(`${name}: transposed conv unsupported in HWC4`);
  }

  const [sH, sW] = parseHw(stride, 'stride');
  const [pH, pW] = parseHw(padding, 'padding');
  const [dH, dW] = parseHw(dilation, 'dilation');

  const [b, ih, iw, ic] = inTensor.shape as [number, number, number, number];
  const [, oh, ow, oc] = outTensor.shape as [number, number, number, number];

  const ic4 = Math.ceil(ic / 4);
  const oc4 = Math.ceil(oc / 4);
  const hasBias = bias !== undefined;
  const isDepthwise = groups === oc && ic === oc;

  const minVal = outputMin !== undefined ? outputMin : -1e30;
  const maxVal = outputMax !== undefined ? outputMax : 1e30;

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const is1x1 =
    sH === 1 &&
    sW === 1 &&
    pH === 0 &&
    pW === 0 &&
    dH === 1 &&
    dW === 1 &&
    groups === 1 &&
    ih === oh &&
    iw === ow;

  const wgSize = 256;
  let code: string;
  let uniformBuffer: GPUBuffer;
  const total = is1x1 ? Math.ceil((b * oh * ow) / 4) * oc4 : b * oh * ow * oc4;
  const totalWorkgroups = Math.ceil(total / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);
  const strideX = workgroupCountX * wgSize;

  if (is1x1) {
    code = isF16 ? SHADER_CONV_WITH_CLAMP_HWC4_1X1_F16 : SHADER_CONV_WITH_CLAMP_HWC4_1X1;
    const buffer = new ArrayBuffer(48);
    const u32View = new Uint32Array(buffer);
    const f32View = new Float32Array(buffer);
    u32View.set([b, ic4, ih, iw, oc4, hasBias ? 1 : 0, strideX, 0]);
    f32View[8] = minVal;
    f32View[9] = maxVal;
    uniformBuffer = ctx.uniformBuffer(buffer);
  } else if (isDepthwise) {
    code = isF16 ? SHADER_CONV_WITH_CLAMP_HWC4_DW_F16 : SHADER_CONV_WITH_CLAMP_HWC4_DW;
    const kh = 3;
    const kw = 3;
    const buffer = new ArrayBuffer(80);
    const u32View = new Uint32Array(buffer);
    const f32View = new Float32Array(buffer);
    // prettier-ignore
    u32View.set([
      b, oc4,
      ih, iw,
      oh, ow,
      kh, kw,
      sH, sW,
      pH, pW,
      dH, dW,
      hasBias ? 1 : 0,
      strideX,
    ]);
    f32View[16] = minVal;
    f32View[17] = maxVal;
    uniformBuffer = ctx.uniformBuffer(buffer);
  } else {
    code = isF16 ? SHADER_CONV_WITH_CLAMP_HWC4_F16 : SHADER_CONV_WITH_CLAMP_HWC4;
    const kh = 3;
    const kw = 3;
    const buffer = new ArrayBuffer(80);
    const u32View = new Uint32Array(buffer);
    const f32View = new Float32Array(buffer);
    // prettier-ignore
    u32View.set([
      b, ic4,
      ih, iw,
      oh, ow,
      oc4,
      kh, kw,
      sH, sW,
      pH, pW,
      dH, dW,
      hasBias ? 1 : 0,
      strideX,
    ]);
    f32View[16] = minVal;
    f32View[17] = maxVal;
    uniformBuffer = ctx.uniformBuffer(buffer);
  }

  const biasBuffer = hasBias ? bias.buffer : weight.buffer;

  const bundle = createComputeBundle(device, code, [
    { binding: 0, buffer: outTensor.buffer },
    { binding: 1, buffer: inTensor.buffer },
    { binding: 2, buffer: weight.buffer },
    { binding: 3, buffer: biasBuffer },
    { binding: 4, buffer: uniformBuffer },
  ]);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for fused 2D convolution with clamp in packed HWC4 layout.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const convWithClampHwc4: Shader<ConvWithClampHwc4Args, typeof name> = {
  name,
  code: {
    hwc41x1: SHADER_CONV_WITH_CLAMP_HWC4_1X1,
    hwc41x1F16: SHADER_CONV_WITH_CLAMP_HWC4_1X1_F16,
    hwc4Dw: SHADER_CONV_WITH_CLAMP_HWC4_DW,
    hwc4DwF16: SHADER_CONV_WITH_CLAMP_HWC4_DW_F16,
    hwc4: SHADER_CONV_WITH_CLAMP_HWC4,
    hwc4F16: SHADER_CONV_WITH_CLAMP_HWC4_F16,
  },
  recordIn,
};
