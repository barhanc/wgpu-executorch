/**
 * Experimental Hugging Face Conv adapter for kernel/model comparisons.
 * Integration: MIT, https://github.com/barhanc/wgpu-executorch
 * Generated shader: Apache-2.0; provenance and license in ./huggingface/NOTICE.
 * This is opt-in benchmark code, not part of the default shader registry.
 */
import type { WgpuExecutionContext } from '../../src';
import { createComputeBundle } from '../../src';
import type { ConvolutionArgs } from '../../src/shaders/convolution';
import generatedJson from './huggingface/conv-variants.json?raw';

const generated = JSON.parse(generatedJson) as { variants: Record<string, string> };

export type HfConvRoute = '1x1' | '3x3' | 'both';

/**
 * Record a supported dense convolution using the published HF register tile.
 *
 * @param ctx Context owning the recorded commands.
 * @param args Existing convolution arguments; no tensor copies are introduced.
 * @param route Which adapted template to enable.
 * @param implicitRows Published implicit-convolution output tile height.
 * @returns Whether this operator was recorded; false means use the usual fallback.
 */
export function recordHfConvolution(
  ctx: WgpuExecutionContext,
  args: ConvolutionArgs,
  route: HfConvRoute = 'both',
  implicitRows: 32 | 64 = 64
): boolean {
  const [input, weight, bias, stride, padding, dilation, transposed, , groups, output] = args;
  if (transposed || groups !== 1 || input.shape.length !== 4 || weight.shape.length !== 4)
    return false;
  if (input.dtype !== 'float32' && input.dtype !== 'float16') return false;
  const [batch, ic, ih, iw] = input.shape;
  const [oc, , kh, kw] = weight.shape;
  const [, , oh, ow] = output.shape;
  const sh = stride[0],
    sw = stride[1] ?? sh;
  const ph = padding[0],
    pw = padding[1] ?? ph;
  const dh = dilation[0],
    dw = dilation[1] ?? dh;
  const pointwise =
    kh === 1 && kw === 1 && sh === 1 && sw === 1 && ph === 0 && pw === 0 && dh === 1 && dw === 1;
  const dense3x3 = kh === 3 && kw === 3;
  if ((!pointwise || route === '3x3') && (!dense3x3 || route === '1x1')) return false;
  const reduction = ic * kh * kw;
  const tileK = reduction % 32 === 0 ? 32 : 16;
  const scalar = input.dtype === 'float16' ? 'f16' : 'f32';
  const rows = pointwise ? 64 : implicitRows;
  const key =
    `${pointwise ? '1x1' : '3x3'}-${scalar}-k${tileK}-m${rows}` as keyof typeof generated.variants;
  /* eslint-disable @typescript-eslint/naming-convention -- Upstream WGSL names */
  const constants: Record<string, number> = pointwise
    ? {}
    : {
        CONV_STRIDE_H: sh,
        CONV_STRIDE_W: sw,
        CONV_DILATION_H: dh,
        CONV_DILATION_W: dw,
        CONV_PAD_TOP: ph,
        CONV_PAD_LEFT: pw,
        CONV_IN_H: ih,
        CONV_IN_W: iw,
        CONV_OUT_W: ow,
        CONV_IN_CHANNELS: ic,
      };
  /* eslint-enable @typescript-eslint/naming-convention */
  const params = ctx.uniformBuffer(new Uint32Array([oc, reduction, oh * ow, bias ? 1 : 0]));
  const bundle = createComputeBundle(
    ctx.device,
    generated.variants[key],
    [
      { binding: 0, buffer: output.buffer },
      { binding: 1, buffer: input.buffer },
      { binding: 2, buffer: weight.buffer },
      { binding: 3, buffer: bias?.buffer ?? weight.buffer },
      { binding: 4, buffer: params },
    ],
    constants
  );
  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX: Math.ceil((oh * ow) / 64),
    workgroupCountY: Math.ceil(oc / rows),
    workgroupCountZ: batch,
  });
  return true;
}
