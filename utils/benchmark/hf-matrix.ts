/**
 * Opt-in adapter of HF's experimental aligned 1x1 and implicit 3x3 subgroup-matrix Conv.
 * Integration MIT, https://github.com/barhanc/wgpu-executorch
 * Upstream shader Apache-2.0: ./huggingface/NOTICE and LICENSE.
 */
import type { WgpuExecutionContext } from '../../src';
import { createComputeBundle } from '../../src';
import type { ConvolutionArgs } from '../../src/shaders/convolution';
import generatedJson from './huggingface/matrix-variants.json?raw';

const generated = JSON.parse(generatedJson) as { variants: Record<string, string> };

/**
 * Record the published aligned matrix route, falling back for unsupported shapes/devices.
 *
 * @param ctx Context with experimental matrix and subgroup-size features enabled.
 * @param args Standard convolution arguments on existing buffers.
 * @returns Whether a matrix dispatch was recorded.
 */
export function recordHfMatrix(ctx: WgpuExecutionContext, args: ConvolutionArgs): boolean {
  const [input, weight, bias, stride, padding, dilation, transposed, , groups, output] = args;
  if (transposed || groups !== 1 || input.shape.length !== 4 || weight.shape.length !== 4)
    return false;
  if (input.dtype !== 'float32' && input.dtype !== 'float16') return false;
  const features = ctx.device.features as unknown as Set<string>;
  if (
    !['subgroups', 'subgroup-size-control', 'chromium-experimental-subgroup-matrix'].every((name) =>
      features.has(name)
    )
  )
    return false;
  const [batch, ic, ih, iw] = input.shape;
  const [oc, , kh, kw] = weight.shape;
  const implicit = kh === 3 && kw === 3;
  if (
    !implicit &&
    (kh !== 1 ||
      kw !== 1 ||
      stride.some((v) => v !== 1) ||
      padding.some((v) => v !== 0) ||
      dilation.some((v) => v !== 1))
  )
    return false;
  const [, , oh, ow] = output.shape;
  const n = oh * ow;
  if (!implicit && (ic < 32 || ic % 32 !== 0 || n % 64 !== 0 || oc < 16)) return false;
  const half = input.dtype === 'float16';
  const direct = half
    ? oc % 32 === 0
    : oc % 64 === 0 && ctx.device.limits.maxComputeWorkgroupStorageSize >= 24576;
  const rows = !half && direct ? 64 : 32;
  const key = `${implicit ? '3x3' : '1x1'}-${half ? 'f16' : 'f32'}-${direct ? 'direct' : 'shared'}`;
  const params = ctx.uniformBuffer(new Uint32Array([bias ? 1 : 0, 0, 0, 0]));
  /* eslint-disable @typescript-eslint/naming-convention -- Published WGSL names */
  const constants = {
    M: oc,
    K: ic * kh * kw,
    N: n,
    ...(implicit
      ? {
          K_LOOP: Math.ceil((ic * kh * kw) / 32) * 32,
          CONV_STRIDE_H: stride[0],
          CONV_STRIDE_W: stride[1] ?? stride[0],
          CONV_DILATION_H: dilation[0],
          CONV_DILATION_W: dilation[1] ?? dilation[0],
          CONV_PAD_TOP: padding[0],
          CONV_PAD_LEFT: padding[1] ?? padding[0],
          CONV_IN_H: ih,
          CONV_IN_W: iw,
          CONV_OUT_W: ow,
          CONV_IN_CHANNELS: ic,
        }
      : {}),
  };
  /* eslint-enable @typescript-eslint/naming-convention */
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
    workgroupCountX: Math.ceil(n / 64),
    workgroupCountY: Math.ceil(oc / rows),
    workgroupCountZ: batch,
  });
  return true;
}
