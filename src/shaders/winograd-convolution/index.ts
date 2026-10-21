import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import type { ConvWithClampArgs } from '../conv-with-clamp';
import { createComputeBundle } from '../../command';
import { bmm } from '../bmm';

import { SHADER_WINOGRAD_INPUT } from './winograd-input.wgsl';
import { SHADER_WINOGRAD_INPUT_F16 } from './winograd-input-f16.wgsl';
import { SHADER_WINOGRAD_OUTPUT } from './winograd-output.wgsl';
import { SHADER_WINOGRAD_OUTPUT_F16 } from './winograd-output-f16.wgsl';

const name = 'wgpu.winograd_convolution.default';

type WinogradArgs = readonly [
  ...OmitLast<ConvWithClampArgs>,
  inputScratch: Tensor,
  productScratch: Tensor,
  output: Tensor,
];
type OmitLast<T extends readonly unknown[]> = T extends readonly [...infer Head, unknown]
  ? Head
  : never;

/**
 * Experimental transformed convolution; weights and reusable scratch are graph inputs.
 *
 * @param ctx Execution context.
 * @param args Convolution parameters, scratch tensors and output.
 */
function recordIn(ctx: WgpuExecutionContext, args: WinogradArgs): void {
  const [
    input,
    weight,
    bias,
    stride,
    padding,
    dilation,
    transposed,
    outputPadding,
    groups,
    minimum,
    maximum,
    inputScratch,
    productScratch,
    output,
  ] = args;
  if (input.shape.length !== 4 || output.shape.length !== 4 || weight.shape.length !== 3) {
    throw new Error(`${name}: expected NCHW input/output and transformed 3D weights`);
  }
  const [batch, ic, ih, iw] = input.shape;
  const [, oc, oh, ow] = output.shape;
  const planes = weight.shape[0];
  const tile = planes === 16 ? 2 : planes === 36 ? 4 : 0;
  if (
    !tile ||
    weight.dtype !== 'float32' ||
    weight.shape[1] !== oc ||
    weight.shape[2] !== ic ||
    groups !== 1 ||
    transposed ||
    stride.some((v) => v !== 1) ||
    dilation.some((v) => v !== 1) ||
    outputPadding.some((v) => v !== 0) ||
    input.dtype !== output.dtype ||
    (input.dtype !== 'float32' && input.dtype !== 'float16')
  )
    throw new Error(`${name}: unsupported geometry/dtype`);
  const ph = padding[0],
    pw = padding[1] ?? ph;
  if (
    output.shape[0] !== batch ||
    oh !== ih + 2 * ph - 2 ||
    ow !== iw + 2 * pw - 2 ||
    inputScratch.dtype !== 'float32' ||
    productScratch.dtype !== 'float32' ||
    (bias && (bias.dtype !== input.dtype || bias.numel !== oc))
  ) {
    throw new Error(`${name}: invalid output, bias or scratch geometry`);
  }
  const th = Math.ceil(oh / tile),
    tw = Math.ceil(ow / tile);
  const n = Math.ceil((batch * th * tw) / 4) * 4;
  if (
    inputScratch.byteLength < planes * ic * n * 4 ||
    productScratch.byteLength < planes * oc * n * 4
  )
    throw new Error(`${name}: scratch too small`);
  const packedInput = ctx.tensor('float32', [planes, ic, n], inputScratch.buffer);
  const products = ctx.tensor('float32', [planes, oc, n], productScratch.buffer);
  const params = new ArrayBuffer(64);
  new Uint32Array(params).set([
    batch,
    ic,
    ih,
    iw,
    oc,
    oh,
    ow,
    padding[0],
    padding[1] ?? padding[0],
    th,
    tw,
    n,
    bias ? 1 : 0,
    0,
  ]);
  new Float32Array(params).set(
    [minimum ?? -3.4028234663852886e38, maximum ?? 3.4028234663852886e38],
    14
  );
  const uniform = ctx.uniformBuffer(params);
  const isF16 = input.dtype === 'float16';
  const inputBundle = createComputeBundle(
    ctx.device,
    isF16 ? SHADER_WINOGRAD_INPUT_F16 : SHADER_WINOGRAD_INPUT,
    [
      { binding: 0, buffer: input.buffer },
      { binding: 1, buffer: packedInput.buffer },
      { binding: 2, buffer: uniform },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { tile_size: tile }
  );
  ctx.addDispatch({
    pipeline: inputBundle.pipeline,
    bindGroup: inputBundle.bindGroup,
    workgroupCountX: Math.ceil(n / 64),
    workgroupCountY: ic,
  });
  ctx.recordShader(bmm, [weight, packedInput, products]);
  const outputBundle = createComputeBundle(
    ctx.device,
    isF16 ? SHADER_WINOGRAD_OUTPUT_F16 : SHADER_WINOGRAD_OUTPUT,
    [
      { binding: 0, buffer: products.buffer },
      { binding: 1, buffer: output.buffer },
      { binding: 2, buffer: uniform },
      { binding: 3, buffer: bias?.buffer ?? input.buffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { tile_size: tile }
  );
  ctx.addDispatch({
    pipeline: outputBundle.pipeline,
    bindGroup: outputBundle.bindGroup,
    workgroupCountX: Math.ceil(n / 64),
    workgroupCountY: oc,
  });
}

export const winogradConvolution: Shader<WinogradArgs, typeof name> = {
  name,
  code: {
    input: SHADER_WINOGRAD_INPUT,
    output: SHADER_WINOGRAD_OUTPUT,
    inputF16: SHADER_WINOGRAD_INPUT_F16,
    outputF16: SHADER_WINOGRAD_OUTPUT_F16,
  },
  recordIn,
};
