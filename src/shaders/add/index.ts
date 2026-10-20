import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { isBroadcastable } from '../utils/broadcasting';
import { encodeTensorMeta } from '../utils/meta';

import { SHADER_ADD_BROADCAST } from './add-broadcast.wgsl';
import { SHADER_ADD_BROADCAST_F16 } from './add-broadcast-f16.wgsl';
import { SHADER_ADD_ELEMENTWISE } from './add-elementwise.wgsl';
import { SHADER_ADD_ELEMENTWISE_F16 } from './add-elementwise-f16.wgsl';
import { SHADER_ADD_ELEMENTWISE_VEC4 } from './add-elementwise-vec4.wgsl';
import { SHADER_ADD_ELEMENTWISE_VEC4_F16 } from './add-elementwise-vec4-f16.wgsl';

const name = 'aten.add.Tensor';

/**
 * Positional argument tuple for `aten::add.Tensor`:
 * Standard ExecuTorch schema: `[in1, in2, alpha, out]`
 * Or 3-arg variant: `[in1, in2, out]`
 */
export type AddArgs =
  | readonly [in1: Tensor, in2: Tensor, out: Tensor]
  | readonly [in1: Tensor, in2: Tensor, alpha: number, out: Tensor];

/**
 * Internal dispatch builder for `aten::add.Tensor`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: AddArgs): void {
  const [in1, in2] = args;
  const [alpha, out] = args.length === 4 ? [args[2], args[3]] : [1.0, args[2]];

  if (
    (in1.dtype !== 'float32' && in1.dtype !== 'float16') ||
    in2.dtype !== in1.dtype ||
    out.dtype !== in1.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (!isBroadcastable(in1.shape, out.shape) || !isBroadcastable(in2.shape, out.shape)) {
    throw new Error(`${name}: Input shapes are not broadcastable to output shape`);
  }

  const device = ctx.device;
  const isF16 = in1.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const sameShape =
    in1.shape.length === out.shape.length &&
    in2.shape.length === out.shape.length &&
    in1.shape.every((s, i) => s === out.shape[i]) &&
    in2.shape.every((s, i) => s === out.shape[i]);

  const wgSize = 256;

  if (sameShape) {
    const numel = out.numel;
    const useVec4 = numel % 4 === 0;
    let code = useVec4 ? SHADER_ADD_ELEMENTWISE_VEC4 : SHADER_ADD_ELEMENTWISE;
    if (isF16) {
      code = useVec4 ? SHADER_ADD_ELEMENTWISE_VEC4_F16 : SHADER_ADD_ELEMENTWISE_F16;
    }
    const effectiveNumel = useVec4 ? numel / 4 : numel;

    const totalWorkgroups = Math.ceil(effectiveNumel / wgSize);
    const workgroupCountX = Math.min(totalWorkgroups, 65535);
    const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

    const paramBuffer = ctx.uniformBuffer(new Uint32Array([effectiveNumel]));

    const bundle = createComputeBundle(
      device,
      code,
      [
        { binding: 0, buffer: in1.buffer },
        { binding: 1, buffer: in2.buffer },
        { binding: 2, buffer: out.buffer },
        { binding: 3, buffer: paramBuffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { wg_size: wgSize, alpha }
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX,
      workgroupCountY,
    });
    return;
  }

  const outRank = out.shape.length;
  const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(out));
  const in1MetaBuffer = ctx.uniformBuffer(encodeTensorMeta(in1, outRank));
  const in2MetaBuffer = ctx.uniformBuffer(encodeTensorMeta(in2, outRank));

  const totalWorkgroups = Math.ceil(out.numel / wgSize);
  const workgroupCountX = Math.min(totalWorkgroups, 65535);
  const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_ADD_BROADCAST_F16 : SHADER_ADD_BROADCAST,
    [
      { binding: 0, buffer: in1.buffer },
      { binding: 1, buffer: in2.buffer },
      { binding: 2, buffer: out.buffer },
      { binding: 3, buffer: outMetaBuffer },
      { binding: 4, buffer: in1MetaBuffer },
      { binding: 5, buffer: in2MetaBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize, alpha }
  );

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::add.Tensor`.
 *
 * Computes elementwise `output = input1 + alpha * input2` with NumPy/PyTorch-style
 * broadcasting (right-aligned, dimensions equal or 1, up to rank 8). Features an
 * automatic fast path when input and output shapes match.
 */
export const add: Shader<AddArgs, typeof name> = {
  name,
  code: {
    broadcast: SHADER_ADD_BROADCAST,
    elementwise: SHADER_ADD_ELEMENTWISE,
    elementwiseVec4: SHADER_ADD_ELEMENTWISE_VEC4,
    broadcastF16: SHADER_ADD_BROADCAST_F16,
    elementwiseF16: SHADER_ADD_ELEMENTWISE_F16,
    elementwiseVec4F16: SHADER_ADD_ELEMENTWISE_VEC4_F16,
  },
  recordIn,
};
