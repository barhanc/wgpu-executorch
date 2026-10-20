import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { encodeTensorMeta } from '../utils/meta';

import { SHADER_PERMUTE } from './permute.wgsl';
import { SHADER_PERMUTE_TRANSPOSE2D } from './permute-transpose2d.wgsl';
import { SHADER_PERMUTE_F16 } from './permute-f16.wgsl';
import { SHADER_PERMUTE_TRANSPOSE2D_F16 } from './permute-transpose2d-f16.wgsl';

const name = 'aten.permute_copy.default';

/**
 * Positional argument tuple for `aten::permute_copy.default`:
 * `[in, dims, out]`
 */
export type PermuteCopyArgs = readonly [
  inTensor: Tensor,
  dims: readonly number[],
  outTensor: Tensor,
];

/**
 * Internal dispatch builder for `aten::permute_copy.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: PermuteCopyArgs): void {
  const [inTensor, dims, outTensor] = args;

  if (
    (inTensor.dtype !== 'float32' && inTensor.dtype !== 'float16') ||
    outTensor.dtype !== inTensor.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }

  const device = ctx.device;
  const isF16 = inTensor.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const rank = inTensor.shape.length;
  if (outTensor.shape.length !== rank) {
    throw new Error(`${name}: output rank != input rank`);
  }
  if (dims.length !== rank) {
    throw new Error(`${name}: dims count (${dims.length}) != input rank (${rank})`);
  }

  const perm = new Uint32Array(8);
  const seen = new Set<number>();
  for (let d = 0; d < rank; d++) {
    let p = dims[d];
    if (p < 0) {
      p += rank;
    }
    if (p < 0 || p >= rank || seen.has(p)) {
      throw new Error(`${name}: invalid permute dims ${dims}`);
    }
    seen.add(p);
    perm[d] = p;
  }
  for (let d = rank; d < 8; d++) {
    perm[d] = d; // identity padding for unused slots
  }
  for (let d = 0; d < rank; d++) {
    if (outTensor.shape[d] !== inTensor.shape[perm[d]]) {
      throw new Error(`${name}: output shape does not match permuted input shape`);
    }
  }

  // Fast path: 2D matrix transpose
  if (rank === 2 && perm[0] === 1 && perm[1] === 0) {
    const rows = inTensor.shape[0];
    const cols = inTensor.shape[1];
    const paramsBuffer = ctx.uniformBuffer(new Uint32Array([rows, cols]));

    const bundle = createComputeBundle(
      device,
      isF16 ? SHADER_PERMUTE_TRANSPOSE2D_F16 : SHADER_PERMUTE_TRANSPOSE2D,
      [
        { binding: 0, buffer: inTensor.buffer },
        { binding: 1, buffer: outTensor.buffer },
        { binding: 2, buffer: paramsBuffer },
      ]
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(cols / 16),
      workgroupCountY: Math.ceil(rows / 16),
    });
    return;
  }

  const wgSize = 64;
  const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(outTensor));
  const inMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(inTensor));
  const paramsBuffer = ctx.uniformBuffer(perm);

  const bundle = createComputeBundle(
    device,
    isF16 ? SHADER_PERMUTE_F16 : SHADER_PERMUTE,
    [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
      { binding: 2, buffer: outMetaBuffer },
      { binding: 3, buffer: inMetaBuffer },
      { binding: 4, buffer: paramsBuffer },
    ],
    // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
    { wg_size: wgSize }
  );

  const totalWorkgroups = Math.ceil(outTensor.numel / wgSize);
  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX: Math.min(totalWorkgroups, 65535),
    workgroupCountY: Math.ceil(totalWorkgroups / 65535),
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::permute_copy.default`.
 *
 * Copies a tensor with its dimensions reordered — a gather where each output
 * coordinate maps to the permuted input coordinate.
 * Supports float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const permuteCopy: Shader<PermuteCopyArgs, typeof name> = {
  name,
  code: {
    default: SHADER_PERMUTE,
    transpose2d: SHADER_PERMUTE_TRANSPOSE2D,
    defaultF16: SHADER_PERMUTE_F16,
    transpose2dF16: SHADER_PERMUTE_TRANSPOSE2D_F16,
  },
  recordIn,
};
