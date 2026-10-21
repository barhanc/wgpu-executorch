import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_BMM_TILED } from './bmm-tiled.wgsl';
import { SHADER_BMM_VEC4 } from './bmm-vec4.wgsl';
import { SHADER_BMM_64X64 } from './bmm-64x64.wgsl';
import { SHADER_BMM_TILED_F16 } from './bmm-tiled-f16.wgsl';
import { SHADER_BMM_VEC4_F16 } from './bmm-vec4-f16.wgsl';
import { SHADER_BMM_64X64_F16 } from './bmm-64x64-f16.wgsl';

const name = 'aten.bmm.default';

/**
 * Positional argument tuple for `aten::bmm.default`:
 * `[a, b, out]`
 */
export type BmmArgs = readonly [a: Tensor, b: Tensor, out: Tensor];

/**
 * Internal dispatch builder for `aten::bmm.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: BmmArgs): void {
  const [a, b, out] = args;

  if (
    (a.dtype !== 'float32' && a.dtype !== 'float16') ||
    b.dtype !== a.dtype ||
    out.dtype !== a.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (a.shape.length !== 3 || b.shape.length !== 3 || out.shape.length !== 3) {
    throw new Error(`${name}: inputs and output must be 3D`);
  }

  const batch = a.shape[0];
  const m = a.shape[1];
  const k = a.shape[2];
  const n = b.shape[2];
  if (b.shape[0] !== batch || b.shape[1] !== k) {
    throw new Error(`${name}: batch/K mismatch between a and b`);
  }
  if (out.shape[0] !== batch || out.shape[1] !== m || out.shape[2] !== n) {
    throw new Error(`${name}: output shape [${out.shape}] != [${batch}, ${m}, ${n}]`);
  }

  const device = ctx.device;
  const isF16 = a.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([batch, m, n, k]));

  // Use 64x64 register-tiled GEMM when dimensions are sufficiently large and divisible by 4,
  // falling back to 32x32 vec4 or 32x32 scalar tiled paths.
  const useVec4 = k % 4 === 0 && n % 4 === 0;
  const use64x64 = useVec4 && m >= 128 && n >= 128 && k >= 16;

  let code = isF16 ? SHADER_BMM_TILED_F16 : SHADER_BMM_TILED;
  let workgroupCountX = Math.ceil(n / 32);
  let workgroupCountY = batch * Math.ceil(m / 32);
  let workgroupCountZ = 1;

  if (use64x64) {
    code = isF16 ? SHADER_BMM_64X64_F16 : SHADER_BMM_64X64;
    workgroupCountX = Math.ceil(n / 64);
    workgroupCountY = Math.ceil(m / 64);
    workgroupCountZ = batch;
  } else if (useVec4) {
    code = isF16 ? SHADER_BMM_VEC4_F16 : SHADER_BMM_VEC4;
    workgroupCountX = Math.ceil(n / 32);
    workgroupCountY = Math.ceil(m / 32);
    workgroupCountZ = batch;
  }

  const bundle = createComputeBundle(device, code, [
    { binding: 0, buffer: a.buffer },
    { binding: 1, buffer: b.buffer },
    { binding: 2, buffer: out.buffer },
    { binding: 3, buffer: paramsBuffer },
  ]);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
    workgroupCountZ,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::bmm.default`.
 *
 * Batched matrix multiplication via shared-memory tiling —
 * Automatically selects 64x64 register-tiled or 32x32 vectorized vec4 path
 * when dimensions permit. Supports float32 and float16 with `shader-f16`.
 */
export const bmm: Shader<BmmArgs, typeof name> = {
  name,
  code: {
    tiled: SHADER_BMM_TILED,
    vec4: SHADER_BMM_VEC4,
    tiled64x64: SHADER_BMM_64X64,
    tiledF16: SHADER_BMM_TILED_F16,
    vec4F16: SHADER_BMM_VEC4_F16,
    tiled64x64F16: SHADER_BMM_64X64_F16,
  },
  recordIn,
};
