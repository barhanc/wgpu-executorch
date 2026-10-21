import type { Shader } from '../../shader';
import type { Tensor } from '../../tensor';
import type { WgpuExecutionContext } from '../../context';
import { createComputeBundle } from '../../command';

import { SHADER_MM_TILED } from './mm-tiled.wgsl';
import { SHADER_MM_VEC4 } from './mm-vec4.wgsl';
import { SHADER_MM_64X64 } from './mm-64x64.wgsl';
import { SHADER_MM_TILED_F16 } from './mm-tiled-f16.wgsl';
import { SHADER_MM_VEC4_F16 } from './mm-vec4-f16.wgsl';
import { SHADER_MM_64X64_F16 } from './mm-64x64-f16.wgsl';

const name = 'aten.mm.default';

/**
 * Output tile dimension per workgroup (32x32).
 */
const TILE = 32;

/**
 * Positional argument tuple for `aten::mm.default`:
 * `[in1, in2, out]`
 */
export type MmArgs = readonly [in1: Tensor, in2: Tensor, out: Tensor];

/**
 * Internal dispatch builder for `aten::mm.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, [in1, in2, out]: MmArgs): void {
  if (
    (in1.dtype !== 'float32' && in1.dtype !== 'float16') ||
    in2.dtype !== in1.dtype ||
    out.dtype !== in1.dtype
  ) {
    throw new Error(`${name}: Tensors must have matching float32 or float16 dtype`);
  }
  if (in1.shape.length !== 2 || in2.shape.length !== 2 || out.shape.length !== 2) {
    throw new Error(`${name}: Tensors must be 2D matrices`);
  }

  const [m, k1] = in1.shape as [number, number];
  const [k2, n] = in2.shape as [number, number];
  const [outM, outN] = out.shape as [number, number];

  if (k1 !== k2) {
    throw new Error(`${name}: Matrix inner dimensions must match (${k1} !== ${k2})`);
  }
  if (outM !== m || outN !== n) {
    throw new Error(`${name}: Output shape [${outM}, ${outN}] !== expected [${m}, ${n}]`);
  }

  const device = ctx.device;
  const isF16 = in1.dtype === 'float16';
  if (isF16 && !device.features.has('shader-f16')) {
    throw new Error(`${name}: Device does not support shader-f16 feature`);
  }

  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([m, n, k1, 0]));

  // Use 64x64 register-tiled GEMM when dimensions are sufficiently large and divisible by 4,
  // falling back to 32x32 vec4 or 32x32 scalar tiled paths.
  const useVec4 = k1 % 4 === 0 && n % 4 === 0;
  const use64x64 = useVec4 && m >= 64 && n >= 64 && k1 >= 16;

  let code = isF16 ? SHADER_MM_TILED_F16 : SHADER_MM_TILED;
  let workgroupCountX = Math.ceil(n / TILE);
  let workgroupCountY = Math.ceil(m / TILE);

  if (use64x64) {
    code = isF16 ? SHADER_MM_64X64_F16 : SHADER_MM_64X64;
    workgroupCountX = Math.ceil(n / 64);
    workgroupCountY = Math.ceil(m / 64);
  } else if (useVec4) {
    code = isF16 ? SHADER_MM_VEC4_F16 : SHADER_MM_VEC4;
  }

  const bundle = createComputeBundle(device, code, [
    { binding: 0, buffer: in1.buffer },
    { binding: 1, buffer: in2.buffer },
    { binding: 2, buffer: out.buffer },
    { binding: 3, buffer: paramsBuffer },
  ]);

  ctx.addDispatch({
    pipeline: bundle.pipeline,
    bindGroup: bundle.bindGroup,
    workgroupCountX,
    workgroupCountY,
  });
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::mm.default`.
 *
 * Computes matrix multiplication `out = A @ B` using a shared-memory tiled GEMM.
 * Automatically selects 64x64 register-tiled or 32x32 128-bit vectorized paths
 * when dimensions permit, falling back to the standard scalar tiled path.
 * Supports both float32 and float16 with the `shader-f16` WebGPU feature extension.
 */
export const mm: Shader<MmArgs, typeof name> = {
  name,
  code: {
    tiled: SHADER_MM_TILED,
    vec4: SHADER_MM_VEC4,
    tiled64x64: SHADER_MM_64X64,
    tiledF16: SHADER_MM_TILED_F16,
    vec4F16: SHADER_MM_VEC4_F16,
    tiled64x64F16: SHADER_MM_64X64_F16,
  },
  recordIn,
};
