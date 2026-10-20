import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';
import { SHADER_MM_TILED } from './mm/mm-tiled.wgsl';
import { SHADER_MM_VEC4 } from './mm/mm-vec4.wgsl';
import { SHADER_MM_64X64 } from './mm/mm-64x64.wgsl';

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
  if (in1.dtype !== 'float32' || in2.dtype !== 'float32' || out.dtype !== 'float32') {
    throw new Error(`${name}: Only float32 tensors are currently supported`);
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
  const paramsBuffer = ctx.uniformBuffer(new Uint32Array([m, n, k1, 0]));

  // Use 64x64 register-tiled GEMM when dimensions are sufficiently large and divisible by 4,
  // falling back to 32x32 vec4 or 32x32 scalar tiled paths.
  const useVec4 = k1 % 4 === 0 && n % 4 === 0;
  const use64x64 = useVec4 && m >= 64 && n >= 64 && k1 >= 16;

  let code = SHADER_MM_TILED;
  let workgroupCountX = Math.ceil(n / TILE);
  let workgroupCountY = Math.ceil(m / TILE);

  if (use64x64) {
    code = SHADER_MM_64X64;
    workgroupCountX = Math.ceil(n / 64);
    workgroupCountY = Math.ceil(m / 64);
  } else if (useVec4) {
    code = SHADER_MM_VEC4;
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
 * Automatically selects 64x64 register-tiled or 32x32 128-bit vectorized `vec4<f32>` paths
 * when dimensions permit, falling back to the standard scalar tiled path.
 */
export const mm: Shader<MmArgs, typeof name> = {
  name,
  code: { tiled: SHADER_MM_TILED, vec4: SHADER_MM_VEC4, tiled64x64: SHADER_MM_64X64 },
  recordIn,
};
