# AGENTS.md

## Project Overview
- **wgpu-executorch**: A high-performance WebGPU runtime and WGSL compute engine for ExecuTorch ML inference in the browser.
- **Tech Stack**: TypeScript, WebGPU / WGSL, Vitest, Vite, ESLint, Prettier, Lefthook.

## Commands & Workflows
- **Build / Bundle**: `npm run build`
- **Typecheck**: `npm run typecheck` (`tsc --noEmit`)
- **Lint & Fix**: `npm run lint` / `npx eslint --fix .`
- **Full Test Suite**: `npm test` (`vitest run` — 25 test suites, 81+ tests)
- **Run Single Test**: `npx vitest run <path/to/test.ts>`
- **Benchmarks**: `npx vitest run tests/benchmark.test.ts`
- **Git Branching**: All performance and kernel work must be developed on the active feature branch (e.g., `perf/kernel-optimizations`) and pushed to remote origin regularly.

## Architectural Boundaries
- **Core vs. Shaders**:
  - Code changes outside `src/shaders/` must be minimal, focused, and well justified.
  - Preserve core API style and existing contracts (`src/tensor.ts`, `src/command.ts`, `src/program.ts`, `src/context.ts`).
- **Shader Modularization**:
  - Complex shaders with multiple variants or subshaders (e.g. `convolution`, `conv-with-clamp`, `mm`, `bmm`) must reside in dedicated operator subdirectories (e.g. `src/shaders/convolution/`, `src/shaders/conv-with-clamp/`, `src/shaders/mm/`).
  - Submodules must follow the naming pattern `<specializedShaderName>.wgsl.ts` (e.g. `conv2d-tile64.wgsl.ts`, `mm-64x64.wgsl.ts`).
  - Submodule files must export strictly the WGSL code template string (`export const SHADER_... = /* wgsl */ \`...\`;`).
  - Driver files (`src/shaders/<operator>.ts`) remain lightweight, importing subshaders and managing argument validation, uniform buffers, dispatch grid sizing, and compute bundle creation.

## WebGPU Kernel Performance Guidelines
- **2D Register Tiling over Naive Spatial Barrier Loops**:
  - Avoid per-channel workgroup loops with frequent `workgroupBarrier()` calls.
  - Prefer 2D register-tiled GEMM architectures (e.g., 64x64 output blocks with 8x8 register accumulators per thread) to maximize arithmetic intensity and minimize barrier synchronization.
- **128-bit Vectorized Memory Transactions**:
  - Use `vec4<f32>` (or `vec4<f16>`) storage buffer reads for aligned tensor buffers to saturate the GPU memory bus with 128-bit memory transactions.
- **Bitwise Indexing**:
  - Replace hardware integer division (`/`) and modulo (`%`) with bitwise shifts (`>>`, `<<`) and bitwise masks (`&`) whenever tile or tensor dimensions are powers of two.

## Benchmarking & Verification Discipline
- **Zero Regressions**: All 25 test suites must pass 100% green against PyTorch reference outputs before and after optimizations.
- **Repeatable Benchmarking**: In `WgpuExecutionContext`, record the program/shaders ONCE outside the benchmark loop. Inside the measurement loop, only execute `ctx.submit()`, followed by `await ctx.sync()`. Never call `recordShader` or `addDispatch` inside the timed loop to avoid command list accumulation.
