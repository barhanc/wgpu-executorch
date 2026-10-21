# AGENTS.md

## Project Overview
- **wgpu-executorch**: A high-performance WebGPU runtime and WGSL compute engine for ExecuTorch ML inference in the browser.
- **Tech Stack**: TypeScript, WebGPU / WGSL, Vitest, Vite, ESLint, Prettier, Lefthook.

## Commands & Workflows
- **Build / Bundle**: `npm run build`
- **Typecheck**: `npm run typecheck` (`tsc --noEmit`)
- **Lint & Fix**: `npm run lint` / `npx eslint --fix .`
- **Full Test Suite**: `npm test` (`vitest run` — 27 test suites, 112+ tests)
- **Run Single Test**: `npx vitest run <path/to/test.ts>`
- **Benchmarks**: `npm run benchmark:web` (starts local benchmark server on `http://localhost:5178/`)
- **Process Safety Guardrail**: NEVER run `taskkill`, `pkill`, or kill external host browser processes (`chrome.exe`, `msedge.exe`). To test or benchmark in browser environments, spin up a dedicated local HTTP / Vite server (`npm run benchmark:web`) with an auto-reporting endpoint (`POST /api/results`), allowing the user to navigate to the URL without process disruption.
- **Git Branching**: All performance and kernel work must be developed on the active feature branch (e.g., `perf/kernel-optimizations`) and pushed to remote origin regularly.

## Architectural Boundaries
- **Core vs. Shaders**:
  - Code changes outside `src/shaders/` must be minimal, focused, and well justified.
  - Preserve core API style and existing contracts (`src/tensor.ts`, `src/command.ts`, `src/program.ts`, `src/context.ts`).
- **Shader Modularization**:
  - Every operator resides in a dedicated operator subdirectory (`src/shaders/<operator>/index.ts`).
  - Submodules must follow the naming pattern `<specializedShaderName>.wgsl.ts` (e.g. `conv2d-tile64.wgsl.ts`, `mm-64x64.wgsl.ts`).
  - Submodule files must export strictly the WGSL code template string (`export const SHADER_... = /* wgsl */ \`...\`;`) and include an MIT license and citation note (`https://github.com/barhanc/wgpu-executorch`).
  - Driver files (`src/shaders/<operator>/index.ts`) must have an empty line between standard imports and shader imports, and between shader imports and the `const name = ...` definition.
  - In WGSL compute shaders, loop index variables must be declared with mutable `var` (e.g., `for (var i = 0u; ...)`), never `let`.

## WebGPU Alignment & Buffer Rules
- **4-Byte Alignment Invariant**:
  - WebGPU `copyBufferToBuffer` requires `size`, `sourceOffset`, and `destinationOffset` to be multiples of 4 bytes.
  - When copying float16 or odd-sized tensor buffers, size must be padded using `Math.ceil(size / 4) * 4`.
- **Node.js FP16 Support**:
  - Node.js LTS environments lack global `Float16Array`; polyfill via `@petamoriken/float16` conditionally in test setups.

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
