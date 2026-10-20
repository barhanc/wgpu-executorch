# wgpu-executorch Development Guidelines

## 1. Shader Architecture & Modularization
- **Specialized Subshaders in Dedicated Subfolders**:
  For shaders with multiple variants or subshaders (e.g., convolution, conv-with-clamp, mm, bmm), separate shader strings into dedicated submodules inside a subfolder named after the operator (e.g., `src/shaders/convolution/`, `src/shaders/mm/`).
- **File Naming & Structure**:
  - Submodule files must be named `<specializedShaderName>.wgsl.ts` (e.g., `conv2d-tile64.wgsl.ts`, `mm-64x64.wgsl.ts`).
  - Each `*.wgsl.ts` file must export strictly the WGSL code template string (`export const SHADER_... = /* wgsl */ \`...\`;`).
  - The driver file (`src/shaders/<operator>.ts`) remains lightweight, importing subshaders and handling parameter validation, uniform buffers, dispatch sizing, and compute bundle creation.

## 2. Core vs. Shaders Boundary
- Changes outside `src/shaders/` must be minimal, focused, and well justified.
- Maintain existing codebase style and public API contracts at all times.
- All performance optimizations must be developed on the designated feature branch (e.g., `perf/kernel-optimizations`) and pushed to remote regularly.

## 3. WebGPU Compute Kernel Best Practices
- **2D Register Tiling over Naive Spatial Loops**:
  Avoid per-channel workgroup loops with frequent `workgroupBarrier()` calls. Prefer 2D register-tiled GEMM architectures (e.g., 64x64 output blocks with 8x8 register accumulators per thread) to maximize arithmetic intensity and minimize barrier synchronization.
- **128-bit Vectorized Memory Transactions**:
  Use `vec4<f32>` (or `vec4<f16>`) storage buffer reads for aligned tensor buffers to saturate GPU memory bandwidth with 128-bit memory bus transactions.
- **Bitwise Indexing**:
  Replace expensive hardware integer division (`/`) and modulo (`%`) with bitwise shifts (`>>`, `<<`) and bitwise masks (`&`) whenever tile or tensor dimensions are powers of two.

## 4. Benchmarking & Verification Discipline
- **Zero Regressions**: Run the full test suite (`npm test`) on every iteration to guarantee zero numerical divergence from PyTorch reference outputs across all 25 test suites.
- **Repeatable Benchmarking**: In `WgpuExecutionContext`, record the program/shaders ONCE outside the benchmark loop. Inside the measurement loop, only execute `ctx.submit()`, followed by `await ctx.sync()`. Never call `recordShader` or `addDispatch` inside the timed loop.
