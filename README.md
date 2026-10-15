# WebGPU ExecuTorch engine

An exercise based on ExecuTorch.

A pure WebGPU inference engine for PyTorch ExecuTorch `.pte` models, written in
TypeScript. Kernels are hand-written WGSL; tensors live in GPU buffers and
operations are executed as recorded command sequences.

> [!TIP]
> **[Try the live demo →](https://barhanc.github.io/wgpu-executorch/)** — camera-driven
> object detection (YOLO26-N), selfie segmentation and depth estimation (DepthART),
> running entirely in your browser on WebGPU. Use a WebGPU-capable browser (e.g. Chrome 113+).

> [!WARNING]
> This is an unfinished, experimental project and a lot of things still break.

## Philosophy

The engine is built around a few core architectural principles:

- **Radical Minimality**: There is no heavy framework, runtime virtual machine, or complex graph compiler. Tensors map directly to `GPUBuffer` views, and commands record directly into WebGPU passes. The entire runtime layer is thin, predictable, and transparent.
- **Self-Contained Shader Encapsulation**: Every operator lives in its own dedicated file under `src/shaders/`. All logic needed by an operator—hand-written WGSL kernel source, shape and dtype validation, uniform layout packing, and dispatch geometry—is cleanly collocated in one place.
- **Effortless Extensibility**: The core runtime is completely decoupled from operator semantics. Adding a new operator requires zero changes to the engine: implement the `Shader` interface in a new file, export it in `src/shaders/index.ts`, and it is automatically registered for both standalone execution and ExecuTorch program graphs.
- **Static Memory Efficiency**: Intermediate activations leverage ExecuTorch's ahead-of-time memory planner. Multiple tensors with non-overlapping lifetimes share pre-allocated GPU storage pools, eliminating runtime allocation overhead and keeping memory consumption deterministic.

## Quick start

```bash
npm install
npm test        # vitest + headless WebGPU (SwiftShader)
npm run example # browser demo at localhost
```

```ts
import { WgpuExecutionContext, shaders } from 'wgpu-executorch';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const ctx = new WgpuExecutionContext(device);

const xData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
const yData = new Float32Array([10.0, 20.0, 30.0]);

// Create tensors via context factory and upload initial data
const in1 = ctx.tensor('float32', [2, 3]).setData(xData);
const in2 = ctx.tensor('float32', [1, 3]).setData(yData);
const out = ctx.tensor('float32', [2, 3]);

// Compute out = in1 + 2.0 * in2
ctx.recordShader(shaders.add, [in1, in2, 2.0, out]);

const t0 = performance.now();
ctx.submit();
await ctx.sync();
const durMs = performance.now() - t0;

const outBytes = await out.getData();
const outArray = new Float32Array(outBytes);
```

### Running programs

You can execute full serialized ExecuTorch models using `recordProgram`. Convert an ExecuTorch `.pte` file (exported for WebGPU) into a `program.json` + `weights.bin` pair using the Python conversion script:

```bash
# Convert an existing .pte model
python utils/pte_to_program.py model_webgpu.pte -o model/

# Or export and convert a demo model
python utils/pte_to_program.py --export-demo -o utils/build/
```

The runtime automatically manages shared GPU memory pools according to the memory plan, uploads constant weights, dispatches the operator chain, and copies outputs back into user tensors:

```ts
import { WgpuExecutionContext, type Program } from 'wgpu-executorch';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();

const ctx = new WgpuExecutionContext(device);

// Program graph: out = (in1 + 1.5 * in2) @ in3
//
// Chain:
//   1. aten.add.Tensor: temp = in1 + 1.5 * in2   (values 0, 1, 2 -> 3)
//   2. aten.mm.default:  out = temp @ in3        (values 3, 4 -> 5)
//
// Memory planning (all tensors [2, 2] = 16 bytes):
//   - in1 and out share pool 0 (non-overlapping lifetimes)
//   - in2 is allocated in pool 1
//   - temp intermediate is allocated in pool 2
//   - in3 is allocated in pool 3
const prog: Program = {
  version: '1',
  chain: [
    { name: 'aten.add.Tensor', args: [0, 1, 2, 3] },
    { name: 'aten.mm.default', args: [3, 4, 5] },
  ],
  values: [
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 0 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 1 },
    { type: 'scalar', value: 1.5 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 2 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 3 },
    { type: 'tensor', shape: [2, 2], dtype: 'float32', memObjId: 0 }, // reuses pool 0 with in1
  ],
  inputIds: [0, 1, 4],
  outputIds: [5],
  memoryPlan: {
    pools: [
      { id: 0, size: 16 },
      { id: 1, size: 16 },
      { id: 2, size: 16 },
      { id: 3, size: 16 },
    ],
  },
};

const weights = new ArrayBuffer(0); // Raw constant tensor data (e.g. weights.bin)

const xData = new Float32Array([1, 2, 3, 4]);
const yData = new Float32Array([10, 20, 30, 40]);
const zData = new Float32Array([1, 2, 3, 4]);

// Prepare inputs and output tensor
const in1 = ctx.tensor('float32', [2, 2]).setData(xData);
const in2 = ctx.tensor('float32', [2, 2]).setData(yData);
const in3 = ctx.tensor('float32', [2, 2]).setData(zData);
const out = ctx.tensor('float32', [2, 2]);

// Record the program: pass inputs followed by outputs ([...inputs, ...outputs])
ctx.recordProgram(prog, weights, [in1, in2, in3, out]);

const t0 = performance.now();
ctx.submit();
await ctx.sync();
const durMs = performance.now() - t0;

const outBytes = await out.getData();
const outArray = new Float32Array(outBytes); // [112, 160, 240, 352]
```

## Live demos

The demo app is hosted on GitHub Pages:

**https://barhanc.github.io/wgpu-executorch/**

`example/` contains the source for that interactive browser application, with
camera-driven demos, each one a minimal task under `example/src/tasks/`.

To run it locally:

```bash
npm run example
```

Open `http://localhost:5173` in a WebGPU-capable browser (e.g. Chrome 113+ on
macOS/Windows/Linux), pick a tile, and click the camera icon.
