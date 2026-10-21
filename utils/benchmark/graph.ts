import { WgpuExecutionContext, type Program, type ProgramTensor } from '../../src';
import { optimizeProgram } from '../optimize-program';
import { winogradProgram } from '../winograd-program';

async function run() {
  const display = document.getElementById('results')!;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('WebGPU adapter unavailable');
  const hasF16 = adapter.features.has('shader-f16') && typeof Float16Array !== 'undefined';
  const device = await adapter.requestDevice({ requiredFeatures: hasF16 ? ['shader-f16'] : [] });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[values.length >> 1];
  const results = [];
  const winogradTile = new URLSearchParams(location.search).get('winograd');
  const tile = winogradTile === '2' ? 2 : winogradTile === '4' ? 4 : undefined;
  for (const model of tile ? ['depthart'] : ['depthart', 'yolo26n']) {
    for (const dtype of hasF16 ? (['float32', 'float16'] as const) : (['float32'] as const)) {
      display.textContent = `Running ${model} ${dtype}…`;
      const suffix = dtype === 'float16' ? '-fp16' : '';
      const responses = await Promise.all([
        fetch(`./assets/${model}/program${suffix}.json`),
        fetch(`./assets/${model}/weights${suffix}.bin`),
      ]);
      if (responses.some((response) => !response.ok)) throw new Error(`Failed loading ${model}`);
      const original = (await responses[0].json()) as Program;
      const weights = await responses[1].arrayBuffer();
      const optimized = optimizeProgram(original, {
        aliasCopies: true,
        fuseConvolutionSilu: true,
        movePointwiseBeforeBilinear: true,
      });
      const candidate = tile
        ? winogradProgram(original, weights, tile)
        : { program: optimized.program, weights };
      const program = candidate.program;
      const stats = tile
        ? {
            tile,
            replaced: 'replaced' in candidate ? candidate.replaced : [],
            scratchBytes: 'scratchBytes' in candidate ? candidate.scratchBytes : 0,
          }
        : optimized.stats;
      const array = dtype === 'float16' ? Float16Array : Float32Array;
      const runs = [{ program: original, weights }, candidate].map(
        ({ program: graph, weights }) => {
          const ctx = new WgpuExecutionContext(device);
          const inputSpec = graph.values[graph.inputIds[0]] as ProgramTensor;
          const outputSpec = graph.values[graph.outputIds[0]] as ProgramTensor;
          const input = ctx.tensor(dtype, inputSpec.shape);
          const output = ctx.tensor(dtype, outputSpec.shape);
          input.setData(
            new array(Array.from({ length: input.numel }, (value, i) => (i % 251) / 250 - 0.5))
          );
          ctx.recordProgram(graph, weights, [input, output]);
          return { ctx, output };
        }
      );
      try {
        const timings: number[][] = [[], []];
        // Record once above; alternate order and await each submission separately.
        for (let i = -5; i < 31; i++) {
          for (const version of i & 1 ? [1, 0] : [0, 1]) {
            const start = performance.now();
            runs[version].ctx.submit();
            await runs[version].ctx.sync();
            if (i >= 0) timings[version].push(performance.now() - start);
          }
        }
        const outputs = await Promise.all(
          runs.map(async ({ output }) => new array(await output.getData()))
        );
        let maxDelta = 0;
        for (let i = 0; i < outputs[0].length; i++) {
          if (!Number.isFinite(outputs[1][i]))
            throw new Error(`Nonfinite output ${model} ${dtype} at ${i}`);
          maxDelta = Math.max(maxDelta, Math.abs(outputs[0][i] - outputs[1][i]));
        }
        results.push({
          model,
          dtype,
          stats,
          originalOps: original.chain.length,
          optimizedOps: program.chain.length,
          originalMs: median(timings[0]),
          optimizedMs: median(timings[1]),
          speedup: median(timings[0]) / median(timings[1]),
          maxDelta,
          timings,
        });
      } finally {
        runs.forEach(({ ctx }) => ctx.destroy());
      }
    }
  }
  if (errors.length) throw new Error(errors.join('\n'));
  const report = {
    baseline: 'Original graph with current production kernels, same precision',
    experiment: tile ? 'winograd' : 'graph-rewrites',
    browser: navigator.userAgent,
    gpu: {
      vendor: adapter.info.vendor,
      architecture: adapter.info.architecture,
      device: adapter.info.device,
      description: adapter.info.description,
    },
    warmupSubmissions: 5,
    measuredSubmissions: 31,
    results,
    errors,
  };
  display.textContent = JSON.stringify(report, null, 2);
  const response = await fetch('/api/graph-results', {
    method: 'POST',
    // eslint-disable-next-line @typescript-eslint/naming-convention
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(report),
  });
  if (!response.ok) throw new Error('Failed saving graph benchmark results');
  document.title = 'GRAPH_BENCHMARK_DONE';
}

run().catch((error: unknown) => {
  document.getElementById('results')!.textContent = String(error);
  document.title = 'GRAPH_BENCHMARK_FAILED';
});
