import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { WgpuExecutionContext, type Program, type ProgramTensor } from '../../src';
import { optimizeProgram } from '../../utils/optimize-program';
import { setupGPU } from '../helpers';

const results: unknown[] = [];
const measuring = process.env.WGPU_GRAPH_RESULTS;
const root = resolve(import.meta.dirname, '../..');
const buffer = (path: string) => new Uint8Array(readFileSync(path)).buffer;
const median = (values: number[]) => [...values].sort((a, b) => a - b)[values.length >> 1];

describe('offline program rewrites', () => {
  it('retains SiLU intermediates when they are separately observable', () => {
    const source = JSON.parse(
      readFileSync(`${root}/example/public/assets/yolo26n/program.json`, 'utf8')
    ) as Program;
    const convIndex = source.chain.findIndex((op) => op.name === 'aten.convolution.default');
    for (const observed of [source.chain[convIndex].args[9], source.chain[convIndex + 1].args[1]]) {
      const exposed = { ...source, outputIds: [...source.outputIds, observed] };
      const before = JSON.stringify(exposed);
      const result = optimizeProgram(exposed, { aliasCopies: true, fuseConvolutionSilu: true });
      expect(result.stats.fusedSilu).toBe(86);
      expect(result.program.outputIds).toHaveLength(2);
      expect(result.program.values[result.program.outputIds[1]]).toMatchObject(
        source.values[observed].type === 'tensor'
          ? {
              type: 'tensor',
              shape: (source.values[observed] as ProgramTensor).shape,
            }
          : {}
      );
      expect(JSON.stringify(exposed)).toBe(before);
    }
  });

  for (const model of ['depthart', 'yolo26n']) {
    for (const dtype of ['float32', 'float16'] as const) {
      it(`${model} ${dtype}: matches original and reference`, { timeout: 600_000 }, async () => {
        const suffix = dtype === 'float16' ? '-fp16' : '';
        const assets = `${root}/example/public/assets/${model}`;
        const original = JSON.parse(
          readFileSync(`${assets}/program${suffix}.json`, 'utf8')
        ) as Program;
        const { program, stats } = optimizeProgram(original, {
          aliasCopies: true,
          fuseConvolutionSilu: true,
          movePointwiseBeforeBilinear: true,
        });
        expect(stats.reorderedPointwise).toBe(model === 'depthart' ? 4 : 0);
        expect(stats.aliasedCopies).toBe(model === 'depthart' ? 0 : 36);
        const device = await setupGPU();
        const weights = buffer(`${assets}/weights${suffix}.bin`);
        const runs = [original, program].map((graph) => {
          const ctx = new WgpuExecutionContext(device);
          const inputSpec = graph.values[graph.inputIds[0]] as ProgramTensor;
          const outputSpec = graph.values[graph.outputIds[0]] as ProgramTensor;
          const input = ctx.tensor(dtype, inputSpec.shape);
          const output = ctx.tensor(dtype, outputSpec.shape);
          const values =
            dtype === 'float16' ? new Float16Array(input.numel) : new Float32Array(input.numel);
          // Match committed PyTorch fixture inputs.
          for (let i = 0; i < values.length; i++) values[i] = (i % 251) / 250 - 0.5;
          input.setData(values);
          ctx.recordProgram(graph, weights, [input, output]);
          return { ctx, input, output, values };
        });
        try {
          const reference = new Float32Array(
            buffer(`${root}/tests/fixtures/${model}/reference-output.f32`)
          );
          const timings: number[][] = [[], []];
          const samples = measuring ? 21 : 1;
          for (let i = -3; i < samples; i++) {
            const order = i & 1 ? [1, 0] : [0, 1];
            for (const version of order) {
              const start = performance.now();
              runs[version].ctx.submit();
              await runs[version].ctx.sync();
              if (i >= 0) timings[version].push(performance.now() - start);
            }
          }
          const outputs = await Promise.all(
            runs.map(async ({ output }) => {
              const data = await output.getData();
              return dtype === 'float16' ? new Float16Array(data) : new Float32Array(data);
            })
          );
          let maxDelta = 0;
          let maxReferenceDelta = 0;
          let beyondReference = 0;
          for (let i = 0; i < outputs[0].length; i++) {
            if (!Number.isFinite(outputs[1][i]))
              throw new Error(`Nonfinite optimized output at ${i}`);
            maxDelta = Math.max(maxDelta, Math.abs(outputs[0][i] - outputs[1][i]));
            const delta = Math.abs(outputs[1][i] - reference[i]);
            maxReferenceDelta = Math.max(maxReferenceDelta, delta);
            if (delta > (dtype === 'float32' ? 0.01 : 0.5)) beyondReference++;
          }
          if (model === 'depthart')
            expect(maxReferenceDelta).toBeLessThan(dtype === 'float32' ? 1 : 500);
          else
            expect(beyondReference / reference.length).toBeLessThanOrEqual(
              dtype === 'float32' ? 0 : 0.05
            );
          if (model === 'yolo26n' && dtype === 'float32') expect(maxDelta).toBeLessThan(0.01);
          const row = {
            model,
            dtype,
            stats,
            originalOps: original.chain.length,
            optimizedOps: program.chain.length,
            originalMs: median(timings[0]),
            optimizedMs: median(timings[1]),
            speedup: median(timings[0]) / median(timings[1]),
            maxDelta,
            maxReferenceDelta,
            timings,
          };
          results.push(row);
          // eslint-disable-next-line no-console
          if (measuring) console.log(JSON.stringify(row));
          if (measuring)
            writeFileSync(
              measuring,
              JSON.stringify(
                { baseline: 'f1f7fdb', adapter: device.adapterInfo, results },
                null,
                2
              ) + '\n'
            );
        } finally {
          runs.forEach(({ ctx }) => ctx.destroy());
        }
      });
    }
  }
});
