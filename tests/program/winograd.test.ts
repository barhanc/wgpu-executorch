import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { WgpuExecutionContext, type Program, type ProgramTensor } from '../../src';
import { winogradProgram } from '../../utils/winograd-program';
import { setupGPU } from '../helpers';

const root = resolve(import.meta.dirname, '../..');
const readBuffer = (path: string) => new Uint8Array(readFileSync(path)).buffer;
const measuring = process.env.WGPU_WINOGRAD_RESULTS;
const results: unknown[] = [];
const median = (values: number[]) => [...values].sort((a, b) => a - b)[values.length >> 1];

describe('experimental Winograd DepthART graph', () => {
  for (const dtype of ['float32', 'float16'] as const)
    for (const tile of [2, 4] as const) {
      it(`${dtype}, F(${tile}x${tile},3x3)`, { timeout: 600_000 }, async () => {
        const device = await setupGPU();
        device.pushErrorScope('validation');
        const suffix = dtype === 'float16' ? '-fp16' : '';
        const source = JSON.parse(
          readFileSync(`${root}/example/public/assets/depthart/program${suffix}.json`, 'utf8')
        ) as Program;
        const weights = readBuffer(`${root}/example/public/assets/depthart/weights${suffix}.bin`);
        const candidate = winogradProgram(source, weights, tile);
        expect(candidate.replaced).toHaveLength(20);
        const array = dtype === 'float16' ? Float16Array : Float32Array;
        const runs = [{ program: source, weights }, candidate].map(({ program, weights }) => {
          const ctx = new WgpuExecutionContext(device);
          const input = ctx.tensor(
            dtype,
            (program.values[program.inputIds[0]] as ProgramTensor).shape
          );
          const output = ctx.tensor(
            dtype,
            (program.values[program.outputIds[0]] as ProgramTensor).shape
          );
          input.setData(
            new array(Array.from({ length: input.numel }, (value, i) => (i % 251) / 250 - 0.5))
          );
          ctx.recordProgram(program, weights, [input, output]);
          return { ctx, output };
        });
        try {
          const timings: number[][] = [[], []];
          for (let i = -3; i < (measuring ? 21 : 1); i++)
            for (const version of i & 1 ? [1, 0] : [0, 1]) {
              const start = performance.now();
              runs[version].ctx.submit();
              await runs[version].ctx.sync();
              if (i >= 0) timings[version].push(performance.now() - start);
            }
          const outputs = await Promise.all(
            runs.map(async ({ output }) => new array(await output.getData()))
          );
          const reference = new Float32Array(
            readBuffer(`${root}/tests/fixtures/depthart/reference-output.f32`)
          );
          let maxReferenceDelta = 0,
            maxOriginalDelta = 0;
          for (let i = 0; i < reference.length; i++) {
            if (!Number.isFinite(outputs[1][i])) throw new Error(`Nonfinite output at ${i}`);
            maxReferenceDelta = Math.max(maxReferenceDelta, Math.abs(outputs[1][i] - reference[i]));
            maxOriginalDelta = Math.max(maxOriginalDelta, Math.abs(outputs[1][i] - outputs[0][i]));
          }
          expect(await device.popErrorScope()).toBeNull();
          expect(maxReferenceDelta).toBeLessThan(dtype === 'float32' ? 1 : 500);
          const row = {
            model: 'depthart',
            dtype,
            tile,
            replaced: candidate.replaced,
            scratchBytes: candidate.scratchBytes,
            originalMs: median(timings[0]),
            candidateMs: median(timings[1]),
            speedup: median(timings[0]) / median(timings[1]),
            maxReferenceDelta,
            maxOriginalDelta,
            timings,
          };
          results.push(row);
          if (measuring) {
            // eslint-disable-next-line no-console
            console.log(JSON.stringify(row));
            writeFileSync(
              measuring,
              JSON.stringify(
                { baseline: 'b09fe1f, original graph and kernels', results },
                null,
                2
              ) + '\n'
            );
          }
        } finally {
          runs.forEach(({ ctx }) => ctx.destroy());
        }
      });
    }
});
