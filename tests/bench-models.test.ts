import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { it, expect } from 'vitest';
import { WgpuExecutionContext, type Program, type ProgramTensor } from '../src';
import { setupGPU } from './helpers';

// WGPU_MODEL_BASELINE=<commit> WGPU_MODEL_RESULTS=/tmp/model-results.json npx vitest run tests/bench-models.test.ts
// Archives source into a temporary directory; both contexts use identical graphs and inputs.
it.skipIf(!process.env.WGPU_MODEL_BASELINE)(
  'compares unchanged model graphs with a git kernel baseline',
  { timeout: 600_000 },
  async () => {
    const baseline = process.env.WGPU_MODEL_BASELINE!;
    const dir = mkdtempSync(join(tmpdir(), 'wgpu-model-baseline-'));
    try {
      const archive = execFileSync('git', ['archive', baseline, 'src']);
      execFileSync('tar', ['-x', '-C', dir], { input: archive });
      const baselinePath = join(dir, 'src/index.ts');
      const { WgpuExecutionContext: baselineContext } = (await import(
        baselinePath
      )) as typeof import('../src');
      const device = await setupGPU();
      const results = [];
      const median = (a: number[]) => [...a].sort((a, b) => a - b)[a.length >> 1];
      for (const model of ['depthart', 'yolo26n'])
        for (const dtype of ['float32', 'float16'] as const) {
          const suffix = dtype === 'float16' ? '-fp16' : '';
          const root = `example/public/assets/${model}`;
          const program = JSON.parse(
            readFileSync(`${root}/program${suffix}.json`, 'utf8')
          ) as Program;
          const weights = new Uint8Array(readFileSync(`${root}/weights${suffix}.bin`)).buffer;
          const array = dtype === 'float16' ? Float16Array : Float32Array;
          const runs = [baselineContext, WgpuExecutionContext].map((contextType) => {
            const ctx = new contextType(device),
              input = ctx.tensor(
                dtype,
                (program.values[program.inputIds[0]] as ProgramTensor).shape
              ),
              out = ctx.tensor(
                dtype,
                (program.values[program.outputIds[0]] as ProgramTensor).shape
              );
            input.setData(
              new array(Array.from({ length: input.numel }, (value, i) => (i % 251) / 250 - 0.5))
            );
            ctx.recordProgram(program, weights, [input, out]);
            return { ctx, out };
          });
          const samples: number[][] = [[], []];
          for (let i = -10; i < 51; i++)
            for (const version of i & 1 ? [1, 0] : [0, 1]) {
              const start = performance.now();
              runs[version].ctx.submit();
              await runs[version].ctx.sync();
              if (i >= 0) samples[version].push(performance.now() - start);
            }
          const output = await Promise.all(runs.map(async (r) => new array(await r.out.getData())));
          let maxDelta = 0;
          for (let i = 0; i < output[0].length; i++) {
            if (!Number.isFinite(output[1][i])) throw Error('nonfinite');
            maxDelta = Math.max(maxDelta, Math.abs(output[0][i] - output[1][i]));
          }
          expect(maxDelta).toBe(0);
          const result = {
            model,
            dtype,
            beforeMs: median(samples[0]),
            afterMs: median(samples[1]),
            speedup: median(samples[0]) / median(samples[1]),
            maxDelta,
            samples,
          };
          results.push(result);
          // eslint-disable-next-line no-console
          console.log(JSON.stringify({ ...result, samples: undefined }));
          runs.forEach((r) => r.ctx.destroy());
        }
      writeFileSync(
        process.env.WGPU_MODEL_RESULTS ?? join(tmpdir(), 'wgpu-model-results.json'),
        JSON.stringify({ baseline, warmups: 10, samples: 51, results }, null, 2) + '\n'
      );
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }
);
