import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it } from 'vitest';
import { WgpuExecutionContext, type Program, shaders } from '../src';
import { setupGPU } from './helpers';

const REPO_ROOT = resolve(import.meta.dirname, '..');

function makeInput(numel: number): Float32Array {
  const arr = new Float32Array(numel);
  for (let i = 0; i < numel; i++) {
    arr[i] = (i % 251) / 250 - 0.5;
  }
  return arr;
}

function readArrayBuffer(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  const copy = new ArrayBuffer(buf.byteLength);
  new Uint8Array(copy).set(buf);
  return copy;
}

describe('Benchmarks', () => {
  it('benchmark models and hot operators', { timeout: 600_000 }, async () => {
    const device = await setupGPU();

    // 1. DepthART benchmark
    {
      const ctx = new WgpuExecutionContext(device);
      const progPath = resolve(REPO_ROOT, 'example/public/assets/depthart/program.json');
      const weightsPath = resolve(REPO_ROOT, 'example/public/assets/depthart/weights.bin');
      const program = JSON.parse(readFileSync(progPath, 'utf8')) as Program;
      const weights = readArrayBuffer(weightsPath);

      const inp = ctx.tensor('float32', [1, 3, 448, 448]).setData(makeInput(3 * 448 * 448));
      const out = ctx.tensor('float32', [1, 1, 448, 448]);
      ctx.recordProgram(program, weights, [inp, out]);

      // Warmup
      ctx.submit();
      await ctx.sync();

      const iters = 5;
      const start = performance.now();
      for (let i = 0; i < iters; i++) {
        ctx.submit();
      }
      await ctx.sync();
      const avgMs = (performance.now() - start) / iters;
      // eslint-disable-next-line no-console
      console.log(`[BENCHMARK] DepthART (1x3x448x448): ${avgMs.toFixed(2)} ms`);
      ctx.destroy();
    }

    // 2. YOLO26-N benchmark
    {
      const ctx = new WgpuExecutionContext(device);
      const progPath = resolve(REPO_ROOT, 'example/public/assets/yolo26n/program.json');
      const weightsPath = resolve(REPO_ROOT, 'example/public/assets/yolo26n/weights.bin');
      const program = JSON.parse(readFileSync(progPath, 'utf8')) as Program;
      const weights = readArrayBuffer(weightsPath);

      const inp = ctx.tensor('float32', [1, 3, 640, 640]).setData(makeInput(3 * 640 * 640));
      const out = ctx.tensor('float32', [1, 84, 8400]);
      ctx.recordProgram(program, weights, [inp, out]);

      // Warmup
      ctx.submit();
      await ctx.sync();

      const iters = 5;
      const start = performance.now();
      for (let i = 0; i < iters; i++) {
        ctx.submit();
      }
      await ctx.sync();
      const avgMs = (performance.now() - start) / iters;
      // eslint-disable-next-line no-console
      console.log(`[BENCHMARK] YOLO26-N (1x3x640x640): ${avgMs.toFixed(2)} ms`);
      ctx.destroy();
    }

    // 3. Conv2D microbenchmark (typical bottleneck layer: 64ch -> 64ch, 56x56, 3x3)
    {
      const ctx = new WgpuExecutionContext(device);
      const inp = ctx.tensor('float32', [1, 64, 56, 56]).setData(makeInput(1 * 64 * 56 * 56));
      const weight = ctx.tensor('float32', [64, 64, 3, 3]).setData(makeInput(64 * 64 * 3 * 3));
      const bias = ctx.tensor('float32', [64]).setData(makeInput(64));
      const out = ctx.tensor('float32', [1, 64, 56, 56]);

      ctx.recordShader(shaders.convolution, [
        inp,
        weight,
        bias,
        [1, 1], // stride
        [1, 1], // padding
        [1, 1], // dilation
        false, // transposed
        [0, 0], // output_padding
        1, // groups
        out,
      ]);

      ctx.submit();
      await ctx.sync();

      const iters = 20;
      const start = performance.now();
      for (let i = 0; i < iters; i++) {
        ctx.submit();
      }
      await ctx.sync();
      const avgMs = (performance.now() - start) / iters;
      // eslint-disable-next-line no-console
      console.log(`[BENCHMARK] Conv2D [1,64,56,56]*[64,64,3,3]: ${avgMs.toFixed(3)} ms`);
      ctx.destroy();
    }

    // 4. BMM microbenchmark (typical attention shape: [4, 64, 64] * [4, 64, 64])
    {
      const ctx = new WgpuExecutionContext(device);
      const a = ctx.tensor('float32', [4, 64, 64]).setData(makeInput(4 * 64 * 64));
      const b = ctx.tensor('float32', [4, 64, 64]).setData(makeInput(4 * 64 * 64));
      const out = ctx.tensor('float32', [4, 64, 64]);

      ctx.recordShader(shaders.bmm, [a, b, out]);

      ctx.submit();
      await ctx.sync();

      const iters = 50;
      const start = performance.now();
      for (let i = 0; i < iters; i++) {
        ctx.submit();
      }
      await ctx.sync();
      const avgMs = (performance.now() - start) / iters;
      // eslint-disable-next-line no-console
      console.log(`[BENCHMARK] BMM [4,64,64]*[4,64,64]: ${avgMs.toFixed(3)} ms`);
      ctx.destroy();
    }

    // 5. MM microbenchmark (GEMM [256, 256] * [256, 256])
    {
      const ctx = new WgpuExecutionContext(device);
      const a = ctx.tensor('float32', [256, 256]).setData(makeInput(256 * 256));
      const b = ctx.tensor('float32', [256, 256]).setData(makeInput(256 * 256));
      const out = ctx.tensor('float32', [256, 256]);

      ctx.recordShader(shaders.mm, [a, b, out]);

      ctx.submit();
      await ctx.sync();

      const iters = 30;
      const start = performance.now();
      for (let i = 0; i < iters; i++) {
        ctx.submit();
      }
      await ctx.sync();
      const avgMs = (performance.now() - start) / iters;
      // eslint-disable-next-line no-console
      console.log(`[BENCHMARK] MM [256,256]*[256,256]: ${avgMs.toFixed(3)} ms`);
      ctx.destroy();
    }
  });
});
