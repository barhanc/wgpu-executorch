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

function makeInputF16(numel: number): Float16Array {
  const arr = new Float16Array(numel);
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
      console.log(`[BENCHMARK] Conv2D [1,64,56,56]*[64,64,3,3] (FP32): ${avgMs.toFixed(3)} ms`);
      ctx.destroy();

      if (device.features.has('shader-f16')) {
        const ctxF16 = new WgpuExecutionContext(device);
        const inpF16 = ctxF16
          .tensor('float16', [1, 64, 56, 56])
          .setData(makeInputF16(1 * 64 * 56 * 56));
        const weightF16 = ctxF16
          .tensor('float16', [64, 64, 3, 3])
          .setData(makeInputF16(64 * 64 * 3 * 3));
        const biasF16 = ctxF16.tensor('float16', [64]).setData(makeInputF16(64));
        const outF16 = ctxF16.tensor('float16', [1, 64, 56, 56]);

        ctxF16.recordShader(shaders.convolution, [
          inpF16,
          weightF16,
          biasF16,
          [1, 1],
          [1, 1],
          [1, 1],
          false,
          [0, 0],
          1,
          outF16,
        ]);

        ctxF16.submit();
        await ctxF16.sync();

        const startF16 = performance.now();
        for (let i = 0; i < iters; i++) {
          ctxF16.submit();
        }
        await ctxF16.sync();
        const avgMsF16 = (performance.now() - startF16) / iters;
        const speedup = avgMs / avgMsF16;
        // eslint-disable-next-line no-console
        console.log(
          `[BENCHMARK] Conv2D [1,64,56,56]*[64,64,3,3] (FP16): ${avgMsF16.toFixed(3)} ms (${speedup.toFixed(2)}x speedup)`
        );
        ctxF16.destroy();
      }
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
      console.log(`[BENCHMARK] BMM [4,64,64]*[4,64,64] (FP32): ${avgMs.toFixed(3)} ms`);
      ctx.destroy();

      if (device.features.has('shader-f16')) {
        const ctxF16 = new WgpuExecutionContext(device);
        const aF16 = ctxF16.tensor('float16', [4, 64, 64]).setData(makeInputF16(4 * 64 * 64));
        const bF16 = ctxF16.tensor('float16', [4, 64, 64]).setData(makeInputF16(4 * 64 * 64));
        const outF16 = ctxF16.tensor('float16', [4, 64, 64]);

        ctxF16.recordShader(shaders.bmm, [aF16, bF16, outF16]);

        ctxF16.submit();
        await ctxF16.sync();

        const startF16 = performance.now();
        for (let i = 0; i < iters; i++) {
          ctxF16.submit();
        }
        await ctxF16.sync();
        const avgMsF16 = (performance.now() - startF16) / iters;
        const speedup = avgMs / avgMsF16;
        // eslint-disable-next-line no-console
        console.log(
          `[BENCHMARK] BMM [4,64,64]*[4,64,64] (FP16): ${avgMsF16.toFixed(3)} ms (${speedup.toFixed(2)}x speedup)`
        );
        ctxF16.destroy();
      }
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
      console.log(`[BENCHMARK] MM [256,256]*[256,256] (FP32): ${avgMs.toFixed(3)} ms`);
      ctx.destroy();

      if (device.features.has('shader-f16')) {
        const ctxF16 = new WgpuExecutionContext(device);
        const aF16 = ctxF16.tensor('float16', [256, 256]).setData(makeInputF16(256 * 256));
        const bF16 = ctxF16.tensor('float16', [256, 256]).setData(makeInputF16(256 * 256));
        const outF16 = ctxF16.tensor('float16', [256, 256]);

        ctxF16.recordShader(shaders.mm, [aF16, bF16, outF16]);

        ctxF16.submit();
        await ctxF16.sync();

        const startF16 = performance.now();
        for (let i = 0; i < iters; i++) {
          ctxF16.submit();
        }
        await ctxF16.sync();
        const avgMsF16 = (performance.now() - startF16) / iters;
        const speedup = avgMs / avgMsF16;
        // eslint-disable-next-line no-console
        console.log(
          `[BENCHMARK] MM [256,256]*[256,256] (FP16): ${avgMsF16.toFixed(3)} ms (${speedup.toFixed(2)}x speedup)`
        );
        ctxF16.destroy();
      }
    }

    // 6. Large MM microbenchmark (GEMM [1024, 1024] * [1024, 1024])
    {
      const ctx = new WgpuExecutionContext(device);
      const a = ctx.tensor('float32', [1024, 1024]).setData(makeInput(1024 * 1024));
      const b = ctx.tensor('float32', [1024, 1024]).setData(makeInput(1024 * 1024));
      const out = ctx.tensor('float32', [1024, 1024]);

      ctx.recordShader(shaders.mm, [a, b, out]);

      ctx.submit();
      await ctx.sync();

      const iters = 15;
      const start = performance.now();
      for (let i = 0; i < iters; i++) {
        ctx.submit();
      }
      await ctx.sync();
      const avgMs = (performance.now() - start) / iters;
      // eslint-disable-next-line no-console
      console.log(`[BENCHMARK] MM [1024,1024]*[1024,1024] (FP32): ${avgMs.toFixed(3)} ms`);
      ctx.destroy();

      if (device.features.has('shader-f16')) {
        const ctxF16 = new WgpuExecutionContext(device);
        const aF16 = ctxF16.tensor('float16', [1024, 1024]).setData(makeInputF16(1024 * 1024));
        const bF16 = ctxF16.tensor('float16', [1024, 1024]).setData(makeInputF16(1024 * 1024));
        const outF16 = ctxF16.tensor('float16', [1024, 1024]);

        ctxF16.recordShader(shaders.mm, [aF16, bF16, outF16]);

        ctxF16.submit();
        await ctxF16.sync();

        const startF16 = performance.now();
        for (let i = 0; i < iters; i++) {
          ctxF16.submit();
        }
        await ctxF16.sync();
        const avgMsF16 = (performance.now() - startF16) / iters;
        const speedup = avgMs / avgMsF16;
        // eslint-disable-next-line no-console
        console.log(
          `[BENCHMARK] MM [1024,1024]*[1024,1024] (FP16): ${avgMsF16.toFixed(3)} ms (${speedup.toFixed(2)}x speedup)`
        );
        ctxF16.destroy();
      }
    }

    // 7. Elementwise microbenchmark (Mul [1, 64, 56, 56])
    {
      const ctx = new WgpuExecutionContext(device);
      const a = ctx.tensor('float32', [1, 64, 56, 56]).setData(makeInput(64 * 56 * 56));
      const b = ctx.tensor('float32', [1, 64, 56, 56]).setData(makeInput(64 * 56 * 56));
      const out = ctx.tensor('float32', [1, 64, 56, 56]);

      ctx.recordShader(shaders.mul, [a, b, out]);

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
      console.log(`[BENCHMARK] Mul [1,64,56,56] (FP32): ${avgMs.toFixed(3)} ms`);
      ctx.destroy();

      if (device.features.has('shader-f16')) {
        const ctxF16 = new WgpuExecutionContext(device);
        const aF16 = ctxF16.tensor('float16', [1, 64, 56, 56]).setData(makeInputF16(64 * 56 * 56));
        const bF16 = ctxF16.tensor('float16', [1, 64, 56, 56]).setData(makeInputF16(64 * 56 * 56));
        const outF16 = ctxF16.tensor('float16', [1, 64, 56, 56]);

        ctxF16.recordShader(shaders.mul, [aF16, bF16, outF16]);

        ctxF16.submit();
        await ctxF16.sync();

        const startF16 = performance.now();
        for (let i = 0; i < iters; i++) {
          ctxF16.submit();
        }
        await ctxF16.sync();
        const avgMsF16 = (performance.now() - startF16) / iters;
        const speedup = avgMs / avgMsF16;
        // eslint-disable-next-line no-console
        console.log(
          `[BENCHMARK] Mul [1,64,56,56] (FP16): ${avgMsF16.toFixed(3)} ms (${speedup.toFixed(2)}x speedup)`
        );
        ctxF16.destroy();
      }
    }
  });
});
