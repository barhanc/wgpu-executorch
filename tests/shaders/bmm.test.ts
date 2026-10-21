import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.bmm.default', () => {
  it('multiplies a single 2x3 by 3x2 batch', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // a: [1, 2, 3] = [[1, 2, 3], [4, 5, 6]]
    // b: [1, 3, 2] = [[7, 8], [9, 10], [11, 12]]
    // out: [1, 2, 2] = [[58, 64], [139, 154]]
    const a = ctx.tensor('float32', [1, 2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    const b = ctx.tensor('float32', [1, 3, 2]).setData(new Float32Array([7, 8, 9, 10, 11, 12]));
    const out = ctx.tensor('float32', [1, 2, 2]);

    ctx.recordShader(shaders.bmm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [58, 64, 139, 154], 1e-3);
    ctx.destroy();
  });

  it('multiplies two independent batches', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // batch 0: [[1, 0], [0, 1]] * [[5, 6], [7, 8]] = [[5, 6], [7, 8]]
    // batch 1: [[2, 0], [0, 3]] * [[5, 6], [7, 8]] = [[10, 12], [21, 24]]
    const a = ctx.tensor('float32', [2, 2, 2]).setData(new Float32Array([1, 0, 0, 1, 2, 0, 0, 3]));
    const b = ctx.tensor('float32', [2, 2, 2]).setData(new Float32Array([5, 6, 7, 8, 5, 6, 7, 8]));
    const out = ctx.tensor('float32', [2, 2, 2]);

    ctx.recordShader(shaders.bmm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [5, 6, 7, 8, 10, 12, 21, 24], 1e-3);
    ctx.destroy();
  });

  it('multiplies float16 batched matrices using shader-f16', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float16', [2, 2, 2]).setData(new Float16Array([1, 0, 0, 1, 2, 0, 0, 3]));
    const b = ctx.tensor('float16', [2, 2, 2]).setData(new Float16Array([5, 6, 7, 8, 5, 6, 7, 8]));
    const out = ctx.tensor('float16', [2, 2, 2]);

    ctx.recordShader(shaders.bmm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float16Array(await out.getData());
    expect(actual[0]).toBeCloseTo(5, 1);
    expect(actual[1]).toBeCloseTo(6, 1);
    expect(actual[2]).toBeCloseTo(7, 1);
    expect(actual[3]).toBeCloseTo(8, 1);
    expect(actual[4]).toBeCloseTo(10, 1);
    expect(actual[5]).toBeCloseTo(12, 1);
    expect(actual[6]).toBeCloseTo(21, 1);
    expect(actual[7]).toBeCloseTo(24, 1);
    ctx.destroy();
  });

  it('multiplies 64x64 batched matrices (tiled BMM path)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const [B, N] = [2, 64];
    const aData = new Float32Array(B * N * N);
    const bData = new Float32Array(B * N * N);

    // Identity matrices for each batch
    for (let b = 0; b < B; b++) {
      for (let i = 0; i < N; i++) {
        aData[b * (N * N) + i * N + i] = 1.0;
        for (let j = 0; j < N; j++) {
          bData[b * (N * N) + i * N + j] = (i + j) * 0.1;
        }
      }
    }

    const a = ctx.tensor('float32', [B, N, N]).setData(aData);
    const b = ctx.tensor('float32', [B, N, N]).setData(bData);
    const out = ctx.tensor('float32', [B, N, N]);

    ctx.recordShader(shaders.bmm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    // A * I = A
    const actual = new Float32Array(await out.getData());
    for (let i = 0; i < actual.length; i++) {
      expect(actual[i]).toBeCloseTo(bData[i], 3);
    }
    ctx.destroy();
  });
});
