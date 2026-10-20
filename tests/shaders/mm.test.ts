import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.mm.default', () => {
  it('multiplies 2x2 matrices', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([5, 6, 7, 8]));
    const out = ctx.tensor('float32', [2, 2]);

    ctx.recordShader(shaders.mm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    await expectTensorClose(out, [19, 22, 43, 50]);
    ctx.destroy();
  });

  it('multiplies 64x64 matrices (vec4 path)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const N = 64;
    const aData = new Float32Array(N * N);
    const bData = new Float32Array(N * N);
    const expected = new Float32Array(N * N);

    for (let i = 0; i < N * N; i++) {
      aData[i] = (i % 7) - 3;
      bData[i] = (i % 5) - 2;
    }

    // CPU reference
    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
          sum += aData[r * N + k] * bData[k * N + c];
        }
        expected[r * N + c] = sum;
      }
    }

    const a = ctx.tensor('float32', [N, N]).setData(aData);
    const b = ctx.tensor('float32', [N, N]).setData(bData);
    const out = ctx.tensor('float32', [N, N]);

    ctx.recordShader(shaders.mm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float32Array(await out.getData());
    for (let i = 0; i < N * N; i++) {
      expect(Math.abs(actual[i] - expected[i])).toBeLessThan(1e-3);
    }
    ctx.destroy();
  });

  it('multiplies 128x128 matrices (64x64 multi-workgroup path)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const N = 128;
    const aData = new Float32Array(N * N);
    const bData = new Float32Array(N * N);
    const expected = new Float32Array(N * N);

    for (let i = 0; i < N * N; i++) {
      aData[i] = (i % 7) - 3;
      bData[i] = (i % 5) - 2;
    }

    for (let r = 0; r < N; r++) {
      for (let c = 0; c < N; c++) {
        let sum = 0;
        for (let k = 0; k < N; k++) {
          sum += aData[r * N + k] * bData[k * N + c];
        }
        expected[r * N + c] = sum;
      }
    }

    const a = ctx.tensor('float32', [N, N]).setData(aData);
    const b = ctx.tensor('float32', [N, N]).setData(bData);
    const out = ctx.tensor('float32', [N, N]);

    ctx.recordShader(shaders.mm, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float32Array(await out.getData());
    for (let i = 0; i < N * N; i++) {
      expect(Math.abs(actual[i] - expected[i])).toBeLessThan(1e-2);
    }
    ctx.destroy();
  });
});
