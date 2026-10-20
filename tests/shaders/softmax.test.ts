import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten._softmax.default', () => {
  it('normalizes along the last dimension', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [1, 2, 3]; softmax over dim -1
    const inp = ctx.tensor('float32', [1, 2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    const out = ctx.tensor('float32', [1, 2, 3]);

    ctx.recordShader(shaders.softmax, [inp, -1, false, out]);
    ctx.submit();
    await ctx.sync();

    const row0 = [1, 2, 3].map((x) => Math.exp(x));
    const s0 = row0.reduce((a, b) => a + b, 0);
    const row1 = [4, 5, 6].map((x) => Math.exp(x));
    const s1 = row1.reduce((a, b) => a + b, 0);
    await expectTensorClose(out, [
      row0[0] / s0,
      row0[1] / s0,
      row0[2] / s0,
      row1[0] / s1,
      row1[1] / s1,
      row1[2] / s1,
    ]);
    ctx.destroy();
  });

  it('normalizes along dim 0 of a 2D tensor', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [2, 2]; softmax over dim 0 (columns)
    const inp = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [2, 2]);

    ctx.recordShader(shaders.softmax, [inp, 0, false, out]);
    ctx.submit();
    await ctx.sync();

    const c0 = [1, 3].map((x) => Math.exp(x));
    const s0 = c0.reduce((a, b) => a + b, 0);
    const c1 = [2, 4].map((x) => Math.exp(x));
    const s1 = c1.reduce((a, b) => a + b, 0);
    await expectTensorClose(out, [c0[0] / s0, c1[0] / s1, c0[1] / s0, c1[1] / s1]);
    ctx.destroy();
  });

  it('normalizes along the last dimension with size divisible by 4 (vec4 path)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const R = 8;
    const inpData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, -1, 0, 1, 2, -3, 4, 0, 1]);
    const inp = ctx.tensor('float32', [2, R]).setData(inpData);
    const out = ctx.tensor('float32', [2, R]);

    ctx.recordShader(shaders.softmax, [inp, -1, false, out]);
    ctx.submit();
    await ctx.sync();

    for (let row = 0; row < 2; row++) {
      const slice = Array.from(inpData.slice(row * R, (row + 1) * R));
      const expSlice = slice.map((x) => Math.exp(x));
      const s = expSlice.reduce((a, b) => a + b, 0);
      const expected = expSlice.map((x) => x / s);
      const actual = new Float32Array(await out.getData()).slice(row * R, (row + 1) * R);
      for (let i = 0; i < R; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThan(1e-4);
      }
    }
    ctx.destroy();
  });
});
