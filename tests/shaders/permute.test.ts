import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.permute_copy.default', () => {
  it('transposes a 2D tensor', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // [2, 3] -> [3, 2]
    const inp = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    const out = ctx.tensor('float32', [3, 2]);

    ctx.recordShader(shaders.permuteCopy, [inp, [1, 0], out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 4, 2, 5, 3, 6]);
    ctx.destroy();
  });

  it('swaps the last two dims of a 4D tensor', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // [1, 2, 2, 3] -> [1, 2, 3, 2]
    const inp = ctx
      .tensor('float32', [1, 2, 2, 3])
      .setData(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    const out = ctx.tensor('float32', [1, 2, 3, 2]);

    ctx.recordShader(shaders.permuteCopy, [inp, [0, 1, 3, 2], out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]);
    ctx.destroy();
  });

  it('transposes a larger matrix with multi-workgroup shared-memory tiling', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const R = 64;
    const C = 48;
    const data = new Float32Array(R * C);
    const expected = new Float32Array(C * R);

    for (let r = 0; r < R; r++) {
      for (let c = 0; c < C; c++) {
        data[r * C + c] = r * 100 + c;
        expected[c * R + r] = r * 100 + c;
      }
    }

    const inp = ctx.tensor('float32', [R, C]).setData(data);
    const out = ctx.tensor('float32', [C, R]);

    ctx.recordShader(shaders.permuteCopy, [inp, [1, 0], out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float32Array(await out.getData());
    for (let i = 0; i < C * R; i++) {
      if (actual[i] !== expected[i]) {
        throw new Error(`Mismatch at ${i}: got ${actual[i]} expected ${expected[i]}`);
      }
    }
    ctx.destroy();
  });
});
