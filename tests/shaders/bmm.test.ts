import { describe, it } from 'vitest';
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
});
