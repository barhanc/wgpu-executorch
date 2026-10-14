import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.max_pool2d_with_indices.default', () => {
  it('pools a 4x4 plane with 2x2 windows and returns indices', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [1, 1, 4, 4]
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    // 13 14 15 16
    // k=2 s=2 -> out [1, 1, 2, 2]: [6, 8, 14, 16]
    // indices are flat in-window offsets (ih*IW+iw): 5, 7, 13, 15
    const inp = ctx
      .tensor('float32', [1, 1, 4, 4])
      .setData(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
    const outVals = ctx.tensor('float32', [1, 1, 2, 2]);
    const outIdx = ctx.tensor('int32', [1, 1, 2, 2]);

    ctx.recordShader(shaders.maxPool2d, [
      inp,
      [2, 2],
      [2, 2],
      [0, 0],
      [1, 1],
      false,
      [outVals, outIdx],
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(outVals, [6, 8, 14, 16]);
    const idx = new Int32Array(await outIdx.getData());
    const expectedIdx = [5, 7, 13, 15];
    for (let i = 0; i < expectedIdx.length; i++) {
      if (idx[i] !== expectedIdx[i]) {
        throw new Error(`indices mismatch at ${i}: got ${idx[i]}, expected ${expectedIdx[i]}`);
      }
    }
    ctx.destroy();
  });

  it('handles padding by skipping pad cells', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [1, 1, 2, 2] = [1, 2, 3, 4], k=3 s=1 p=1 -> out [1, 1, 2, 2]
    // 3x3 windows overlap the whole 2x2 plane from every output position,
    // so each output is the global max 4.
    const inp = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const outVals = ctx.tensor('float32', [1, 1, 2, 2]);
    const outIdx = ctx.tensor('int32', [1, 1, 2, 2]);

    ctx.recordShader(shaders.maxPool2d, [
      inp,
      [3, 3],
      [1, 1],
      [1, 1],
      [1, 1],
      false,
      [outVals, outIdx],
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(outVals, [4, 4, 4, 4]);
    ctx.destroy();
  });
});
