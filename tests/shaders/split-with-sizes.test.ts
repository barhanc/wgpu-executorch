import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.split_with_sizes_copy.default', () => {
  it('splits along the channel dimension', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [1, 4, 1, 2] split dim=1 into [2, 2] -> two [1, 2, 1, 2]
    const inp = ctx
      .tensor('float32', [1, 4, 1, 2])
      .setData(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]));
    const out0 = ctx.tensor('float32', [1, 2, 1, 2]);
    const out1 = ctx.tensor('float32', [1, 2, 1, 2]);

    ctx.recordShader(shaders.splitWithSizesCopy, [inp, [2, 2], 1, [out0, out1]]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out0, [1, 2, 3, 4]);
    await expectTensorClose(out1, [5, 6, 7, 8]);
    ctx.destroy();
  });

  it('splits an uneven chunk along the last dimension', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [2, 3] split dim=1 into [1, 2] -> out0 [2, 1], out1 [2, 2]
    const inp = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    const out0 = ctx.tensor('float32', [2, 1]);
    const out1 = ctx.tensor('float32', [2, 2]);

    ctx.recordShader(shaders.splitWithSizesCopy, [inp, [1, 2], -1, [out0, out1]]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out0, [1, 4]);
    await expectTensorClose(out1, [2, 3, 5, 6]);
    ctx.destroy();
  });
});
