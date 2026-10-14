import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.upsample_nearest2d.vec', () => {
  it('upsamples a 2x2 plane to 4x4 (scale 2)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [1, 1, 2, 2] = [1, 2, 3, 4] -> out [1, 1, 4, 4]
    // 1 1 2 2
    // 1 1 2 2
    // 3 3 4 4
    // 3 3 4 4
    const inp = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [1, 1, 4, 4]);

    ctx.recordShader(shaders.upsampleNearest2d, [inp, undefined, [2, 2], out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    ctx.destroy();
  });

  it('upsamples two channels independently', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // in: [1, 2, 1, 2] = ch0 [5, 6], ch1 [7, 8] -> out [1, 2, 2, 4]
    const inp = ctx.tensor('float32', [1, 2, 1, 2]).setData(new Float32Array([5, 6, 7, 8]));
    const out = ctx.tensor('float32', [1, 2, 2, 4]);

    ctx.recordShader(shaders.upsampleNearest2d, [inp, undefined, [2, 2], out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [5, 5, 6, 6, 5, 5, 6, 6, 7, 7, 8, 8, 7, 7, 8, 8]);
    ctx.destroy();
  });
});
