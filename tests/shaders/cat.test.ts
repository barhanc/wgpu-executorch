import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.cat.default', () => {
  it('concatenates along the channel dimension', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // a: [1, 2, 1, 2], b: [1, 1, 1, 2] -> out: [1, 3, 1, 2]
    const a = ctx.tensor('float32', [1, 2, 1, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [1, 1, 1, 2]).setData(new Float32Array([5, 6]));
    const out = ctx.tensor('float32', [1, 3, 1, 2]);

    ctx.recordShader(shaders.cat, [[a, b], 1, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 2, 3, 4, 5, 6]);
    ctx.destroy();
  });

  it('concatenates three tensors along the last dimension', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // a: [2, 1], b: [2, 2], c: [2, 1] -> out: [2, 4]
    const a = ctx.tensor('float32', [2, 1]).setData(new Float32Array([1, 2]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([3, 4, 5, 6]));
    const c = ctx.tensor('float32', [2, 1]).setData(new Float32Array([7, 8]));
    const out = ctx.tensor('float32', [2, 4]);

    ctx.recordShader(shaders.cat, [[a, b, c], -1, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 3, 4, 7, 2, 5, 6, 8]);
    ctx.destroy();
  });
});
