import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('flat-copy ops (view/clone/squeeze/unsqueeze/prepack)', () => {
  it('aten.view_copy.default reshapes without remapping data', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    const out = ctx.tensor('float32', [3, 2]);

    ctx.recordShader(shaders.viewCopy, [inp, [3, 2], out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 2, 3, 4, 5, 6]);
    ctx.destroy();
  });

  it('aten.clone.default copies data', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [2, 2]);

    ctx.recordShader(shaders.clone, [inp, undefined, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 2, 3, 4]);
    ctx.destroy();
  });

  it('aten.unsqueeze_copy.default adds a leading dim', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float32', [3]).setData(new Float32Array([7, 8, 9]));
    const out = ctx.tensor('float32', [1, 3]);

    ctx.recordShader(shaders.unsqueezeCopy, [inp, 0, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [7, 8, 9]);
    ctx.destroy();
  });

  it('aten.squeeze_copy.dims drops size-1 dims', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float32', [1, 3]).setData(new Float32Array([7, 8, 9]));
    const out = ctx.tensor('float32', [3]);

    ctx.recordShader(shaders.squeezeCopy, [inp, [0], out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [7, 8, 9]);
    ctx.destroy();
  });

  it('et_vk.prepack.default materializes constants', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const out = ctx.tensor('float32', [2, 2]);

    ctx.recordShader(shaders.prepack, [inp, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [1, 2, 3, 4]);
    ctx.destroy();
  });
});
