import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.sigmoid.default', () => {
  it('computes element-wise sigmoid correctly', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const values = [-10.0, -2.0, 0.0, 2.0, 10.0];
    const expected = values.map((x) => 1.0 / (1.0 + Math.exp(-x)));

    const inp = ctx.tensor('float32', [5]).setData(new Float32Array(values));
    const out = ctx.tensor('float32', [5]);

    ctx.recordShader(shaders.sigmoid, [inp, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, expected);
    ctx.destroy();
  });

  it('computes element-wise sigmoid correctly for float16', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const values = [-10.0, -2.0, 0.0, 2.0, 10.0];
    const expected = values.map((x) => 1.0 / (1.0 + Math.exp(-x)));

    const inp = ctx.tensor('float16', [5]).setData(new Float16Array(values));
    const out = ctx.tensor('float16', [5]);

    ctx.recordShader(shaders.sigmoid, [inp, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, expected, 1e-2);
    ctx.destroy();
  });
});
