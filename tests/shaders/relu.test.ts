import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.relu.default', () => {
  it('computes element-wise relu correctly', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const values = [-2.0, -0.5, 0.0, 0.5, 2.0];
    const expected = values.map((x) => Math.max(x, 0));

    const inp = ctx.tensor('float32', [5]).setData(new Float32Array(values));
    const out = ctx.tensor('float32', [5]);

    ctx.recordShader(shaders.relu, [inp, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, expected);
    ctx.destroy();
  });

  it('computes element-wise relu correctly for float16', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const values = [-2.0, -0.5, 0.0, 0.5, 2.0];
    const expected = values.map((x) => Math.max(x, 0));

    const inp = ctx.tensor('float16', [5]).setData(new Float16Array(values));
    const out = ctx.tensor('float16', [5]);

    ctx.recordShader(shaders.relu, [inp, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, expected, 1e-2);
    ctx.destroy();
  });
});
