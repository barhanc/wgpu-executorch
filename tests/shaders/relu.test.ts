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
});
