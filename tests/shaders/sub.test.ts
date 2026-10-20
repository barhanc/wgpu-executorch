import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.sub.Tensor', () => {
  it('computes elementwise subtraction (identical shapes)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float32', [2, 3]).setData(new Float32Array([5, 7, 9, 4, 3, 1]));
    const b = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    const out = ctx.tensor('float32', [2, 3]);

    ctx.recordShader(shaders.sub, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [4, 5, 6, 0, -2, -5]);
    ctx.destroy();
  });

  it('computes alpha-scaled subtraction with broadcast', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float32', [1, 3]).setData(new Float32Array([10, 20, 30]));
    const b = ctx.tensor('float32', [2, 3]).setData(new Float32Array([1, 1, 1, 2, 2, 2]));
    const out = ctx.tensor('float32', [2, 3]);

    ctx.recordShader(shaders.sub, [a, b, 2.0, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [8, 18, 28, 6, 16, 26]);
    ctx.destroy();
  });

  it('computes float16 elementwise subtraction', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float16', [2, 3]).setData(new Float16Array([5, 7, 9, 4, 3, 1]));
    const b = ctx.tensor('float16', [2, 3]).setData(new Float16Array([1, 2, 3, 4, 5, 6]));
    const out = ctx.tensor('float16', [2, 3]);

    ctx.recordShader(shaders.sub, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [4, 5, 6, 0, -2, -5], 1e-2);
    ctx.destroy();
  });

  it('computes float16 alpha-scaled subtraction with broadcast', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float16', [1, 3]).setData(new Float16Array([10, 20, 30]));
    const b = ctx.tensor('float16', [2, 3]).setData(new Float16Array([1, 1, 1, 2, 2, 2]));
    const out = ctx.tensor('float16', [2, 3]);

    ctx.recordShader(shaders.sub, [a, b, 2.0, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [8, 18, 28, 6, 16, 26], 1e-2);
    ctx.destroy();
  });
});
