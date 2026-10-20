import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.mul.Tensor', () => {
  it('computes elementwise multiplication (identical shapes)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float32', [2, 2]).setData(new Float32Array([1, 2, 3, 4]));
    const b = ctx.tensor('float32', [2, 2]).setData(new Float32Array([5, 6, 7, 8]));
    const out = ctx.tensor('float32', [2, 2]);

    ctx.recordShader(shaders.mul, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [5, 12, 21, 32]);
    ctx.destroy();
  });

  it('computes broadcasted multiplication across channels', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // [1, 2, 2, 2] * [1, 2, 1, 1] -> [1, 2, 2, 2]
    // ch 0 scaled by 2, ch 1 scaled by 0.5
    const a = ctx.tensor('float32', [1, 2, 2, 2]).setData(
      new Float32Array([
        // channel 0
        1, 2, 3, 4,
        // channel 1
        10, 20, 30, 40,
      ])
    );
    const b = ctx.tensor('float32', [1, 2, 1, 1]).setData(new Float32Array([2.0, 0.5]));
    const out = ctx.tensor('float32', [1, 2, 2, 2]);

    ctx.recordShader(shaders.mul, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [2, 4, 6, 8, 5, 10, 15, 20]);
    ctx.destroy();
  });

  it('computes float16 elementwise multiplication', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float16', [2, 2]).setData(new Float16Array([1, 2, 3, 4]));
    const b = ctx.tensor('float16', [2, 2]).setData(new Float16Array([5, 6, 7, 8]));
    const out = ctx.tensor('float16', [2, 2]);

    ctx.recordShader(shaders.mul, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [5, 12, 21, 32], 1e-2);
    ctx.destroy();
  });

  it('computes float16 broadcasted multiplication', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const a = ctx
      .tensor('float16', [1, 2, 2, 2])
      .setData(new Float16Array([1, 2, 3, 4, 10, 20, 30, 40]));
    const b = ctx.tensor('float16', [1, 2, 1, 1]).setData(new Float16Array([2.0, 0.5]));
    const out = ctx.tensor('float16', [1, 2, 2, 2]);

    ctx.recordShader(shaders.mul, [a, b, out]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [2, 4, 6, 8, 5, 10, 15, 20], 1e-2);
    ctx.destroy();
  });
});
