import { describe, it, expect } from 'vitest';
import { setupGPU } from '../helpers';
import { WgpuExecutionContext, shaders } from '../../src';

describe('to_hwc4 and to_chw layout conversions', () => {
  it('correctly packs float32 [1, 3, 2, 2] to HWC4 and restores back to NCHW', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // [1, 3, 2, 2] -> 12 values
    const b = 1;
    const c = 3;
    const h = 2;
    const w = 2;
    const c4 = Math.ceil(c / 4); // 1

    const inputData = new Float32Array([
      // c=0
      1.0, 2.0, 3.0, 4.0,
      // c=1
      5.0, 6.0, 7.0, 8.0,
      // c=2
      9.0, 10.0, 11.0, 12.0,
    ]);

    const inTensor = ctx.tensor('float32', [b, c, h, w]).setData(inputData);
    // HWC4 has shape [b, h, w, c4 * 4]
    const hwc4Tensor = ctx.tensor('float32', [b, h, w, c4 * 4]);
    const restoredTensor = ctx.tensor('float32', [b, c, h, w]);

    ctx.recordShader(shaders.toHwc4, [inTensor, hwc4Tensor]);
    ctx.recordShader(shaders.toChw, [hwc4Tensor, restoredTensor]);

    ctx.submit();
    await ctx.sync();

    // Check intermediate HWC4 buffer: at (h=0, w=0), channels are [c0=1, c1=5, c2=9, c3=0]
    const hwc4Data = new Float32Array(await hwc4Tensor.getData());
    expect(hwc4Data[0]).toBeCloseTo(1.0);
    expect(hwc4Data[1]).toBeCloseTo(5.0);
    expect(hwc4Data[2]).toBeCloseTo(9.0);
    expect(hwc4Data[3]).toBeCloseTo(0.0); // padded zero

    // Check restored data matches original input
    const restoredData = new Float32Array(await restoredTensor.getData());
    expect(restoredData.length).toBe(inputData.length);
    for (let i = 0; i < inputData.length; i++) {
      expect(restoredData[i]).toBeCloseTo(inputData[i]);
    }
  });

  it('correctly converts float16 [1, 8, 4, 4] with native shader-f16', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;

    const ctx = new WgpuExecutionContext(device);
    const b = 1;
    const c = 8;
    const h = 4;
    const w = 4;
    const c4 = Math.ceil(c / 4); // 2

    const numel = b * c * h * w;
    const inputData = new Float16Array(numel);
    for (let i = 0; i < numel; i++) {
      inputData[i] = ((i * 0.5) % 15) - 7.0;
    }

    const inTensor = ctx.tensor('float16', [b, c, h, w]).setData(inputData);
    const hwc4Tensor = ctx.tensor('float16', [b, h, w, c4 * 4]);
    const restoredTensor = ctx.tensor('float16', [b, c, h, w]);

    ctx.recordShader(shaders.toHwc4, [inTensor, hwc4Tensor]);
    ctx.recordShader(shaders.toChw, [hwc4Tensor, restoredTensor]);

    ctx.submit();
    await ctx.sync();

    const restoredData = new Float16Array(await restoredTensor.getData());
    for (let i = 0; i < numel; i++) {
      expect(restoredData[i]).toBeCloseTo(inputData[i], 2);
    }
  });
});
