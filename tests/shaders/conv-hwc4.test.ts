import { describe, it, expect } from 'vitest';
import { setupGPU } from '../helpers';
import { WgpuExecutionContext, shaders } from '../../src';
import {
  packConvHwc4F32,
  packDwHwc4F32,
  packConvHwc4F16,
  packDwHwc4F16,
} from '../../utils/pack_hwc4';

describe('HWC4 convolution operators', () => {
  it('matches NCHW reference for 1x1 convolution in float32', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const b = 1;
    const ic = 8;
    const oc = 8;
    const h = 4;
    const w = 4;

    // Standard NCHW input and weights
    const inData = new Float32Array(b * ic * h * w);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 17) / 10 - 0.8;

    const wtData = new Float32Array(oc * ic * 1 * 1);
    for (let i = 0; i < wtData.length; i++) wtData[i] = (i % 13) / 10 - 0.6;

    const biasData = new Float32Array(oc);
    for (let i = 0; i < oc; i++) biasData[i] = 0.25;

    // Standard NCHW convolution
    const inNchw = ctx.tensor('float32', [b, ic, h, w]).setData(inData);
    const wtNchw = ctx.tensor('float32', [oc, ic, 1, 1]).setData(wtData);
    const biasNchw = ctx.tensor('float32', [oc]).setData(biasData);
    const outNchw = ctx.tensor('float32', [b, oc, h, w]);

    ctx.recordShader(shaders.convolution, [
      inNchw,
      wtNchw,
      biasNchw,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      outNchw,
    ]);

    // HWC4 execution
    const inHwc4 = ctx.tensor('float32', [b, h, w, ic]);
    ctx.recordShader(shaders.toHwc4, [inNchw, inHwc4]);

    const packedWt = packConvHwc4F32(wtData, oc, ic, 1, 1);
    const wtHwc4 = ctx.tensor('float32', [packedWt.length]).setData(packedWt);

    const outHwc4 = ctx.tensor('float32', [b, h, w, oc]);
    ctx.recordShader(shaders.convolutionHwc4, [
      inHwc4,
      wtHwc4,
      biasNchw,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      outHwc4,
    ]);

    const restoredOut = ctx.tensor('float32', [b, oc, h, w]);
    ctx.recordShader(shaders.toChw, [outHwc4, restoredOut]);

    ctx.submit();
    await ctx.sync();

    const expected = new Float32Array(await outNchw.getData());
    const actual = new Float32Array(await restoredOut.getData());

    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 4);
    }
  });

  it('matches NCHW reference for 3x3 depthwise convolution in float32', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const b = 1;
    const c = 8;
    const h = 6;
    const w = 6;

    const inData = new Float32Array(b * c * h * w);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 23) / 10 - 1.1;

    const wtData = new Float32Array(c * 1 * 3 * 3);
    for (let i = 0; i < wtData.length; i++) wtData[i] = (i % 11) / 10 - 0.5;

    const biasData = new Float32Array(c);
    for (let i = 0; i < c; i++) biasData[i] = 0.1;

    // Standard NCHW DW conv (stride=1, pad=1)
    const inNchw = ctx.tensor('float32', [b, c, h, w]).setData(inData);
    const wtNchw = ctx.tensor('float32', [c, 1, 3, 3]).setData(wtData);
    const biasNchw = ctx.tensor('float32', [c]).setData(biasData);
    const outNchw = ctx.tensor('float32', [b, c, h, w]);

    ctx.recordShader(shaders.convolution, [
      inNchw,
      wtNchw,
      biasNchw,
      [1, 1],
      [1, 1],
      [1, 1],
      false,
      [0, 0],
      c,
      outNchw,
    ]);

    // HWC4 DW execution
    const inHwc4 = ctx.tensor('float32', [b, h, w, c]);
    ctx.recordShader(shaders.toHwc4, [inNchw, inHwc4]);

    const packedWt = packDwHwc4F32(wtData, c, 3, 3);
    const wtHwc4 = ctx.tensor('float32', [packedWt.length]).setData(packedWt);

    const outHwc4 = ctx.tensor('float32', [b, h, w, c]);
    ctx.recordShader(shaders.convolutionHwc4, [
      inHwc4,
      wtHwc4,
      biasNchw,
      [1, 1],
      [1, 1],
      [1, 1],
      false,
      [0, 0],
      c,
      outHwc4,
    ]);

    const restoredOut = ctx.tensor('float32', [b, c, h, w]);
    ctx.recordShader(shaders.toChw, [outHwc4, restoredOut]);

    ctx.submit();
    await ctx.sync();

    const expected = new Float32Array(await outNchw.getData());
    const actual = new Float32Array(await restoredOut.getData());

    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 4);
    }
  });

  it('matches NCHW reference for 3x3 depthwise convolution in float16', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;

    const ctx = new WgpuExecutionContext(device);

    const b = 1;
    const c = 8;
    const h = 6;
    const w = 6;

    const inData = new Float16Array(b * c * h * w);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 23) / 10 - 1.1;

    const wtData = new Float16Array(c * 1 * 3 * 3);
    for (let i = 0; i < wtData.length; i++) wtData[i] = (i % 11) / 10 - 0.5;

    const biasData = new Float16Array(c);
    for (let i = 0; i < c; i++) biasData[i] = 0.1;

    // Standard NCHW DW conv
    const inNchw = ctx.tensor('float16', [b, c, h, w]).setData(inData);
    const wtNchw = ctx.tensor('float16', [c, 1, 3, 3]).setData(wtData);
    const biasNchw = ctx.tensor('float16', [c]).setData(biasData);
    const outNchw = ctx.tensor('float16', [b, c, h, w]);

    ctx.recordShader(shaders.convolution, [
      inNchw,
      wtNchw,
      biasNchw,
      [1, 1],
      [1, 1],
      [1, 1],
      false,
      [0, 0],
      c,
      outNchw,
    ]);

    // HWC4 DW execution
    const inHwc4 = ctx.tensor('float16', [b, h, w, c]);
    ctx.recordShader(shaders.toHwc4, [inNchw, inHwc4]);

    const packedWt = packDwHwc4F16(wtData, c, 3, 3);
    const wtHwc4 = ctx.tensor('float16', [packedWt.length]).setData(packedWt);

    const outHwc4 = ctx.tensor('float16', [b, h, w, c]);
    ctx.recordShader(shaders.convolutionHwc4, [
      inHwc4,
      wtHwc4,
      biasNchw,
      [1, 1],
      [1, 1],
      [1, 1],
      false,
      [0, 0],
      c,
      outHwc4,
    ]);

    const restoredOut = ctx.tensor('float16', [b, c, h, w]);
    ctx.recordShader(shaders.toChw, [outHwc4, restoredOut]);

    ctx.submit();
    await ctx.sync();

    const expected = new Float16Array(await outNchw.getData());
    const actual = new Float16Array(await restoredOut.getData());

    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 1);
    }
  });

  it('matches NCHW reference for 1x1 convolution with clamp in float16', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;

    const ctx = new WgpuExecutionContext(device);

    const b = 1;
    const ic = 16;
    const oc = 16;
    const h = 4;
    const w = 4;

    const inData = new Float16Array(b * ic * h * w);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 17) / 10 - 0.8;

    const wtData = new Float16Array(oc * ic * 1 * 1);
    for (let i = 0; i < wtData.length; i++) wtData[i] = (i % 13) / 10 - 0.6;

    const biasData = new Float16Array(oc);
    for (let i = 0; i < oc; i++) biasData[i] = 0.25;

    // Standard NCHW conv with clamp [0, 1]
    const inNchw = ctx.tensor('float16', [b, ic, h, w]).setData(inData);
    const wtNchw = ctx.tensor('float16', [oc, ic, 1, 1]).setData(wtData);
    const biasNchw = ctx.tensor('float16', [oc]).setData(biasData);
    const outNchw = ctx.tensor('float16', [b, oc, h, w]);

    ctx.recordShader(shaders.convWithClamp, [
      inNchw,
      wtNchw,
      biasNchw,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      0.0,
      1.0,
      outNchw,
    ]);

    // HWC4 execution with clamp [0, 1]
    const inHwc4 = ctx.tensor('float16', [b, h, w, ic]);
    ctx.recordShader(shaders.toHwc4, [inNchw, inHwc4]);

    const packedWt = packConvHwc4F16(wtData, oc, ic, 1, 1);
    const wtHwc4 = ctx.tensor('float16', [packedWt.length]).setData(packedWt);

    const outHwc4 = ctx.tensor('float16', [b, h, w, oc]);
    ctx.recordShader(shaders.convWithClampHwc4, [
      inHwc4,
      wtHwc4,
      biasNchw,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      0.0,
      1.0,
      outHwc4,
    ]);

    const restoredOut = ctx.tensor('float16', [b, oc, h, w]);
    ctx.recordShader(shaders.toChw, [outHwc4, restoredOut]);

    ctx.submit();
    await ctx.sync();

    const expected = new Float16Array(await outNchw.getData());
    const actual = new Float16Array(await restoredOut.getData());

    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i++) {
      expect(actual[i]).toBeCloseTo(expected[i], 1);
    }
  });
});
