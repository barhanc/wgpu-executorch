import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('et_vk.conv_with_clamp.default', () => {
  it('computes 2D dense convolution with clamp and bias', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 1, 3, 3]
    const inp = ctx
      .tensor('float32', [1, 1, 3, 3])
      .setData(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));

    // weight: [1, 1, 2, 2]
    const weight = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 0, 0, 1]));

    // bias: [1]
    const bias = ctx.tensor('float32', [1]).setData(new Float32Array([10]));

    // out: [1, 1, 2, 2] with stride 1, pad 0, dilation 1
    // Unclamped conv results: [16, 18, 22, 24]
    // Clamped with min: 17.0, max: 23.0 -> [17, 18, 22, 23]
    const out = ctx.tensor('float32', [1, 1, 2, 2]);

    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      bias,
      [1, 1], // stride
      [0, 0], // padding
      [1, 1], // dilation
      false, // transposed
      [0, 0], // output_padding
      1, // groups
      17.0, // min
      23.0, // max
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [17, 18, 22, 23]);
    ctx.destroy();
  });

  it('computes depthwise grouped convolution with clamp (ReLU6 style)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 2, 2, 2]
    const inp = ctx.tensor('float32', [1, 2, 2, 2]).setData(
      new Float32Array([
        // channel 0
        -5, -2, 1, 2,
        // channel 1
        2, 4, 1, 3,
      ])
    );

    // weight: [2, 1, 2, 2] with groups = 2 (depthwise)
    const weight = ctx.tensor('float32', [2, 1, 2, 2]).setData(
      new Float32Array([
        // filter 0 for channel 0: sum -> -5 + -2 + 1 + 2 = -4
        1, 1, 1, 1,
        // filter 1 for channel 1: sum -> 2 + 4 + 1 + 3 = 10
        1, 1, 1, 1,
      ])
    );

    const out = ctx.tensor('float32', [1, 2, 1, 1]);

    // clamp [0.0, 6.0]
    // ch0: clamp(-4, 0, 6) = 0
    // ch1: clamp(10, 0, 6) = 6
    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      undefined, // no bias
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      2, // groups
      0.0,
      6.0,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [0, 6]);
    ctx.destroy();
  });

  it('computes float16 2D dense convolution with clamp and bias', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx
      .tensor('float16', [1, 1, 3, 3])
      .setData(new Float16Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const weight = ctx.tensor('float16', [1, 1, 2, 2]).setData(new Float16Array([1, 0, 0, 1]));
    const bias = ctx.tensor('float16', [1]).setData(new Float16Array([10]));
    const out = ctx.tensor('float16', [1, 1, 2, 2]);

    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      bias,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      17.0,
      23.0,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [17, 18, 22, 23], 1e-2);
    ctx.destroy();
  });

  it('computes float16 64x64 register-tiled dense convolution with clamp', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const B = 1;
    const IC = 64;
    const IH = 8;
    const IW = 8;
    const OC = 64;
    const KH = 1;
    const KW = 1;

    const inData = new Float16Array(B * IC * IH * IW);
    for (let i = 0; i < inData.length; i++) inData[i] = ((i % 5) - 2) * 0.5;
    const wData = new Float16Array(OC * IC * KH * KW);
    for (let i = 0; i < wData.length; i++) wData[i] = ((i % 7) - 3) * 0.25;

    const inp = ctx.tensor('float16', [B, IC, IH, IW]).setData(inData);
    const weight = ctx.tensor('float16', [OC, IC, KH, KW]).setData(wData);
    const out = ctx.tensor('float16', [B, OC, IH, IW]);

    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      -1.0,
      1.0,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float16Array(await out.getData());
    for (let i = 0; i < actual.length; i++) {
      expect(actual[i]).toBeGreaterThanOrEqual(-1.01);
      expect(actual[i]).toBeLessThanOrEqual(1.01);
    }
    ctx.destroy();
  });

  it('computes float16 depthwise grouped convolution with clamp', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx
      .tensor('float16', [1, 2, 2, 2])
      .setData(new Float16Array([-5, -2, 1, 2, 2, 4, 1, 3]));
    const weight = ctx
      .tensor('float16', [2, 1, 2, 2])
      .setData(new Float16Array([1, 1, 1, 1, 1, 1, 1, 1]));
    const out = ctx.tensor('float16', [1, 2, 1, 1]);

    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      2,
      0.0,
      6.0,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [0, 6], 1e-2);
    ctx.destroy();
  });

  it('computes 1x1 conv-with-clamp with 64x64 register-tiled path', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const [B, IC, IH, IW, OC] = [1, 16, 8, 8, 64];
    const inData = new Float32Array(B * IC * IH * IW);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 13) * 0.5 - 2.0;

    const wData = new Float32Array(OC * IC * 1 * 1);
    for (let i = 0; i < wData.length; i++) wData[i] = (i % 7) * 0.1;

    const inp = ctx.tensor('float32', [B, IC, IH, IW]).setData(inData);
    const weight = ctx.tensor('float32', [OC, IC, 1, 1]).setData(wData);
    const out = ctx.tensor('float32', [B, OC, IH, IW]);

    ctx.recordShader(shaders.convWithClamp, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      0.0,
      5.0,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // Verify all outputs are clamped to [0.0, 5.0]
    const actual = new Float32Array(await out.getData());
    for (let i = 0; i < actual.length; i++) {
      expect(actual[i]).toBeGreaterThanOrEqual(0.0);
      expect(actual[i]).toBeLessThanOrEqual(5.0001);
    }
    ctx.destroy();
  });
});
