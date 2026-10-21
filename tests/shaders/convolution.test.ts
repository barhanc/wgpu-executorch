import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

describe('aten.convolution.default', () => {
  it('computes 2D dense convolution with bias', async () => {
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
    const out = ctx.tensor('float32', [1, 1, 2, 2]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      bias,
      [1, 1], // stride
      [0, 0], // padding
      [1, 1], // dilation
      false, // transposed
      [0, 0], // output_padding
      1, // groups
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // patch (0,0): 1*1 + 5*1 + 10 = 16
    // patch (0,1): 2*1 + 6*1 + 10 = 18
    // patch (1,0): 4*1 + 8*1 + 10 = 22
    // patch (1,1): 5*1 + 9*1 + 10 = 24
    await expectTensorClose(out, [16, 18, 22, 24]);
    ctx.destroy();
  });

  it('computes depthwise grouped 2D convolution', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 2, 2, 2]
    const inp = ctx.tensor('float32', [1, 2, 2, 2]).setData(
      new Float32Array([
        // channel 0
        1, 2, 3, 4,
        // channel 1
        10, 20, 30, 40,
      ])
    );

    // weight: [2, 1, 2, 2] with groups = 2 (depthwise)
    const weight = ctx.tensor('float32', [2, 1, 2, 2]).setData(
      new Float32Array([
        // filter 0 for channel 0: sum
        1, 1, 1, 1,
        // filter 1 for channel 1: weighted
        1, 0, 0, 1,
      ])
    );

    const out = ctx.tensor('float32', [1, 2, 1, 1]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined, // no bias
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      2, // groups
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // ch0: 1 + 2 + 3 + 4 = 10
    // ch1: 10*1 + 40*1 = 50
    await expectTensorClose(out, [10, 50]);
    ctx.destroy();
  });

  it('computes transposed 2D convolution', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 1, 2, 2]
    const inp = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 2, 3, 4]));

    // transposed weight: [IC=1, OC/groups=1, KH=2, KW=2]
    const weight = ctx.tensor('float32', [1, 1, 2, 2]).setData(new Float32Array([1, 1, 1, 1]));

    // stride 2 -> upsample to [1, 1, 3, 3] with stride=2, pad=0, kernel=2, but formula:
    // OH = (2 - 1)*2 - 0 + 1*(2-1) + 0 + 1 = 2 + 1 + 1 = 4. With OH=4, OW=4:
    const out = ctx.tensor('float32', [1, 1, 4, 4]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined,
      [2, 2], // stride 2
      [0, 0], // pad 0
      [1, 1], // dilation 1
      true, // transposed
      [0, 0], // output_padding
      1, // groups
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // Scatter 2x2 into 4x4 with 2x2 box filter:
    // [1, 1, 2, 2]
    // [1, 1, 2, 2]
    // [3, 3, 4, 4]
    // [3, 3, 4, 4]
    await expectTensorClose(out, [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    ctx.destroy();
  });

  it('computes non-trivial multi-channel conv2d with padding, stride, and bias', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 2, 5, 5] = 50 elements (1..50)
    const inpData = Float32Array.from({ length: 50 }, (unused, i) => i + 1);
    const inp = ctx.tensor('float32', [1, 2, 5, 5]).setData(inpData);

    // weight: [3, 2, 3, 3] = 54 elements
    const weightData = Float32Array.from({ length: 54 }, (unused, i) => (i + 1) * 0.1);
    const weight = ctx.tensor('float32', [3, 2, 3, 3]).setData(weightData);

    // bias: [3]
    const bias = ctx.tensor('float32', [3]).setData(new Float32Array([2.5, -1.0, 0.5]));

    // stride: [2, 2], padding: [1, 1], dilation: [1, 1]
    // OH = (5 + 2*1 - 1*(3-1) - 1)/2 + 1 = 3, OW = 3 -> out: [1, 3, 3, 3]
    const out = ctx.tensor('float32', [1, 3, 3, 3]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      bias,
      [2, 2],
      [1, 1],
      [1, 1],
      false,
      [0, 0],
      1,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(
      out,
      [
        202.5, 312.9, 214.5, 370.3, 559.0, 373.9, 264.9, 392.1, 257.7, 436.6, 698.2, 491.8, 885.2,
        1381.7, 953.6, 715.0, 1101.4, 751.0, 675.7, 1088.5, 774.1, 1405.1, 2209.4, 1538.3, 1170.1,
        1815.7, 1249.3,
      ],
      1e-3
    );
    ctx.destroy();
  });

  it('computes non-trivial multi-channel conv_transpose2d with stride, padding, and output_padding', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // input: [1, 2, 3, 3] = 18 elements (1..18)
    const inpData = Float32Array.from({ length: 18 }, (unused, i) => i + 1);
    const inp = ctx.tensor('float32', [1, 2, 3, 3]).setData(inpData);

    // transposed weight: [IC=2, OC=2, KH=3, KW=3] = 36 elements
    const weightData = Float32Array.from({ length: 36 }, (unused, i) => (i + 1) * 0.05);
    const weight = ctx.tensor('float32', [2, 2, 3, 3]).setData(weightData);

    // bias: [2]
    const bias = ctx.tensor('float32', [2]).setData(new Float32Array([1.5, -2.0]));

    // stride: [2, 2], padding: [1, 1], output_padding: [1, 1], dilation: [1, 1]
    // OH = (3 - 1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 4 - 2 + 2 + 1 + 1 = 6, OW = 6 -> out: [1, 2, 6, 6]
    const out = ctx.tensor('float32', [1, 2, 6, 6]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      bias,
      [2, 2],
      [1, 1],
      [1, 1],
      true,
      [1, 1],
      1,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(
      out,
      [
        13.25, 26.3, 14.65, 29.1, 16.05, 16.8, 28.3, 57.7, 31.1, 63.3, 33.9, 35.7, 17.45, 34.7,
        18.85, 37.5, 20.25, 21.3, 36.7, 74.5, 39.5, 80.1, 42.3, 44.7, 21.65, 43.1, 23.05, 45.9,
        24.45, 25.8, 25.1, 50.3, 26.8, 53.7, 28.5, 29.85,

        14.7, 33.6, 17.0, 38.2, 19.3, 20.05, 37.4, 81.2, 42.0, 90.4, 46.6, 48.4, 21.6, 47.4, 23.9,
        52.0, 26.2, 27.25, 51.2, 108.8, 55.8, 118.0, 60.4, 62.8, 28.5, 61.2, 30.8, 65.8, 33.1,
        34.45, 31.95, 68.4, 34.55, 73.6, 37.15, 38.5,
      ]
    );
    ctx.destroy();
  });

  it('computes float16 2D dense convolution with bias (f16 GEMM)', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx
      .tensor('float16', [1, 1, 3, 3])
      .setData(new Float16Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const weight = ctx.tensor('float16', [1, 1, 2, 2]).setData(new Float16Array([1, 0, 0, 1]));
    const bias = ctx.tensor('float16', [1]).setData(new Float16Array([10]));
    const out = ctx.tensor('float16', [1, 1, 2, 2]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      bias,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [16, 18, 22, 24], 1e-2);
    ctx.destroy();
  });

  it('computes float16 64x64 register-tiled dense convolution', async () => {
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

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // Verify first few elements manually
    const actual = new Float16Array(await out.getData());
    // For oc = 0, spatial (0, 0): sum over ic of in[0, ic, 0, 0] * w[0, ic, 0, 0]
    let expected00 = 0;
    for (let ic = 0; ic < IC; ic++) {
      const inVal = inData[ic * (IH * IW)];
      const wVal = wData[ic];
      expected00 += inVal * wVal;
    }
    expect(actual[0]).toBeCloseTo(expected00, 1);
    ctx.destroy();
  });

  it('computes float16 depthwise convolution', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx
      .tensor('float16', [1, 2, 2, 2])
      .setData(new Float16Array([1, 2, 3, 4, 10, 20, 30, 40]));
    const weight = ctx.tensor('float16', [2, 1, 1, 1]).setData(new Float16Array([2, 0.5]));
    const out = ctx.tensor('float16', [1, 2, 2, 2]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      2,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out, [2, 4, 6, 8, 5, 10, 15, 20], 1e-2);
    ctx.destroy();
  });

  it('computes 1x1 convolution with Vec4 path', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const [B, IC, IH, IW, OC] = [1, 4, 2, 2, 4];
    const inData = new Float32Array(B * IC * IH * IW);
    for (let i = 0; i < inData.length; i++) inData[i] = i + 1;

    // Identity-like weights for 1x1
    const wData = new Float32Array(OC * IC * 1 * 1);
    for (let o = 0; o < OC; o++) {
      for (let c = 0; c < IC; c++) {
        wData[o * IC + c] = o === c ? 2 : 0;
      }
    }

    const inp = ctx.tensor('float32', [B, IC, IH, IW]).setData(inData);
    const weight = ctx.tensor('float32', [OC, IC, 1, 1]).setData(wData);
    const out = ctx.tensor('float32', [B, OC, IH, IW]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float32Array(await out.getData());
    for (let o = 0; o < OC; o++) {
      for (let s = 0; s < IH * IW; s++) {
        expect(actual[o * (IH * IW) + s]).toBeCloseTo(inData[o * (IH * IW) + s] * 2, 4);
      }
    }
    ctx.destroy();
  });

  it('computes 1x1 convolution with 64x64 register-tiled path', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const [B, IC, IH, IW, OC] = [1, 16, 8, 8, 64];
    const inData = new Float32Array(B * IC * IH * IW);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 13) * 0.1;

    const wData = new Float32Array(OC * IC * 1 * 1);
    for (let i = 0; i < wData.length; i++) wData[i] = (i % 7) * 0.1;

    const inp = ctx.tensor('float32', [B, IC, IH, IW]).setData(inData);
    const weight = ctx.tensor('float32', [OC, IC, 1, 1]).setData(wData);
    const out = ctx.tensor('float32', [B, OC, IH, IW]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      1,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // Verify element (0, 0, 0, 0): dot product of out_ch 0 with in_pos 0 across all IC
    let expected0 = 0;
    for (let c = 0; c < IC; c++) {
      expected0 += inData[c * (IH * IW)] * wData[0 * IC + c];
    }
    const actual = new Float32Array(await out.getData());
    expect(actual[0]).toBeCloseTo(expected0, 3);
    ctx.destroy();
  });

  it('computes general grouped convolution (groups=2 fallback)', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // groups = 2, IC = 4 (2 per group), OC = 4 (2 per group)
    const inp = ctx.tensor('float32', [1, 4, 2, 2]).setData(new Float32Array(16).fill(1));
    const weight = ctx.tensor('float32', [4, 2, 1, 1]).setData(new Float32Array(8).fill(2));
    const out = ctx.tensor('float32', [1, 4, 2, 2]);

    ctx.recordShader(shaders.convolution, [
      inp,
      weight,
      undefined,
      [1, 1],
      [0, 0],
      [1, 1],
      false,
      [0, 0],
      2,
      out,
    ]);
    ctx.submit();
    await ctx.sync();

    // Each out channel = sum over 2 input channels of 1 * 2 = 4
    await expectTensorClose(out, new Array(16).fill(4), 1e-3);
    ctx.destroy();
  });
});
