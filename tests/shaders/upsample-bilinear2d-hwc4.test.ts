import { describe, it, expect } from 'vitest';
import { setupGPU } from '../helpers';
import { WgpuExecutionContext, shaders } from '../../src';

describe('upsampleBilinear2dHwc4 operator', () => {
  it('matches NCHW reference for bilinear upsampling in float32', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const b = 1;
    const c = 8;
    const ih = 4;
    const iw = 4;
    const oh = 8;
    const ow = 8;

    const inData = new Float32Array(b * c * ih * iw);
    for (let i = 0; i < inData.length; i++) inData[i] = (i % 19) / 10 - 0.9;

    // Standard NCHW
    const inNchw = ctx.tensor('float32', [b, c, ih, iw]).setData(inData);
    const outNchw = ctx.tensor('float32', [b, c, oh, ow]);
    ctx.recordShader(shaders.upsampleBilinear2d, [inNchw, [oh, ow], false, undefined, outNchw]);

    // HWC4
    const inHwc4 = ctx.tensor('float32', [b, ih, iw, c]);
    ctx.recordShader(shaders.toHwc4, [inNchw, inHwc4]);

    const outHwc4 = ctx.tensor('float32', [b, oh, ow, c]);
    ctx.recordShader(shaders.upsampleBilinear2dHwc4, [inHwc4, [oh, ow], false, undefined, outHwc4]);

    const restoredOut = ctx.tensor('float32', [b, c, oh, ow]);
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
});
