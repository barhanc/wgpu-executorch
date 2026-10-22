import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU, expectTensorClose } from '../helpers';

const cases = [
  { ic: 3, oc: 13, h: 131, w: 137, batch: 2, stride: 2, dilation: 1, bias: true },
  { ic: 3, oc: 8, h: 129, w: 129, batch: 1, stride: 2, dilation: 1, bias: false },
  { ic: 3, oc: 16, h: 128, w: 128, batch: 1, stride: 2, dilation: 1, bias: true },
  { ic: 4, oc: 8, h: 65, w: 67, batch: 2, stride: 1, dilation: 1, bias: false },
  { ic: 8, oc: 7, h: 64, w: 66, batch: 1, stride: 1, dilation: 1, bias: true },
  { ic: 16, oc: 4, h: 64, w: 66, batch: 1, stride: 1, dilation: 1, bias: false },
  { ic: 16, oc: 8, h: 65, w: 66, batch: 2, stride: 1, dilation: 1, bias: true },
  { ic: 16, oc: 8, h: 128, w: 130, batch: 1, stride: 2, dilation: 1, bias: true },
  { ic: 4, oc: 8, h: 64, w: 67, batch: 1, stride: 1, dilation: 2, bias: true },
  // Generic FP32 fallback must multiply the batch offset by the input channels.
  { ic: 3, oc: 17, h: 17, w: 19, batch: 2, stride: 2, dilation: 1, bias: false },
  // Neighbors of the specialization boundaries keep using the existing kernels.
  { ic: 3, oc: 17, h: 128, w: 128, batch: 1, stride: 2, dilation: 1, bias: false },
  { ic: 12, oc: 8, h: 63, w: 65, batch: 1, stride: 1, dilation: 1, bias: true },
];

describe('convolution specialization boundaries', () => {
  for (const dtype of ['float32', 'float16'] as const)
    for (const c of cases)
      it(`${dtype} ${JSON.stringify(c)}`, async () => {
        const device = await setupGPU();
        const ctx = new WgpuExecutionContext(device);
        device.pushErrorScope('validation');
        const array = dtype === 'float16' ? Float16Array : Float32Array;
        // Exact dyadic products and bounded sums separate indexing errors from
        // accumulation rounding, including on Node's Float16Array polyfill.
        const fill = (length: number) =>
          Float32Array.from({ length }, (unused, i) => (((i * 17 + (i >> 3)) % 3) - 1) / 8);
        const input = fill(c.batch * c.ic * c.h * c.w);
        const weight = fill(c.oc * c.ic * 9),
          bias = fill(c.oc);
        const oh = Math.floor((c.h - 1) / c.stride) + 1;
        const ow = Math.floor((c.w - 1) / c.stride) + 1;
        const expected = new Float32Array(c.batch * c.oc * oh * ow);
        for (let b = 0; b < c.batch; b++)
          for (let oc = 0; oc < c.oc; oc++)
            for (let y = 0; y < oh; y++)
              for (let x = 0; x < ow; x++) {
                let sum = c.bias ? bias[oc] : 0;
                for (let ic = 0; ic < c.ic; ic++)
                  for (let ky = 0; ky < 3; ky++)
                    for (let kx = 0; kx < 3; kx++) {
                      const iy = y * c.stride - c.dilation + ky * c.dilation;
                      const ix = x * c.stride - c.dilation + kx * c.dilation;
                      if (iy >= 0 && iy < c.h && ix >= 0 && ix < c.w)
                        sum +=
                          input[((b * c.ic + ic) * c.h + iy) * c.w + ix] *
                          weight[((oc * c.ic + ic) * 3 + ky) * 3 + kx];
                    }
                expected[((b * c.oc + oc) * oh + y) * ow + x] = sum;
              }
        try {
          const x = ctx.tensor(dtype, [c.batch, c.ic, c.h, c.w]).setData(new array(input));
          const w = ctx.tensor(dtype, [c.oc, c.ic, 3, 3]).setData(new array(weight));
          const b = c.bias ? ctx.tensor(dtype, [c.oc]).setData(new array(bias)) : undefined;
          const out = ctx
            .tensor(dtype, [c.batch, c.oc, oh, ow])
            .setData(new array(expected.length).fill(NaN));
          ctx.recordShader(shaders.convolution, [
            x,
            w,
            b,
            [c.stride, c.stride],
            [c.dilation, c.dilation],
            [c.dilation, c.dilation],
            false,
            [0, 0],
            1,
            out,
          ]);
          ctx.submit();
          await ctx.sync();
          await expectTensorClose(out, Array.from(expected), 1e-5);
          const error = await device.popErrorScope();
          if (error) throw new Error(error.message);
        } finally {
          ctx.destroy();
        }
      });
});
