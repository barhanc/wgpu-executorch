import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext } from '../../src';
import { recordHfMatrix } from '../../utils/benchmark/hf-matrix';
import { recordHfConvolution } from '../../utils/benchmark/hf-convolution';
import { setupGPU, expectTensorClose } from '../helpers';

const cases = [
  { batch: 2, ic: 5, oc: 13, h: 3, w: 23, kernel: 1, stride: 1, pad: 0, dilation: 1, bias: true },
  { batch: 1, ic: 32, oc: 68, h: 4, w: 17, kernel: 1, stride: 1, pad: 0, dilation: 1, bias: false },
  { batch: 1, ic: 64, oc: 80, h: 3, w: 23, kernel: 1, stride: 1, pad: 0, dilation: 1, bias: true },
  { batch: 2, ic: 3, oc: 13, h: 9, w: 11, kernel: 3, stride: 2, pad: 1, dilation: 1, bias: true },
  { batch: 2, ic: 4, oc: 68, h: 11, w: 17, kernel: 3, stride: 1, pad: 2, dilation: 2, bias: false },
  { batch: 1, ic: 32, oc: 16, h: 5, w: 19, kernel: 3, stride: 1, pad: 0, dilation: 1, bias: true },
  { batch: 1, ic: 4, oc: 16, h: 1, w: 69, kernel: 3, stride: 1, pad: 1, dilation: 1, bias: false },
];

describe('pinned Hugging Face convolution adapters', () => {
  it('falls back when experimental matrix features are not enabled', async () => {
    const ctx = new WgpuExecutionContext(await setupGPU());
    try {
      const input = ctx.tensor('float32', [1, 32, 8, 8]);
      const weight = ctx.tensor('float32', [32, 32, 1, 1]);
      const output = ctx.tensor('float32', [1, 32, 8, 8]);
      expect(
        recordHfMatrix(ctx, [
          input,
          weight,
          undefined,
          [1, 1],
          [0, 0],
          [1, 1],
          false,
          [0, 0],
          1,
          output,
        ])
      ).toBe(false);
    } finally {
      ctx.destroy();
    }
  });
  for (const dtype of ['float32', 'float16'] as const)
    for (const rows of [64, 32] as const)
      for (const c of cases.filter((c) => rows === 64 || c.kernel === 3))
        it(`${dtype} m${rows} ${JSON.stringify(c)}`, async () => {
          const device = await setupGPU();
          device.pushErrorScope('validation');
          const ctx = new WgpuExecutionContext(device);
          const array = dtype === 'float16' ? Float16Array : Float32Array;
          const fill = (length: number) =>
            new array(Array.from({ length }, (unused, i) => (((i * 7) % 13) - 6) / 16));
          const oh = Math.floor((c.h + 2 * c.pad - c.dilation * (c.kernel - 1) - 1) / c.stride) + 1;
          const ow = Math.floor((c.w + 2 * c.pad - c.dilation * (c.kernel - 1) - 1) / c.stride) + 1;
          const input = fill(c.batch * c.ic * c.h * c.w);
          const weight = fill(c.oc * c.ic * c.kernel * c.kernel);
          const bias = fill(c.oc);
          const expected: number[] = [];
          for (let b = 0; b < c.batch; b++)
            for (let oc = 0; oc < c.oc; oc++)
              for (let y = 0; y < oh; y++)
                for (let x = 0; x < ow; x++) {
                  let sum = c.bias ? bias[oc] : 0;
                  for (let ic = 0; ic < c.ic; ic++)
                    for (let ky = 0; ky < c.kernel; ky++)
                      for (let kx = 0; kx < c.kernel; kx++) {
                        const iy = y * c.stride - c.pad + ky * c.dilation;
                        const ix = x * c.stride - c.pad + kx * c.dilation;
                        if (iy >= 0 && iy < c.h && ix >= 0 && ix < c.w)
                          sum +=
                            input[((b * c.ic + ic) * c.h + iy) * c.w + ix] *
                            weight[((oc * c.ic + ic) * c.kernel + ky) * c.kernel + kx];
                      }
                  expected.push(sum);
                }
          try {
            const x = ctx.tensor(dtype, [c.batch, c.ic, c.h, c.w]).setData(input);
            const w = ctx.tensor(dtype, [c.oc, c.ic, c.kernel, c.kernel]).setData(weight);
            const b = c.bias ? ctx.tensor(dtype, [c.oc]).setData(bias) : undefined;
            const out = ctx
              .tensor(dtype, [c.batch, c.oc, oh, ow])
              .setData(new array(c.batch * c.oc * oh * ow).fill(NaN));
            expect(
              recordHfConvolution(
                ctx,
                [
                  x,
                  w,
                  b,
                  [c.stride, c.stride],
                  [c.pad, c.pad],
                  [c.dilation, c.dilation],
                  false,
                  [0, 0],
                  1,
                  out,
                ],
                'both',
                rows
              )
            ).toBe(true);
            ctx.submit();
            await ctx.sync();
            await expectTensorClose(out, Array.from(new array(expected)), 1e-5);
            const error = await device.popErrorScope();
            if (error) throw new Error(error.message);
          } finally {
            ctx.destroy();
          }
        });
});
