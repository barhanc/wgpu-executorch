import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { expectTensorClose, setupGPU } from '../helpers';

// Exercise all tile heights, NCHW batches, spatial/channel tails, rectangular
// kernels, stride and dilation. Small dyadic values make a precise CPU oracle.
const cases = [
  {
    batch: 2,
    ic: 4,
    oc: 12,
    h: 9,
    w: 11,
    kh: 2,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: true,
  },
  {
    batch: 2,
    ic: 4,
    oc: 48,
    h: 11,
    w: 13,
    kh: 3,
    kw: 2,
    stride: [2, 1],
    pad: [1, 0],
    dilation: [2, 2],
    bias: false,
  },
  {
    batch: 1,
    ic: 8,
    oc: 16,
    h: 9,
    w: 9,
    kh: 1,
    kw: 3,
    stride: [1, 1],
    pad: [0, 1],
    dilation: [1, 1],
    bias: true,
  },
  // Large enough spatial grids select the narrow tile, including tails,
  // multiple NCHW batches, stride/dilation and an all-border input.
  {
    batch: 2,
    ic: 4,
    oc: 13,
    h: 67,
    w: 131,
    kh: 3,
    kw: 3,
    stride: [2, 2],
    pad: [2, 2],
    dilation: [2, 2],
    bias: true,
  },
  {
    batch: 1,
    ic: 4,
    oc: 8,
    h: 65,
    w: 65,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: false,
  },
  {
    batch: 1,
    ic: 4,
    oc: 16,
    h: 1,
    w: 4097,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: false,
  },
  // Small-grid fallbacks: channel tails, dilation, stride and batch crossings.
  {
    batch: 2,
    ic: 4,
    oc: 13,
    h: 11,
    w: 17,
    kh: 3,
    kw: 3,
    stride: [2, 1],
    pad: [2, 2],
    dilation: [2, 2],
    bias: true,
  },
  {
    batch: 1,
    ic: 8,
    oc: 8,
    h: 9,
    w: 11,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: false,
  },
  // Just above the narrow range must keep the 32-row fallback.
  {
    batch: 1,
    ic: 4,
    oc: 17,
    h: 9,
    w: 11,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: true,
  },
  // 3x3: interior and border gathers, row/batch crossings, dilation/stride,
  // channel/spatial tails, and an image with no full-height interior footprint.
  {
    batch: 2,
    ic: 4,
    oc: 12,
    h: 9,
    w: 11,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: true,
  },
  {
    batch: 2,
    ic: 4,
    oc: 48,
    h: 11,
    w: 13,
    kh: 3,
    kw: 3,
    stride: [2, 1],
    pad: [1, 1],
    dilation: [1, 2],
    bias: false,
  },
  {
    batch: 1,
    ic: 4,
    oc: 16,
    h: 10,
    w: 11,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [0, 0],
    dilation: [1, 1],
    bias: true,
  },
  {
    batch: 1,
    ic: 4,
    oc: 48,
    h: 5,
    w: 16,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [2, 2],
    dilation: [1, 1],
    bias: true,
  },
  {
    batch: 1,
    ic: 4,
    oc: 16,
    h: 1,
    w: 64,
    kh: 3,
    kw: 3,
    stride: [1, 1],
    pad: [1, 1],
    dilation: [1, 1],
    bias: false,
  },
];

describe('dense convolution geometry and output indexing', () => {
  for (const dtype of ['float32', 'float16'] as const) {
    for (const fused of [false, true]) {
      for (const c of cases) {
        it(`${dtype}, clamp=${fused}, ${c.batch}x${c.ic}x${c.h}x${c.w}, kernel ${c.oc}x${c.kh}x${c.kw}`, async () => {
          const device = await setupGPU();
          device.pushErrorScope('validation');
          const ctx = new WgpuExecutionContext(device);
          const array = dtype === 'float16' ? Float16Array : Float32Array;
          const oh =
            Math.floor((c.h + 2 * c.pad[0] - c.dilation[0] * (c.kh - 1) - 1) / c.stride[0]) + 1;
          const ow =
            Math.floor((c.w + 2 * c.pad[1] - c.dilation[1] * (c.kw - 1) - 1) / c.stride[1]) + 1;
          const input = new array(
            Array.from(
              { length: c.batch * c.ic * c.h * c.w },
              (value, i) => (((i * 7) % 17) - 8) / 8
            )
          );
          const weight = new array(
            Array.from(
              { length: c.oc * c.ic * c.kh * c.kw },
              (value, i) => (((i * 3) % 11) - 5) / 8
            )
          );
          const bias = new array(Array.from({ length: c.oc }, (value, i) => ((i % 5) - 2) / 4));
          const expected: number[] = [];
          for (let b = 0; b < c.batch; b++)
            for (let oc = 0; oc < c.oc; oc++) {
              for (let y = 0; y < oh; y++)
                for (let x = 0; x < ow; x++) {
                  let sum = c.bias ? bias[oc] : 0;
                  for (let ic = 0; ic < c.ic; ic++)
                    for (let ky = 0; ky < c.kh; ky++)
                      for (let kx = 0; kx < c.kw; kx++) {
                        const iy = y * c.stride[0] - c.pad[0] + ky * c.dilation[0];
                        const ix = x * c.stride[1] - c.pad[1] + kx * c.dilation[1];
                        if (iy >= 0 && iy < c.h && ix >= 0 && ix < c.w) {
                          sum +=
                            input[((b * c.ic + ic) * c.h + iy) * c.w + ix] *
                            weight[((oc * c.ic + ic) * c.kh + ky) * c.kw + kx];
                        }
                      }
                  expected.push(fused ? Math.min(1.5, Math.max(-0.5, sum)) : sum);
                }
            }
          try {
            const x = ctx.tensor(dtype, [c.batch, c.ic, c.h, c.w]).setData(input);
            const w = ctx.tensor(dtype, [c.oc, c.ic, c.kh, c.kw]).setData(weight);
            const b = c.bias ? ctx.tensor(dtype, [c.oc]).setData(bias) : undefined;
            const out = ctx.tensor(dtype, [c.batch, c.oc, oh, ow]);
            if (fused)
              ctx.recordShader(shaders.convWithClamp, [
                x,
                w,
                b,
                c.stride,
                c.pad,
                c.dilation,
                false,
                [0, 0],
                1,
                -0.5,
                1.5,
                out,
              ]);
            else
              ctx.recordShader(shaders.convolution, [
                x,
                w,
                b,
                c.stride,
                c.pad,
                c.dilation,
                false,
                [0, 0],
                1,
                out,
              ]);
            ctx.submit();
            await ctx.sync();
            await expectTensorClose(out, expected, dtype === 'float16' ? 0.02 : 1e-5);
            const error = await device.popErrorScope();
            if (error) throw error;
          } finally {
            ctx.destroy();
          }
        });
      }
    }
  }
});
