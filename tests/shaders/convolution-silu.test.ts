import { describe, expect, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { setupGPU } from '../helpers';
import type { ConvolutionArgs } from '../../src/shaders/convolution';

// Exercise direct/grouped, depthwise, transposed, dense tiles and pointwise tiles.
const cases = [
  { ic: 16, oc: 16, k: 1, groups: 1 },
  { ic: 16, oc: 48, k: 1, groups: 1 },
  { ic: 4, oc: 12, k: 3, groups: 1 },
  { ic: 4, oc: 48, k: 3, groups: 1 },
  { ic: 3, oc: 5, k: 3, groups: 1 },
  { ic: 4, oc: 8, k: 3, groups: 2 },
  { ic: 4, oc: 4, k: 3, groups: 4 },
  { ic: 4, oc: 4, k: 5, groups: 4 },
  { ic: 4, oc: 8, k: 3, groups: 1, transposed: true },
];

describe('convolution SiLU epilogue', () => {
  for (const dtype of ['float32', 'float16'] as const) {
    for (const c of cases) {
      it(`${dtype}: ${JSON.stringify(c)}`, async () => {
        const device = await setupGPU();
        device.pushErrorScope('validation');
        const ctx = new WgpuExecutionContext(device);
        try {
          const array = dtype === 'float16' ? Float16Array : Float32Array;
          const input = ctx.tensor(dtype, [1, c.ic, 8, 12]);
          const weight = ctx.tensor(
            dtype,
            c.transposed ? [c.ic, c.oc, c.k, c.k] : [c.oc, c.ic / c.groups, c.k, c.k]
          );
          const bias = ctx.tensor(dtype, [c.oc]);
          input.setData(
            new array(Array.from({ length: input.numel }, (value, i) => (((i * 7) % 31) - 15) / 8))
          );
          weight.setData(
            new array(Array.from({ length: weight.numel }, (value, i) => (((i * 3) % 17) - 8) / 32))
          );
          bias.setData(
            new array(Array.from({ length: bias.numel }, (value, i) => ((i % 5) - 2) / 4))
          );
          const plain = ctx.tensor(dtype, [1, c.oc, 8, 12]);
          const gate = ctx.tensor(dtype, plain.shape);
          const expected = ctx.tensor(dtype, plain.shape);
          const actual = ctx.tensor(dtype, plain.shape);
          const args: ConvolutionArgs = [
            input,
            weight,
            bias,
            [1, 1],
            [c.k >> 1, c.k >> 1],
            [1, 1],
            c.transposed ?? false,
            [0, 0],
            c.groups,
            plain,
          ];
          ctx.recordShader(shaders.convolution, args);
          ctx.recordShader(shaders.sigmoid, [plain, gate]);
          ctx.recordShader(shaders.mul, [plain, gate, expected]);
          ctx.recordShader(shaders.convolutionSilu, [
            ...args.slice(0, -1),
            actual,
          ] as unknown as ConvolutionArgs);
          ctx.submit();
          await ctx.sync();
          const a = new array(await actual.getData());
          const b = new array(await expected.getData());
          let maxDelta = 0;
          for (let i = 0; i < a.length; i++) {
            if (!Number.isFinite(a[i])) throw new Error(`Nonfinite activation at ${i}`);
            maxDelta = Math.max(maxDelta, Math.abs(a[i] - b[i]));
          }
          expect(maxDelta).toBeLessThan(dtype === 'float16' ? 0.005 : 1e-5);
          expect(await device.popErrorScope()).toBeNull();
        } finally {
          ctx.destroy();
        }
      });
    }
  }
});
