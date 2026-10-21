import { describe, expect, it } from 'vitest';
import { WgpuExecutionContext, type Program } from '../../src';
import { winogradProgram } from '../../utils/winograd-program';
import { setupGPU } from '../helpers';

// Odd IC/OC, multiple batches, non-square images, tiles crossing borders, and
// a one-pixel output exercise scalar BMM tails and zero-filled padded tiles.
const cases = [
  { batch: 2, ic: 3, oc: 5, h: 7, w: 9, pad: 1, bias: true, clamp: false },
  { batch: 1, ic: 4, oc: 8, h: 5, w: 7, pad: 2, bias: false, clamp: true },
  { batch: 1, ic: 1, oc: 1, h: 3, w: 3, pad: 0, bias: true, clamp: false },
];

describe('Winograd convolution CPU oracle', () => {
  for (const dtype of ['float32', 'float16'] as const)
    for (const tile of [2, 4] as const)
      for (const c of cases) {
        it(`${dtype}, tile=${tile}, ${JSON.stringify(c)}`, async () => {
          const array = dtype === 'float16' ? Float16Array : Float32Array;
          const input = new array(
            Array.from(
              { length: c.batch * c.ic * c.h * c.w },
              (value, i) => (((i * 7) % 17) - 8) / 4
            )
          );
          const weight = new array(
            Array.from({ length: c.oc * c.ic * 9 }, (value, i) => (((i * 3) % 11) - 5) / 8)
          );
          const bias = new array(Array.from({ length: c.oc }, (value, i) => ((i % 3) - 1) / 4));
          const oh = c.h + 2 * c.pad - 2,
            ow = c.w + 2 * c.pad - 2;
          const outputShape = [c.batch, c.oc, oh, ow];
          const weights = new ArrayBuffer(weight.byteLength + bias.byteLength);
          new Uint8Array(weights).set(new Uint8Array(weight.buffer));
          new Uint8Array(weights).set(new Uint8Array(bias.buffer), weight.byteLength);
          const program: Program = {
            version: '1',
            inputIds: [0],
            outputIds: [11],
            memoryPlan: { pools: [] },
            values: [
              { type: 'tensor', dtype, shape: [c.batch, c.ic, c.h, c.w] },
              {
                type: 'tensor',
                dtype,
                shape: [c.oc, c.ic, 3, 3],
                weightsOffset: 0,
                weightsLength: weight.byteLength,
              },
              c.bias
                ? {
                    type: 'tensor',
                    dtype,
                    shape: [c.oc],
                    weightsOffset: weight.byteLength,
                    weightsLength: bias.byteLength,
                  }
                : { type: 'null' },
              { type: 'list', items: [1, 1] },
              { type: 'list', items: [c.pad, c.pad] },
              { type: 'scalar', value: false },
              { type: 'list', items: [0, 0] },
              { type: 'scalar', value: 1 },
              c.clamp ? { type: 'scalar', value: 0 } : { type: 'null' },
              c.clamp ? { type: 'scalar', value: 3 } : { type: 'null' },
              { type: 'null' },
              { type: 'tensor', dtype, shape: outputShape },
            ],
            chain: [
              {
                name: 'et_vk.conv_with_clamp.default',
                args: [0, 1, 2, 3, 4, 3, 5, 6, 7, 8, 9, 11],
              },
            ],
          };
          const candidate = winogradProgram(program, weights, tile);
          const expected = new array(c.batch * c.oc * oh * ow);
          for (let b = 0; b < c.batch; b++)
            for (let oc = 0; oc < c.oc; oc++)
              for (let y = 0; y < oh; y++)
                for (let x = 0; x < ow; x++) {
                  let sum = c.bias ? bias[oc] : 0;
                  for (let ic = 0; ic < c.ic; ic++)
                    for (let ky = 0; ky < 3; ky++)
                      for (let kx = 0; kx < 3; kx++) {
                        const iy = y - c.pad + ky,
                          ix = x - c.pad + kx;
                        if (iy >= 0 && iy < c.h && ix >= 0 && ix < c.w)
                          sum +=
                            input[((b * c.ic + ic) * c.h + iy) * c.w + ix] *
                            weight[(oc * c.ic + ic) * 9 + ky * 3 + kx];
                      }
                  expected[((b * c.oc + oc) * oh + y) * ow + x] = c.clamp
                    ? Math.min(3, Math.max(0, sum))
                    : sum;
                }
          const device = await setupGPU();
          device.pushErrorScope('validation');
          const ctx = new WgpuExecutionContext(device);
          try {
            const inp = ctx.tensor(dtype, [c.batch, c.ic, c.h, c.w]).setData(input),
              out = ctx.tensor(dtype, outputShape);
            ctx.recordProgram(candidate.program, candidate.weights, [inp, out]);
            ctx.submit();
            await ctx.sync();
            const actual = new array(await out.getData());
            let maxDelta = 0;
            for (let i = 0; i < actual.length; i++) {
              if (!Number.isFinite(actual[i])) throw new Error(`Nonfinite output at ${i}`);
              maxDelta = Math.max(maxDelta, Math.abs(actual[i] - expected[i]));
            }
            expect(maxDelta).toBeLessThan(dtype === 'float16' ? 0.005 : 0.0001);
            expect(await device.popErrorScope()).toBeNull();
          } finally {
            ctx.destroy();
          }
        });
      }
});
