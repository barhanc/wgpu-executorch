import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { expectTensorClose, setupGPU } from '../helpers';

describe('rectangular and batched matrix multiplication', () => {
  for (const dtype of ['float32', 'float16'] as const) {
    for (const [batch, m, k, n] of [
      [1, 65, 68, 100],
      [2, 67, 400, 76],
      [2, 33, 36, 44],
      [1, 64, 16, 64],
      [2, 400, 32, 400],
      [2, 64, 400, 400],
    ]) {
      it(`${dtype}: [${batch},${m},${k}] x [${batch},${k},${n}]`, async () => {
        const device = await setupGPU();
        device.pushErrorScope('validation');
        const ctx = new WgpuExecutionContext(device);
        const array = dtype === 'float16' ? Float16Array : Float32Array;
        const a = new array(
          Array.from({ length: batch * m * k }, (value, i) => (((i * 3) % 7) - 3) / 8)
        );
        const b = new array(
          Array.from({ length: batch * k * n }, (value, i) => (((i * 7) % 5) - 2) / 8)
        );
        const expected: number[] = [];
        for (let z = 0; z < batch; z++)
          for (let row = 0; row < m; row++)
            for (let col = 0; col < n; col++) {
              let sum = 0;
              for (let inner = 0; inner < k; inner++)
                sum += a[(z * m + row) * k + inner] * b[(z * k + inner) * n + col];
              expected.push(sum);
            }
        try {
          const x = ctx.tensor(dtype, batch === 1 ? [m, k] : [batch, m, k]).setData(a);
          const y = ctx.tensor(dtype, batch === 1 ? [k, n] : [batch, k, n]).setData(b);
          const out = ctx.tensor(dtype, batch === 1 ? [m, n] : [batch, m, n]);
          ctx.recordShader(batch === 1 ? shaders.mm : shaders.bmm, [x, y, out]);
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
});
