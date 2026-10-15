import { describe, it } from 'vitest';
import { WgpuExecutionContext, shaders } from '../../src';
import { expectTensorClose, f32ToF16Array, setupGPU } from '../helpers';

/**
 * FP16 kernel variants: the same operator kernels run against binary16
 * storage (f32 accumulation, f16 load/store). Inputs use binary16-exact
 * values so expected results need no tolerance beyond f16 rounding of the
 * outputs. The full-model FP16 path is exercised by the program suites; this
 * file pins down individual ops in isolation.
 */
describe('float16 kernel variants', () => {
  it('aten.add.Tensor adds with broadcast and alpha', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const in1 = ctx.tensor('float16', [2, 3]).setData(f32ToF16Array([1, 2, 3, 4, 5, 6]));
    const in2 = ctx.tensor('float16', [1, 3]).setData(f32ToF16Array([10, 20, 30]));
    const out = ctx.tensor('float16', [2, 3]);

    ctx.recordShader(shaders.add, [in1, in2, 2.0, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [21, 42, 63, 24, 45, 66], 0.01);
    ctx.destroy();
  });

  it('aten.sub.Tensor subtracts in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const in1 = ctx.tensor('float16', [4]).setData(f32ToF16Array([10, 8, 6, 4]));
    const in2 = ctx.tensor('float16', [4]).setData(f32ToF16Array([1, 2, 3, 4]));
    const out = ctx.tensor('float16', [4]);

    ctx.recordShader(shaders.sub, [in1, in2, 1.5, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [8.5, 5, 1.5, -2], 0.01);
    ctx.destroy();
  });

  it('aten.mul.Tensor multiplies in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const in1 = ctx.tensor('float16', [2, 2]).setData(f32ToF16Array([1.5, -2, 4, 0.5]));
    const in2 = ctx.tensor('float16', [2, 2]).setData(f32ToF16Array([2, 3, -1, 8]));
    const out = ctx.tensor('float16', [2, 2]);

    ctx.recordShader(shaders.mul, [in1, in2, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [3, -6, -4, 4], 0.01);
    ctx.destroy();
  });

  it('aten.sigmoid.default and aten.relu.default in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [4]).setData(f32ToF16Array([-2, -0.5, 0.5, 2]));
    const sig = ctx.tensor('float16', [4]);
    const rel = ctx.tensor('float16', [4]);

    ctx.recordShader(shaders.sigmoid, [inp, sig]).recordShader(shaders.relu, [inp, rel]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(sig, [0.1192, 0.3775, 0.6225, 0.8808], 0.002);
    await expectTensorClose(rel, [0, 0, 0.5, 2], 0.01);
    ctx.destroy();
  });

  it('aten.hardswish.default in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [5]).setData(f32ToF16Array([-4, -3, -1.5, 1.5, 4]));
    const out = ctx.tensor('float16', [5]);

    ctx.recordShader(shaders.hardswish, [inp, out]);
    ctx.submit();
    await ctx.sync();
    // x * clamp(x + 3, 0, 6) / 6 -> [0, 0, -0.375, 1.125, 4]
    await expectTensorClose(out, [0, 0, -0.375, 1.125, 4], 0.01);
    ctx.destroy();
  });

  it('aten.mean.dim reduces in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [2, 2, 2]).setData(f32ToF16Array([1, 3, 5, 7, 2, 4, 6, 8]));
    const out = ctx.tensor('float16', [2, 1, 2]);

    ctx.recordShader(shaders.meanDim, [inp, [1], true, undefined, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [3, 5, 4, 6], 0.01);
    ctx.destroy();
  });

  it('aten._softmax.default in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [1, 3]).setData(f32ToF16Array([1, 2, 3]));
    const out = ctx.tensor('float16', [1, 3]);

    ctx.recordShader(shaders.softmax, [inp, 1, false, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [0.09, 0.2447, 0.6652], 0.002);
    ctx.destroy();
  });

  it('aten.constant_pad_nd.default in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [1, 1, 2, 2]).setData(f32ToF16Array([1, 2, 3, 4]));
    const out = ctx.tensor('float16', [1, 1, 3, 4]);

    // pad = [1, 1, 0, 1] -> (left, right) for last dim then dim -2
    ctx.recordShader(shaders.constantPadNd, [inp, [1, 1, 0, 1], -1, out]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(out, [-1, 1, 2, -1, -1, 3, 4, -1, -1, -1, -1, -1], 0.01);
    ctx.destroy();
  });

  it('aten.cat.default and aten.split_with_sizes_copy.default in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float16', [1, 2]).setData(f32ToF16Array([1, 2]));
    const b = ctx.tensor('float16', [1, 1]).setData(f32ToF16Array([3]));
    const cat = ctx.tensor('float16', [1, 3]);
    ctx.recordShader(shaders.cat, [[a, b], 1, cat]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(cat, [1, 2, 3], 0.01);

    const s0 = ctx.tensor('float16', [1, 2]);
    const s1 = ctx.tensor('float16', [1, 1]);
    ctx.recordShader(shaders.splitWithSizesCopy, [cat, [2, 1], 1, [s0, s1]]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(s0, [1, 2], 0.01);
    await expectTensorClose(s1, [3], 0.01);
    ctx.destroy();
  });

  it('aten.mm.default tiled and vec4 paths in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    // 3x4 @ 4x2 (K, N not divisible by 4 -> tiled)
    const a1 = ctx
      .tensor('float16', [3, 4])
      .setData(f32ToF16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]));
    const b1 = ctx.tensor('float16', [4, 2]).setData(f32ToF16Array([1, 0, 0, 1, 1, 1, 2, 2]));
    const o1 = ctx.tensor('float16', [3, 2]);
    ctx.recordShader(shaders.mm, [a1, b1, o1]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(o1, [12, 13, 28, 29, 44, 45], 0.05);

    // 4x4 @ 4x4 (K, N divisible by 4 -> vec4)
    const a2 = ctx
      .tensor('float16', [4, 4])
      .setData(f32ToF16Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]));
    const b2 = ctx
      .tensor('float16', [4, 4])
      .setData(f32ToF16Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]));
    const o2 = ctx.tensor('float16', [4, 4]);
    ctx.recordShader(shaders.mm, [a2, b2, o2]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(o2, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 0.05);
    ctx.destroy();
  });

  it('et_vk.conv_with_clamp.default in binary16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx
      .tensor('float16', [1, 1, 3, 3])
      .setData(f32ToF16Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const weight = ctx.tensor('float16', [1, 1, 2, 2]).setData(f32ToF16Array([1, 0, 0, 1]));
    const bias = ctx.tensor('float16', [1]).setData(f32ToF16Array([10]));
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
    await expectTensorClose(out, [17, 18, 22, 23], 0.01);
    ctx.destroy();
  });

  it('aten.max_pool2d_with_indices.default in binary16 with int32 indices', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [1, 1, 2, 2]).setData(f32ToF16Array([1, 3, 2, 4]));
    const vals = ctx.tensor('float16', [1, 1, 1, 1]);
    const idx = ctx.tensor('int32', [1, 1, 1, 1]);

    ctx.recordShader(shaders.maxPool2d, [inp, [2, 2], [2, 2], [0, 0], [1, 1], false, [vals, idx]]);
    ctx.submit();
    await ctx.sync();
    await expectTensorClose(vals, [4], 0.01);
    const idxData = new Int32Array(await idx.getData());
    if (idxData[0] !== 3) {
      throw new Error(`indices mismatch: got ${idxData[0]}, expected 3`);
    }
    ctx.destroy();
  });
});
