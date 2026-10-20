import { describe, it, expect } from 'vitest';
import { setupGPU, expectTensorClose } from '../helpers';
import { WgpuExecutionContext, createComputeBundle, shaders } from '../../src';

describe('shader-f16 feature', () => {
  it('device has shader-f16 feature and executes WGSL f16 shader', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;

    const ctx = new WgpuExecutionContext(device);
    const inTensor = ctx.tensor('float16', [4]);
    const outTensor = ctx.tensor('float16', [4]);

    const inData = new Float16Array([1.5, -2.0, 3.25, 0.0]);
    inTensor.setData(inData);

    const WGSL_FP16 = /* wgsl */ `
    enable f16;

    @group(0) @binding(0) var<storage, read> a: array<f16>;
    @group(0) @binding(1) var<storage, read_write> b: array<f16>;

    @compute @workgroup_size(4, 1, 1)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      b[gid.x] = a[gid.x] * 2.0h;
    }
    `;

    const bundle = createComputeBundle(device, WGSL_FP16, [
      { binding: 0, buffer: inTensor.buffer },
      { binding: 1, buffer: outTensor.buffer },
    ]);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: 1,
      workgroupCountY: 1,
      workgroupCountZ: 1,
    });

    ctx.submit();
    await ctx.sync();

    const raw = await outTensor.getData();
    const result = new Float16Array(raw);
    expect(result[0]).toBeCloseTo(3.0);
    expect(result[1]).toBeCloseTo(-4.0);
    expect(result[2]).toBeCloseTo(6.5);
    expect(result[3]).toBeCloseTo(0.0);

    ctx.destroy();
  });

  it('computes float16 addition with shaders.add (elementwise and broadcast)', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    // 1. Elementwise identical shapes
    const a = ctx.tensor('float16', [4]).setData(new Float16Array([1.0, 2.5, 3.0, 4.5]));
    const b = ctx.tensor('float16', [4]).setData(new Float16Array([0.5, 1.5, -1.0, 2.0]));
    const out1 = ctx.tensor('float16', [4]);

    ctx.recordShader(shaders.add, [a, b, out1]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(out1, [1.5, 4.0, 2.0, 6.5], 1e-2);

    // 2. Broadcasted addition with alpha
    const x = ctx.tensor('float16', [1, 2, 1]).setData(new Float16Array([10.0, 20.0]));
    const y = ctx.tensor('float16', [1, 2, 2]).setData(new Float16Array([1.0, 2.0, 3.0, 4.0]));
    const out2 = ctx.tensor('float16', [1, 2, 2]);

    ctx.recordShader(shaders.add, [x, y, 2.0, out2]);
    ctx.submit();
    await ctx.sync();

    // [10 + 2*1, 10 + 2*2, 20 + 2*3, 20 + 2*4] = [12, 14, 26, 28]
    await expectTensorClose(out2, [12, 14, 26, 28], 1e-2);
    ctx.destroy();
  });

  it('computes float16 subtraction and multiplication (sub, mul)', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const a = ctx.tensor('float16', [4]).setData(new Float16Array([10.0, 20.0, 30.0, 40.0]));
    const b = ctx.tensor('float16', [4]).setData(new Float16Array([1.0, 2.0, 3.0, 4.0]));
    const outSub = ctx.tensor('float16', [4]);
    const outMul = ctx.tensor('float16', [4]);

    ctx.recordShader(shaders.sub, [a, b, 2.0, outSub]);
    ctx.recordShader(shaders.mul, [a, b, outMul]);
    ctx.submit();
    await ctx.sync();

    await expectTensorClose(outSub, [8.0, 16.0, 24.0, 32.0], 1e-2);
    await expectTensorClose(outMul, [10.0, 40.0, 90.0, 160.0], 1e-2);
    ctx.destroy();
  });

  it('computes float16 activations (relu, sigmoid, hardswish)', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    const inp = ctx.tensor('float16', [4]).setData(new Float16Array([-4.0, -1.0, 0.0, 3.0]));
    const outRelu = ctx.tensor('float16', [4]);
    const outSigmoid = ctx.tensor('float16', [4]);
    const outHardswish = ctx.tensor('float16', [4]);

    ctx.recordShader(shaders.relu, [inp, outRelu]);
    ctx.recordShader(shaders.sigmoid, [inp, outSigmoid]);
    ctx.recordShader(shaders.hardswish, [inp, outHardswish]);
    ctx.submit();
    await ctx.sync();

    // relu: max(x, 0) -> [0, 0, 0, 3]
    await expectTensorClose(outRelu, [0, 0, 0, 3], 1e-2);

    // sigmoid: 1 / (1 + exp(-x))
    const sigm = (x: number) => 1 / (1 + Math.exp(-x));
    await expectTensorClose(outSigmoid, [-4, -1, 0, 3].map(sigm), 1e-2);

    // hardswish: x * clamp(x + 3, 0, 6) / 6
    // -4: -4 * 0 / 6 = 0
    // -1: -1 * 2 / 6 = -0.3333
    // 0: 0
    // 3: 3 * 6 / 6 = 3.0
    await expectTensorClose(outHardswish, [0.0, -0.3333, 0.0, 3.0], 1e-2);
    ctx.destroy();
  });

  it('computes float16 reductions (softmax, meanDim)', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    // Softmax
    const smIn = ctx.tensor('float16', [2, 4]).setData(new Float16Array([1, 2, 3, 4, 0, 0, 0, 0]));
    const smOut = ctx.tensor('float16', [2, 4]);

    ctx.recordShader(shaders.softmax, [smIn, 1, false, smOut]);

    // MeanDim
    const meanIn = ctx
      .tensor('float16', [2, 3])
      .setData(new Float16Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    const meanOut = ctx.tensor('float16', [2]);

    ctx.recordShader(shaders.meanDim, [meanIn, [1], false, undefined, meanOut]);
    ctx.submit();
    await ctx.sync();

    const expectedSmRow0 = [1, 2, 3, 4].map((v) => Math.exp(v));
    const sum0 = expectedSmRow0.reduce((a, b) => a + b, 0);
    await expectTensorClose(
      smOut,
      [...expectedSmRow0.map((v) => v / sum0), 0.25, 0.25, 0.25, 0.25],
      1e-2
    );

    // Row 0 mean = (1+2+3)/3 = 2, Row 1 mean = (4+5+6)/3 = 5
    await expectTensorClose(meanOut, [2.0, 5.0], 1e-2);
    ctx.destroy();
  });

  it('computes float16 maxPool2d, upsampleNearest2d, upsampleBilinear2d', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    // 1. MaxPool2D
    const poolIn = ctx
      .tensor('float16', [1, 1, 4, 4])
      .setData(new Float16Array([1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]));
    const poolVals = ctx.tensor('float16', [1, 1, 2, 2]);
    const poolIdx = ctx.tensor('int32', [1, 1, 2, 2]);

    ctx.recordShader(shaders.maxPool2d, [
      poolIn,
      [2, 2], // kernel
      [2, 2], // stride
      [0, 0], // pad
      [1, 1], // dilation
      false, // ceil_mode
      [poolVals, poolIdx],
    ]);

    // 2. Nearest upsample [1, 1, 2, 2] -> [1, 1, 4, 4]
    const nearIn = ctx.tensor('float16', [1, 1, 2, 2]).setData(new Float16Array([10, 20, 30, 40]));
    const nearOut = ctx.tensor('float16', [1, 1, 4, 4]);
    ctx.recordShader(shaders.upsampleNearest2d, [nearIn, undefined, [2, 2], nearOut]);

    // 3. Bilinear upsample [1, 1, 2, 2] -> [1, 1, 3, 3] with align_corners=true
    const biIn = ctx.tensor('float16', [1, 1, 2, 2]).setData(new Float16Array([0, 10, 20, 30]));
    const biOut = ctx.tensor('float16', [1, 1, 3, 3]);
    ctx.recordShader(shaders.upsampleBilinear2d, [biIn, [3, 3], true, undefined, biOut]);

    ctx.submit();
    await ctx.sync();

    // Pool results: 2x2 max blocks: [4, 8, 12, 16]
    await expectTensorClose(poolVals, [4, 8, 12, 16], 1e-2);

    // Nearest upsample 2x2 -> 4x4
    await expectTensorClose(
      nearOut,
      [10, 10, 20, 20, 10, 10, 20, 20, 30, 30, 40, 40, 30, 30, 40, 40],
      1e-2
    );

    // Bilinear upsample 2x2 -> 3x3 align_corners=true
    await expectTensorClose(biOut, [0, 5, 10, 10, 15, 20, 20, 25, 30], 1e-2);

    ctx.destroy();
  });

  it('computes float16 data-movement operators (cat, constantPadNd, split, permute)', async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;
    const ctx = new WgpuExecutionContext(device);

    // 1. Cat
    const c1 = ctx.tensor('float16', [1, 2]).setData(new Float16Array([1.0, 2.0]));
    const c2 = ctx.tensor('float16', [1, 2]).setData(new Float16Array([3.0, 4.0]));
    const catOut = ctx.tensor('float16', [1, 4]);
    ctx.recordShader(shaders.cat, [[c1, c2], 1, catOut]);

    // 2. ConstantPadNd
    const padIn = ctx.tensor('float16', [1, 1, 2, 2]).setData(new Float16Array([1, 2, 3, 4]));
    const padOut = ctx.tensor('float16', [1, 1, 4, 4]);
    ctx.recordShader(shaders.constantPadNd, [padIn, [1, 1, 1, 1], 9.0, padOut]);

    // 3. SplitWithSizes
    const splitIn = ctx.tensor('float16', [4]).setData(new Float16Array([10, 20, 30, 40]));
    const s1 = ctx.tensor('float16', [1]);
    const s2 = ctx.tensor('float16', [3]);
    ctx.recordShader(shaders.splitWithSizesCopy, [splitIn, [1, 3], 0, [s1, s2]]);

    // 4. Permute (2D transpose)
    const permIn = ctx.tensor('float16', [2, 3]).setData(new Float16Array([1, 2, 3, 4, 5, 6]));
    const permOut = ctx.tensor('float16', [3, 2]);
    ctx.recordShader(shaders.permuteCopy, [permIn, [1, 0], permOut]);

    ctx.submit();
    await ctx.sync();

    // Cat: [1, 2, 3, 4]
    await expectTensorClose(catOut, [1.0, 2.0, 3.0, 4.0], 1e-2);

    // Pad: 2x2 in 4x4 with 1-pad border of 9.0
    await expectTensorClose(padOut, [9, 9, 9, 9, 9, 1, 2, 9, 9, 3, 4, 9, 9, 9, 9, 9], 1e-2);

    // Split: s1=[10], s2=[20, 30, 40]
    await expectTensorClose(s1, [10], 1e-2);
    await expectTensorClose(s2, [20, 30, 40], 1e-2);

    // Permute 2x3 -> 3x2 transpose
    await expectTensorClose(permOut, [1, 4, 2, 5, 3, 6], 1e-2);

    ctx.destroy();
  });
});
