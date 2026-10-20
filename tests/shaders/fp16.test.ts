import { describe, it, expect } from 'vitest';
import { setupGPU, expectTensorClose } from '../helpers';
import { WgpuExecutionContext, createComputeBundle, shaders } from '../../src';

describe('shader-f16 feature', () => {
  it('device has shader-f16 feature and executes WGSL f16 shader', async () => {
    const device = await setupGPU();
    expect(device.features.has('shader-f16')).toBe(true);

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
});
