import { describe, it, expect } from 'vitest';
import { WgpuExecutionContext, Tensor, MAX_NDIM } from '../../src';
import { setupGPU } from '../helpers';

describe('Tensor', () => {
  it('creates a tensor with valid shape', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.numel).toBe(6);
    expect(t.byteLength).toBe(24);
    expect(t.dtype).toBe('float32');
    ctx.destroy();
  });

  it('rejects rank > MAX_NDIM', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const shape = new Array(MAX_NDIM + 1).fill(1);
    expect(() => ctx.tensor('float32', shape)).toThrow(/exceeds maximum rank/);
    ctx.destroy();
  });

  it('rejects zero dimensions', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    expect(() => ctx.tensor('float32', [0, 3])).toThrow(/positive integers/);
    ctx.destroy();
  });

  it('rejects negative dimensions', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    expect(() => ctx.tensor('float32', [-1, 3])).toThrow(/positive integers/);
    ctx.destroy();
  });

  it('rejects non-integer dimensions', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    expect(() => ctx.tensor('float32', [1.5, 3])).toThrow(/positive integers/);
    ctx.destroy();
  });

  it('rejects buffer too small', async () => {
    const device = await setupGPU();
    const buf = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    expect(() => new Tensor('float32', [2, 3], device, buf)).toThrow(/< required/);
    buf.destroy();
  });

  it('rejects buffer without STORAGE usage', async () => {
    const device = await setupGPU();
    const buf = device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    expect(() => new Tensor('float32', [2, 3], device, buf)).toThrow(/STORAGE flag/);
    buf.destroy();
  });

  it('setData uploads TypedArray data', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [2, 3]);
    t.setData(new Float32Array([1, 2, 3, 4, 5, 6]));

    const data = new Float32Array(await t.getData());
    expect(Array.from(data)).toEqual([1, 2, 3, 4, 5, 6]);
    ctx.destroy();
  });

  it('setData uploads ArrayBuffer data', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [2, 2]);
    const buf = new Float32Array([1, 2, 3, 4]);
    t.setData(buf.buffer);

    const data = new Float32Array(await t.getData());
    expect(Array.from(data)).toEqual([1, 2, 3, 4]);
    ctx.destroy();
  });

  it('setData writes a prefix when the source is smaller than the tensor', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [2, 3]);
    t.setData(new Float32Array([1, 2, 3, 4, 5, 6]));
    t.setData(new Float32Array([7, 8]));

    const data = new Float32Array(await t.getData());
    expect(Array.from(data)).toEqual([7, 8, 3, 4, 5, 6]);
    ctx.destroy();
  });

  it('setData rejects source larger than tensor', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [2, 3]);
    expect(() => t.setData(new Float32Array(8))).toThrow(/pass a view/);
    ctx.destroy();
  });

  it('setData rejects buffer without COPY_DST', async () => {
    const device = await setupGPU();
    const buf = device.createBuffer({
      size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const t = new Tensor('float32', [2, 3], device, buf);
    expect(() => t.setData(new Float32Array([1, 2, 3, 4, 5, 6]))).toThrow(/COPY_DST/);
    buf.destroy();
  });

  it('setData/getData roundtrip preserves data', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    const t = ctx.tensor('float32', [4, 4]);
    const original = Float32Array.from({ length: 16 }, (unused, i) => i * 0.5);
    t.setData(original);

    const data = new Float32Array(await t.getData());
    expect(Array.from(data)).toEqual(Array.from(original));
    ctx.destroy();
  });

  it('handles unaligned byte sizes in setData and getData', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);
    // 3 bytes (unaligned to 4)
    const t = ctx.tensor('uint8', [3]);
    const original = new Uint8Array([7, 42, 99]);
    t.setData(original);

    const data = new Uint8Array(await t.getData());
    expect(Array.from(data)).toEqual([7, 42, 99]);
    ctx.destroy();
  });
});
