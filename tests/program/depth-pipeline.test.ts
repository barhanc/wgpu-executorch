import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { expect, it } from 'vitest';
import { WgpuExecutionContext, type Program } from '../../src';
import {
  clearStats,
  depthStatsF16,
  drawDepthF16,
  preprocessF16,
} from '../../example/src/tasks/depth-estimation/shaders';
import { setupGPU } from '../helpers';

it('renders finite FP16 depth across changing camera frames', async () => {
  const device = await setupGPU();
  device.pushErrorScope('validation');
  const ctx = new WgpuExecutionContext(device);
  const width = 448;
  const height = 448;
  const video = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const image = device.createTexture({
    size: [width, height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  });
  const inp = ctx.tensor('float16', [1, 3, height, width]);
  const out = ctx.tensor('float16', [1, 1, height, width]);
  const stats = ctx.storageBuffer(16);
  const dir = resolve(import.meta.dirname, '../../example/public/assets/depthart');
  const program = JSON.parse(readFileSync(resolve(dir, 'program-fp16.json'), 'utf8')) as Program;
  const weights = Uint8Array.from(readFileSync(resolve(dir, 'weights-fp16.bin'))).buffer;
  const reference = new WgpuExecutionContext(device);
  const referenceIn = reference.tensor('float32', [1, 3, height, width]);
  const referenceOut = reference.tensor('float32', [1, 1, height, width]);
  reference.recordProgram(
    JSON.parse(readFileSync(resolve(dir, 'program.json'), 'utf8')) as Program,
    Uint8Array.from(readFileSync(resolve(dir, 'weights.bin'))).buffer,
    [referenceIn, referenceOut]
  );
  ctx
    .recordShader(preprocessF16, [video, inp])
    .recordProgram(program, weights, [inp, out])
    .recordShader(clearStats, [stats])
    .recordShader(depthStatsF16, [out, stats])
    .recordShader(drawDepthF16, [out, stats, image]);

  const readback = device.createBuffer({
    size: width * height * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    // A dark circle on white drives the unbalanced decoder above FP16's range;
    // uniform frames and the original small reference input do not expose it.
    for (const frame of ['circle', 'gradient', 'black', 'white'] as const) {
      const pixels = new Uint8Array(width * height * 4);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = (y * width + x) * 4;
          pixels[i] =
            frame === 'gradient'
              ? Math.round((255 * x) / (width - 1))
              : frame === 'white'
                ? 255
                : 0;
          pixels[i + 1] = frame === 'gradient' ? Math.round((255 * y) / (height - 1)) : pixels[i];
          pixels[i + 2] = frame === 'gradient' ? 128 : pixels[i];
          if (frame === 'circle') {
            pixels.fill(
              (x - width / 2) ** 2 + (y - height / 2) ** 2 < 120 ** 2 ? 0 : 255,
              i,
              i + 3
            );
          }
          pixels[i + 3] = 255;
        }
      }
      device.queue.writeTexture({ texture: video }, pixels, { bytesPerRow: width * 4 }, [
        width,
        height,
      ]);
      ctx.submit();
      await ctx.sync();
      const input = new Float16Array(await inp.getData());
      expect(input[0]).toBeCloseTo((pixels[0] / 255 - 0.485) / 0.229, 2);
      const depth = new Float16Array(await out.getData());
      referenceIn.setData(Float32Array.from(input));
      reference.submit();
      await reference.sync();
      const expected = new Float32Array(await referenceOut.getData());
      let maxError = 0;
      let referenceMax = 0;
      let depthMin = Infinity;
      let depthMax = -Infinity;
      for (let i = 0; i < depth.length; i++) {
        if (!Number.isFinite(depth[i]))
          throw new Error(`${frame}: non-finite depth at ${i}: ${depth[i]}`);
        if (!Number.isFinite(expected[i]))
          throw new Error(`${frame}: non-finite FP32 reference at ${i}`);
        maxError = Math.max(maxError, Math.abs(depth[i] - expected[i]));
        referenceMax = Math.max(referenceMax, Math.abs(expected[i]));
        depthMin = Math.min(depthMin, depth[i]);
        depthMax = Math.max(depthMax, depth[i]);
      }
      expect(maxError / referenceMax, `${frame}: FP16 versus FP32 depth`).toBeLessThan(0.05);
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer(
        { texture: image },
        { buffer: readback, bytesPerRow: width * 4 },
        [width, height]
      );
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const rgba = new Uint8Array(readback.getMappedRange());
      let min = 255;
      let max = 0;
      for (let i = 0; i < rgba.length; i += 4) {
        const gray = Math.round(255 * (1 - (depth[i / 4] - depthMin) / (depthMax - depthMin)));
        if (
          Math.abs(rgba[i] - gray) > 1 ||
          rgba[i + 1] !== rgba[i] ||
          rgba[i + 2] !== rgba[i] ||
          rgba[i + 3] !== 255
        ) {
          throw new Error(
            `${frame}: incorrect rendered pixel ${i / 4}: got ${rgba[i]}, expected ${gray}`
          );
        }
        min = Math.min(min, rgba[i]);
        max = Math.max(max, rgba[i]);
      }
      readback.unmap();
      expect(max - min, `${frame}: depth contrast`).toBeGreaterThan(200);
    }
    expect(await device.popErrorScope()).toBeNull();
  } finally {
    ctx.destroy();
    reference.destroy();
    video.destroy();
    image.destroy();
    readback.destroy();
  }
}, 600_000);
