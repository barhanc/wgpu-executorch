import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it, expect } from 'vitest';

import { WgpuExecutionContext, type Program } from '../../src';
import { setupGPU } from '../helpers';

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const PROGRAM_HWC4_PATH = resolve(REPO_ROOT, 'example/public/assets/depthart/program-hwc4.json');
const WEIGHTS_HWC4_PATH = resolve(REPO_ROOT, 'example/public/assets/depthart/weights-hwc4.bin');
const PROGRAM_HWC4_FP16_PATH = resolve(
  REPO_ROOT,
  'example/public/assets/depthart/program-hwc4-fp16.json'
);
const WEIGHTS_HWC4_FP16_PATH = resolve(
  REPO_ROOT,
  'example/public/assets/depthart/weights-hwc4-fp16.bin'
);

const INPUT_SHAPE = [1, 3, 448, 448] as const;
const OUTPUT_SHAPE = [1, 1, 448, 448] as const;
const INPUT_NUMEL = 3 * 448 * 448;

function makeInput(): Float32Array {
  const arr = new Float32Array(INPUT_NUMEL);
  for (let i = 0; i < INPUT_NUMEL; i++) {
    arr[i] = (i % 251) / 250 - 0.5;
  }
  return arr;
}

function makeInputF16(): Float16Array {
  const arr = new Float16Array(INPUT_NUMEL);
  for (let i = 0; i < INPUT_NUMEL; i++) {
    arr[i] = (i % 251) / 250 - 0.5;
  }
  return arr;
}

function readArrayBuffer(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  const copy = new ArrayBuffer(buf.byteLength);
  new Uint8Array(copy).set(buf);
  return copy;
}

describe('depthart-hwc4 program (full ExecuTorch graph in packed HWC4)', () => {
  it('executes the full HWC4 operator chain in float32', { timeout: 600_000 }, async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const program = JSON.parse(readFileSync(PROGRAM_HWC4_PATH, 'utf8')) as Program;
    const weights = readArrayBuffer(WEIGHTS_HWC4_PATH);

    const inp = ctx.tensor('float32', [...INPUT_SHAPE]).setData(makeInput());
    const out = ctx.tensor('float32', [...OUTPUT_SHAPE]);

    ctx.recordProgram(program, weights, [inp, out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float32Array(await out.getData());
    expect(actual.length).toBe(
      OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2] * OUTPUT_SHAPE[3]
    );
    expect(Number.isFinite(actual[0])).toBe(true);
    expect(Number.isNaN(actual[0])).toBe(false);
  });

  it('executes the full HWC4 operator chain in float16', { timeout: 600_000 }, async () => {
    const device = await setupGPU();
    if (!device.features.has('shader-f16')) return;

    const ctx = new WgpuExecutionContext(device);

    const program = JSON.parse(readFileSync(PROGRAM_HWC4_FP16_PATH, 'utf8')) as Program;
    const weights = readArrayBuffer(WEIGHTS_HWC4_FP16_PATH);

    const inp = ctx.tensor('float16', [...INPUT_SHAPE]).setData(makeInputF16());
    const out = ctx.tensor('float16', [...OUTPUT_SHAPE]);

    ctx.recordProgram(program, weights, [inp, out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float16Array(await out.getData());
    expect(actual.length).toBe(
      OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2] * OUTPUT_SHAPE[3]
    );
    expect(Number.isFinite(actual[0])).toBe(true);
    expect(Number.isNaN(actual[0])).toBe(false);
  });
});
