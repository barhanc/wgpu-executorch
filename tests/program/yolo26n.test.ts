import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it } from 'vitest';

import { WgpuExecutionContext, type Program } from '../../src';
import { setupGPU } from '../helpers';

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const PROGRAM_PATH = resolve(REPO_ROOT, 'example/public/assets/yolo26n/program.json');
const WEIGHTS_PATH = resolve(REPO_ROOT, 'example/public/assets/yolo26n/weights.bin');
const REFERENCE_PATH = resolve(import.meta.dirname, '../fixtures/yolo26n/reference-output.f32');

const INPUT_SHAPE = [1, 3, 640, 640] as const;
const OUTPUT_SHAPE = [1, 84, 8400] as const;
const INPUT_NUMEL = 3 * 640 * 640;

// Deterministic input pattern (must match make_reference_input in the
// YOLO26-N export script): value[i] = (i % 251) / 250 - 0.5.
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

describe('yolo26n program (full ExecuTorch graph)', () => {
  it(
    'executes the full operator chain and matches the PyTorch reference output',
    { timeout: 600_000 },
    async () => {
      const device = await setupGPU();
      const ctx = new WgpuExecutionContext(device);

      const program = JSON.parse(readFileSync(PROGRAM_PATH, 'utf8')) as Program;
      const weights = readArrayBuffer(WEIGHTS_PATH);

      const inp = ctx.tensor('float32', [...INPUT_SHAPE]).setData(makeInput());
      const out = ctx.tensor('float32', [...OUTPUT_SHAPE]);

      ctx.recordProgram(program, weights, [inp, out]);
      ctx.submit();
      await ctx.sync();

      const actual = new Float32Array(await out.getData());
      const expected = new Float32Array(readArrayBuffer(REFERENCE_PATH));

      if (actual.length !== expected.length) {
        throw new Error(
          `Output length mismatch: got ${actual.length}, expected ${expected.length}`
        );
      }

      // Box coordinates span [0, 640] and class scores are sigmoid outputs in
      // [0, 1]; fp32 accumulation-order differences between CPU torch and GPU
      // warrant an absolute tolerance well above elementwise epsilon but far
      // below any real detection error.
      let maxDiff = 0;
      let maxIdx = 0;
      let beyondTol = 0;
      const tolerance = 0.01;
      for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > maxDiff) {
          maxDiff = diff;
          maxIdx = i;
        }
        if (diff > tolerance) {
          beyondTol++;
        }
      }

      if (beyondTol > 0 || maxDiff > tolerance) {
        throw new Error(
          `Output mismatch: ${beyondTol}/${actual.length} elements exceed tolerance ` +
            `${tolerance} (max diff ${maxDiff.toExponential(3)} at index ${maxIdx}: ` +
            `got ${actual[maxIdx]}, expected ${expected[maxIdx]})`
        );
      }

      ctx.destroy();
    }
  );

  it('executes the full operator chain in float16', { timeout: 600_000 }, async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const progPath = resolve(REPO_ROOT, 'example/public/assets/yolo26n/program-fp16.json');
    const weightsPath = resolve(REPO_ROOT, 'example/public/assets/yolo26n/weights-fp16.bin');
    const program = JSON.parse(readFileSync(progPath, 'utf8')) as Program;
    const weights = readArrayBuffer(weightsPath);

    const inp = ctx.tensor('float16', [...INPUT_SHAPE]).setData(makeInputF16());
    const out = ctx.tensor('float16', [...OUTPUT_SHAPE]);

    ctx.recordProgram(program, weights, [inp, out]);
    ctx.submit();
    await ctx.sync();

    const actual = new Float16Array(await out.getData());
    const expected = new Float32Array(readArrayBuffer(REFERENCE_PATH));

    if (actual.length !== expected.length) {
      throw new Error(`Output length mismatch: got ${actual.length}, expected ${expected.length}`);
    }

    // Check against FP32 reference with FP16 tolerance
    let beyondTol = 0;
    const tolerance = 0.5;
    for (let i = 0; i < actual.length; i++) {
      const diff = Math.abs(actual[i] - expected[i]);
      if (diff > tolerance) beyondTol++;
    }

    if (beyondTol > actual.length * 0.05) {
      throw new Error(
        `Output mismatch: ${beyondTol}/${actual.length} elements exceed tolerance ${tolerance}`
      );
    }

    ctx.destroy();
  });
});
