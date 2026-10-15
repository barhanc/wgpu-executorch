import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it } from 'vitest';

import { WgpuExecutionContext, type Program } from '../../src';
import { f32ToF16Array, readTensorF32, setupGPU } from '../helpers';

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const PROGRAM_PATH = resolve(REPO_ROOT, 'example/public/assets/depthart/program.json');
const WEIGHTS_PATH = resolve(REPO_ROOT, 'example/public/assets/depthart/weights.bin');
const REFERENCE_PATH = resolve(import.meta.dirname, '../fixtures/depthart/reference-output.f32');

const INPUT_SHAPE = [1, 3, 448, 448] as const;
const OUTPUT_SHAPE = [1, 1, 448, 448] as const;
const INPUT_NUMEL = 3 * 448 * 448;

// Deterministic input pattern (must match make_reference_input in the
// DepthART export script): value[i] = (i % 251) / 250 - 0.5.
function makeInput(): Float32Array {
  const arr = new Float32Array(INPUT_NUMEL);
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

describe('depthart program (full ExecuTorch graph)', () => {
  it(
    'executes the full operator chain and matches the PyTorch reference output',
    { timeout: 600_000 },
    async () => {
      const device = await setupGPU();
      const ctx = new WgpuExecutionContext(device);

      const program = JSON.parse(readFileSync(PROGRAM_PATH, 'utf8')) as Program;
      const weights = readArrayBuffer(WEIGHTS_PATH);

      // FP16 model: inputs are uploaded as binary16 (matching the PyTorch
      // half reference, which rounds the deterministic pattern to f16).
      const inp = ctx.tensor('float16', [...INPUT_SHAPE]).setData(f32ToF16Array(makeInput()));
      const out = ctx.tensor('float16', [...OUTPUT_SHAPE]);

      ctx.recordProgram(program, weights, [inp, out]);
      ctx.submit();
      await ctx.sync();

      const actual = await readTensorF32(out);
      const expected = new Float32Array(readArrayBuffer(REFERENCE_PATH));

      if (actual.length !== expected.length) {
        throw new Error(
          `Output length mismatch: got ${actual.length}, expected ${expected.length}`
        );
      }

      // Affine-invariant depth is unbounded and reaches ~1e4 for this input.
      // Both sides compute in f32 and round every layer to binary16; the
      // residual drift between two binary16 realizations reaches ~0.4% of the
      // value scale here, so the tolerance combines a small absolute term
      // with a relative term (still far below any real depth error).
      let maxDiff = 0;
      let maxIdx = 0;
      let beyondTol = 0;
      for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > maxDiff) {
          maxDiff = diff;
          maxIdx = i;
        }
        const tolerance = 4.0 + 0.01 * Math.abs(expected[i]);
        if (diff > tolerance) {
          beyondTol++;
        }
      }

      if (beyondTol > 0) {
        throw new Error(
          `Output mismatch: ${beyondTol}/${actual.length} elements exceed tolerance ` +
            `(max diff ${maxDiff.toExponential(3)} at index ${maxIdx}: ` +
            `got ${actual[maxIdx]}, expected ${expected[maxIdx]})`
        );
      }

      ctx.destroy();
    }
  );
});
