import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it } from 'vitest';

import { WgpuExecutionContext, type Program } from '../../src';
import { f32ToF16Array, readTensorF32, setupGPU } from '../helpers';

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const PROGRAM_PATH = resolve(REPO_ROOT, 'example/public/assets/selfie-segmentation/program.json');
const WEIGHTS_PATH = resolve(REPO_ROOT, 'example/public/assets/selfie-segmentation/weights.bin');

const INPUT_SHAPE = [1, 3, 256, 256] as const;
const OUTPUT_SHAPE = [1, 1, 256, 256] as const;
const INPUT_NUMEL = 3 * 256 * 256;

// Deterministic input pattern shared with the other program suites:
// value[i] = (i % 251) / 250 - 0.5.
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

describe('selfie-segmentation program (full ExecuTorch graph)', () => {
  it(
    'executes the full FP16 operator chain and produces a sane segmentation mask',
    { timeout: 600_000 },
    async () => {
      const device = await setupGPU();
      const ctx = new WgpuExecutionContext(device);

      const program = JSON.parse(readFileSync(PROGRAM_PATH, 'utf8')) as Program;
      const weights = readArrayBuffer(WEIGHTS_PATH);

      // FP16 model: inputs are uploaded as binary16.
      const inp = ctx.tensor('float16', [...INPUT_SHAPE]).setData(f32ToF16Array(makeInput()));
      const out = ctx.tensor('float16', [...OUTPUT_SHAPE]);

      ctx.recordProgram(program, weights, [inp, out]);
      ctx.submit();
      await ctx.sync();

      // No PyTorch source model exists for this conversion-only export, so
      // there is no reference-output ground truth. Sanity-check the mask
      // instead: the full 118-op FP16 chain must produce finite sigmoid-range
      // values. (The portrait segmenter saturates to ~0 on face-free
      // synthetic inputs, so contrast is not assertable here.)
      const actual = await readTensorF32(out);

      for (let i = 0; i < actual.length; i++) {
        const v = actual[i];
        if (!Number.isFinite(v)) {
          throw new Error(`mask contains non-finite value at index ${i}: ${v}`);
        }
        if (v < -0.01 || v > 1.01) {
          throw new Error(`mask out of sigmoid range at index ${i}: ${v}`);
        }
      }

      ctx.destroy();
    }
  );
});
