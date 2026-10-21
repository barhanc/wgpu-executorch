import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, it } from 'vitest';

import { WgpuExecutionContext, type Program } from '../../src';
import { setupGPU } from '../helpers';

const REPO_ROOT = resolve(import.meta.dirname, '../..');
const PROG_PATH = resolve(REPO_ROOT, 'example/public/assets/selfie-segmentation/program-fp16.json');
const WEIGHTS_PATH = resolve(
  REPO_ROOT,
  'example/public/assets/selfie-segmentation/weights-fp16.bin'
);

function readArrayBuffer(path: string): ArrayBuffer {
  const buf = readFileSync(path);
  const copy = new ArrayBuffer(buf.byteLength);
  new Uint8Array(copy).set(buf);
  return copy;
}

describe('selfie-segmentation fp16', () => {
  it('records and submits selfie-segmentation graph in float16', async () => {
    const device = await setupGPU();
    const ctx = new WgpuExecutionContext(device);

    const program = JSON.parse(readFileSync(PROG_PATH, 'utf8')) as Program;
    const weights = readArrayBuffer(WEIGHTS_PATH);

    const inp = ctx.tensor('float16', [1, 3, 256, 256]);
    const out = ctx.tensor('float16', [1, 1, 256, 256]);

    ctx.recordProgram(program, weights, [inp, out]);
    ctx.submit();
    await ctx.sync();
    ctx.destroy();
  });
});
