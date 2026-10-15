import type { Tensor } from '../../src';

import { f16BitsToF32Array } from './f16';

/**
 * Reads a tensor from the GPU into float32 values.
 *
 * float16 tensors are expanded exactly; all other dtypes are reinterpreted
 * as float32 (the dtype under test in the float kernel suites).
 *
 * @param tensor The tensor to read.
 * @returns The tensor's values as a Float32Array.
 */
export async function readTensorF32(tensor: Tensor): Promise<Float32Array> {
  const raw = await tensor.getData();
  if (tensor.dtype === 'float16') {
    return f16BitsToF32Array(new Uint16Array(raw));
  }
  return new Float32Array(raw);
}

/**
 * Reads a tensor from the GPU and asserts its values are close to expected.
 *
 * @param tensor The output tensor to check.
 * @param expected Expected values (flattened, row-major).
 * @param tolerance Maximum absolute difference per element (default: 1e-5).
 */
export async function expectTensorClose(
  tensor: Tensor,
  expected: number[],
  tolerance = 1e-5
): Promise<void> {
  const actual = await readTensorF32(tensor);

  if (actual.length !== expected.length) {
    throw new Error(`Tensor length mismatch: got ${actual.length}, expected ${expected.length}`);
  }

  let maxDiff = 0;
  let maxIdx = 0;
  for (let i = 0; i < actual.length; i++) {
    const diff = Math.abs(actual[i] - expected[i]);
    if (diff > maxDiff) {
      maxDiff = diff;
      maxIdx = i;
    }
  }

  if (maxDiff > tolerance) {
    const a = actual[maxIdx];
    const e = expected[maxIdx];
    throw new Error(
      `Tensor mismatch at index ${maxIdx}: got ${a}, expected ${e} ` +
        `(max diff ${maxDiff.toExponential(3)}, tolerance ${tolerance})`
    );
  }
}
