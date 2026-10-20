import type { Tensor } from '../../src';

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
  const data = await tensor.getData();
  const actual = tensor.dtype === 'float16' ? new Float16Array(data) : new Float32Array(data);

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
