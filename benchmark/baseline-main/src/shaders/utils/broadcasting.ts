/**
 * Checks whether an input shape is broadcastable to the target output shape.
 * Follows NumPy / PyTorch broadcasting semantics:
 * - Dimensions are compared from right to left (trailing dimensions first).
 * - Two dimensions are compatible if they are equal or if one of them is 1.
 * - Input rank cannot exceed the target rank.
 *
 * @param shape The shape to check.
 * @param targetShape The destination broadcast shape.
 * @returns True if `shape` can be broadcast to `targetShape`.
 */
export function isBroadcastable(shape: readonly number[], targetShape: readonly number[]): boolean {
  if (shape.length > targetShape.length) {
    return false;
  }
  for (let i = 0; i < shape.length; i++) {
    const dim = shape[shape.length - 1 - i];
    const targetDim = targetShape[targetShape.length - 1 - i];
    if (dim !== 1 && dim !== targetDim) {
      return false;
    }
  }
  return true;
}

/**
 * Computes the broadcasted output shape of two tensor shapes.
 * Returns undefined if the shapes are incompatible under NumPy / PyTorch broadcasting rules.
 *
 * @param a First shape.
 * @param b Second shape.
 * @returns Broadcasted shape array, or undefined if incompatible.
 */
export function computeBroadcastShape(
  a: readonly number[],
  b: readonly number[]
): number[] | undefined {
  const maxRank = Math.max(a.length, b.length);
  const outShape = new Array<number>(maxRank);

  for (let i = 0; i < maxRank; i++) {
    const dimA = a[a.length - 1 - i] ?? 1;
    const dimB = b[b.length - 1 - i] ?? 1;

    if (dimA !== 1 && dimB !== 1 && dimA !== dimB) {
      return undefined;
    }
    outShape[maxRank - 1 - i] = Math.max(dimA, dimB);
  }

  return outShape;
}
