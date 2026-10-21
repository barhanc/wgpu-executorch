import { MAX_NDIM } from '../../tensor';
import type { Tensor } from '../../tensor';

/**
 * Total byte size of the std140 TensorMeta uniform buffer (80 bytes).
 * Layout:
 * - 0..3:   ndim (u32)
 * - 4..7:   numel (u32)
 * - 8..15:  padding (8 bytes)
 * - 16..47: sizes (8 x u32, packaged as 2 x vec4<u32>)
 * - 48..79: strides (8 x u32, packaged as 2 x vec4<u32>)
 */
export const TENSOR_META_BYTE_SIZE = 80;

/**
 * Common WGSL struct definition for std140 TensorMeta uniform buffers.
 * Defines ndim, numel, and 8-element sizes/strides packaged as 2 x vec4<u32>.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/TensorMeta.h)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/TensorMeta.h
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
export const TENSOR_META_WGSL = /* wgsl */ `
struct TensorMeta {
  ndim: u32,
  numel: u32,
  sizes: array<vec4<u32>, 2>,
  strides: array<vec4<u32>, 2>,
}
`;

/** std140 32-bit word offsets */
const OFFSET_NDIM = 0;
const OFFSET_NUMEL = 1;
const OFFSET_SIZES = 4; // Byte offset 16 (16 / 4 = 4)
const OFFSET_STRIDES = 12; // Byte offset 48 (48 / 4 = 12)

/**
 * Encodes a Tensor's metadata into an 80-byte std140 ArrayBuffer matching ExecuTorch's
 * `TensorMeta` layout (ndim, numel, 8-element sizes, 8-element strides).
 *
 * If `targetRank` is provided, dimensions are right-aligned to match the broadcast output rank.
 *
 * @param tensor The Tensor instance to encode.
 * @param targetRank Optional target rank to right-align dimensions for broadcasting.
 * @returns ArrayBuffer containing the 80-byte std140 uniform data.
 */
export function encodeTensorMeta(
  tensor: Tensor,
  targetRank: number = tensor.shape.length
): ArrayBuffer {
  const shape = tensor.shape;

  if (shape.length > MAX_NDIM || targetRank > MAX_NDIM) {
    throw new Error(`Tensor rank exceeds maximum supported rank of ${MAX_NDIM}`);
  }
  if (shape.length > targetRank) {
    throw new Error(`Shape rank (${shape.length}) exceeds targetRank (${targetRank})`);
  }

  const buffer = new ArrayBuffer(TENSOR_META_BYTE_SIZE);
  const u32 = new Uint32Array(buffer);

  for (let i = 0; i < MAX_NDIM; i++) {
    u32[OFFSET_SIZES + i] = 1;
    u32[OFFSET_STRIDES + i] = 0;
  }

  u32[OFFSET_NDIM] = targetRank;
  u32[OFFSET_NUMEL] = tensor.numel;

  let stride = 1;
  for (let i = shape.length - 1; i >= 0; i--) {
    const slot = targetRank - shape.length + i;
    const dim = shape[i];
    u32[OFFSET_SIZES + slot] = dim;
    u32[OFFSET_STRIDES + slot] = stride;
    stride *= dim;
  }

  return buffer;
}
