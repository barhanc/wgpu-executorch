/**
 * Packing utilities for HWC4 layout.
 *
 * Pack weights from standard ExecuTorch/PyTorch OIHW format to HWC4 matrix tiles:
 * - Dense Conv: [OC, IC, KH, KW] -> [OC4, IC4, KH*KW, 4, 4]
 * - Depthwise Conv: [C, 1, KH, KW] -> [C4, KH*KW, 4]
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */

import type { TypedArray } from '../src/tensor';

/**
 * Packs dense conv2d weights into HWC4 matrix tiles of 4x4.
 * Each tile holds 4 vec4s: for input lane iL in 0..3, tile[iL] holds outputs for 4 output lanes oL in 0..3.
 *
 * @param src Source tensor buffer or typed array.
 * @param cOut Number of output channels.
 * @param cIn Number of input channels.
 * @param kH Kernel height.
 * @param kW Kernel width.
 * @returns Packed typed array with HWC4 layout.
 */
export function packConvHwc4F16<T extends TypedArray>(
  src: T,
  cOut: number,
  cIn: number,
  kH: number,
  kW: number
): T {
  const kk = kH * kW;
  const ob = Math.ceil(cOut / 4);
  const ib = Math.ceil(cIn / 4);
  const ctor = (src as unknown as { constructor: new (len: number) => T }).constructor;
  const dst = new ctor(ob * ib * kk * 16);
  for (let o = 0; o < cOut; o++) {
    for (let i = 0; i < cIn; i++) {
      for (let t = 0; t < kk; t++) {
        const tile = ((o >> 2) * ib + (i >> 2)) * kk + t;
        dst[tile * 16 + (i & 3) * 4 + (o & 3)] = src[(o * cIn + i) * kk + t];
      }
    }
  }
  return dst;
}

/**
 * Packs depthwise conv2d weights into HWC4 vec4 vectors.
 *
 * @param src Source tensor buffer or typed array.
 * @param c Number of channels.
 * @param kH Kernel height.
 * @param kW Kernel width.
 * @returns Packed typed array with HWC4 layout.
 */
export function packDwHwc4F16<T extends TypedArray>(src: T, c: number, kH: number, kW: number): T {
  const kk = kH * kW;
  const cb = Math.ceil(c / 4);
  const ctor = (src as unknown as { constructor: new (len: number) => T }).constructor;
  const dst = new ctor(cb * kk * 4);
  for (let ch = 0; ch < c; ch++) {
    for (let t = 0; t < kk; t++) {
      dst[((ch >> 2) * kk + t) * 4 + (ch & 3)] = src[ch * kk + t];
    }
  }
  return dst;
}

/**
 * Packs dense conv2d weights in float32.
 *
 * @param src Source tensor buffer or typed array.
 * @param cOut Number of output channels.
 * @param cIn Number of input channels.
 * @param kH Kernel height.
 * @param kW Kernel width.
 * @returns Packed Float32Array with HWC4 layout.
 */
export function packConvHwc4F32(
  src: ArrayLike<number>,
  cOut: number,
  cIn: number,
  kH: number,
  kW: number
): Float32Array {
  const kk = kH * kW;
  const ob = Math.ceil(cOut / 4);
  const ib = Math.ceil(cIn / 4);
  const dst = new Float32Array(ob * ib * kk * 16);
  for (let o = 0; o < cOut; o++) {
    for (let i = 0; i < cIn; i++) {
      for (let t = 0; t < kk; t++) {
        const tile = ((o >> 2) * ib + (i >> 2)) * kk + t;
        dst[tile * 16 + (i & 3) * 4 + (o & 3)] = src[(o * cIn + i) * kk + t];
      }
    }
  }
  return dst;
}

/**
 * Packs depthwise conv2d weights in float32.
 *
 * @param src Source tensor buffer or typed array.
 * @param c Number of channels.
 * @param kH Kernel height.
 * @param kW Kernel width.
 * @returns Packed Float32Array with HWC4 layout.
 */
export function packDwHwc4F32(
  src: ArrayLike<number>,
  c: number,
  kH: number,
  kW: number
): Float32Array {
  const kk = kH * kW;
  const cb = Math.ceil(c / 4);
  const dst = new Float32Array(cb * kk * 4);
  for (let ch = 0; ch < c; ch++) {
    for (let t = 0; t < kk; t++) {
      dst[((ch >> 2) * kk + t) * 4 + (ch & 3)] = src[ch * kk + t];
    }
  }
  return dst;
}
