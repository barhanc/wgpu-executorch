/**
 * IEEE binary16 (f16) <-> float32 conversion helpers for tests.
 *
 * The runtime stores FP16 tensors as raw binary16 values; tests convert with
 * round-to-nearest-even on write (matching PyTorch's `.half()`) and exact
 * expansion on read.
 */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/**
 * Converts a float32 value to its IEEE binary16 bit pattern
 * (round-to-nearest-even, ties to even).
 *
 * @param value The float32 value.
 * @returns The 16-bit binary16 representation as an unsigned integer.
 */
export function f32ToF16Bits(value: number): number {
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const abs = x & 0x7fffffff;

  if (abs > 0x7f800000) return 0x7e00; // NaN -> quiet NaN
  if (abs >= 0x47800000) return sign | 0x7c00; // >= 2^16 -> Infinity
  if (abs === 0) return sign; // +/-0

  if (abs < 0x38800000) {
    // |x| < 2^-14: binary16 subnormal or zero. Round |x| in units of 2^-24.
    const scaled = Math.abs(value) * 2 ** 24;
    let k = Math.floor(scaled);
    const frac = scaled - k;
    if (frac > 0.5 || (frac === 0.5 && (k & 1) === 1)) {
      k += 1;
    }
    return sign | (k >= 0x400 ? 0x400 : k);
  }

  // Normal range: round the 24-bit significand to 11 bits (RNE).
  const e = abs >>> 23;
  let out = (abs & 0x7fffff) + 0x800000;
  const rem = out & 0x1fff;
  out = out >>> 13;
  if (rem > 0x1000 || (rem === 0x1000 && (out & 1) === 1)) {
    out += 1;
  }
  let hExp = e - 112;
  if (out >= 0x800) {
    out = out >>> 1;
    hExp += 1;
  }
  if (hExp >= 0x1f) {
    return sign | 0x7c00;
  }
  return sign | (hExp << 10) | (out & 0x3ff);
}

/**
 * Converts an IEEE binary16 bit pattern to its exact float32 value.
 *
 * @param bits The 16-bit binary16 representation.
 * @returns The expanded float32 value.
 */
export function f16BitsToF32(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exp = (bits >>> 10) & 0x1f;
  const mant = bits & 0x3ff;
  if (exp === 0x1f) {
    return mant ? NaN : sign * Infinity;
  }
  if (exp === 0) {
    return sign * mant * 2 ** -24;
  }
  return sign * (1024 + mant) * 2 ** (exp - 25);
}

/**
 * Converts a float32 array to binary16 bit patterns for tensor upload.
 *
 * @param values Source float32 values.
 * @returns Binary16 bits as a Uint16Array.
 */
export function f32ToF16Array(values: ArrayLike<number>): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) {
    out[i] = f32ToF16Bits(values[i]);
  }
  return out;
}

/**
 * Converts raw binary16 tensor data to float32 values.
 *
 * @param bits Binary16 values (e.g. from `Tensor.getData()`).
 * @returns Expanded float32 values.
 */
export function f16BitsToF32Array(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  for (let i = 0; i < bits.length; i++) {
    out[i] = f16BitsToF32(bits[i]);
  }
  return out;
}
