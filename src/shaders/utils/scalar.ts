import type { Tensor } from '../../tensor';

/**
 * Float storage scalar type of a kernel's tensor buffers.
 *
 * Every float kernel is written once against `array<scalar_t>` storage and
 * f32 arithmetic: values are widened on load (`f32(x)`) and narrowed on
 * store (`scalar_t(x)`), so an f16 dispatch stores half-precision values
 * while all math (including accumulation) stays in f32.
 */
export type FloatKind = 'f32' | 'f16';

/** The WGSL type alias every float kernel source must declare. */
export const SCALAR_ALIAS = 'alias scalar_t = f32;';

/**
 * Generates the WGSL source variants of a float kernel — one per
 * {@link FloatKind}. The source must declare `alias scalar_t = f32;`; the f16
 * variant swaps the alias for `f16` and enables the `f16` shader extension
 * (requires the `shader-f16` device feature).
 *
 * @param source Kernel source with `alias scalar_t = f32;`.
 * @returns Source variants keyed by float kind.
 */
export function scalarCode(source: string): Record<FloatKind, string> {
  if (!source.includes(SCALAR_ALIAS)) {
    throw new Error(`kernel source is missing '${SCALAR_ALIAS}'`);
  }
  return {
    f32: source,
    f16: `enable f16;\n${source.replace(SCALAR_ALIAS, 'alias scalar_t = f16;')}`,
  };
}

/**
 * Flattens per-variant {@link scalarCode} maps into `variant:kind` keys for a
 * {@link Shader} `code` dictionary.
 *
 * @param variants Named kernel variants, each with f32/f16 sources.
 * @returns Flat map keyed `<variant>:<kind>`.
 */
export function variantCode(
  variants: Record<string, Record<FloatKind, string>>
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [variant, codes] of Object.entries(variants)) {
    for (const [kind, source] of Object.entries(codes)) {
      out[`${variant}:${kind}`] = source;
    }
  }
  return out;
}

/**
 * Validates that every given tensor is a float tensor and that all share one
 * float storage type, returning the {@link FloatKind} to dispatch with.
 *
 * @param opName Operator name for error messages.
 * @param tensors Tensors taking part in the dispatch (undefined entries skipped).
 * @returns The shared float kind of the tensors.
 * @throws {Error} If a tensor is not float32/float16 or the kinds are mixed.
 */
export function floatKind(opName: string, tensors: readonly (Tensor | undefined)[]): FloatKind {
  let kind: FloatKind | undefined;

  for (const tensor of tensors) {
    if (tensor === undefined) {
      continue;
    }
    if (tensor.dtype !== 'float32' && tensor.dtype !== 'float16') {
      throw new Error(
        `${opName}: only float32/float16 tensors are supported (got ${tensor.dtype})`
      );
    }
    const tensorKind: FloatKind = tensor.dtype === 'float16' ? 'f16' : 'f32';
    if (kind !== undefined && kind !== tensorKind) {
      throw new Error(`${opName}: mixed float dtypes (${kind} and ${tensorKind}) in one dispatch`);
    }
    kind = tensorKind;
  }

  if (kind === undefined) {
    throw new Error(`${opName}: expected at least one tensor`);
  }
  return kind;
}
