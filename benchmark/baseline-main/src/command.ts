/**
 * A buffer binding specification for constructing bind groups.
 */
export type BufferBindingSpec = {
  /** The shader binding index corresponding to `@binding(n)` in WGSL. */
  readonly binding: number;
  /** The underlying WebGPU buffer to bind. */
  readonly buffer: GPUBuffer;
  /** Optional byte offset from the start of the buffer (defaults to 0). */
  readonly offset?: number;
  /** Optional size in bytes of the buffer slice accessible to the shader. */
  readonly size?: number;
};

/**
 * A generic WebGPU resource binding specification (e.g. texture view, sampler).
 */
export type ResourceBindingSpec = {
  /** The shader binding index corresponding to `@binding(n)` in WGSL. */
  readonly binding: number;
  /** The WebGPU binding resource (texture view, sampler, external texture, etc.). */
  readonly resource: GPUBindingResource;
};

/**
 * Binding specification for constructing bind groups.
 */
export type BindingSpec = BufferBindingSpec | ResourceBindingSpec;

/**
 * Result of creating a compute pipeline and matching bind group.
 */
export type ComputePipelineBundle = {
  /** The compiled or cached WebGPU compute pipeline. */
  readonly pipeline: GPUComputePipeline;
  /** The bind group pre-configured to match the pipeline's layout. */
  readonly bindGroup: GPUBindGroup;
};

/**
 * A recorded GPU compute dispatch command.
 */
export type WgpuDispatch = {
  /** The compute pipeline to bind for execution. */
  readonly pipeline: GPUComputePipeline;
  /** The bind group containing bound buffers and uniforms. */
  readonly bindGroup: GPUBindGroup;
  /** Number of workgroups to dispatch in the X dimension. */
  readonly workgroupCountX: number;
  /** Number of workgroups to dispatch in the Y dimension. */
  readonly workgroupCountY: number;
  /** Optional number of workgroups to dispatch in the Z dimension (defaults to 1). */
  readonly workgroupCountZ?: number;
};

/**
 * A recorded GPU buffer-to-buffer copy command.
 */
export type WgpuCopy = {
  /** Source buffer to copy from. */
  readonly src: GPUBuffer;
  /** Destination buffer to copy to. */
  readonly dst: GPUBuffer;
  /** Number of bytes to copy. */
  readonly size: number;
};

/**
 * A recorded GPU command — either a compute dispatch or a buffer copy.
 */
export type WgpuCommand =
  | { readonly kind: 'dispatch'; readonly dispatch: WgpuDispatch }
  | { readonly kind: 'copy'; readonly copy: WgpuCopy };

const cache = new WeakMap<GPUDevice, Map<string, GPUComputePipeline>>();

/**
 * Serializes pipeline override constants into a deterministic,
 * order-independent cache key string.
 *
 * @param constants Pipeline override constants.
 * @returns Serialized key string.
 */
function serializeConstants(constants?: Record<string, number>): string {
  if (!constants) return '';

  const entries = Object.entries(constants);
  if (entries.length === 0) return '';

  return entries
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join(';');
}

/**
 * Builds and caches a compute pipeline along with its bind group from a list of
 * bindings.
 *
 * @param device WebGPU device instance.
 * @param code WGSL shader code string.
 * @param bindings List of buffer bindings for `@group(0)`.
 * @param constants Optional pipeline override constants.
 * @param entryPoint Shader entry point function name (default: 'main').
 * @returns The compiled pipeline and its matching bind group.
 */
export function createComputeBundle(
  device: GPUDevice,
  code: string,
  bindings: readonly BindingSpec[],
  constants: Record<string, number> = {},
  entryPoint: string = 'main'
): ComputePipelineBundle {
  let deviceMap = cache.get(device);
  if (!deviceMap) {
    deviceMap = new Map();
    cache.set(device, deviceMap);
  }

  const serializedConsts = serializeConstants(constants);
  const cacheKey = `${entryPoint}#${serializedConsts}#${code}`;

  let pipeline = deviceMap.get(cacheKey);
  if (!pipeline) {
    const module = device.createShaderModule({ code });
    const computeStage: GPUProgrammableStage = { module, entryPoint, constants };
    pipeline = device.createComputePipeline({ layout: 'auto', compute: computeStage });
    deviceMap.set(cacheKey, pipeline);
  }

  const bindGroupEntries: GPUBindGroupEntry[] = bindings.map((b) => {
    if ('buffer' in b) {
      return { binding: b.binding, resource: { buffer: b.buffer, offset: b.offset, size: b.size } };
    }
    return { binding: b.binding, resource: b.resource };
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindGroupEntries,
  });

  return { pipeline, bindGroup };
}
