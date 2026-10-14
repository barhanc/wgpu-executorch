/**
 * Supported tensor data types in WebGPU storage buffers.
 */
export type DType =
  'float32' | 'float16' | 'int32' | 'int8' | 'uint8' | 'bool' | 'float64' | 'int64';

/**
 * Supported typed arrays that can be uploaded to a Tensor.
 */
export type TypedArray =
  Float32Array | Int32Array | Int8Array | Uint8Array | Uint16Array | Float64Array | BigInt64Array;

/**
 * Maximum tensor rank supported by WebGPU kernels (std140 layout limit).
 */
export const MAX_NDIM = 8;

// prettier-ignore
export const DTYPE_BYTESIZE: Record<DType, number> = {
  float32: 4, float16: 2, float64: 8,
  int32:   4, int64:   8,
  int8:    1, uint8:   1, bool:    1,
} as const;

const STAGING_BUFFER_USAGE = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST;

/**
 * A non-owning WebGPU-backed Tensor view.
 *
 * Encapsulates an interpretation of an allocated `GPUBuffer` in device VRAM
 * along with its metadata (`dtype`, `shape`, `numel`, `byteLength`). Use
 * {@link Tensor.getData} to read raw bytes back into host memory and
 * {@link Tensor.setData} to upload host data into it. The underlying
 * `GPUBuffer` lifetime is managed externally.
 */
export class Tensor {
  readonly #dtype: DType;
  readonly #shape: readonly number[];
  readonly #numel: number;
  readonly #buffer: GPUBuffer;
  readonly #device: GPUDevice;

  constructor(dtype: DType, shape: readonly number[], device: GPUDevice, buffer: GPUBuffer) {
    if (shape.length > MAX_NDIM) {
      throw new Error(`Tensor rank ${shape.length} exceeds maximum rank of ${MAX_NDIM}`);
    }
    if (shape.some((dim) => dim <= 0 || !Number.isInteger(dim))) {
      throw new Error('Tensor dimensions must be positive integers');
    }

    this.#dtype = dtype;
    this.#shape = shape;
    this.#numel = shape.reduce((a, b) => a * b, 1);
    this.#buffer = buffer;
    this.#device = device;

    const requiredBytes = this.#numel * DTYPE_BYTESIZE[dtype];
    if (buffer.size < requiredBytes) {
      throw new Error(`GPUBuffer size (${buffer.size}B) < required (${requiredBytes}B)`);
    }
    if ((buffer.usage & GPUBufferUsage.STORAGE) === 0) {
      throw new Error('GPUBuffer must have GPUBufferUsage.STORAGE flag');
    }
  }

  /** The element data type of the tensor. */
  get dtype(): DType {
    return this.#dtype;
  }

  /** Total number of elements across all dimensions. */
  get numel(): number {
    return this.#numel;
  }

  /** The dimensions of the tensor. */
  get shape(): readonly number[] {
    return this.#shape;
  }

  /** The underlying WebGPU storage buffer in VRAM. */
  get buffer(): GPUBuffer {
    return this.#buffer;
  }

  /** The WebGPU device that owns this tensor's buffer. */
  get device(): GPUDevice {
    return this.#device;
  }

  /** The total byte size of the tensor data. */
  get byteLength(): number {
    return this.#numel * DTYPE_BYTESIZE[this.#dtype];
  }

  /**
   * Uploads data from a host TypedArray or ArrayBuffer into this tensor's
   * storage buffer.
   *
   * @param src Source data to copy from.
   * @returns This tensor instance.
   * @throws {Error} If source byte size is smaller than this tensor's byte length.
   */
  setData(src: TypedArray | ArrayBuffer): this {
    const byteLength = this.byteLength;

    if ((this.#buffer.usage & GPUBufferUsage.COPY_DST) === 0) {
      throw new Error('setData: buffer missing COPY_DST usage');
    }
    if (src.byteLength < byteLength) {
      throw new Error(`setData: source (${src.byteLength}B) < required (${byteLength}B)`);
    }

    const alignedSize = Math.ceil(byteLength / 4) * 4;
    const bytes = new Uint8Array(alignedSize);
    bytes.set(
      src instanceof ArrayBuffer
        ? new Uint8Array(src, 0, byteLength)
        : new Uint8Array(src.buffer, src.byteOffset, byteLength)
    );

    this.#device.queue.writeBuffer(this.#buffer, 0, bytes.buffer, bytes.byteOffset, alignedSize);
    return this;
  }

  /**
   * Reads raw bytes back from the GPU storage buffer into host memory.
   *
   * @returns Raw byte buffer containing the tensor data.
   */
  async getData(): Promise<ArrayBuffer> {
    const byteLength = this.byteLength;
    const copySize = Math.ceil(byteLength / 4) * 4;
    const stagingSize = Math.max(16, copySize);

    const encoder = this.#device.createCommandEncoder();
    const staging = this.#device.createBuffer({ size: stagingSize, usage: STAGING_BUFFER_USAGE });

    try {
      encoder.copyBufferToBuffer(this.#buffer, 0, staging, 0, copySize);
      this.#device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      return staging.getMappedRange(0, stagingSize).slice(0, byteLength);
    } finally {
      staging.destroy();
    }
  }
}
