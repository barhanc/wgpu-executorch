import type { Shader } from './shader';
import type { Program } from './program';
import type { WgpuDispatch, WgpuCommand } from './command';

import { Tensor, DTYPE_BYTESIZE, type DType } from './tensor';
import { resolveArgs } from './program';
import { shaderRegistry } from './shaders';

const UNIFORM_BUFFER_USAGE = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
const STORAGE_BUFFER_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

/**
 * Execution context that manages GPU buffers and the sequence of recorded
 * kernel dispatches.
 */
export class WgpuExecutionContext {
  readonly #device: GPUDevice;
  readonly #commands: WgpuCommand[] = [];
  readonly #ownedBuffers = new Set<GPUBuffer>();
  #destroyed = false;

  /**
   * Constructs a new WgpuExecutionContext.
   *
   * @param device The WebGPU device instance.
   */
  constructor(device: GPUDevice) {
    this.#device = device;
  }

  /** The WebGPU device used by this context. */
  get device(): GPUDevice {
    return this.#device;
  }

  /**
   * Transfers ownership of a buffer to the context to ensure it remains
   * allocated for the model lifetime and is destroyed on context disposal.
   *
   * @param buffer The GPUBuffer to take ownership of.
   */
  #ownBuffer(buffer: GPUBuffer): void {
    this.#ownedBuffers.add(buffer);
  }

  /**
   * Throws if this context has been destroyed.
   *
   * @throws {Error} If {@link destroy} has been called.
   */
  #assertNotDestroyed(): void {
    if (this.#destroyed) {
      throw new Error('WgpuExecutionContext is destroyed');
    }
  }

  /**
   * Destroys all owned buffers and releases resources.
   */
  destroy(): void {
    if (this.#destroyed) {
      return;
    }
    this.#destroyed = true;

    for (const buffer of this.#ownedBuffers) {
      buffer.destroy();
    }
    this.#ownedBuffers.clear();
    this.#commands.length = 0;
  }

  // =========================================================================
  // Buffer Allocation
  // =========================================================================

  /**
   * Creates an empty WebGPU storage buffer whose lifetime is managed by this
   * context.
   *
   * @param size Size in bytes to allocate.
   * @returns The newly allocated and owned GPUBuffer with STORAGE | COPY_SRC |
   * COPY_DST usage.
   */
  storageBuffer(size: number): GPUBuffer {
    this.#assertNotDestroyed();
    const alignedSize = Math.max(4, Math.ceil(size / 4) * 4);
    const buffer = this.#device.createBuffer({ size: alignedSize, usage: STORAGE_BUFFER_USAGE });
    this.#ownBuffer(buffer);
    return buffer;
  }

  /**
   * Creates and populates a WebGPU uniform buffer whose lifetime is managed by
   * this context.
   *
   * @param data Binary data to copy into the uniform buffer.
   * @returns The newly allocated and owned GPUBuffer.
   */
  uniformBuffer(data: ArrayBufferView | ArrayBuffer): GPUBuffer {
    this.#assertNotDestroyed();

    const byteLength = data.byteLength;
    const alignedSize = Math.max(16, Math.ceil(byteLength / 4) * 4);
    const buffer = this.#device.createBuffer({ size: alignedSize, usage: UNIFORM_BUFFER_USAGE });

    if (data instanceof ArrayBuffer) {
      this.#device.queue.writeBuffer(buffer, 0, data, 0, byteLength);
    } else {
      this.#device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, byteLength);
    }
    this.#ownBuffer(buffer);
    return buffer;
  }

  /**
   * Creates a Tensor view over a GPU buffer.
   *
   * Buffer ownership behavior:
   * 1. **`buffer` omitted**: The context allocates a fresh `GPUBuffer` via
   *    {@link storageBuffer}. This buffer is tracked and owned by the context,
   *    and will be destroyed when {@link destroy} is called on this context.
   * 2. **`buffer` provided**: The context wraps the given `GPUBuffer` without
   *    altering its lifecycle or ownership. If it was already managed (e.g. via
   *    {@link storageBuffer}), it remains managed; if it is external, its
   *    lifetime remains with the caller.
   *
   * @param dtype Element data type of the tensor.
   * @param shape Tensor dimensions.
   * @param buffer Optional existing GPUBuffer to wrap. If omitted, the context
   * allocates and owns a new buffer.
   * @returns A new Tensor view.
   */
  tensor(dtype: DType, shape: readonly number[], buffer?: GPUBuffer): Tensor {
    this.#assertNotDestroyed();

    if (buffer !== undefined) {
      return new Tensor(dtype, shape, this.#device, buffer);
    }

    const numel = shape.reduce((a, b) => a * b, 1);
    const byteLength = numel * DTYPE_BYTESIZE[dtype];
    const newBuffer = this.storageBuffer(byteLength);

    return new Tensor(dtype, shape, this.#device, newBuffer);
  }

  // =========================================================================
  // Dispatch Recording & Execution
  // =========================================================================

  /**
   * Records a compute dispatch into the context execution queue.
   *
   * @param dispatch The dispatch descriptor to record.
   */
  addDispatch(dispatch: WgpuDispatch): void {
    this.#assertNotDestroyed();
    this.#commands.push({ kind: 'dispatch', dispatch });
  }

  /**
   * Records a deferred GPU buffer-to-buffer copy into the context execution
   * queue. The copy is encoded and submitted when {@link submit} is called,
   * preserving ordering relative to recorded dispatches.
   *
   * @param src Source tensor to copy from.
   * @param dst Destination tensor to copy to.
   * @returns This context instance.
   */
  copy(src: Tensor, dst: Tensor): this {
    this.#assertNotDestroyed();
    if ((src.buffer.usage & GPUBufferUsage.COPY_SRC) === 0) {
      throw new Error('copy: source buffer missing COPY_SRC usage');
    }
    if ((dst.buffer.usage & GPUBufferUsage.COPY_DST) === 0) {
      throw new Error('copy: destination buffer missing COPY_DST usage');
    }
    if (dst.byteLength < src.byteLength) {
      throw new Error(`copy: destination (${dst.byteLength}B) < source (${src.byteLength}B)`);
    }
    this.#commands.push({
      kind: 'copy',
      copy: { src: src.buffer, dst: dst.buffer, size: src.byteLength },
    });
    return this;
  }

  /**
   * Records a compute shader dispatch into this context.
   *
   * @param shader The compute shader to execute.
   * @param args Positional arguments expected by the operator.
   * @returns This context instance.
   */
  recordShader<TArgs extends readonly unknown[]>(shader: Shader<TArgs>, args: TArgs): this {
    this.#assertNotDestroyed();
    shader.recordIn(this, args);
    return this;
  }

  /**
   * Records a full serialized program — allocates buffers, uploads weights,
   * copies inputs, dispatches the operator chain, and copies outputs.
   *
   * All work is deferred until {@link submit} is called.
   *
   * @param program The serialized program descriptor (program.json).
   * @param weights Raw constant tensor data (weights.bin).
   * @param args I/O tensors and scalars: `[in1, ..., out1, ...]` matching
   * `input_ids` and `output_ids`.
   * @returns This context instance.
   */
  recordProgram(
    program: Program,
    weights: ArrayBuffer,
    args: readonly (Tensor | number | boolean | string)[]
  ): this {
    this.#assertNotDestroyed();

    const numInputs = program.inputIds.length;
    const numOutputs = program.outputIds.length;
    if (args.length !== numInputs + numOutputs) {
      throw new Error(`program: expected ${numInputs + numOutputs} args, got ${args.length}`);
    }

    // Allocate shared pool buffers
    const pools = new Map<number, GPUBuffer>();
    for (const pool of program.memoryPlan.pools) {
      pools.set(pool.id, this.storageBuffer(pool.size));
    }

    // Allocate buffers and create Tensor views for all values
    const tensors = new Map<number, Tensor>();
    for (const [i, v] of program.values.entries()) {
      if (v.type !== 'tensor') continue;

      let buffer: GPUBuffer;

      if (v.weightsOffset !== undefined && v.weightsLength !== undefined) {
        // Constant — dedicated buffer
        buffer = this.storageBuffer(v.weightsLength);
      } else if (v.memObjId !== undefined) {
        // Intermediate - shared pool buffer
        if (!pools.has(v.memObjId)) {
          throw new Error(`program: no memory pool ${v.memObjId} for value ${i}`);
        }
        buffer = pools.get(v.memObjId)!;
      } else {
        // Dedicated buffer - no pool assignment
        const numel = v.shape.reduce((a, b) => a * b, 1);
        buffer = this.storageBuffer(numel * DTYPE_BYTESIZE[v.dtype]);
      }

      tensors.set(i, this.tensor(v.dtype, v.shape, buffer));
    }

    // Upload weights into constant tensors
    for (const [i, v] of program.values.entries()) {
      if (
        v.type === 'tensor' && // prettier-ignore
        v.weightsOffset !== undefined &&
        v.weightsLength !== undefined
      ) {
        const t = tensors.get(i);
        const w = weights.slice(v.weightsOffset, v.weightsOffset + v.weightsLength);
        if (t === undefined) {
          throw new Error(`program: no tensor for constant value ${i}`);
        }
        t.setData(w);
      }
    }

    // Copy inputs: user args → internal tensors
    for (let i = 0; i < numInputs; i++) {
      const src = args[i];
      const dst = tensors.get(program.inputIds[i]);

      if (!(src instanceof Tensor)) {
        continue;
      }
      if (dst === undefined) {
        throw new Error(`program: no tensor for input value ${program.inputIds[i]}`);
      }
      this.copy(src, dst);
    }

    // Dispatch operator chain
    for (const op of program.chain) {
      const shader = shaderRegistry[op.name] as Shader | undefined;
      if (!shader) {
        throw new Error(`program: unknown operator '${op.name}'`);
      }
      this.recordShader(shader, resolveArgs(op, program.values, tensors));
    }

    // Copy outputs: internal tensors → user args
    for (let i = 0; i < numOutputs; i++) {
      const src = tensors.get(program.outputIds[i]);
      const dst = args[numInputs + i];

      if (!(dst instanceof Tensor)) {
        throw new Error(`program: output ${i} must be a Tensor`);
      }
      if (src === undefined) {
        throw new Error(`program: no tensor for output value ${program.outputIds[i]}`);
      }
      this.copy(src, dst);
    }

    return this;
  }

  /**
   * Encodes all recorded commands (dispatches and copies) into a single
   * command buffer and submits to the GPU queue.
   */
  submit(): void {
    this.#assertNotDestroyed();
    if (this.#commands.length === 0) {
      return;
    }

    const encoder = this.#device.createCommandEncoder();
    let pass: GPUComputePassEncoder | undefined;

    for (const cmd of this.#commands) {
      switch (cmd.kind) {
        case 'copy': {
          if (pass) {
            pass.end();
            pass = undefined;
          }
          encoder.copyBufferToBuffer(cmd.copy.src, 0, cmd.copy.dst, 0, cmd.copy.size);
          break;
        }

        case 'dispatch': {
          if (!pass) {
            pass = encoder.beginComputePass();
          }
          const d = cmd.dispatch;
          pass.setPipeline(d.pipeline);
          pass.setBindGroup(0, d.bindGroup);
          pass.dispatchWorkgroups(d.workgroupCountX, d.workgroupCountY, d.workgroupCountZ ?? 1);
          break;
        }

        default: {
          const exhaustive: never = cmd;
          throw new Error(`Unhandled command: ${JSON.stringify(exhaustive)}`);
        }
      }
    }

    if (pass) {
      pass.end();
    }
    this.#device.queue.submit([encoder.finish()]);
  }

  /**
   * Waits for all previously submitted GPU work to complete.
   *
   * @returns A promise that resolves when the GPU queue is idle.
   */
  async sync(): Promise<void> {
    this.#assertNotDestroyed();
    await this.#device.queue.onSubmittedWorkDone();
  }

  /**
   * Clears all recorded commands to prepare for a fresh recording pass while
   * keeping owned buffers intact.
   */
  reset(): void {
    this.#assertNotDestroyed();
    this.#commands.length = 0;
  }
}
