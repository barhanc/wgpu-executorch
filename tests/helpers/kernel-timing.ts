import { WgpuExecutionContext } from '../../src';
import type { WgpuDispatch } from '../../src/command';

/** Operator-only benchmark context with optional GPU timestamps per submission. */
export class KernelTimingContext extends WgpuExecutionContext {
  readonly #dispatches: WgpuDispatch[] = [];
  readonly #queries?: GPUQuerySet;
  readonly #resolve?: GPUBuffer;
  readonly #readback?: GPUBuffer;
  gpuMs = 0;

  /**
   * @param device Device with optional timestamp-query support enabled.
   */
  constructor(device: GPUDevice) {
    super(device);
    if (device.features.has('timestamp-query')) {
      this.#queries = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.#resolve = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
      this.#readback = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
    }
  }

  /**
   * @param dispatch Dispatch recorded once before warmup and measurement.
   */
  override addDispatch(dispatch: WgpuDispatch): void {
    super.addDispatch(dispatch);
    this.#dispatches.push(dispatch);
  }

  /** Submit the fixed operator sequence, measuring only its compute pass. */
  override submit(): void {
    if (!this.#queries) return super.submit();
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass({
      timestampWrites: {
        querySet: this.#queries,
        beginningOfPassWriteIndex: 0,
        endOfPassWriteIndex: 1,
      },
    });
    for (const dispatch of this.#dispatches) {
      pass.setPipeline(dispatch.pipeline);
      pass.setBindGroup(0, dispatch.bindGroup);
      pass.dispatchWorkgroups(
        dispatch.workgroupCountX,
        dispatch.workgroupCountY,
        dispatch.workgroupCountZ ?? 1
      );
    }
    pass.end();
    encoder.resolveQuerySet(this.#queries, 0, 2, this.#resolve!, 0);
    encoder.copyBufferToBuffer(this.#resolve!, 0, this.#readback!, 0, 16);
    this.device.queue.submit([encoder.finish()]);
  }

  /** Wait for completion and read the GPU duration, in milliseconds. */
  override async sync(): Promise<void> {
    await super.sync();
    if (!this.#readback) return;
    await this.#readback.mapAsync(GPUMapMode.READ);
    const times = new BigUint64Array(this.#readback.getMappedRange());
    this.gpuMs = Number(times[1] - times[0]) / 1e6;
    this.#readback.unmap();
  }

  /** Release the timestamp resources and context buffers. */
  override destroy(): void {
    this.#queries?.destroy();
    this.#resolve?.destroy();
    this.#readback?.destroy();
    super.destroy();
  }
}
