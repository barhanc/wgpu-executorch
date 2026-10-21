// Runs only inside a dedicated profiling page. Hooks are restored after each run;
// the example and runtime do not acquire profiling overhead or new API contracts.
export async function profileLivePipeline(options) {
  const frames = [];
  const pending = [];
  const resources = [];
  let frame = null;
  let state = null;
  let encoding = false;
  let stop;
  let timer;
  let stream;
  const errors = [];
  const originals = {
    requestDevice: GPUAdapter.prototype.requestDevice,
    createEncoder: GPUDevice.prototype.createCommandEncoder,
    createTexture: GPUDevice.prototype.createTexture,
    configure: GPUCanvasContext.prototype.configure,
    copyTexture: GPUCommandEncoder.prototype.copyTextureToTexture,
    drawImage: OffscreenCanvasRenderingContext2D.prototype.drawImage,
    upload: GPUQueue.prototype.copyExternalImageToTexture,
    map: GPUBuffer.prototype.mapAsync,
  };
  const { WgpuExecutionContext } = await import('/@fs' + options.repo + '/src/index.ts');
  originals.submit = WgpuExecutionContext.prototype.submit;
  originals.recordShader = WgpuExecutionContext.prototype.recordShader;
  const preferred = options.presentation === 'preferred';
  let presentation;
  try {
    GPUAdapter.prototype.requestDevice = async function (descriptor = {}) {
      if (!this.features.has('timestamp-query'))
        throw new Error('timestamp-query is required for GPU breakdowns');
      const device = await originals.requestDevice.call(this, {
        ...descriptor,
        requiredFeatures: [...new Set([...(descriptor.requiredFeatures ?? []), 'timestamp-query'])],
      });
      device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
      state = {
        device,
        gpu: { vendor: this.info.vendor, architecture: this.info.architecture },
        queries: device.createQuerySet({ type: 'timestamp', count: 6 }),
        resolve: device.createBuffer({
          size: 48,
          usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        }),
        readback: device.createBuffer({
          size: 48,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        passes: 0,
      };
      resources.push(state);
      return device;
    };
    GPUDevice.prototype.createCommandEncoder = function (descriptor) {
      const encoder = originals.createEncoder.call(this, descriptor);
      if (!encoding) return encoder;
      const current = state;
      const begin = encoder.beginComputePass.bind(encoder);
      const finish = encoder.finish.bind(encoder);
      encoder.beginComputePass = (descriptor = {}) => {
        const slot = current.passes++;
        if (slot >= 3)
          throw new Error('Unexpected compute pass layout; update profiler phase labels');
        return begin({
          ...descriptor,
          timestampWrites: {
            querySet: current.queries,
            beginningOfPassWriteIndex: slot * 2,
            endOfPassWriteIndex: slot * 2 + 1,
          },
        });
      };
      encoder.finish = (descriptor) => {
        if (current.passes !== 3)
          throw new Error('Expected preprocess / model / postprocess compute passes');
        encoder.resolveQuerySet(current.queries, 0, 6, current.resolve, 0);
        encoder.copyBufferToBuffer(current.resolve, 0, current.readback, 0, 48);
        return finish(descriptor);
      };
      return encoder;
    };
    if (preferred) {
      GPUCanvasContext.prototype.configure = function (descriptor) {
        return originals.configure.call(this, {
          ...descriptor,
          format: navigator.gpu.getPreferredCanvasFormat(),
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
      };
      GPUDevice.prototype.createTexture = function (descriptor) {
        const usage =
          descriptor.usage & GPUTextureUsage.STORAGE_BINDING
            ? descriptor.usage | GPUTextureUsage.TEXTURE_BINDING
            : descriptor.usage;
        return originals.createTexture.call(this, { ...descriptor, usage });
      };
      GPUCommandEncoder.prototype.copyTextureToTexture = function (source, destination, size) {
        // Only the example's final result-to-canvas copy is replaced.
        if (!(source.texture.usage & GPUTextureUsage.STORAGE_BINDING)) {
          return originals.copyTexture.call(this, source, destination, size);
        }
        if (!presentation) {
          const device = state.device;
          const module = device.createShaderModule({
            code: `
            @group(0) @binding(0) var image: texture_2d<f32>;
            @vertex fn vertex(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
              let positions = array<vec2<f32>, 3>(vec2(-1., -1.), vec2(3., -1.), vec2(-1., 3.));
              return vec4(positions[i], 0., 1.);
            }
            @fragment fn fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
              return textureLoad(image, vec2<i32>(p.xy), 0);
            }`,
          });
          const pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vertex' },
            fragment: {
              module,
              entryPoint: 'fragment',
              targets: [{ format: destination.texture.format }],
            },
          });
          const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: source.texture.createView() }],
          });
          presentation = { pipeline, bindGroup };
        }
        const pass = this.beginRenderPass({
          colorAttachments: [
            { view: destination.texture.createView(), loadOp: 'clear', storeOp: 'store' },
          ],
        });
        pass.setPipeline(presentation.pipeline);
        pass.setBindGroup(0, presentation.bindGroup);
        pass.draw(3);
        pass.end();
      };
    }
    OffscreenCanvasRenderingContext2D.prototype.drawImage = function (...args) {
      if (!(args[0] instanceof HTMLVideoElement)) return originals.drawImage.apply(this, args);
      const start = performance.now();
      frame = { startMs: start };
      const result = originals.drawImage.apply(this, args);
      frame.cropCpuMs = performance.now() - start;
      frames.push(frame);
      return result;
    };
    GPUQueue.prototype.copyExternalImageToTexture = function (...args) {
      const start = performance.now();
      const result = originals.upload.apply(this, args);
      if (frame) frame.uploadCallCpuMs = performance.now() - start;
      return result;
    };
    WgpuExecutionContext.prototype.submit = function () {
      const currentFrame = frame;
      const current = state;
      const start = performance.now();
      current.passes = 0;
      encoding = true;
      try {
        originals.submit.call(this);
      } finally {
        encoding = false;
      }
      currentFrame.encodeCpuMs = performance.now() - start;
      const read = current.readback.mapAsync(GPUMapMode.READ).then(() => {
        const t = [...new BigUint64Array(current.readback.getMappedRange())];
        current.readback.unmap();
        currentFrame.preprocessGpuMs = Number(t[1] - t[0]) / 1e6;
        currentFrame.modelGpuMs = Number(t[3] - t[2]) / 1e6;
        currentFrame.postprocessGpuMs = Number(t[5] - t[4]) / 1e6;
        currentFrame.modelCopyAndPassGapGpuMs = Number(t[2] - t[1] + (t[4] - t[3])) / 1e6;
        currentFrame.computeSpanGpuMs = Number(t[5] - t[0]) / 1e6;
      });
      pending.push(read);
    };
    GPUBuffer.prototype.mapAsync = function (...args) {
      const currentFrame = frame;
      const start = performance.now();
      const result = originals.map.apply(this, args);
      if (this.size !== 16 || !currentFrame) return result;
      currentFrame.cpuBeforeFenceMs = start - currentFrame.startMs;
      return result.then((value) => {
        currentFrame.frameMs = performance.now() - currentFrame.startMs;
        currentFrame.fenceWaitWallMs = performance.now() - start;
        return value;
      });
    };
    if (options.detections !== undefined) {
      WgpuExecutionContext.prototype.recordShader = function (shader, args) {
        if (shader.name === 'yolo.draw_detections') {
          originals.recordShader.call(
            this,
            {
              name: 'profile.force_detections',
              code: '',
              recordIn(ctx) {
                const module = ctx.device.createShaderModule({
                  code: `
              @group(0) @binding(0) var<storage, read_write> counter: array<u32>;
              @group(0) @binding(1) var<storage, read_write> dets: array<f32>;
              @compute @workgroup_size(64) fn main(@builtin(local_invocation_id) tid: vec3<u32>) {
                if (tid.x == 0u) { counter[0] = ${options.detections}u; }
                if (tid.x >= ${options.detections}u) { return; }
                let i = tid.x; let b = i * 6u;
                dets[b] = f32(16u + (i % 8u) * 64u); dets[b+1u] = f32(16u + (i / 8u) * 64u);
                dets[b+2u] = dets[b] + 100.; dets[b+3u] = dets[b+1u] + 120.;
                dets[b+4u] = .9; dets[b+5u] = f32(i);
              }`,
                });
                const pipeline = ctx.device.createComputePipeline({
                  layout: 'auto',
                  compute: { module, entryPoint: 'main' },
                });
                const bindGroup = ctx.device.createBindGroup({
                  layout: pipeline.getBindGroupLayout(0),
                  entries: [
                    { binding: 0, resource: { buffer: args[2] } },
                    { binding: 1, resource: { buffer: args[3] } },
                  ],
                });
                ctx.addDispatch({ pipeline, bindGroup, workgroupCountX: 1, workgroupCountY: 1 });
              },
            },
            []
          );
        }
        return originals.recordShader.call(this, shader, args);
      };
    }
    const source = document.createElement('canvas');
    source.width = 1280;
    source.height = 720;
    const paint = source.getContext('2d');
    let tick = 0;
    timer = setInterval(() => {
      paint.fillStyle = 'white';
      paint.fillRect(0, 0, 1280, 720);
      paint.fillStyle = 'black';
      paint.beginPath();
      paint.arc(640 + Math.sin(tick++ / 20) * 20, 360, 120, 0, Math.PI * 2);
      paint.fill();
    }, 33);
    stream = source.captureStream(30);
    const video = document.createElement('video');
    video.muted = true;
    video.autoplay = true;
    video.playsInline = true;
    video.srcObject = stream;
    const canvas = document.createElement('canvas');
    document.body.replaceChildren(video, canvas);
    await video.play();
    const { startFrameLoop } = await import(`/src/tasks/${options.task}/loop.ts`);
    let finish;
    const completed = new Promise((resolve) => {
      finish = resolve;
    });
    const total = (options.warmups ?? 30) + (options.samples ?? 80);
    let count = 0;
    stop = await startFrameLoop(
      video,
      canvas,
      () => {
        if (++count >= total) finish();
      },
      options.precision
    );
    let timeout;
    try {
      await Promise.race([
        completed,
        new Promise((resolve, reject) => {
          timeout = setTimeout(
            () => reject(new Error(`Only ${count}/${total} frames completed`)),
            30000
          );
        }),
      ]);
    } finally {
      clearTimeout(timeout);
    }
    stop();
    stop = null;
    await Promise.all(pending);
    if (errors.length) throw new Error(errors.join('\n'));
    const measured = frames
      .slice(options.warmups ?? 30)
      .filter((item) => item.frameMs !== undefined && item.computeSpanGpuMs !== undefined);
    for (const item of measured)
      item.outsideComputeSpanWallMs = item.frameMs - item.computeSpanGpuMs;
    const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
    const keys = Object.keys(measured[0]).filter((key) => key !== 'startMs');
    const configuration = { ...options };
    delete configuration.repo;
    delete configuration.reportEndpoint;
    const result = {
      ...configuration,
      gpu: state.gpu,
      browser: navigator.userAgent,
      source: '1280x720 animated canvas captured as a 30fps HTMLVideoElement',
      timing:
        'GPU timestamps on the actual three compute passes; CPU clocks for crop/upload calls, encoding and frame-fence completion. One additional timestamp readback alongside the existing frame fence adds profiling overhead. Medians are not additive because CPU and GPU work can overlap.',
      medians: Object.fromEntries(
        keys.map((key) => [key, median(measured.map((item) => item[key]))])
      ),
      frames: measured,
      note: 'outsideComputeSpanWallMs includes crop, upload/conversion, presentation, queue scheduling and browser callback latency; it is not all postprocessing work.',
      errors,
    };
    if (options.reportEndpoint) {
      const response = await fetch(options.reportEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result),
      });
      if (!response.ok) throw new Error(`Result upload failed: ${response.status}`);
    }
    return result;
  } finally {
    stop?.();
    clearInterval(timer);
    stream?.getTracks().forEach((track) => track.stop());
    await Promise.allSettled(pending);
    GPUAdapter.prototype.requestDevice = originals.requestDevice;
    GPUDevice.prototype.createCommandEncoder = originals.createEncoder;
    GPUDevice.prototype.createTexture = originals.createTexture;
    GPUCanvasContext.prototype.configure = originals.configure;
    GPUCommandEncoder.prototype.copyTextureToTexture = originals.copyTexture;
    OffscreenCanvasRenderingContext2D.prototype.drawImage = originals.drawImage;
    GPUQueue.prototype.copyExternalImageToTexture = originals.upload;
    GPUBuffer.prototype.mapAsync = originals.map;
    WgpuExecutionContext.prototype.submit = originals.submit;
    WgpuExecutionContext.prototype.recordShader = originals.recordShader;
    for (const resource of resources) {
      resource.queries.destroy();
      resource.resolve.destroy();
      resource.readback.destroy();
      resource.device.destroy();
    }
  }
}
