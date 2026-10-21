// Controlled ablation of the example workload. Record once, then vary only
// crop/upload, pre/post shaders, destination texture and completion mechanism.
export async function profilePipelineStages(options) {
  const { WgpuExecutionContext } = await import('/@fs' + options.repo + '/src/index.ts');
  const shaders = await import(`/src/tasks/${options.task}/shaders.ts`);
  const depth = options.task === 'depth-estimation';
  const half = options.precision === 'float16';
  const extent = depth ? 448 : 640;
  const size = [extent, extent];
  const adapter = await navigator.gpu.requestAdapter();
  const timestamps = options.timestamps !== false;
  const device = await adapter.requestDevice({
    requiredFeatures: [...(half ? ['shader-f16'] : []), ...(timestamps ? ['timestamp-query'] : [])],
  });
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  const contexts = [];
  let stream;
  let timer;
  const createEncoder = device.createCommandEncoder.bind(device);
  let capture = false;
  let passCount = 0;
  const queries = timestamps ? device.createQuerySet({ type: 'timestamp', count: 6 }) : null;
  const resolve = timestamps
    ? device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
    : null;
  const readback = timestamps
    ? device.createBuffer({ size: 48, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
    : null;
  try {
    if (timestamps)
      device.createCommandEncoder = function (descriptor) {
        const encoder = createEncoder(descriptor);
        if (!capture) return encoder;
        const begin = encoder.beginComputePass.bind(encoder);
        const finish = encoder.finish.bind(encoder);
        encoder.beginComputePass = (descriptor = {}) => {
          const index = passCount++ * 2;
          if (index >= 6) throw new Error('Unexpected compute pass count');
          return begin({
            ...descriptor,
            timestampWrites: {
              querySet: queries,
              beginningOfPassWriteIndex: index,
              endOfPassWriteIndex: index + 1,
            },
          });
        };
        encoder.finish = (descriptor) => {
          encoder.resolveQuerySet(queries, 0, passCount * 2, resolve, 0);
          encoder.copyBufferToBuffer(resolve, 0, readback, 0, passCount * 16);
          return finish(descriptor);
        };
        return encoder;
      };
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = extent;
    const canvasContext = canvas.getContext('webgpu');
    canvasContext.configure({
      device,
      format: 'rgba8unorm',
      alphaMode: 'premultiplied',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });
    const videoTexture = device.createTexture({
      size,
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING,
    });
    const resultTexture = device.createTexture({
      size,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING,
    });
    const scratchTexture = device.createTexture({
      size,
      format: 'rgba8unorm',
      usage: GPUTextureUsage.COPY_DST,
    });
    const fenceSource = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
    const fence = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const crop = new OffscreenCanvas(extent, extent);
    const cropContext = crop.getContext('2d');
    const source = document.createElement('canvas');
    source.width = 1280;
    source.height = 720;
    const paint = source.getContext('2d');
    const draw = () => {
      // Repaint identical pixels so every model receives the same tensor while
      // captureStream continues delivering fresh video frames.
      paint.fillStyle = 'white';
      paint.fillRect(0, 0, 1280, 720);
      paint.fillStyle = 'black';
      paint.beginPath();
      paint.arc(640, 360, 120, 0, 2 * Math.PI);
      paint.fill();
    };
    draw();
    timer = setInterval(draw, 33);
    stream = source.captureStream(30);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    document.body.replaceChildren(video, canvas);
    await video.play();
    const nextFrame = () =>
      new Promise((done, reject) => {
        const timeout = setTimeout(() => reject(new Error('Video frame timed out')), 5000);
        video.requestVideoFrameCallback(() => {
          clearTimeout(timeout);
          done();
        });
      });
    await nextFrame();
    const cropFrame = () => cropContext.drawImage(video, 280, 0, 720, 720, 0, 0, extent, extent);
    const upload = () =>
      device.queue.copyExternalImageToTexture({ source: crop }, { texture: videoTexture }, size);
    cropFrame();
    upload();
    const model = new WgpuExecutionContext(device);
    const full = new WgpuExecutionContext(device);
    const init = new WgpuExecutionContext(device);
    contexts.push(model, full, init);
    const input = model.tensor(options.precision, [1, 3, extent, extent]);
    const output = model.tensor(options.precision, depth ? [1, 1, extent, extent] : [1, 84, 8400]);
    const preprocess = half ? shaders.preprocessF16 : shaders.preprocess;
    init.recordShader(preprocess, [videoTexture, input]);
    init.submit();
    await init.sync();
    const base = `/assets/${depth ? 'depthart' : 'yolo26n'}/`;
    const suffix = half ? '-fp16' : '';
    const responses = await Promise.all([
      fetch(`${base}program${suffix}.json`),
      fetch(`${base}weights${suffix}.bin`),
    ]);
    if (responses.some((response) => !response.ok)) throw new Error('Model artifacts unavailable');
    const program = await responses[0].json();
    const weights = await responses[1].arrayBuffer();
    model.recordProgram(program, weights, [input, output]);
    full
      .recordShader(preprocess, [videoTexture, input])
      .recordProgram(program, weights, [input, output]);
    if (depth) {
      const stats = full.storageBuffer(16);
      full
        .recordShader(shaders.clearStats, [stats])
        .recordShader(half ? shaders.depthStatsF16 : shaders.depthStats, [output, stats])
        .recordShader(half ? shaders.drawDepthF16 : shaders.drawDepth, [
          output,
          stats,
          resultTexture,
        ]);
    } else {
      const counter = full.storageBuffer(16);
      const detections = full.storageBuffer(shaders.MAX_DETECTIONS * 6 * 4);
      full
        .recordShader(shaders.clearDetections, [counter])
        .recordShader(half ? shaders.compactDetectionsF16 : shaders.compactDetections, [
          output,
          counter,
          detections,
        ])
        .recordShader(shaders.drawDetections, [videoTexture, resultTexture, counter, detections]);
    }
    const cases = [
      { name: 'model-sync', sync: true },
      { name: 'model-fence' },
      { name: 'model-paced', paced: true },
      { name: 'crop', paced: true, crop: true },
      { name: 'upload', paced: true, crop: true, upload: true },
      { name: 'prepost', paced: true, crop: true, upload: true, full: true },
      { name: 'texture-copy', paced: true, crop: true, upload: true, full: true, copy: true },
      {
        name: 'canvas',
        paced: true,
        crop: true,
        upload: true,
        full: true,
        copy: true,
        canvas: true,
      },
    ];
    async function run(test) {
      if (test.paced) await nextFrame(); // Callback wait is excluded from all timers, as in the example.
      const start = performance.now();
      if (test.crop) cropFrame();
      const cropped = performance.now();
      if (test.upload) upload();
      const uploaded = performance.now();
      passCount = 0;
      capture = timestamps;
      try {
        (test.full ? full : model).submit();
      } finally {
        capture = false;
      }
      const submitted = performance.now();
      if (test.sync) await model.sync();
      else {
        const encoder = device.createCommandEncoder();
        if (test.copy)
          encoder.copyTextureToTexture(
            { texture: resultTexture },
            { texture: test.canvas ? canvasContext.getCurrentTexture() : scratchTexture },
            size
          );
        encoder.copyBufferToBuffer(fenceSource, 0, fence, 0, 16);
        device.queue.submit([encoder.finish()]);
        await fence.mapAsync(GPUMapMode.READ);
        fence.unmap();
      }
      const end = performance.now();
      const result = {
        wallMs: end - start,
        cropCpuMs: cropped - start,
        uploadCallCpuMs: uploaded - cropped,
        submitCpuMs: submitted - uploaded,
        completionWallMs: end - submitted,
      };
      // Read timestamps AFTER the wall timer. No extra mapAsync callback is in
      // the measured interval; the same query instrumentation is used in every case.
      if (timestamps) {
        await readback.mapAsync(GPUMapMode.READ);
        const t = [...new BigUint64Array(readback.getMappedRange())];
        readback.unmap();
        const index = test.full ? 2 : 0;
        result.modelGpuMs = Number(t[index + 1] - t[index]) / 1e6;
        result.computeSpanGpuMs = Number(t[passCount * 2 - 1] - t[0]) / 1e6;
      }
      return result;
    }
    // Seeded shuffle avoids consistently assigning earlier/higher GPU clocks
    // to the same case. Every round contributes a paired observation.
    let seed = 479;
    const shuffle = () => {
      const ordered = [...cases];
      for (let i = ordered.length - 1; i > 0; i--) {
        seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
        const j = seed % (i + 1);
        [ordered[i], ordered[j]] = [ordered[j], ordered[i]];
      }
      return ordered;
    };
    const rounds = [];
    for (let round = -(options.warmups ?? 8); round < (options.samples ?? 61); round++) {
      const results = {};
      for (const test of shuffle()) results[test.name] = await run(test);
      if (round >= 0) rounds.push(results);
    }
    // Sustained ABBA blocks reproduce continuous benchmark submissions and
    // the example's repeated canvas presentation, rather than only isolated
    // interleaved samples. This also checks presentation at its normal cadence.
    const sustained = [];
    for (const name of ['model-sync', 'canvas', 'canvas', 'model-sync']) {
      const test = cases.find((item) => item.name === name);
      const samples = [];
      for (let i = -4; i < 31; i++) {
        const sample = await run(test);
        if (i >= 0) samples.push(sample);
      }
      sustained.push({ name, samples });
    }
    if (errors.length) throw new Error(errors.join('\n'));
    const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
    const comparisons = [
      ['model-fence', 'model-sync'],
      ['model-paced', 'model-fence'],
      ['crop', 'model-paced'],
      ['upload', 'crop'],
      ['prepost', 'upload'],
      ['texture-copy', 'prepost'],
      ['canvas', 'texture-copy'],
      ['canvas', 'model-sync'],
    ];
    const result = {
      task: options.task,
      precision: options.precision,
      timestamps,
      warmups: options.warmups ?? 8,
      samples: rounds.length,
      gpu: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
      browser: navigator.userAgent,
      source: 'Identical 1280x720 synthetic video frames at 30fps',
      method:
        'Single page/device; two recorded model contexts sharing input/output tensors; shuffled paired rounds. All shaders recorded outside measurement. Video callback wait excluded. Timestamp readback occurs after the wall timer. Additional sustained ABBA blocks compare continuous model submissions to video-paced canvas output. Canvas case matches example commands; wall time ends at GPU completion notification, not display scanout.',
      interpretation:
        'Paired deltas are medians of within-round differences, not differences of medians. They are not additive. wallMinusModelGpuDeltaMs subtracts the paired change in model GPU duration; it is a diagnostic residual, not an independently measured stage duration. Synthetic video and headless Chrome do not reproduce every camera/compositor cost.',
      medians: Object.fromEntries(
        cases.map(({ name }) => [
          name,
          Object.fromEntries(
            Object.keys(rounds[0][name]).map((key) => [
              key,
              median(rounds.map((r) => r[name][key])),
            ])
          ),
        ])
      ),
      pairedDeltas: comparisons.map(([to, from]) => ({
        from,
        to,
        ...(timestamps
          ? {
              wallMinusModelGpuDeltaMs: median(
                rounds.map(
                  (r) => r[to].wallMs - r[from].wallMs - (r[to].modelGpuMs - r[from].modelGpuMs)
                )
              ),
            }
          : {}),
        ...Object.fromEntries(
          Object.keys(rounds[0][to]).map((key) => [
            key,
            median(rounds.map((r) => r[to][key] - r[from][key])),
          ])
        ),
      })),
      sustainedWarmupsPerBlock: 4,
      sustainedSamplesPerBlock: 31,
      sustained: sustained.map(({ name, samples }) => ({
        name,
        medians: Object.fromEntries(
          Object.keys(samples[0]).map((key) => [key, median(samples.map((r) => r[key]))])
        ),
        samples,
      })),
      rounds,
      errors,
    };
    const response = await fetch(options.reportEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(result),
    });
    if (!response.ok) throw new Error(`Results upload failed: ${response.status}`);
    return result;
  } finally {
    clearInterval(timer);
    stream?.getTracks().forEach((track) => track.stop());
    await device.queue.onSubmittedWorkDone();
    contexts.reverse().forEach((ctx) => ctx.destroy());
    device.destroy();
  }
}
