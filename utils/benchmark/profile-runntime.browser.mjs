// Compare unmodified published ruNNtime APIs with this runtime on one device.
export async function profileRunntime(options) {
  const { WgpuExecutionContext } = await import('/@fs' + options.repo + '/src/index.ts');
  const {
    tgpu,
    initRunntime,
    createObjectDetector,
    createDepthEstimator,
    imageBuffer,
    models,
    kernelInternals,
  } = await import('/@fs' + options.repo + '/utils/benchmark/runntime/entry.mjs');
  const depth = options.task === 'depth-estimation';
  const extent = depth ? 448 : 640;
  const modelName = depth ? 'depthart' : 'yolo26n';
  const modelAsset = options.runntimeAssets.provenance.find(
    (p) => p.model === (depth ? 'depthart' : 'yolo26')
  );
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice({
    requiredFeatures: [
      'shader-f16',
      'timestamp-query',
      ...(adapter.features.has('subgroups') ? ['subgroups'] : []),
    ],
  });
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  const root = tgpu.initFromDevice({ device });
  initRunntime(root);
  let runner;
  const owned = [];
  const originalEncoder = device.createCommandEncoder.bind(device);
  let instrument = false,
    queryCount = 0;
  const queries = device.createQuerySet({ type: 'timestamp', count: 128 });
  const queryResolve = device.createBuffer({
    size: 128 * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });
  const queryRead = device.createBuffer({
    size: 128 * 8,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  device.createCommandEncoder = function (descriptor) {
    const encoder = originalEncoder(descriptor);
    if (!instrument) return encoder;
    const begin = encoder.beginComputePass.bind(encoder);
    encoder.beginComputePass = (descriptor = {}) => {
      const first = queryCount;
      queryCount += 2;
      if (queryCount > 128) throw new Error('Too many compute passes for GPU timing');
      return begin({
        ...descriptor,
        timestampWrites: {
          querySet: queries,
          beginningOfPassWriteIndex: first,
          endOfPassWriteIndex: first + 1,
        },
      });
    };
    return encoder;
  };
  const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
  const summarize = (a) => ({
    median: median(a),
    p10: [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.1)],
    p90: [...a].sort((x, y) => x - y)[Math.floor(a.length * 0.9)],
    samples: a,
  });
  try {
    const loadStart = performance.now();
    const config = {
      ...(depth ? models.depthEstimation.DEPTHART.S : models.objectDetection.YOLO26.N),
      modelPath: '/@fs' + modelAsset.path,
      resizeMode: 'crop',
    };
    runner = depth ? await createDepthEstimator(config) : await createObjectDetector(config);
    const runntimeLoadMs = performance.now() - loadStart;
    const shaders = await import(`/src/tasks/${options.task}/shaders.ts`);
    const resources = await Promise.all(
      ['program-fp16.json', 'weights-fp16.bin', 'program.json', 'weights.bin'].map((name) =>
        fetch(`/assets/${modelName}/${name}`)
      )
    );
    if (resources.some((r) => !r.ok)) throw new Error('Missing local model artifacts');
    const [program, weights, program32, weights32] = await Promise.all([
      resources[0].json(),
      resources[1].arrayBuffer(),
      resources[2].json(),
      resources[3].arrayBuffer(),
    ]);
    const oursLoadStart = performance.now();
    const context = new WgpuExecutionContext(device);
    const oracle = new WgpuExecutionContext(device);
    owned.push(context, oracle);
    const input = context.tensor('float16', [1, 3, extent, extent]);
    const output = context.tensor('float16', depth ? [1, 1, extent, extent] : [1, 84, 8400]);
    const x32 = oracle.tensor('float32', input.shape),
      y32 = oracle.tensor('float32', output.shape);
    const texture = device.createTexture({
      size: [extent, extent],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });
    context
      .recordShader(shaders.preprocessF16, [texture, input])
      .recordProgram(program, weights, [input, output]);
    oracle
      .recordShader(shaders.preprocess, [texture, x32])
      .recordProgram(program32, weights32, [x32, y32]);
    const oursRecordMs = performance.now() - oursLoadStart;
    const crop = new OffscreenCanvas(extent, extent),
      cropContext = crop.getContext('2d');
    const sourceCanvas = new OffscreenCanvas(1280, 720),
      sourceContext = sourceCanvas.getContext('2d');
    const photograph = await createImageBitmap(
      await (await fetch('/@fs' + options.runntimeAssets.image.path)).blob()
    );
    function makeImage(width, height, gradient = false) {
      const canvas = new OffscreenCanvas(width, height),
        ctx = canvas.getContext('2d');
      if (gradient) {
        const image = ctx.createImageData(width, height);
        for (let i = 0; i < width * height; i++) {
          image.data[4 * i] = ((i % width) * 255) / width;
          image.data[4 * i + 1] = (Math.floor(i / width) * 255) / height;
          image.data[4 * i + 2] = (i % 47) * 5;
          image.data[4 * i + 3] = 255;
        }
        ctx.putImageData(image, 0, 0);
      } else ctx.drawImage(photograph, 0, 0, width, height);
      const imageData = ctx.getImageData(0, 0, width, height);
      return {
        imageData,
        buffer: imageBuffer(imageData.data, width, height, 'rgba'),
        width,
        height,
      };
    }
    const images = [
      { name: 'photograph-native', ...makeImage(extent, extent) },
      { name: 'photograph-1280x720-crop', ...makeImage(1280, 720) },
      { name: 'gradient-native', ...makeImage(extent, extent, true) },
    ];
    photograph.close();
    function upload(image) {
      if (image.width === extent && image.height === extent) {
        device.queue.writeTexture({ texture }, image.imageData.data, { bytesPerRow: extent * 4 }, [
          extent,
          extent,
        ]);
      } else {
        sourceContext.putImageData(image.imageData, 0, 0);
        const side = Math.min(image.width, image.height);
        cropContext.drawImage(
          sourceCanvas,
          (image.width - side) / 2,
          (image.height - side) / 2,
          side,
          side,
          0,
          0,
          extent,
          extent
        );
        device.queue.copyExternalImageToTexture({ source: crop }, { texture }, [extent, extent]);
      }
    }
    function decode(predictions, image) {
      const count = 8400,
        hits = [],
        scale = Math.min(image.width, image.height) / extent;
      const offsetX = (image.width - Math.min(image.width, image.height)) / 2;
      const offsetY = (image.height - Math.min(image.width, image.height)) / 2;
      for (let i = 0; i < count; i++) {
        let confidence = -Infinity,
          classId = 0;
        for (let c = 0; c < 80; c++) {
          const score = predictions[(4 + c) * count + i];
          if (score > confidence) {
            confidence = score;
            classId = c;
          }
        }
        if (confidence < 0.35) continue;
        const box = {
          format: 'xyxy',
          xmin: Math.max(0, Math.min(image.width, predictions[i] * scale + offsetX)),
          ymin: Math.max(0, Math.min(image.height, predictions[count + i] * scale + offsetY)),
          xmax: Math.max(0, Math.min(image.width, predictions[2 * count + i] * scale + offsetX)),
          ymax: Math.max(0, Math.min(image.height, predictions[3 * count + i] * scale + offsetY)),
        };
        hits.push({ classId, label: runner.labels[classId], confidence, box });
      }
      return hits.sort((a, b) => b.confidence - a.confidence).slice(0, 64);
    }
    const inferenceOptions = { confidenceThreshold: 0.35, maxDetections: 64 };
    const execute = [
      async (image) => {
        upload(image);
        context.submit();
        const values = new Float16Array(await output.getData());
        return depth
          ? { width: extent, height: extent, data: new Float32Array(values) }
          : decode(values, image);
      },
      async (image) =>
        depth
          ? runner.estimateDepth(image.buffer)
          : runner.detectObjects(image.buffer, inferenceOptions),
    ];
    const cases = [];
    for (const image of images) {
      const samples = [[], []],
        outputs = [];
      for (let i = -options.warmups; i < options.samples; i++) {
        for (const engine of i & 1 ? [1, 0] : [0, 1]) {
          const start = performance.now();
          outputs[engine] = await execute[engine](image);
          if (i >= 0) samples[engine].push(performance.now() - start);
        }
      }
      const gpuSamples = [[], []],
        diagnosticWall = [[], []],
        residuals = [[], []],
        passCounts = [[], []];
      // Instrument in a separate phase, leaving the primary wall clock untouched.
      for (let i = 0; i < 15; i++)
        for (const engine of i & 1 ? [1, 0] : [0, 1]) {
          queryCount = 0;
          instrument = true;
          const phaseStart = performance.now();
          await execute[engine](image);
          const phaseMs = performance.now() - phaseStart;
          instrument = false;
          const encoder = originalEncoder();
          encoder.resolveQuerySet(queries, 0, queryCount, queryResolve, 0);
          encoder.copyBufferToBuffer(queryResolve, 0, queryRead, 0, queryCount * 8);
          device.queue.submit([encoder.finish()]);
          await queryRead.mapAsync(GPUMapMode.READ);
          const times = new BigUint64Array(queryRead.getMappedRange());
          let total = 0;
          for (let q = 0; q < queryCount; q += 2) total += Number(times[q + 1] - times[q]) / 1e6;
          queryRead.unmap();
          gpuSamples[engine].push(total);
          diagnosticWall[engine].push(phaseMs);
          residuals[engine].push(phaseMs - total);
          passCounts[engine].push(queryCount / 2);
        }
      upload(image);
      oracle.submit();
      const referenceRaw = new Float32Array(await y32.getData());
      const reference = depth
        ? { width: extent, height: extent, data: referenceRaw }
        : decode(referenceRaw, image);
      function iou(a, b) {
        const overlap =
          Math.max(0, Math.min(a.xmax, b.xmax) - Math.max(a.xmin, b.xmin)) *
          Math.max(0, Math.min(a.ymax, b.ymax) - Math.max(a.ymin, b.ymin));
        return (
          overlap /
          Math.max(
            1e-9,
            (a.xmax - a.xmin) * (a.ymax - a.ymin) + (b.xmax - b.xmin) * (b.ymax - b.ymin) - overlap
          )
        );
      }
      function validate(result, engine) {
        if (depth) {
          if (result.data.length !== extent * extent) throw new Error('Invalid depth shape');
          let min = Infinity,
            max = -Infinity,
            sum = 0,
            sumSq = 0,
            sumRef = 0,
            sumRefSq = 0,
            cross = 0,
            abs = 0;
          for (let i = 0; i < result.data.length; i++) {
            const x = result.data[i],
              y = reference.data[i];
            if (!Number.isFinite(x)) throw new Error('Nonfinite depth result');
            min = Math.min(min, x);
            max = Math.max(max, x);
            sum += x;
            sumSq += x * x;
            sumRef += y;
            sumRefSq += y * y;
            cross += x * y;
            abs += Math.abs(x - y);
          }
          const n = result.data.length;
          return {
            min,
            max,
            mean: sum / n,
            meanAbsoluteDifferenceToOursF32: engine === 0 ? abs / n : undefined,
            correlationToOursF32:
              (cross - (sum * sumRef) / n) /
              Math.sqrt((sumSq - (sum * sum) / n) * (sumRefSq - (sumRef * sumRef) / n)),
          };
        }
        for (const d of result)
          if (
            ![d.confidence, ...Object.values(d.box).filter((v) => typeof v === 'number')].every(
              Number.isFinite
            )
          )
            throw new Error('Nonfinite detection');
        const used = new Set();
        const matches = reference.map((r) => {
          let best = -1,
            overlap = 0;
          for (let j = 0; j < result.length; j++)
            if (
              !used.has(j) &&
              r.classId === result[j].classId &&
              iou(r.box, result[j].box) > overlap
            ) {
              best = j;
              overlap = iou(r.box, result[j].box);
            }
          if (best >= 0 && overlap >= 0.8) used.add(best);
          return {
            classId: r.classId,
            confidence: r.confidence,
            bestIoU: overlap,
            confidenceDifference:
              best >= 0 ? Math.abs(r.confidence - result[best].confidence) : null,
          };
        });
        const passesReferenceCheck =
          used.size === reference.length &&
          result.length === used.size &&
          matches.every((m) => m.confidenceDifference <= 0.05);
        if (!passesReferenceCheck)
          throw new Error('YOLO outputs failed detection agreement with local FP32 reference');
        return {
          passesReferenceCheck,
          detections: result,
          referenceDetections: reference.length,
          matchedAtIoU08: used.size,
          matches,
          unmatchedResults: result.length - used.size,
        };
      }
      const wall = samples.map(summarize),
        gpu = gpuSamples.map(summarize);
      cases.push({
        name: image.name,
        width: image.width,
        height: image.height,
        engines: ['wgpu-executorch-fp16', 'runntime-0.1.0'],
        wallMs: wall,
        gpuComputeMs: gpu,
        computePassCounts: passCounts,
        diagnosticWallMs: diagnosticWall.map(summarize),
        nonComputeResidualMs: residuals.map(summarize),
        runntimeOverOursWall: wall[1].median / wall[0].median,
        runntimeOverOursGpu: gpu[1].median / gpu[0].median,
        validation: outputs.map(validate),
        modelEquivalence: depth
          ? 'Different backbones: ours MNV4-S, ruNNtime TinyViM-S. Workload comparison only, not a runtime speedup claim.'
          : 'Both YOLO26-N one-to-one head; same input size and detection options. ruNNtime decodes raw distances/logits on CPU; our export decodes boxes/scores on GPU.',
      });
    }
    let nativeKernels;
    if (!depth) {
      const { compareRunntimeKernels } = await import(
        '/@fs' + options.repo + '/utils/benchmark/runntime/kernels.browser.mjs'
      );
      nativeKernels = await compareRunntimeKernels({
        repo: options.repo,
        device,
        root,
        lib: kernelInternals,
        gpuTime: async (submit) => {
          queryCount = 0;
          instrument = true;
          submit();
          instrument = false;
          const encoder = originalEncoder();
          encoder.resolveQuerySet(queries, 0, queryCount, queryResolve, 0);
          encoder.copyBufferToBuffer(queryResolve, 0, queryRead, 0, queryCount * 8);
          device.queue.submit([encoder.finish()]);
          await queryRead.mapAsync(GPUMapMode.READ);
          const times = new BigUint64Array(queryRead.getMappedRange());
          let total = 0;
          for (let q = 0; q < queryCount; q += 2) total += Number(times[q + 1] - times[q]) / 1e6;
          queryRead.unmap();
          return total;
        },
      });
    }
    if (errors.length) throw new Error(errors.join('\n'));
    const result = {
      nativeKernels,
      task: options.task,
      precision: 'float16',
      gpu: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
      browser: navigator.userAgent,
      features: [...device.features],
      runntimeLoadMs,
      oursRecordMs,
      packages: { runntime: '0.1.0', typegpu: '0.12.8' },
      assets: options.runntimeAssets,
      method:
        'One adapter/device; published ruNNtime APIs unmodified. Alternating wall-clock measurements: RGBA CPU bytes in -> CPU boxes/depth out, includes preprocessing/upload/inference/readback/CPU decode, excludes downloads/compilation/image decode/camera acquisition/rendering. Ours uses existing GPU preprocess shaders and model graphs; CPU crop for non-native sizes. Separate GPU timestamp phase measures summed compute pass durations; excludes CPU and copies, includes model/input conversion kernels. No extra timestamps in primary wall-clock samples. Both paths cache their command lists before measurement. DepthART backbones differ.',
      warmups: options.warmups,
      samples: options.samples,
      results: cases,
    };
    const response = await fetch(options.reportEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(result),
    });
    if (!response.ok) throw new Error('Failed to save results');
    texture.destroy();
    return result;
  } finally {
    instrument = false;
    device.createCommandEncoder = originalEncoder;
    runner?.dispose();
    owned.forEach((c) => c.destroy());
    queries.destroy();
    queryResolve.destroy();
    queryRead.destroy();
    root.destroy();
    device.destroy();
  }
}
