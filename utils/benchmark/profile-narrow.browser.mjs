// Compare the new narrow convolution with the existing 32-row tile, then
// compare unchanged model graphs against a complete archived source baseline.
export async function profileNarrowConvolution(options) {
  const { WgpuExecutionContext, createComputeBundle } = await import(
    '/@fs' + options.repo + '/src/index.ts'
  );
  const { WgpuExecutionContext: BaselineContext } = await import(
    '/@fs' + options.baselineRepo + '/src/index.ts'
  );
  const { KernelTimingContext } = await import(
    '/@fs' + options.repo + '/tests/helpers/kernel-timing.ts'
  );
  const adapter = await navigator.gpu.requestAdapter();
  const half = options.precision === 'float16';
  const device = await adapter.requestDevice({
    requiredFeatures: ['timestamp-query', ...(half ? ['shader-f16'] : [])],
  });
  device.pushErrorScope('validation');
  const suffix = half ? '-f16' : '';
  const array = half ? Float16Array : Float32Array;
  const fill = (length) =>
    new array(Array.from({ length }, (_, i) => (((i * 17 + 11) % 61) - 30) / 64));
  const code = [];
  for (const height of [32, 16]) {
    const module = await import(
      `/@fs${height === 32 ? options.baselineRepo : options.repo}/src/shaders/convolution/conv2d-tile${height}${suffix}.wgsl.ts`
    );
    code.push(Object.values(module)[0]);
  }
  const results = [];
  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  async function compare(contexts, repeats, outputs, gpu) {
    const samples = [[], []];
    for (let i = -8; i < 51; i++)
      for (const version of i & 1 ? [1, 0] : [0, 1]) {
        const start = performance.now();
        contexts[version].submit();
        await contexts[version].sync();
        if (i >= 0)
          samples[version].push(
            (gpu ? contexts[version].gpuMs : performance.now() - start) / repeats
          );
      }
    const values = [];
    for (let i = 0; i < 2; i++) {
      contexts[i].submit();
      await contexts[i].sync();
      values.push(new array(await outputs[i].getData()));
    }
    let maxDelta = 0;
    for (let i = 0; i < values[0].length; i++) {
      if (!Number.isFinite(values[0][i]) || !Number.isFinite(values[1][i]))
        throw new Error('Nonfinite output');
      maxDelta = Math.max(maxDelta, Math.abs(values[0][i] - values[1][i]));
    }
    if (maxDelta !== 0) throw new Error(`Output mismatch: ${maxDelta}`);
    const beforeMs = median(samples[0]),
      afterMs = median(samples[1]);
    return { beforeMs, afterMs, speedup: beforeMs / afterMs, maxDelta, repeats, samples };
  }
  try {
    const shapes = [
      [16, 8, 160, 1],
      [8, 16, 160, 1],
      [32, 16, 80, 1],
      [64, 16, 80, 1],
      [16, 16, 64, 1],
      [16, 16, 80, 1],
      [8, 16, 80, 1],
      [16, 8, 80, 1],
      [16, 16, 160, 2],
      [32, 16, 40, 1],
      [32, 16, 20, 1],
      [4, 13, 17, 1],
      [24, 16, 448, 1],
    ];
    for (const shape of shapes) {
      const [ic, oc, h, stride] = shape;
      const oh = Math.floor((h - 1) / stride) + 1;
      const ctx = new WgpuExecutionContext(device);
      const input = ctx.tensor(options.precision, [1, ic, h, h]).setData(fill(ic * h * h));
      const weight = ctx.tensor(options.precision, [oc, ic, 3, 3]).setData(fill(oc * ic * 9));
      const bias = ctx.tensor(options.precision, [oc]).setData(fill(oc));
      const out = ctx.tensor(options.precision, [1, oc, oh, oh]);
      const params = new Uint32Array(20);
      params.set([1, ic, h, h, oc, oh, oh, 3, 3, stride, stride, 1, 1, 1, 1, 1, 1]);
      const uniform = ctx.uniformBuffer(params);
      const bindings = [out, input, weight, bias].map((tensor, binding) => ({
        binding,
        buffer: tensor.buffer,
      }));
      bindings.push({ binding: 4, buffer: uniform });
      const repeats = Math.min(256, Math.max(8, Math.ceil(1e9 / (out.numel * ic * 9))));
      const contexts = code.map((shader) => {
        const timing = new KernelTimingContext(device);
        const bundle = createComputeBundle(device, shader, bindings, { kernel_h: 3, kernel_w: 3 });
        for (let i = 0; i < repeats; i++)
          timing.addDispatch({
            pipeline: bundle.pipeline,
            bindGroup: bundle.bindGroup,
            workgroupCountX: Math.ceil((oh * oh) / 64),
            workgroupCountY: 1,
          });
        return timing;
      });
      results.push({
        category: 'kernel',
        shape,
        selected: oh * oh >= 4096,
        ...(await compare(contexts, repeats, [out, out], true)),
      });
      contexts.forEach((context) => context.destroy());
      ctx.destroy();
    }
    for (const name of ['yolo26n', 'depthart']) {
      const base = `/assets/${name}/`;
      const responses = await Promise.all([
        fetch(`${base}program${half ? '-fp16' : ''}.json`),
        fetch(`${base}weights${half ? '-fp16' : ''}.bin`),
      ]);
      if (responses.some((response) => !response.ok))
        throw new Error('Model artifacts unavailable');
      const program = await responses[0].json(),
        weights = await responses[1].arrayBuffer();
      const outputs = [];
      const contexts = [BaselineContext, WgpuExecutionContext].map((Context) => {
        const ctx = new Context(device);
        const input = ctx.tensor(options.precision, program.values[program.inputIds[0]].shape);
        input.setData(
          new array(Array.from({ length: input.numel }, (_, i) => (i % 251) / 250 - 0.5))
        );
        const output = ctx.tensor(options.precision, program.values[program.outputIds[0]].shape);
        ctx.recordProgram(program, weights, [input, output]);
        outputs.push(output);
        return ctx;
      });
      results.push({ category: 'model', name, ...(await compare(contexts, 1, outputs, false)) });
      contexts.forEach((ctx) => ctx.destroy());
    }
    const error = await device.popErrorScope();
    if (error) throw error;
    const result = {
      precision: options.precision,
      gpu: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
      browser: navigator.userAgent,
      method:
        'Same-device alternating baseline/candidate measurements; 8 warmups, 51 samples. Kernels: recorded repeated dispatches, GPU timestamps per dispatch. Models: unchanged graphs and identical inputs, submit + sync wall clock. Model baseline is archived source; raw kernel comparison uses the unchanged 32-row tile. selected marks the conservative spatial threshold. All outputs must match bitwise.',
      results,
    };
    const response = await fetch(options.reportEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(result),
    });
    if (!response.ok) throw new Error(`Results upload failed: ${response.status}`);
    return result;
  } finally {
    device.destroy();
  }
}
