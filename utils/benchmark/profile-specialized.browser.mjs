// Same-device regression mesh and unchanged full graphs against archived kernels.
export async function profileConvolutionSpecializations(options) {
  const { WgpuExecutionContext, shaders } = await import('/@fs' + options.repo + '/src/index.ts');
  const { WgpuExecutionContext: BaselineContext, shaders: baselineShaders } = await import(
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
  const errors = [];
  device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
  const array = half ? Float16Array : Float32Array;
  const fill = (n) =>
    new array(
      Array.from({ length: n }, (_, i) =>
        options.fractionalInputs
          ? ((Math.imul(i, 1664525) + 1013904223) >>> 0) / 4294967296 - 0.5
          : (((i * 17 + 11) % 61) - 30) / 64
      )
    );
  const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
  async function compare(contexts, outputs, repeats, gpu) {
    const samples = contexts.map(() => []);
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
    const bytes = [];
    const values = [];
    for (let version = 0; version < 2; version++) {
      outputs[version].setData(new array(outputs[version].numel).fill(NaN));
      contexts[version].submit();
      await contexts[version].sync();
      const buffer = await outputs[version].getData();
      bytes.push(new Uint8Array(buffer));
      values.push(new array(buffer));
    }
    const differences = [];
    let maxAbs = 0,
      changed = 0;
    for (let i = 0; i < values[0].length; i++) {
      if (!Number.isFinite(values[0][i]) || !Number.isFinite(values[1][i]))
        throw new Error('Nonfinite or missing convolution output');
      const delta = Math.abs(values[0][i] - values[1][i]);
      maxAbs = Math.max(maxAbs, delta);
      if (delta) {
        changed++;
        if (differences.length < 6) differences.push([i, values[0][i], values[1][i]]);
      }
    }
    const differentBytes = bytes[0].reduce((n, v, i) => n + Number(v !== bytes[1][i]), 0);
    const medians = samples.map(median);
    return {
      beforeMs: medians[0],
      afterMs: medians[1],
      speedup: medians[0] / medians[1],
      maxAbs,
      differentBytes,
      differences,
      changed,
      repeats,
      samples,
    };
  }
  const results = [];
  try {
    // [IC, OC, IH, IW, stride, dilation, batch]; 3x3 same-padding.
    const cases = [
      [3, 16, 640, 640, 2, 1, 1],
      [3, 8, 640, 640, 2, 1, 1],
      [3, 12, 448, 448, 2, 1, 1],
      [3, 16, 320, 320, 2, 1, 1],
      [3, 16, 128, 128, 2, 1, 1],
      [3, 13, 137, 145, 2, 1, 1],
      [3, 16, 64, 64, 2, 1, 1],
      [3, 20, 640, 640, 2, 1, 1],
      [3, 32, 448, 448, 2, 1, 1],
      [3, 16, 128, 128, 1, 1, 1],
      [16, 8, 160, 160, 1, 1, 1],
      [8, 8, 80, 80, 1, 1, 1],
      [32, 8, 80, 80, 1, 1, 1],
      [64, 8, 80, 80, 1, 1, 1],
      [128, 8, 80, 80, 1, 1, 1],
      [256, 8, 80, 80, 1, 1, 1],
      [16, 4, 80, 80, 1, 1, 1],
      [16, 7, 80, 80, 1, 1, 1],
      [16, 9, 80, 80, 1, 1, 1],
      [16, 8, 40, 40, 1, 1, 1],
      [16, 8, 160, 160, 2, 1, 1],
      [16, 8, 80, 80, 1, 2, 1],
      [4, 8, 65, 67, 1, 1, 2],
      [16, 8, 80, 80, 3, 1, 6],
      [8, 8, 80, 80, 1, 3, 1],
      [16, 16, 80, 80, 1, 1, 1],
      [24, 16, 448, 448, 1, 1, 1],
    ];
    for (const shape of cases) {
      const [ic, oc, ih, iw, stride, dilation, batch] = shape,
        pad = dilation;
      const oh = Math.floor((ih - 1) / stride) + 1,
        ow = Math.floor((iw - 1) / stride) + 1;
      const owner = new WgpuExecutionContext(device);
      const x = owner
        .tensor(options.precision, [batch, ic, ih, iw])
        .setData(fill(batch * ic * ih * iw));
      const w = owner.tensor(options.precision, [oc, ic, 3, 3]).setData(fill(oc * ic * 9));
      const bias = owner.tensor(options.precision, [oc]).setData(fill(oc));
      const sharedOutput = owner.tensor(options.precision, [batch, oc, oh, ow]);
      const outputs = [sharedOutput, sharedOutput];
      const repeats = Math.min(128, Math.max(4, Math.ceil(2e8 / (outputs[0].numel * ic * 9))));
      class RepeatedContext extends KernelTimingContext {
        addDispatch(d) {
          for (let i = 0; i < repeats; i++) super.addDispatch(d);
        }
      }
      const contexts = [new RepeatedContext(device), new RepeatedContext(device)];
      device.pushErrorScope('validation');
      for (let version = 0; version < 2; version++)
        contexts[version].recordShader((version ? shaders : baselineShaders).convolution, [
          x,
          w,
          bias,
          [stride, stride],
          [pad, pad],
          [dilation, dilation],
          false,
          [0, 0],
          1,
          outputs[version],
        ]);
      const error = await device.popErrorScope();
      if (error) throw new Error(error.message);
      const timing = await compare(contexts, outputs, repeats, true);
      if (timing.differentBytes !== 0)
        throw new Error(
          `Raw convolution changed ${shape}: ${JSON.stringify({ max: timing.maxAbs, examples: timing.differences, precision: options.precision })}`
        );
      results.push({ category: 'kernel', shape, ...timing });
      contexts.forEach((c) => c.destroy());
      owner.destroy();
    }
    for (const name of ['yolo26n', 'depthart']) {
      const suffix = half ? '-fp16' : '';
      const responses = await Promise.all([
        fetch(`/assets/${name}/program${suffix}.json`),
        fetch(`/assets/${name}/weights${suffix}.bin`),
      ]);
      if (responses.some((r) => !r.ok)) throw new Error('Missing model artifacts');
      const program = await responses[0].json(),
        weights = await responses[1].arrayBuffer(),
        outputs = [];
      const contexts = [BaselineContext, WgpuExecutionContext].map((Context) => {
        const ctx = new Context(device);
        const x = ctx
          .tensor(options.precision, program.values[program.inputIds[0]].shape)
          .setData(
            new array(
              Array.from(
                { length: program.values[program.inputIds[0]].shape.reduce((a, b) => a * b, 1) },
                (_, i) => (i % 251) / 250 - 0.5
              )
            )
          );
        const y = ctx.tensor(options.precision, program.values[program.outputIds[0]].shape);
        outputs.push(y);
        ctx.recordProgram(program, weights, [x, y]);
        return ctx;
      });
      const timing = await compare(contexts, outputs, 1, false);
      // The candidate preserves the original reduction order; require bitwise parity.
      if (timing.differentBytes !== 0) throw new Error(`Model changed ${name}: ${timing.maxAbs}`);
      results.push({ category: 'model', name, ...timing });
      contexts.forEach((c) => c.destroy());
    }
    if (errors.length) throw new Error(errors.join('\n'));
    const result = {
      inputPattern: options.fractionalInputs ? 'fractional-rounded-products' : 'dyadic',
      precision: options.precision,
      gpu: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
      browser: navigator.userAgent,
      method:
        'Archived baseline vs current automatic convolution routing, same device. Commands recorded once. 8 warmups, 51 alternating samples. Kernels: identical input/weight/output buffers, repeated dispatches and GPU timestamps, with boundary/neighbor fallback cases. Full unchanged YOLO and DepthART graphs: submit+sync wall time on identical fixture inputs. Poisoned outputs, all outputs required bitwise equal. No pre/post/rendering in model timings.',
      results,
    };
    const response = await fetch(options.reportEndpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(result),
    });
    if (!response.ok) throw new Error('Results upload failed');
    return result;
  } finally {
    device.destroy();
  }
}
