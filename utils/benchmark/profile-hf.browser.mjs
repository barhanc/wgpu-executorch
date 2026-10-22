// Compare pinned HF shader adaptations with this runtime on identical buffers.
export async function profileHuggingFace(options) {
  const { WgpuExecutionContext, shaders } = await import('/@fs' + options.repo + '/src/index.ts');
  const { WgpuExecutionContext: BaselineContext, shaders: baselineShaders } = await import(
    '/@fs' + options.baselineRepo + '/src/index.ts'
  );
  const { KernelTimingContext } = await import(
    '/@fs' + options.repo + '/tests/helpers/kernel-timing.ts'
  );
  const { recordHfConvolution } = await import(
    '/@fs' + options.repo + '/utils/benchmark/hf-convolution.ts'
  );
  const { recordHfMatrix } = await import('/@fs' + options.repo + '/utils/benchmark/hf-matrix.ts');
  const adapter = await navigator.gpu.requestAdapter();
  const half = options.precision === 'float16';
  const matrixFeatures = [
    'subgroups',
    'subgroup-size-control',
    'chromium-experimental-subgroup-matrix',
  ];
  const matrixAvailable =
    matrixFeatures.every((name) => adapter.features.has(name)) &&
    adapter.info.subgroupMinSize <= 32 &&
    adapter.info.subgroupMaxSize >= 32 &&
    Array.from(adapter.info.subgroupMatrixConfigs ?? []).some(
      (c) => c.componentType === (half ? 'f16' : 'f32') && c.M === 8 && c.N === 8 && c.K === 8
    );
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupStorageSize: Math.min(
        adapter.limits.maxComputeWorkgroupStorageSize,
        32768
      ),
    },
    requiredFeatures: [
      'timestamp-query',
      ...(half ? ['shader-f16'] : []),
      ...(matrixAvailable ? matrixFeatures : []),
    ],
  });
  const errors = [];
  device.addEventListener('uncapturederror', (event) => errors.push(event.error.message));
  const array = half ? Float16Array : Float32Array;
  const fill = (length) =>
    new array(Array.from({ length }, (_, i) => (((i * 17 + 11) % 61) - 30) / 64));
  const median = (values) => [...values].sort((a, b) => a - b)[values.length >> 1];
  async function measure(contexts, repeats = 1, gpu = false) {
    const samples = contexts.map(() => []);
    const warmups = options.warmups ?? 8,
      count = options.samples ?? 51;
    for (let i = -warmups; i < count; i++) {
      // Rotate and reverse so each variant occupies every position repeatedly.
      const order = contexts.map((_, j) => (j + i + warmups) % contexts.length);
      if (i & 1) order.reverse();
      for (const version of order) {
        const start = performance.now();
        contexts[version].submit();
        await contexts[version].sync();
        if (i >= 0)
          samples[version].push(
            (gpu ? contexts[version].gpuMs : performance.now() - start) / repeats
          );
      }
    }
    const medians = samples.map(median);
    return { medians, samples, speedups: medians.map((ms) => medians[0] / ms), repeats };
  }
  const results = [];
  try {
    const matrixOracleChecks = [];
    if (matrixAvailable) {
      // Independent CPU oracle: batches, partial M/N/K tiles and absent bias.
      for (const kernel of [1, 3]) {
        const batch = 2,
          ic = kernel === 1 ? 32 : 3,
          oc = 17;
        const h = kernel === 1 ? 8 : 9,
          w = kernel === 1 ? 8 : 11;
        const stride = kernel === 1 ? 1 : 2,
          pad = kernel === 1 ? 0 : 1;
        const oh = Math.floor((h + 2 * pad - kernel) / stride) + 1;
        const ow = Math.floor((w + 2 * pad - kernel) / stride) + 1;
        const ctx = new WgpuExecutionContext(device);
        const xv = fill(batch * ic * h * w),
          wv = fill(oc * ic * kernel * kernel);
        const x = ctx.tensor(options.precision, [batch, ic, h, w]).setData(xv);
        const weight = ctx.tensor(options.precision, [oc, ic, kernel, kernel]).setData(wv);
        const y = ctx
          .tensor(options.precision, [batch, oc, oh, ow])
          .setData(new array(batch * oc * oh * ow).fill(NaN));
        device.pushErrorScope('validation');
        if (
          !recordHfMatrix(ctx, [
            x,
            weight,
            undefined,
            [stride, stride],
            [pad, pad],
            [1, 1],
            false,
            [0, 0],
            1,
            y,
          ])
        )
          throw new Error('Matrix oracle route unavailable');
        const compileError = await device.popErrorScope();
        if (compileError) throw new Error(compileError.message);
        ctx.submit();
        await ctx.sync();
        const actual = new array(await y.getData());
        let maxAbs = 0;
        for (let b = 0; b < batch; b++)
          for (let o = 0; o < oc; o++)
            for (let iy = 0; iy < oh; iy++)
              for (let ix = 0; ix < ow; ix++) {
                let sum = 0;
                for (let c = 0; c < ic; c++)
                  for (let ky = 0; ky < kernel; ky++)
                    for (let kx = 0; kx < kernel; kx++) {
                      const sy = iy * stride - pad + ky,
                        sx = ix * stride - pad + kx;
                      if (sy >= 0 && sy < h && sx >= 0 && sx < w)
                        sum +=
                          xv[((b * ic + c) * h + sy) * w + sx] *
                          wv[((o * ic + c) * kernel + ky) * kernel + kx];
                    }
                const value = actual[((b * oc + o) * oh + iy) * ow + ix];
                if (!Number.isFinite(value))
                  throw new Error('Nonfinite/missing matrix oracle output');
                maxAbs = Math.max(maxAbs, Math.abs(value - new array([sum])[0]));
              }
        if (maxAbs > 1e-5) throw new Error(`Matrix CPU oracle error: ${maxAbs}`);
        matrixOracleChecks.push({ kernel, batch, ic, oc, h, w, stride, pad, maxAbs });
        ctx.destroy();
      }
    }
    const cases = [
      [48, 64, 160, 1, 1, 0],
      [64, 64, 80, 1, 1, 0],
      [96, 128, 80, 1, 1, 0],
      [192, 128, 40, 1, 1, 0],
      [256, 256, 20, 1, 1, 0],
      [384, 256, 20, 1, 1, 0],
      [80, 80, 80, 1, 1, 0],
      [80, 80, 20, 1, 1, 0],
      [32, 16, 80, 1, 1, 0],
      [3, 16, 640, 3, 2, 1],
      [16, 32, 320, 3, 2, 1],
      [16, 8, 160, 3, 1, 1],
      [32, 32, 40, 3, 1, 1],
      [64, 64, 80, 3, 2, 1],
      [128, 128, 40, 3, 2, 1],
    ];
    for (const shape of cases) {
      const [ic, oc, h, kernel, stride, pad] = shape;
      const oh = Math.floor((h + 2 * pad - kernel) / stride) + 1;
      const owner = new WgpuExecutionContext(device);
      const inputValues = fill(ic * h * h),
        weightValues = fill(oc * ic * kernel * kernel),
        biasValues = fill(oc);
      const input = owner.tensor(options.precision, [1, ic, h, h]).setData(inputValues);
      const weight = owner
        .tensor(options.precision, [oc, ic, kernel, kernel])
        .setData(weightValues);
      const bias = owner.tensor(options.precision, [oc]).setData(biasValues);
      const out = owner.tensor(options.precision, [1, oc, oh, oh]);
      const args = [
        input,
        weight,
        bias,
        [stride, stride],
        [pad, pad],
        [1, 1],
        false,
        [0, 0],
        1,
        out,
      ];
      const repeats = Math.min(
        128,
        Math.max(4, Math.ceil(2e8 / (out.numel * ic * kernel * kernel)))
      );
      class RepeatedContext extends KernelTimingContext {
        addDispatch(dispatch) {
          for (let i = 0; i < repeats; i++) super.addDispatch(dispatch);
        }
      }
      const contexts = Array.from(
        { length: kernel === 3 ? 3 : 2 },
        () => new RepeatedContext(device)
      );
      device.pushErrorScope('validation');
      contexts[0].recordShader(baselineShaders.convolution, args);
      if (!recordHfConvolution(contexts[1], args))
        throw new Error('Unsupported HF benchmark shape');
      if (kernel === 3) recordHfConvolution(contexts[2], args, 'both', 32);
      const variants = kernel === 3 ? ['baseline', 'hf-m64', 'hf-m32'] : ['baseline', 'hf-m64'];
      if (matrixAvailable) {
        const matrixContext = new RepeatedContext(device);
        if (recordHfMatrix(matrixContext, args)) {
          contexts.push(matrixContext);
          variants.push('hf-matrix');
        } else matrixContext.destroy();
      }
      const compileError = await device.popErrorScope();
      if (compileError) throw new Error(compileError.message);
      const timing = await measure(contexts, repeats, true);
      const values = [];
      for (const ctx of contexts) {
        out.setData(new array(out.numel).fill(NaN));
        ctx.submit();
        await ctx.sync();
        values.push(new array(await out.getData()));
      }
      // Same rounded inputs, F32 arithmetic: tests the mixed-precision HF path
      // independently of the baseline's FP16 accumulation differences.
      const x32 = owner.tensor('float32', [1, ic, h, h]).setData(new Float32Array(inputValues));
      const w32 = owner
        .tensor('float32', [oc, ic, kernel, kernel])
        .setData(new Float32Array(weightValues));
      const b32 = owner.tensor('float32', [oc]).setData(new Float32Array(biasValues));
      const y32 = owner.tensor('float32', [1, oc, oh, oh]);
      owner.recordShader(shaders.convolution, [
        x32,
        w32,
        b32,
        [stride, stride],
        [pad, pad],
        [1, 1],
        false,
        [0, 0],
        1,
        y32,
      ]);
      owner.submit();
      await owner.sync();
      const reference = new array(new Float32Array(await y32.getData()));
      const errorsVsReference = values.map((value) => {
        let maxAbs = 0;
        for (let i = 0; i < value.length; i++) {
          if (!Number.isFinite(value[i])) throw new Error(`Nonfinite/missing output for ${shape}`);
          maxAbs = Math.max(maxAbs, Math.abs(value[i] - reference[i]));
        }
        return maxAbs;
      });
      if (errorsVsReference.slice(1).some((error) => error > 1e-5))
        throw new Error(`HF reference error for ${shape}: ${errorsVsReference[1]}`);
      results.push({
        category: 'kernel',
        shape,
        variants,
        ...timing,
        errorsVsFloat32Reference: errorsVsReference,
      });
      contexts.forEach((ctx) => ctx.destroy());
      owner.destroy();
    }
    const suffix = half ? '-fp16' : '';
    const [programResponse, weightsResponse, referenceResponse] = await Promise.all([
      fetch(`/assets/yolo26n/program${suffix}.json`),
      fetch(`/assets/yolo26n/weights${suffix}.bin`),
      fetch(`/@fs${options.repo}/tests/fixtures/yolo26n/reference-output.f32`),
    ]);
    if (![programResponse, weightsResponse, referenceResponse].every((r) => r.ok))
      throw new Error('Missing model artifacts/reference');
    const program = await programResponse.json(),
      weights = await weightsResponse.arrayBuffer();
    const reference = new Float32Array(await referenceResponse.arrayBuffer());
    class HfContext extends WgpuExecutionContext {
      constructor(device, route) {
        super(device);
        this.route = route;
        this.replaced = 0;
      }
      recordShader(shader, args) {
        if (
          shader.name === 'aten.convolution.default' &&
          (this.route === 'matrix'
            ? recordHfMatrix(this, args)
            : recordHfConvolution(
                this,
                args,
                this.route.replace('-m32', ''),
                this.route.endsWith('-m32') ? 32 : 64
              ))
        ) {
          this.replaced++;
          return this;
        }
        return super.recordShader(shader, args);
      }
    }
    const routes = [
      'baseline',
      '1x1',
      '3x3',
      'both',
      '3x3-m32',
      'both-m32',
      ...(matrixAvailable ? ['matrix'] : []),
    ];
    const outputs = [];
    const contexts = routes.map((route) => {
      const ctx = route === 'baseline' ? new BaselineContext(device) : new HfContext(device, route);
      const input = ctx.tensor(options.precision, program.values[program.inputIds[0]].shape);
      input.setData(
        new array(Array.from({ length: input.numel }, (_, i) => (i % 251) / 250 - 0.5))
      );
      const out = ctx.tensor(options.precision, program.values[program.outputIds[0]].shape);
      out.setData(new array(out.numel).fill(NaN));
      outputs.push(out);
      ctx.recordProgram(program, weights, [input, out]);
      return ctx;
    });
    const timing = await measure(contexts);
    const validation = [];
    for (let version = 0; version < contexts.length; version++) {
      contexts[version].submit();
      await contexts[version].sync();
      const actual = new array(await outputs[version].getData());
      let maxAbs = 0,
        sumAbs = 0,
        beyondTolerance = 0;
      const tolerance = half ? 0.5 : 0.01;
      for (let i = 0; i < actual.length; i++) {
        if (!Number.isFinite(actual[i])) throw new Error('Nonfinite model output');
        const delta = Math.abs(actual[i] - reference[i]);
        maxAbs = Math.max(maxAbs, delta);
        sumAbs += delta;
        if (delta > tolerance) beyondTolerance++;
      }
      validation.push({
        maxAbs,
        meanAbs: sumAbs / actual.length,
        beyondTolerance,
        tolerance,
        allowedBeyondTolerance: half ? Math.floor(actual.length * 0.05) : 0,
        passesProjectReferenceTolerance: half
          ? beyondTolerance <= actual.length * 0.05
          : beyondTolerance === 0,
      });
    }
    if (validation.some((v) => !v.passesProjectReferenceTolerance))
      throw new Error('HF model variant failed the project PyTorch reference tolerance');
    results.push({
      category: 'model',
      name: 'yolo26n',
      routes,
      replacedConvolutions: contexts.map((ctx) => ctx.replaced ?? 0),
      ...timing,
      validation,
    });
    contexts.forEach((ctx) => ctx.destroy());
    if (errors.length) throw new Error(errors.join('\n'));
    const result = {
      precision: options.precision,
      upstreamRevision: '7b19a1ea4fd2abef3203991ca933405f12cfcc62',
      gpu: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
      matrixAvailable,
      matrixOracleChecks,
      maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
      subgroupInfo: {
        min: adapter.info.subgroupMinSize,
        max: adapter.info.subgroupMaxSize,
        matrixConfigs: Array.from(adapter.info.subgroupMatrixConfigs ?? [], (c) => ({
          componentType: c.componentType,
          resultComponentType: c.resultComponentType,
          M: c.M,
          N: c.N,
          K: c.K,
        })),
      },
      availableFeatures: [...adapter.features],
      browser: navigator.userAgent,
      method:
        'Pinned HF register-tiled Conv template, 64x64 and 32x64/8x4 register layouts, carried implicit gathering (published default mode=1), K=32 when reduction divisible by 32 else 16; FP32 accumulation preserved. Bindings and geometry pipeline constants adapted. This is not the full HF loader/route-selection policy; split-K excluded. Aligned 1x1 and implicit 3x3 subgroup-matrix routes measured separately when experimental features are available; published f32 wide/f16 narrow tiles, direct inputs when aligned, subgroup size pinned to 32. Identical buffers/graphs, commands recorded once. Kernel timings: repeated dispatches, GPU timestamps. Model timings: rotated alternating submit+sync wall clock, no camera/pre/post/presentation. CPU-oracle tests cover the adapters; raw shaders compared to F32 arithmetic; models validated against project PyTorch fixtures.',
      warmups: options.warmups ?? 8,
      samples: options.samples ?? 51,
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
