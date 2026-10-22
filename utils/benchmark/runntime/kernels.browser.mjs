// Native ruNNtime operators: packing/capture outside timing, no shader ports.
export async function compareRunntimeKernels({ repo, device, root, lib, gpuTime }) {
  const { WgpuExecutionContext, shaders } = await import(
    /* @vite-ignore */ '/@fs' + repo + '/src/index.ts'
  );
  lib.initRunntime(root);
  const fill = (n) => Float32Array.from({ length: n }, (_, i) => (((i * 17 + 11) % 61) - 30) / 128);
  const median = (a) => [...a].sort((x, y) => x - y)[a.length >> 1];
  const results = [];
  const programs = await Promise.all(
    ['yolo26n', 'depthart'].map(async (name) => {
      const response = await fetch(`/assets/${name}/program.json`);
      if (!response.ok) throw new Error('Missing model descriptor for kernel membership');
      return [name, await response.json()];
    })
  );
  function membership(c) {
    return Object.fromEntries(
      programs.map(([name, program]) => [
        name,
        program.chain.filter((op) => {
          const a = program.values[op.args[0]]?.shape,
            b = program.values[op.args[1]]?.shape;
          if (!a || !b) return false;
          if (c.op === 'conv2d') {
            const [ic, oc, h, k, stride, groups] = c.shape;
            return (
              ['aten.convolution.default', 'et_vk.conv_with_clamp.default'].includes(op.name) &&
              a.length === 4 &&
              a[1] === ic &&
              a[2] === h &&
              a[3] === h &&
              b[0] === oc &&
              b[1] === ic / groups &&
              b[2] === k &&
              b[3] === k &&
              program.values[op.args[3]]?.items?.[0] === stride
            );
          }
          const [m, k, n] = c.shape;
          return (
            ['aten.mm.default', 'aten.bmm.default'].includes(op.name) &&
            a.at(-2) === m &&
            a.at(-1) === k &&
            b.at(-1) === n
          );
        }).length,
      ])
    );
  }
  const cases = [
    ...[
      [48, 64, 160, 1, 1, 1],
      [64, 64, 80, 1, 1, 1],
      [192, 128, 40, 1, 1, 1],
      [256, 256, 20, 1, 1, 1],
      [80, 80, 20, 1, 1, 1],
      [32, 16, 80, 1, 1, 1],
      [960, 1280, 1, 1, 1, 1],
      [3, 16, 640, 3, 2, 1],
      [16, 32, 320, 3, 2, 1],
      [64, 64, 160, 3, 2, 1],
      [128, 256, 40, 3, 2, 1],
      [64, 16, 80, 3, 1, 1],
      [16, 8, 160, 3, 1, 1],
      [64, 64, 80, 3, 1, 64],
      [576, 576, 14, 5, 1, 576],
    ].map((shape) => ({ op: 'conv2d', dtype: 'float16', shape })),
    ...['float16', 'float32'].flatMap((dtype) =>
      [
        [400, 32, 400],
        [64, 400, 400],
        [128, 384, 1152],
        [32, 1536, 384],
        [1, 288, 864],
      ].map((shape) => ({ op: 'matmul', dtype, shape }))
    ),
  ];
  for (const c of cases) {
    const ours = new WgpuExecutionContext(device),
      oracle = new WgpuExecutionContext(device),
      ex = lib.gpuExecutor(root);
    const array = c.dtype === 'float16' ? Float16Array : Float32Array,
      dtype = c.dtype === 'float16' ? 'f16' : 'f32';
    const nativeInputs = [];
    let build,
      output,
      referenceOutput,
      unlayout = (a) => a;
    if (c.op === 'conv2d') {
      const [ic, oc, h, k, s, groups] = c.shape,
        pad = (k - 1) >> 1,
        oh = Math.floor((h + 2 * pad - k) / s) + 1;
      const xv = fill(ic * h * h),
        wv = fill(oc * (ic / groups) * k * k),
        bv = fill(oc);
      const x = ours.tensor(c.dtype, [1, ic, h, h]).setData(new array(xv));
      const w = ours.tensor(c.dtype, [oc, ic / groups, k, k]).setData(new array(wv));
      const b = ours.tensor(c.dtype, [oc]).setData(new array(bv));
      output = ours.tensor(c.dtype, [1, oc, oh, oh]);
      ours.recordShader(shaders.convolution, [
        x,
        w,
        b,
        [s, s],
        [pad, pad],
        [1, 1],
        false,
        [0, 0],
        groups,
        output,
      ]);
      const x32 = oracle.tensor('float32', x.shape).setData(xv),
        w32 = oracle.tensor('float32', w.shape).setData(wv),
        b32 = oracle.tensor('float32', b.shape).setData(bv);
      referenceOutput = oracle.tensor('float32', output.shape);
      oracle.recordShader(shaders.convolution, [
        x32,
        w32,
        b32,
        [s, s],
        [pad, pad],
        [1, 1],
        false,
        [0, 0],
        groups,
        referenceOutput,
      ]);
      const xm = lib.hwc4Meta(ic, h, h),
        c4 = xm.elems / (h * h),
        packedX = new Float32Array(xm.elems);
      for (let channel = 0; channel < ic; channel++)
        for (let p = 0; p < h * h; p++) packedX[p * c4 + channel] = xv[channel * h * h + p];
      const nx = lib.tensor(root, packedX, xm),
        bits = lib.f32ArrayToF16Bits(wv).data;
      const packedW =
        groups === 1 ? lib.packConvHwc4F16(bits, oc, ic, k, k) : lib.packDwHwc4F16(bits, ic, k, k);
      const nw = lib.materialized(
        { elems: wv.length, dtype: 'f16', dims: [oc, (ic / groups) * k * k], layout: 'hwc4' },
        lib.uploadF16(root, packedW)
      );
      const nb = lib.tensor(root, bv, { elems: oc, dtype: 'f16', dims: [oc] });
      nativeInputs.push(nx, nw, nb);
      build = () => lib.conv2d(nx, nw, nb, { kernelSize: k, stride: s, padding: pad, groups });
      unlayout = (raw) => {
        const result = new Float32Array(oc * oh * oh),
          oc4 = Math.ceil(oc / 4) * 4;
        for (let channel = 0; channel < oc; channel++)
          for (let p = 0; p < oh * oh; p++) result[channel * oh * oh + p] = raw[p * oc4 + channel];
        return result;
      };
    } else {
      const [m, k, n] = c.shape,
        av = fill(m * k),
        bv = fill(k * n);
      const a = ours.tensor(c.dtype, [m, k]).setData(new array(av)),
        b = ours.tensor(c.dtype, [k, n]).setData(new array(bv));
      output = ours.tensor(c.dtype, [m, n]);
      ours.recordShader(shaders.mm, [a, b, output]);
      const a32 = oracle.tensor('float32', [m, k]).setData(av),
        b32 = oracle.tensor('float32', [k, n]).setData(bv);
      referenceOutput = oracle.tensor('float32', [m, n]);
      oracle.recordShader(shaders.mm, [a32, b32, referenceOutput]);
      const na = lib.tensor(root, av, { elems: m * k, dtype, dims: [m, k] }),
        nb = lib.tensor(root, bv, { elems: k * n, dtype, dims: [k, n] });
      nativeInputs.push(na, nb);
      build = () => lib.matmul(na, nb);
    }
    device.pushErrorScope('validation');
    const captured = ex.captureFrame(() => {
      const out = build();
      lib.evalValues([out], ex);
      return out;
    });
    const validationError = await device.popErrorScope();
    if (validationError) throw new Error(validationError.message);
    const runs = [() => ours.submit(), () => captured.replay()];
    const repeats = 16;
    for (let i = 0; i < 8; i++)
      for (const run of runs) {
        for (let r = 0; r < repeats; r++) run();
        await device.queue.onSubmittedWorkDone();
      }
    const samples = [[], []];
    for (let i = 0; i < 41; i++)
      for (const engine of i & 1 ? [1, 0] : [0, 1])
        samples[engine].push(
          (await gpuTime(() => {
            for (let r = 0; r < repeats; r++) runs[engine]();
          })) / repeats
        );
    oracle.submit();
    const reference = new array(new Float32Array(await referenceOutput.getData()));
    const oursValues = new array(await output.getData());
    const nativeValues = unlayout(
      await ex.readback(captured.targets.buffer, { elems: captured.targets.shape.elems, dtype })
    );
    const validation = [oursValues, nativeValues].map((values) => {
      if (values.length !== reference.length)
        throw new Error('Native kernel output shape mismatch');
      let maxAbs = 0,
        error2 = 0,
        signal2 = 0;
      for (let i = 0; i < values.length; i++) {
        if (!Number.isFinite(values[i])) throw new Error('Nonfinite native kernel output');
        const error = values[i] - reference[i];
        maxAbs = Math.max(maxAbs, Math.abs(error));
        error2 += error * error;
        signal2 += reference[i] * reference[i];
      }
      return {
        maxAbsVsRoundedF32: maxAbs,
        relativeL2VsRoundedF32: Math.sqrt(error2 / Math.max(signal2, 1e-30)),
      };
    });
    // Detect indexing/packing failures; mixed-precision accumulation need not be bitwise equal.
    if (validation.some((v) => v.relativeL2VsRoundedF32 > 0.03))
      throw new Error(
        `Kernel reference mismatch ${JSON.stringify(c)}: ${JSON.stringify(validation)}`
      );
    const medians = samples.map(median);
    results.push({
      ...c,
      repeats,
      modelReferenceCounts: membership(c),
      engines: ['wgpu-executorch', 'runntime-native'],
      gpuMs: medians,
      runntimeOverOurs: medians[1] / medians[0],
      samples,
      validation,
    });
    nativeInputs.forEach((v) => v.buffer.destroy());
    ex.dispose();
    ours.destroy();
    oracle.destroy();
  }
  return {
    method:
      'Published package implementation plus named exports only. Native conv uses HWC4/mat4 weight packing; ours uses NCHW. Packing, upload, graph capture and correctness readback excluded. No activation/residual fusion in these raw cases. Both libraries select their own kernels. 8 warmups, 41 alternating GPU timestamp samples, 16 replays per sample, one recorded invocation per submit, summed GPU compute duration divided by 16. Validation requires relative L2 error <=3% for these FP16 accumulation workloads and records max absolute error. F16 numerical differences compared with F32 on identical rounded dyadic inputs. These are native-layout cached-input microbenchmarks; use whole-model results for application performance.',
    results,
  };
}
