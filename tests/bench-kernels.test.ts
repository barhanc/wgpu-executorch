import { readFileSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { it } from 'vitest';
import { WgpuExecutionContext, createComputeBundle } from '../src';
import { KernelTimingContext } from './helpers/kernel-timing';

// Opt-in comparison against any commit, e.g.:
// WGPU_KERNEL_BASELINE=224da79 npx vitest run tests/bench-kernels.test.ts --disableConsoleIntercept
// Both versions use the same device, inputs and dispatch counts, with alternating
// measurement order. Raw samples are saved for inspecting variance.
it.skipIf(!process.env.WGPU_KERNEL_BASELINE)(
  'compares convolution and GEMM with a git baseline',
  async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No suitable GPU adapter found');
    const requiredFeatures = (['shader-f16', 'timestamp-query'] as GPUFeatureName[]).filter(
      (feature) => adapter.features.has(feature)
    );
    const device = await adapter.requestDevice({ requiredFeatures });
    const gpuTiming = device.features.has('timestamp-query');
    device.pushErrorScope('validation');
    const baseline = process.env.WGPU_KERNEL_BASELINE!;
    const report: unknown[] = [];
    const matrixCases = [
      [1, 64, 64, 64],
      [1, 256, 256, 256],
      [1, 1024, 1024, 1024],
      [1, 128, 256, 512],
      [1, 65, 68, 100],
      [1, 33, 36, 44],
      [2, 400, 32, 400],
      [2, 64, 400, 400],
    ];
    const convCases = [
      [64, 64, 56, 3, 1, 1],
      [960, 48, 14, 3, 1, 1],
      [64, 64, 80, 3, 2, 1],
      [32, 32, 40, 3, 1, 1],
      [24, 16, 448, 3, 1, 1],
      [48, 48, 112, 3, 1, 1],
      [48, 24, 224, 3, 1, 1],
      [8, 12, 17, 2, 1, 0],
    ];
    for (const dtype of ['float32', 'float16'] as const) {
      if (dtype === 'float16' && !device.features.has('shader-f16')) continue;
      const array = dtype === 'float16' ? Float16Array : Float32Array;
      const suffix = dtype === 'float16' ? '-f16' : '';
      const fill = (length: number) => {
        const data = new array(length);
        for (let i = 0; i < length; i++) data[i] = (((i * 17 + 11) % 61) - 30) / 64;
        return data;
      };
      for (const category of ['matrix', 'convolution', 'clamp']) {
        for (const shape of category === 'matrix' ? matrixCases : convCases) {
          const ctx = new WgpuExecutionContext(device);
          let path: string;
          let counts: number[];
          let params: GPUBuffer;
          let tensors;
          let constants: Record<string, number> = {};
          let out;
          if (category === 'matrix') {
            const [batch, m, k, n] = shape;
            const group = batch === 1 ? 'mm' : 'bmm';
            const variant = m >= 64 && n >= 64 && k >= 16 ? '64x64' : 'vec4';
            path = `src/shaders/${group}/${group}-${variant}${suffix}.wgsl.ts`;
            const a = ctx.tensor(dtype, [batch, m, k]).setData(fill(batch * m * k));
            const b = ctx.tensor(dtype, [batch, k, n]).setData(fill(batch * k * n));
            out = ctx.tensor(dtype, [batch, m, n]);
            params = ctx.uniformBuffer(
              new Uint32Array(batch === 1 ? [m, n, k, 0] : [batch, m, n, k])
            );
            tensors = [a, b, out];
            counts = [
              Math.ceil(n / (variant === '64x64' ? 64 : 32)),
              Math.ceil(m / (variant === '64x64' ? 64 : 32)),
              batch,
            ];
          } else {
            const [ic, oc, h, kernel, stride, pad] = shape;
            const oh = Math.floor((h + 2 * pad - kernel) / stride) + 1;
            const group = category === 'clamp' ? 'conv-with-clamp' : 'convolution';
            const prefix = category === 'clamp' ? 'conv-with-clamp' : 'conv2d';
            path = `src/shaders/${group}/${prefix}-tile${oc <= 32 ? '32' : '64'}${suffix}.wgsl.ts`;
            const input = ctx.tensor(dtype, [1, ic, h, h]).setData(fill(ic * h * h));
            const weight = ctx
              .tensor(dtype, [oc, ic, kernel, kernel])
              .setData(fill(oc * ic * kernel * kernel));
            const bias = ctx.tensor(dtype, [oc]).setData(fill(oc));
            out = ctx.tensor(dtype, [1, oc, oh, oh]);
            const data = new ArrayBuffer(category === 'clamp' ? 96 : 84);
            new Uint32Array(data).set(
              category === 'clamp'
                ? [
                    1,
                    ic,
                    h,
                    h,
                    oc,
                    oh,
                    oh,
                    kernel,
                    kernel,
                    stride,
                    stride,
                    pad,
                    pad,
                    1,
                    1,
                    1,
                    oc * oh * oh,
                    1,
                    ic,
                    0,
                    0,
                    0,
                  ]
                : [
                    1,
                    ic,
                    h,
                    h,
                    oc,
                    oh,
                    oh,
                    kernel,
                    kernel,
                    stride,
                    stride,
                    pad,
                    pad,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    0,
                  ]
            );
            if (category === 'clamp') {
              new Float32Array(data)[22] = 0;
              new Float32Array(data)[23] = Infinity;
            }
            params = ctx.uniformBuffer(data);
            tensors = [out, input, weight, bias];
            // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
            constants = { kernel_h: kernel, kernel_w: kernel };
            counts = [Math.ceil((oh * oh) / 64), Math.ceil(oc / (oc <= 32 ? 32 : 64)), 1];
          }
          const bindings = tensors.map((t, binding) => ({ binding, buffer: t.buffer }));
          bindings.push({ binding: bindings.length, buffer: params });
          const candidate = readFileSync(path, 'utf8').split('`')[1];
          device.pushErrorScope('validation');
          const original = execFileSync('git', ['show', `${baseline}:${path}`], {
            encoding: 'utf8',
          }).split('`')[1];
          // Apply the same geometry specialization when the baseline supports it;
          // otherwise this comparison would also measure an already-landed change.
          const beforeConstants = original.includes('override kernel_h') ? constants : {};
          const before = createComputeBundle(device, original, bindings, beforeConstants);
          const after = createComputeBundle(device, candidate, bindings, constants);
          const compileError = await device.popErrorScope();
          if (compileError) throw compileError;
          const contexts = [new KernelTimingContext(device), new KernelTimingContext(device)];
          // Amortize submission overhead and timestamp quantization on short kernels.
          // Scale by mathematical work, independently of GPU vendor or model.
          const work =
            category === 'matrix'
              ? shape.reduce((product, dimension) => product * dimension, 1)
              : out.numel * shape[0] * shape[3] * shape[3];
          const repeats = Math.min(512, Math.max(8, Math.ceil(2e9 / work)));
          for (let version = 0; version < 2; version++) {
            const bundle = version === 0 ? before : after;
            for (let i = 0; i < repeats; i++)
              contexts[version].addDispatch({
                pipeline: bundle.pipeline,
                bindGroup: bundle.bindGroup,
                workgroupCountX: counts[0],
                workgroupCountY: counts[1],
                workgroupCountZ: counts[2],
              });
            for (let i = 0; i < 3; i++) {
              contexts[version].submit();
              await contexts[version].sync();
            }
          }
          const samples: number[][] = [[], []];
          for (let i = 0; i < 11; i++)
            for (const version of i % 2 ? [1, 0] : [0, 1]) {
              const start = performance.now();
              contexts[version].submit();
              await contexts[version].sync();
              samples[version].push(
                (gpuTiming ? contexts[version].gpuMs : performance.now() - start) / repeats
              );
            }
          const median = samples.map((s) => [...s].sort((a, b) => a - b)[5]);
          const entry = {
            category,
            dtype,
            shape,
            repeats,
            beforeMs: median[0],
            afterMs: median[1],
            speedup: median[0] / median[1],
            samples,
          };
          report.push(entry);
          // eslint-disable-next-line no-console
          console.log(JSON.stringify({ ...entry, samples: undefined }));
          ctx.destroy();
          contexts.forEach((c) => c.destroy());
        }
      }
    }
    const error = await device.popErrorScope();
    if (error) throw error;
    writeFileSync(
      process.env.WGPU_KERNEL_RESULTS ?? resolve(tmpdir(), 'wgpu-conv-mm-bench.json'),
      JSON.stringify(
        {
          baseline,
          gpu: { vendor: adapter.info.vendor, architecture: adapter.info.architecture },
          method: `8–512 recorded dispatches per submit, scaled by operation count; submit + await sync; 3 warmups, 11 alternating samples per version; median milliseconds per dispatch; ${gpuTiming ? 'GPU timestamps' : 'wall clock fallback'}`,
          results: report,
        },
        null,
        2
      )
    );
    device.destroy();
  },
  600_000
);
