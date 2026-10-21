import { WgpuExecutionContext, shaders, type Program } from '../src';
import {
  WgpuExecutionContext as BaselineContext,
  shaders as baselineShaders,
} from './baseline-main/src';
import { packConvHwc4F32, packDwHwc4F32, packConvHwc4F16, packDwHwc4F16 } from '../utils/pack_hwc4';

interface BenchItem {
  category: string;
  name: string;
  shape: string;
  baselineMs: number | null;
  fp32Ms: number;
  fp16Ms: number | null;
  speedupFp32: string;
  speedupFp16: string;
}

function makeF32(numel: number): Float32Array {
  const arr = new Float32Array(numel);
  for (let i = 0; i < numel; i++) arr[i] = (i % 251) / 250 - 0.5;
  return arr;
}

function makeF16(numel: number): Float16Array {
  const arr = new Float16Array(numel);
  for (let i = 0; i < numel; i++) arr[i] = (i % 251) / 250 - 0.5;
  return arr;
}

async function run() {
  const statusEl = document.getElementById('status')!;
  const infoEl = document.getElementById('info')!;
  const tableEl = document.getElementById('results-table')!;
  const bodyEl = document.getElementById('results-body')!;
  const jsonEl = document.getElementById('json-output')!;

  if (!navigator.gpu) {
    statusEl.innerText = 'WebGPU is not supported in this browser.';
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    statusEl.innerText = 'Failed to obtain WebGPU adapter.';
    return;
  }

  const hasF16 = adapter.features.has('shader-f16');
  const requiredFeatures: GPUFeatureName[] = [];
  if (hasF16) requiredFeatures.push('shader-f16');

  const device = await adapter.requestDevice({ requiredFeatures });
  const info = adapter.info;

  infoEl.innerHTML = `
    <p><strong>GPU Hardware:</strong> ${info.vendor || 'Unknown'} (${info.architecture || 'Unknown architecture'}) - ${info.device || 'Default device'}</p>
    <p><strong>shader-f16 Feature:</strong> ${hasF16 ? '<span style="color:#98c379">YES (Native 16-bit ALUs & memory bandwidth enabled)</span>' : '<span style="color:#e06c75">NO</span>'}</p>
  `;

  const results: BenchItem[] = [];
  bodyEl.innerHTML = '';

  const addRow = (item: BenchItem) => {
    results.push(item);
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.category}</td>
      <td><strong>${item.name}</strong></td>
      <td><code>${item.shape}</code></td>
      <td style="text-align: right;">${item.baselineMs !== null ? item.baselineMs.toFixed(3) + ' ms' : 'N/A'}</td>
      <td style="text-align: right;">${item.fp32Ms.toFixed(3)} ms</td>
      <td style="text-align: right;">${item.fp16Ms !== null ? item.fp16Ms.toFixed(3) + ' ms' : 'N/A'}</td>
      <td class="speedup" style="text-align: right;">${item.speedupFp32}</td>
      <td class="speedup" style="text-align: right;">${item.speedupFp16}</td>
    `;
    bodyEl.appendChild(tr);
    tableEl.style.display = 'table';
  };

  // 0. End-to-End ExecuTorch Models
  const benchmarkModel = async (
    modelName: string,
    displayName: string,
    inShape: number[],
    outShape: number[],
    iters = 5
  ) => {
    statusEl.innerText = `Benchmarking full model: ${displayName} (Baseline - main)...`;

    let baselineMs: number | null = null;
    try {
      const progRes = await fetch(`/assets/${modelName}/program.json`);
      const program = (await progRes.json()) as Program;
      const weightsRes = await fetch(`/assets/${modelName}/weights.bin`);
      const weights = await weightsRes.arrayBuffer();

      const baseCtx = new BaselineContext(device);
      const inNumel = inShape.reduce((a, b) => a * b, 1);
      const inp = baseCtx.tensor('float32', inShape).setData(makeF32(inNumel));
      const out = baseCtx.tensor('float32', outShape);

      baseCtx.recordProgram(program, weights, [inp, out]);
      baseCtx.submit();
      await baseCtx.sync();

      const tb = performance.now();
      for (let i = 0; i < iters; i++) baseCtx.submit();
      await baseCtx.sync();
      baselineMs = (performance.now() - tb) / iters;
      baseCtx.destroy();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn(`Could not run ${displayName} Baseline:`, e);
    }

    statusEl.innerText = `Benchmarking full model: ${displayName} (Opt FP32)...`;
    let fp32Ms = 0;
    try {
      const progRes = await fetch(`/assets/${modelName}/program.json`);
      const program = (await progRes.json()) as Program;
      const weightsRes = await fetch(`/assets/${modelName}/weights.bin`);
      const weights = await weightsRes.arrayBuffer();

      const ctx32 = new WgpuExecutionContext(device);
      const inNumel = inShape.reduce((a, b) => a * b, 1);
      const inp = ctx32.tensor('float32', inShape).setData(makeF32(inNumel));
      const out = ctx32.tensor('float32', outShape);

      ctx32.recordProgram(program, weights, [inp, out]);
      ctx32.submit();
      await ctx32.sync();

      const t0 = performance.now();
      for (let i = 0; i < iters; i++) ctx32.submit();
      await ctx32.sync();
      fp32Ms = (performance.now() - t0) / iters;
      ctx32.destroy();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.warn(`Could not run ${displayName} FP32:`, e);
      return;
    }

    let fp16Ms: number | null = null;
    let speedupFp16 = 'N/A';
    if (hasF16) {
      statusEl.innerText = `Benchmarking full model: ${displayName} (Native FP16)...`;
      try {
        const progRes = await fetch(`/assets/${modelName}/program-fp16.json`);
        const program = (await progRes.json()) as Program;
        const weightsRes = await fetch(`/assets/${modelName}/weights-fp16.bin`);
        const weights = await weightsRes.arrayBuffer();

        const ctx16 = new WgpuExecutionContext(device);
        const inNumel = inShape.reduce((a, b) => a * b, 1);
        const inp = ctx16.tensor('float16', inShape).setData(makeF16(inNumel));
        const out = ctx16.tensor('float16', outShape);

        ctx16.recordProgram(program, weights, [inp, out]);
        ctx16.submit();
        await ctx16.sync();

        const t1 = performance.now();
        for (let i = 0; i < iters; i++) ctx16.submit();
        await ctx16.sync();
        fp16Ms = (performance.now() - t1) / iters;
        speedupFp16 = baselineMs
          ? `${(baselineMs / fp16Ms).toFixed(2)}x`
          : `${(fp32Ms / fp16Ms).toFixed(2)}x`;
        ctx16.destroy();
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn(`Could not run ${displayName} FP16:`, e);
      }
    }

    const speedupFp32 = baselineMs ? `${(baselineMs / fp32Ms).toFixed(2)}x` : '1.00x';

    addRow({
      category: 'End-to-End Model',
      name: displayName,
      shape: `[${inShape.join(',')}] → [${outShape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  };

  await benchmarkModel(
    'depthart',
    'DepthART Monocular Depth',
    [1, 3, 448, 448],
    [1, 1, 448, 448],
    4
  );
  await benchmarkModel('yolo26n', 'YOLO26-N Object Detection', [1, 3, 640, 640], [1, 84, 8400], 4);
  await benchmarkModel(
    'selfie-segmentation',
    'Selfie Segmentation',
    [1, 3, 256, 256],
    [1, 1, 256, 256],
    5
  );

  // 1. Matrix Multiplication (MM)
  const mmConfigs = [
    [256, 256, 25],
    [512, 512, 20],
    [1024, 1024, 15],
    [2048, 2048, 8],
  ] as const;

  for (const [M, N, iters] of mmConfigs) {
    statusEl.innerText = `Benchmarking MM ${M}x${N}...`;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const ba = baseCtx.tensor('float32', [M, N]).setData(makeF32(M * N));
    const bb = baseCtx.tensor('float32', [N, N]).setData(makeF32(N * N));
    const bout = baseCtx.tensor('float32', [M, N]);
    baseCtx.recordShader(baselineShaders.mm, [ba, bb, bout]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < iters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / iters;
    baseCtx.destroy();

    // Optimized FP32
    const ctx32 = new WgpuExecutionContext(device);
    const a32 = ctx32.tensor('float32', [M, N]).setData(makeF32(M * N));
    const b32 = ctx32.tensor('float32', [N, N]).setData(makeF32(N * N));
    const out32 = ctx32.tensor('float32', [M, N]);
    ctx32.recordShader(shaders.mm, [a32, b32, out32]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < iters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / iters;
    ctx32.destroy();

    // Native FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const a16 = ctx16.tensor('float16', [M, N]).setData(makeF16(M * N));
      const b16 = ctx16.tensor('float16', [N, N]).setData(makeF16(N * N));
      const out16 = ctx16.tensor('float16', [M, N]);
      ctx16.recordShader(shaders.mm, [a16, b16, out16]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < iters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / iters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Linear Algebra',
      name: `MatMul (MM ${M >= 1024 ? '64x64' : 'Vec4'})`,
      shape: `[${M}, ${N}] * [${N}, ${N}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 2. Batched Matrix Multiplication (BMM)
  const bmmConfigs = [
    [4, 64, 64, 30],
    [8, 128, 128, 20],
  ] as const;

  for (const [B, M, N, iters] of bmmConfigs) {
    statusEl.innerText = `Benchmarking BMM [${B}, ${M}, ${N}]...`;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const ba = baseCtx.tensor('float32', [B, M, N]).setData(makeF32(B * M * N));
    const bb = baseCtx.tensor('float32', [B, N, N]).setData(makeF32(B * N * N));
    const bout = baseCtx.tensor('float32', [B, M, N]);
    baseCtx.recordShader(baselineShaders.bmm, [ba, bb, bout]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < iters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / iters;
    baseCtx.destroy();

    // Optimized FP32
    const ctx32 = new WgpuExecutionContext(device);
    const a32 = ctx32.tensor('float32', [B, M, N]).setData(makeF32(B * M * N));
    const b32 = ctx32.tensor('float32', [B, N, N]).setData(makeF32(B * N * N));
    const out32 = ctx32.tensor('float32', [B, M, N]);
    ctx32.recordShader(shaders.bmm, [a32, b32, out32]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < iters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / iters;
    ctx32.destroy();

    // Native FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const a16 = ctx16.tensor('float16', [B, M, N]).setData(makeF16(B * M * N));
      const b16 = ctx16.tensor('float16', [B, N, N]).setData(makeF16(B * N * N));
      const out16 = ctx16.tensor('float16', [B, M, N]);
      ctx16.recordShader(shaders.bmm, [a16, b16, out16]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < iters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / iters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Linear Algebra',
      name: 'Batch MatMul (BMM)',
      shape: `[${B}, ${M}, ${N}] * [${B}, ${N}, ${N}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 3. Convolutions (Tile64 3x3, Pointwise 1x1, Depthwise, ConvWithClamp)
  const convConfigs = [
    {
      shader: shaders.convolution,
      category: 'Convolution',
      label: 'Conv2D 3x3 (Tile64)',
      inShape: [1, 64, 56, 56],
      wShape: [64, 64, 3, 3],
      groups: 1,
      stride: [1, 1],
      pad: [1, 1],
      iters: 20,
    },
    {
      shader: shaders.convolution,
      category: 'Convolution',
      label: 'Conv2D 1x1 (Tile64)',
      inShape: [1, 128, 56, 56],
      wShape: [128, 128, 1, 1],
      groups: 1,
      stride: [1, 1],
      pad: [0, 0],
      iters: 20,
    },
    {
      shader: shaders.convolution,
      category: 'Convolution',
      label: 'Conv2D 3x3 Depthwise',
      inShape: [1, 64, 56, 56],
      wShape: [64, 1, 3, 3],
      groups: 64,
      stride: [1, 1],
      pad: [1, 1],
      iters: 25,
    },
    {
      shader: shaders.convWithClamp,
      category: 'Convolution',
      label: 'ConvWithClamp 3x3 (Tile64)',
      inShape: [1, 64, 56, 56],
      wShape: [64, 64, 3, 3],
      groups: 1,
      stride: [1, 1],
      pad: [1, 1],
      iters: 20,
    },
    {
      shader: shaders.convWithClamp,
      category: 'Convolution',
      label: 'ConvWithClamp 1x1 (Tile64)',
      inShape: [1, 128, 56, 56],
      wShape: [128, 128, 1, 1],
      groups: 1,
      stride: [1, 1],
      pad: [0, 0],
      iters: 20,
    },
  ];

  for (const c of convConfigs) {
    statusEl.innerText = `Benchmarking ${c.label}...`;

    const inNumel = c.inShape.reduce((a, b) => a * b, 1);
    const wNumel = c.wShape.reduce((a, b) => a * b, 1);
    const outShape = [c.inShape[0], c.wShape[0], c.inShape[2], c.inShape[3]];

    // Baseline
    const baseCtx = new BaselineContext(device);
    const bin = baseCtx.tensor('float32', c.inShape).setData(makeF32(inNumel));
    const bw = baseCtx.tensor('float32', c.wShape).setData(makeF32(wNumel));
    const bb = baseCtx.tensor('float32', [c.wShape[0]]).setData(makeF32(c.wShape[0]));
    const bout = baseCtx.tensor('float32', outShape);
    if (c.shader === shaders.convolution) {
      baseCtx.recordShader(baselineShaders.convolution, [
        bin,
        bw,
        bb,
        c.stride,
        c.pad,
        [1, 1],
        false,
        [0, 0],
        c.groups,
        bout,
      ]);
    } else {
      baseCtx.recordShader(baselineShaders.convWithClamp, [
        bin,
        bw,
        bb,
        c.stride,
        c.pad,
        [1, 1],
        false,
        [0, 0],
        c.groups,
        0.0,
        6.0,
        bout,
      ]);
    }
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < c.iters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / c.iters;
    baseCtx.destroy();

    // FP32
    const ctx32 = new WgpuExecutionContext(device);
    const in32 = ctx32.tensor('float32', c.inShape).setData(makeF32(inNumel));
    const w32 = ctx32.tensor('float32', c.wShape).setData(makeF32(wNumel));
    const b32 = ctx32.tensor('float32', [c.wShape[0]]).setData(makeF32(c.wShape[0]));
    const out32 = ctx32.tensor('float32', outShape);

    if (c.shader === shaders.convolution) {
      ctx32.recordShader(shaders.convolution, [
        in32,
        w32,
        b32,
        c.stride,
        c.pad,
        [1, 1],
        false,
        [0, 0],
        c.groups,
        out32,
      ]);
    } else {
      ctx32.recordShader(shaders.convWithClamp, [
        in32,
        w32,
        b32,
        c.stride,
        c.pad,
        [1, 1],
        false,
        [0, 0],
        c.groups,
        0.0,
        6.0,
        out32,
      ]);
    }

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < c.iters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / c.iters;
    ctx32.destroy();

    // FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const in16 = ctx16.tensor('float16', c.inShape).setData(makeF16(inNumel));
      const w16 = ctx16.tensor('float16', c.wShape).setData(makeF16(wNumel));
      const b16 = ctx16.tensor('float16', [c.wShape[0]]).setData(makeF16(c.wShape[0]));
      const out16 = ctx16.tensor('float16', outShape);

      if (c.shader === shaders.convolution) {
        ctx16.recordShader(shaders.convolution, [
          in16,
          w16,
          b16,
          c.stride,
          c.pad,
          [1, 1],
          false,
          [0, 0],
          c.groups,
          out16,
        ]);
      } else {
        ctx16.recordShader(shaders.convWithClamp, [
          in16,
          w16,
          b16,
          c.stride,
          c.pad,
          [1, 1],
          false,
          [0, 0],
          c.groups,
          0.0,
          6.0,
          out16,
        ]);
      }

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < c.iters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / c.iters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: c.category,
      name: c.label,
      shape: `[${c.inShape.join(',')}] * [${c.wShape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 3.5. HWC4 Vectorized Convolutions (Channels-Last)
  const hwc4Configs = [
    {
      category: 'HWC4 Convolution',
      label: 'Conv2D 1x1 HWC4 (Barrier-free)',
      b: 1,
      cIn: 128,
      cOut: 128,
      h: 56,
      w: 56,
      kh: 1,
      kw: 1,
      stride: [1, 1],
      pad: [0, 0],
      groups: 1,
      withClamp: false,
      iters: 25,
    },
    {
      category: 'HWC4 Convolution',
      label: 'Conv2D 3x3 DW HWC4 (4x Vectorized)',
      b: 1,
      cIn: 64,
      cOut: 64,
      h: 56,
      w: 56,
      kh: 3,
      kw: 3,
      stride: [1, 1],
      pad: [1, 1],
      groups: 64,
      withClamp: false,
      iters: 25,
    },
    {
      category: 'HWC4 Convolution',
      label: 'ConvWithClamp 1x1 HWC4 (Barrier-free)',
      b: 1,
      cIn: 128,
      cOut: 128,
      h: 56,
      w: 56,
      kh: 1,
      kw: 1,
      stride: [1, 1],
      pad: [0, 0],
      groups: 1,
      withClamp: true,
      iters: 25,
    },
    {
      category: 'HWC4 Convolution',
      label: 'ConvWithClamp 3x3 DW HWC4 (4x Vectorized)',
      b: 1,
      cIn: 64,
      cOut: 64,
      h: 56,
      w: 56,
      kh: 3,
      kw: 3,
      stride: [1, 1],
      pad: [1, 1],
      groups: 64,
      withClamp: true,
      iters: 25,
    },
  ];

  for (const hc of hwc4Configs) {
    statusEl.innerText = `Benchmarking ${hc.label}...`;

    const inNchwShape = [hc.b, hc.cIn, hc.h, hc.w];
    const wNchwShape =
      hc.groups === hc.cOut ? [hc.cOut, 1, hc.kh, hc.kw] : [hc.cOut, hc.cIn, hc.kh, hc.kw];
    const outNchwShape = [hc.b, hc.cOut, hc.h, hc.w];

    const inNumel = inNchwShape.reduce((a, b) => a * b, 1);
    const wNumel = wNchwShape.reduce((a, b) => a * b, 1);

    // Baseline (NCHW)
    const baseCtx = new BaselineContext(device);
    const bin = baseCtx.tensor('float32', inNchwShape).setData(makeF32(inNumel));
    const bw = baseCtx.tensor('float32', wNchwShape).setData(makeF32(wNumel));
    const bb = baseCtx.tensor('float32', [hc.cOut]).setData(makeF32(hc.cOut));
    const bout = baseCtx.tensor('float32', outNchwShape);

    if (hc.withClamp) {
      baseCtx.recordShader(baselineShaders.convWithClamp, [
        bin,
        bw,
        bb,
        hc.stride,
        hc.pad,
        [1, 1],
        false,
        [0, 0],
        hc.groups,
        0.0,
        6.0,
        bout,
      ]);
    } else {
      baseCtx.recordShader(baselineShaders.convolution, [
        bin,
        bw,
        bb,
        hc.stride,
        hc.pad,
        [1, 1],
        false,
        [0, 0],
        hc.groups,
        bout,
      ]);
    }
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < hc.iters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / hc.iters;
    baseCtx.destroy();

    // FP32 HWC4
    const rawW32 = makeF32(wNumel);
    const packedW32 =
      hc.groups === hc.cOut
        ? packDwHwc4F32(rawW32, hc.cOut, hc.kh, hc.kw)
        : packConvHwc4F32(rawW32, hc.cOut, hc.cIn, hc.kh, hc.kw);

    const ctx32 = new WgpuExecutionContext(device);
    const in32 = ctx32.tensor('float32', [hc.b, hc.h, hc.w, hc.cIn]).setData(makeF32(inNumel));
    const w32 = ctx32.tensor('float32', [packedW32.length]).setData(packedW32);
    const b32 = ctx32.tensor('float32', [hc.cOut]).setData(makeF32(hc.cOut));
    const out32 = ctx32.tensor('float32', [hc.b, hc.h, hc.w, hc.cOut]);

    if (hc.withClamp) {
      ctx32.recordShader(shaders.convWithClampHwc4, [
        in32,
        w32,
        b32,
        hc.stride,
        hc.pad,
        [1, 1],
        false,
        [0, 0],
        hc.groups,
        0.0,
        6.0,
        out32,
      ]);
    } else {
      ctx32.recordShader(shaders.convolutionHwc4, [
        in32,
        w32,
        b32,
        hc.stride,
        hc.pad,
        [1, 1],
        false,
        [0, 0],
        hc.groups,
        out32,
      ]);
    }
    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < hc.iters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / hc.iters;
    ctx32.destroy();

    // FP16 HWC4
    let fp16Ms: number | null = null;
    if (hasF16) {
      const rawW16 = makeF16(wNumel);
      const packedW16 =
        hc.groups === hc.cOut
          ? packDwHwc4F16(rawW16, hc.cOut, hc.kh, hc.kw)
          : packConvHwc4F16(rawW16, hc.cOut, hc.cIn, hc.kh, hc.kw);

      const ctx16 = new WgpuExecutionContext(device);
      const in16 = ctx16.tensor('float16', [hc.b, hc.h, hc.w, hc.cIn]).setData(makeF16(inNumel));
      const w16 = ctx16.tensor('float16', [packedW16.length]).setData(packedW16);
      const b16 = ctx16.tensor('float16', [hc.cOut]).setData(makeF16(hc.cOut));
      const out16 = ctx16.tensor('float16', [hc.b, hc.h, hc.w, hc.cOut]);

      if (hc.withClamp) {
        ctx16.recordShader(shaders.convWithClampHwc4, [
          in16,
          w16,
          b16,
          hc.stride,
          hc.pad,
          [1, 1],
          false,
          [0, 0],
          hc.groups,
          0.0,
          6.0,
          out16,
        ]);
      } else {
        ctx16.recordShader(shaders.convolutionHwc4, [
          in16,
          w16,
          b16,
          hc.stride,
          hc.pad,
          [1, 1],
          false,
          [0, 0],
          hc.groups,
          out16,
        ]);
      }
      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < hc.iters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / hc.iters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: hc.category,
      name: hc.label,
      shape: `[${inNchwShape.join(',')}] * [${wNchwShape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 4. Activations
  const actConfigs = [
    { name: 'ReLU (Vec4)', shader: shaders.relu, baseShader: baselineShaders.relu },
    { name: 'Sigmoid (Vec4)', shader: shaders.sigmoid, baseShader: baselineShaders.sigmoid },
    { name: 'Hardswish (Vec4)', shader: shaders.hardswish, baseShader: baselineShaders.hardswish },
  ];

  const actShape = [1, 64, 56, 56];
  const actNumel = 1 * 64 * 56 * 56;
  const actIters = 40;

  for (const a of actConfigs) {
    statusEl.innerText = `Benchmarking ${a.name}...`;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const bin = baseCtx.tensor('float32', actShape).setData(makeF32(actNumel));
    const bout = baseCtx.tensor('float32', actShape);
    baseCtx.recordShader(a.baseShader, [bin, bout]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < actIters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / actIters;
    baseCtx.destroy();

    // FP32
    const ctx32 = new WgpuExecutionContext(device);
    const in32 = ctx32.tensor('float32', actShape).setData(makeF32(actNumel));
    const out32 = ctx32.tensor('float32', actShape);
    ctx32.recordShader(a.shader, [in32, out32]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < actIters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / actIters;
    ctx32.destroy();

    // FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const in16 = ctx16.tensor('float16', actShape).setData(makeF16(actNumel));
      const out16 = ctx16.tensor('float16', actShape);
      ctx16.recordShader(a.shader, [in16, out16]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < actIters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / actIters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Activation',
      name: a.name,
      shape: `[${actShape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 5. Reductions (Softmax, MeanDim)
  {
    statusEl.innerText = 'Benchmarking Softmax (Vec4)...';
    const smShape = [128, 1024];
    const smNumel = 128 * 1024;
    const smIters = 30;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const bin = baseCtx.tensor('float32', smShape).setData(makeF32(smNumel));
    const bout = baseCtx.tensor('float32', smShape);
    baseCtx.recordShader(baselineShaders.softmax, [bin, 1, false, bout]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < smIters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / smIters;
    baseCtx.destroy();

    // FP32
    const ctx32 = new WgpuExecutionContext(device);
    const in32 = ctx32.tensor('float32', smShape).setData(makeF32(smNumel));
    const out32 = ctx32.tensor('float32', smShape);
    ctx32.recordShader(shaders.softmax, [in32, 1, false, out32]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < smIters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / smIters;
    ctx32.destroy();

    // FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const in16 = ctx16.tensor('float16', smShape).setData(makeF16(smNumel));
      const out16 = ctx16.tensor('float16', smShape);
      ctx16.recordShader(shaders.softmax, [in16, 1, false, out16]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < smIters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / smIters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Reduction',
      name: 'Softmax (Last-Dim Vec4)',
      shape: `[${smShape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 6. Pooling & Resampling
  {
    statusEl.innerText = 'Benchmarking MaxPool2D (2D Spatial)...';
    const pShape = [1, 64, 112, 112];
    const pNumel = 1 * 64 * 112 * 112;
    const pIters = 25;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const bin = baseCtx.tensor('float32', pShape).setData(makeF32(pNumel));
    const boutVals = baseCtx.tensor('float32', [1, 64, 56, 56]);
    const boutIdx = baseCtx.tensor('int32', [1, 64, 56, 56]);
    baseCtx.recordShader(baselineShaders.maxPool2d, [
      bin,
      [2, 2],
      [2, 2],
      [0, 0],
      [1, 1],
      false,
      [boutVals, boutIdx],
    ]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < pIters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / pIters;
    baseCtx.destroy();

    // FP32
    const ctx32 = new WgpuExecutionContext(device);
    const in32 = ctx32.tensor('float32', pShape).setData(makeF32(pNumel));
    const outVals32 = ctx32.tensor('float32', [1, 64, 56, 56]);
    const outIdx32 = ctx32.tensor('int32', [1, 64, 56, 56]);
    ctx32.recordShader(shaders.maxPool2d, [
      in32,
      [2, 2],
      [2, 2],
      [0, 0],
      [1, 1],
      false,
      [outVals32, outIdx32],
    ]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < pIters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / pIters;
    ctx32.destroy();

    // FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const in16 = ctx16.tensor('float16', pShape).setData(makeF16(pNumel));
      const outVals16 = ctx16.tensor('float16', [1, 64, 56, 56]);
      const outIdx16 = ctx16.tensor('int32', [1, 64, 56, 56]);
      ctx16.recordShader(shaders.maxPool2d, [
        in16,
        [2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        false,
        [outVals16, outIdx16],
      ]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < pIters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / pIters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Pooling & Resampling',
      name: 'MaxPool2D (2D Spatial Grid)',
      shape: `[${pShape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 7. Elementwise Arithmetic (Mul & Add)
  {
    statusEl.innerText = 'Benchmarking Elementwise Mul (Vec4)...';
    const shape = [1, 64, 56, 56];
    const numel = 1 * 64 * 56 * 56;
    const iters = 40;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const ba = baseCtx.tensor('float32', shape).setData(makeF32(numel));
    const bb = baseCtx.tensor('float32', shape).setData(makeF32(numel));
    const bout = baseCtx.tensor('float32', shape);
    baseCtx.recordShader(baselineShaders.mul, [ba, bb, bout]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < iters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / iters;
    baseCtx.destroy();

    // FP32
    const ctx32 = new WgpuExecutionContext(device);
    const a32 = ctx32.tensor('float32', shape).setData(makeF32(numel));
    const b32 = ctx32.tensor('float32', shape).setData(makeF32(numel));
    const out32 = ctx32.tensor('float32', shape);
    ctx32.recordShader(shaders.mul, [a32, b32, out32]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < iters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / iters;
    ctx32.destroy();

    // FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const a16 = ctx16.tensor('float16', shape).setData(makeF16(numel));
      const b16 = ctx16.tensor('float16', shape).setData(makeF16(numel));
      const out16 = ctx16.tensor('float16', shape);
      ctx16.recordShader(shaders.mul, [a16, b16, out16]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < iters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / iters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Elementwise Math',
      name: 'Elementwise Mul (Vec4)',
      shape: `[${shape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // 8. Data Movement (Permute 2D Matrix Transpose)
  {
    statusEl.innerText = 'Benchmarking Matrix Transpose (Permute)...';
    const shape = [1024, 1024];
    const numel = 1024 * 1024;
    const iters = 25;

    // Baseline
    const baseCtx = new BaselineContext(device);
    const ba = baseCtx.tensor('float32', shape).setData(makeF32(numel));
    const bout = baseCtx.tensor('float32', shape);
    baseCtx.recordShader(baselineShaders.permuteCopy, [ba, [1, 0], bout]);
    baseCtx.submit();
    await baseCtx.sync();
    const tb = performance.now();
    for (let i = 0; i < iters; i++) baseCtx.submit();
    await baseCtx.sync();
    const baselineMs = (performance.now() - tb) / iters;
    baseCtx.destroy();

    // FP32
    const ctx32 = new WgpuExecutionContext(device);
    const a32 = ctx32.tensor('float32', shape).setData(makeF32(numel));
    const out32 = ctx32.tensor('float32', shape);
    ctx32.recordShader(shaders.permuteCopy, [a32, [1, 0], out32]);

    ctx32.submit();
    await ctx32.sync();

    const t0 = performance.now();
    for (let i = 0; i < iters; i++) ctx32.submit();
    await ctx32.sync();
    const fp32Ms = (performance.now() - t0) / iters;
    ctx32.destroy();

    // FP16
    let fp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const a16 = ctx16.tensor('float16', shape).setData(makeF16(numel));
      const out16 = ctx16.tensor('float16', shape);
      ctx16.recordShader(shaders.permuteCopy, [a16, [1, 0], out16]);

      ctx16.submit();
      await ctx16.sync();

      const t1 = performance.now();
      for (let i = 0; i < iters; i++) ctx16.submit();
      await ctx16.sync();
      fp16Ms = (performance.now() - t1) / iters;
      ctx16.destroy();
    }

    const speedupFp32 = `${(baselineMs / fp32Ms).toFixed(2)}x`;
    const speedupFp16 = fp16Ms !== null ? `${(baselineMs / fp16Ms).toFixed(2)}x` : 'N/A';

    addRow({
      category: 'Data Movement',
      name: 'Matrix Transpose (Shared-Memory 16x16)',
      shape: `[${shape.join(',')}]`,
      baselineMs,
      fp32Ms,
      fp16Ms,
      speedupFp32,
      speedupFp16,
    });
  }

  // Layout Conversions: to_hwc4 & to_chw
  {
    const lShape = [1, 64, 56, 56];
    const lNumel = 1 * 64 * 56 * 56;
    const lHwc4Shape = [1, 56, 56, 64];
    const iters = 40;

    // FP32 to_hwc4
    statusEl.innerText = `Benchmarking layout conversion to_hwc4...`;
    const ctx32 = new WgpuExecutionContext(device);
    const in32 = ctx32.tensor('float32', lShape).setData(makeF32(lNumel));
    const hwc4Fp32 = ctx32.tensor('float32', lHwc4Shape);
    ctx32.recordShader(shaders.toHwc4, [in32, hwc4Fp32]);
    ctx32.submit();
    await ctx32.sync();
    const t0 = performance.now();
    for (let i = 0; i < iters; i++) ctx32.submit();
    await ctx32.sync();
    const toHwc4Fp32Ms = (performance.now() - t0) / iters;

    // FP32 to_chw
    const outChw32 = ctx32.tensor('float32', lShape);
    const ctxChw32 = new WgpuExecutionContext(device);
    ctxChw32.recordShader(shaders.toChw, [hwc4Fp32, outChw32]);
    ctxChw32.submit();
    await ctxChw32.sync();
    const t0chw = performance.now();
    for (let i = 0; i < iters; i++) ctxChw32.submit();
    await ctxChw32.sync();
    const toChwFp32Ms = (performance.now() - t0chw) / iters;
    ctx32.destroy();
    ctxChw32.destroy();

    // FP16
    let toHwc4Fp16Ms: number | null = null;
    let toChwFp16Ms: number | null = null;
    if (hasF16) {
      const ctx16 = new WgpuExecutionContext(device);
      const in16 = ctx16.tensor('float16', lShape).setData(makeF16(lNumel));
      const hwc4Fp16 = ctx16.tensor('float16', lHwc4Shape);
      ctx16.recordShader(shaders.toHwc4, [in16, hwc4Fp16]);
      ctx16.submit();
      await ctx16.sync();
      const t1 = performance.now();
      for (let i = 0; i < iters; i++) ctx16.submit();
      await ctx16.sync();
      toHwc4Fp16Ms = (performance.now() - t1) / iters;

      const outChw16 = ctx16.tensor('float16', lShape);
      const ctxChw16 = new WgpuExecutionContext(device);
      ctxChw16.recordShader(shaders.toChw, [hwc4Fp16, outChw16]);
      ctxChw16.submit();
      await ctxChw16.sync();
      const t1chw = performance.now();
      for (let i = 0; i < iters; i++) ctxChw16.submit();
      await ctxChw16.sync();
      toChwFp16Ms = (performance.now() - t1chw) / iters;

      ctx16.destroy();
      ctxChw16.destroy();
    }

    addRow({
      category: 'Layout Conversion',
      name: 'to_hwc4 (NCHW -> HWC4)',
      shape: `[${lShape.join(',')}]`,
      baselineMs: null,
      fp32Ms: toHwc4Fp32Ms,
      fp16Ms: toHwc4Fp16Ms,
      speedupFp32: 'Direct',
      speedupFp16: toHwc4Fp16Ms !== null ? `${(toHwc4Fp32Ms / toHwc4Fp16Ms).toFixed(2)}x` : 'N/A',
    });

    addRow({
      category: 'Layout Conversion',
      name: 'to_chw (HWC4 -> NCHW)',
      shape: `[${lShape.join(',')}]`,
      baselineMs: null,
      fp32Ms: toChwFp32Ms,
      fp16Ms: toChwFp16Ms,
      speedupFp32: 'Direct',
      speedupFp16: toChwFp16Ms !== null ? `${(toChwFp32Ms / toChwFp16Ms).toFixed(2)}x` : 'N/A',
    });
  }

  statusEl.innerHTML = '<span style="color:#98c379">✓ Benchmark Complete!</span>';
  document.title = 'BENCHMARK_DONE';
  const outJson = JSON.stringify(
    {
      gpu: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        hasF16,
      },
      results,
    },
    null,
    2
  );
  jsonEl.innerText = outJson;
  // eslint-disable-next-line no-console
  console.log('__BENCHMARK_RESULTS__:' + outJson);

  try {
    await fetch('/api/results', {
      method: 'POST',
      // eslint-disable-next-line @typescript-eslint/naming-convention
      headers: { 'Content-Type': 'application/json' },
      body: outJson,
    });
  } catch {}
}

const runBtn = document.getElementById('run-btn');
if (runBtn) {
  runBtn.addEventListener('click', () => {
    runBtn.setAttribute('disabled', 'true');
    run().finally(() => runBtn.removeAttribute('disabled'));
  });
}

run();
