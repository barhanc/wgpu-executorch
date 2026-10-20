import { WgpuExecutionContext, shaders } from '../src';

interface BenchItem {
  category: string;
  name: string;
  shape: string;
  fp32Ms: number;
  fp16Ms: number | null;
  speedup: string;
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
      <td style="text-align: right;">${item.fp32Ms.toFixed(3)} ms</td>
      <td style="text-align: right;">${item.fp16Ms !== null ? item.fp16Ms.toFixed(3) + ' ms' : 'N/A'}</td>
      <td class="speedup" style="text-align: right;">${item.speedup}</td>
    `;
    bodyEl.appendChild(tr);
    tableEl.style.display = 'table';
  };

  // 1. Matrix Multiplication (MM)
  const mmConfigs = [
    [256, 256, 25],
    [512, 512, 20],
    [1024, 1024, 15],
    [2048, 2048, 8],
  ] as const;

  for (const [M, N, iters] of mmConfigs) {
    statusEl.innerText = `Benchmarking MM ${M}x${N}...`;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Linear Algebra',
      name: `MatMul (MM ${M >= 1024 ? '64x64' : 'Vec4'})`,
      shape: `[${M}, ${N}] * [${N}, ${N}]`,
      fp32Ms,
      fp16Ms,
      speedup,
    });
  }

  // 2. Batched Matrix Multiplication (BMM)
  const bmmConfigs = [
    [4, 64, 64, 30],
    [8, 128, 128, 20],
  ] as const;

  for (const [B, M, N, iters] of bmmConfigs) {
    statusEl.innerText = `Benchmarking BMM [${B}, ${M}, ${N}]...`;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Linear Algebra',
      name: 'Batch MatMul (BMM)',
      shape: `[${B}, ${M}, ${N}] * [${B}, ${N}, ${N}]`,
      fp32Ms,
      fp16Ms,
      speedup,
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
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: c.category,
      name: c.label,
      shape: `[${c.inShape.join(',')}] * [${c.wShape.join(',')}]`,
      fp32Ms,
      fp16Ms,
      speedup,
    });
  }

  // 4. Activations
  const actConfigs = [
    { name: 'ReLU (Vec4)', shader: shaders.relu },
    { name: 'Sigmoid (Vec4)', shader: shaders.sigmoid },
    { name: 'Hardswish (Vec4)', shader: shaders.hardswish },
  ];

  const actShape = [1, 64, 56, 56];
  const actNumel = 1 * 64 * 56 * 56;
  const actIters = 40;

  for (const a of actConfigs) {
    statusEl.innerText = `Benchmarking ${a.name}...`;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Activation',
      name: a.name,
      shape: `[${actShape.join(',')}]`,
      fp32Ms,
      fp16Ms,
      speedup,
    });
  }

  // 5. Reductions (Softmax, MeanDim)
  {
    statusEl.innerText = 'Benchmarking Softmax (Vec4)...';
    const smShape = [128, 1024];
    const smNumel = 128 * 1024;
    const smIters = 30;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Reduction',
      name: 'Softmax (Last-Dim Vec4)',
      shape: `[${smShape.join(',')}]`,
      fp32Ms,
      fp16Ms,
      speedup,
    });
  }

  // 6. Pooling & Resampling
  {
    statusEl.innerText = 'Benchmarking MaxPool2D (2D Spatial)...';
    const pShape = [1, 64, 112, 112];
    const pNumel = 1 * 64 * 112 * 112;
    const pIters = 25;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Pooling & Resampling',
      name: 'MaxPool2D (2D Spatial Grid)',
      shape: `[${pShape.join(',')}]`,
      fp32Ms,
      fp16Ms,
      speedup,
    });
  }

  // 7. Elementwise Arithmetic (Mul & Add)
  {
    statusEl.innerText = 'Benchmarking Elementwise Mul (Vec4)...';
    const shape = [1, 64, 56, 56];
    const numel = 1 * 64 * 56 * 56;
    const iters = 40;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Elementwise Math',
      name: 'Elementwise Mul (Vec4)',
      shape: `[${shape.join(',')}]`,
      fp32Ms,
      fp16Ms,
      speedup,
    });
  }

  // 8. Data Movement (Permute 2D Matrix Transpose)
  {
    statusEl.innerText = 'Benchmarking Matrix Transpose (Permute)...';
    const shape = [1024, 1024];
    const numel = 1024 * 1024;
    const iters = 25;

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

    let fp16Ms: number | null = null;
    let speedup = 'N/A';
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
      speedup = `${(fp32Ms / fp16Ms).toFixed(2)}x`;
      ctx16.destroy();
    }

    addRow({
      category: 'Data Movement',
      name: 'Matrix Transpose (Shared-Memory 16x16)',
      shape: `[${shape.join(',')}]`,
      fp32Ms,
      fp16Ms,
      speedup,
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
