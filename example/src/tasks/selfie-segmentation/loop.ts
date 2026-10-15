/**
 * Per-frame WebGPU loop for the selfie-segmentation demo: a record-once
 * pipeline (preprocess -> model -> mask composite) replayed every camera frame.
 */

import type { Program } from '../../../../src';
import { WgpuExecutionContext } from '../../../../src';
import { postprocess, preprocess } from './shaders';

const size = [256, 256];

function onNextVideoFrame(video: HTMLVideoElement, callback: (now: number) => void): void {
  if ('requestVideoFrameCallback' in video) {
    video.requestVideoFrameCallback((now) => callback(now));
  } else {
    requestAnimationFrame((now) => callback(now));
  }
}

/**
 * Starts the per-frame loop for the given camera video.
 *
 * @param video The playing camera video to source frames from.
 * @param canvas The canvas element to render the segmented video output to.
 * @param onMetrics Optional callback invoked with smoothed FPS and inference time in ms.
 * @returns A stop function that halts the frame loop and cleans up resources.
 */
export async function startFrameLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  onMetrics?: (fps: number, ms: number) => void
): Promise<() => void> {
  // Acquire WebGPU device
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter?.requestDevice();

  if (device === undefined) {
    throw new Error('no WebGPU device');
  }

  // Configure canvas context
  const canvasCtx = canvas.getContext('webgpu');

  if (canvasCtx === null) {
    throw new Error('failed to get WebGPU context for canvas');
  }

  canvas.width = size[0];
  canvas.height = size[1];

  // rgba8unorm keeps COPY_DST support (bgra8unorm lacks it on some backends)
  canvasCtx.configure({
    device,
    format: 'rgba8unorm',
    alphaMode: 'premultiplied',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
  });

  // Persistent textures — keep bind groups valid across frames
  const vidUsage =
    GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
  const resUsage = GPUTextureUsage.COPY_SRC | GPUTextureUsage.STORAGE_BINDING;

  const vidTexture = device.createTexture({ size, format: 'rgba8unorm', usage: vidUsage });
  const resTexture = device.createTexture({ size, format: 'rgba8unorm', usage: resUsage });

  // Frame-completion fence: a 16-byte copy whose mapAsync resolves exactly
  // when the frame's GPU work has landed.
  const fenceSrc = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC });
  const fenceDst = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  // Crop + scale the centre square on the CPU: copyExternalImageToTexture cannot scale.
  const cropCanvas = new OffscreenCanvas(size[0], size[1]);
  const cropCtx = cropCanvas.getContext('2d')!;

  // Load model program descriptor and weights
  const base = import.meta.env.BASE_URL;
  const [programRes, weightsRes] = await Promise.all([
    fetch(`${base}assets/selfie-segmentation/program.json`),
    fetch(`${base}assets/selfie-segmentation/weights.bin`),
  ]);

  if (!programRes.ok || !weightsRes.ok) {
    throw new Error('failed to fetch selfie segmentation model artifacts');
  }

  const program = (await programRes.json()) as Program;
  const weights = await weightsRes.arrayBuffer();

  // Record the compute pipeline
  const ctx = new WgpuExecutionContext(device);
  const inp = ctx.tensor('float32', [1, 3, ...size]);
  const out = ctx.tensor('float32', [1, 1, ...size]);
  ctx
    .recordShader(preprocess, [vidTexture, inp])
    .recordProgram(program, weights, [inp, out])
    .recordShader(postprocess, [vidTexture, out, resTexture]);

  let running = true;
  let smoothedMs = 0;
  const times: number[] = [];

  const onFrame = (now: number): void => {
    if (!running) return;

    // FPS over a sliding 2s window: smooths a variable-rate camera's jitter
    // while tracking the current rate (a cumulative average lags behind).
    times.push(now);
    while (now - times[0] > 2000) {
      times.shift();
    }
    const fps = times.length > 1 ? ((times.length - 1) * 1000) / (now - times[0]) : 0;
    const start = performance.now();

    // Crop and scale the centred square of the video to 256×256 using drawImage,
    // then upload the scaled result to the GPU texture.
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const sx = Math.floor((video.videoWidth - minDim) / 2);
    const sy = Math.floor((video.videoHeight - minDim) / 2);

    cropCtx.drawImage(video, sx, sy, minDim, minDim, 0, 0, size[0], size[1]);
    device.queue.copyExternalImageToTexture({ source: cropCanvas }, { texture: vidTexture }, size);

    // Re-encode and submit recorded compute commands
    ctx.submit();

    // Blit the result texture to the canvas and append the fence copy
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture(
      { texture: resTexture },
      { texture: canvasCtx.getCurrentTexture() },
      size
    );
    encoder.copyBufferToBuffer(fenceSrc, 0, fenceDst, 0, 16);
    device.queue.submit([encoder.finish()]);

    // Inference time runs to the fence: mapAsync resolves exactly when the
    // copy lands, unlike onSubmittedWorkDone which can lag to a frame boundary.
    void fenceDst.mapAsync(GPUMapMode.READ).then(
      () => {
        if (!running) return;
        fenceDst.unmap();
        if (onMetrics) {
          const ms = performance.now() - start;
          smoothedMs = smoothedMs ? smoothedMs * 0.8 + ms * 0.2 : ms;
          onMetrics(Math.round(fps), smoothedMs);
        }
        onNextVideoFrame(video, onFrame);
      },
      () => {} // loop stopped with the fence pending
    );
  };

  onNextVideoFrame(video, onFrame);

  return () => {
    running = false;
    ctx.destroy();
    vidTexture.destroy();
    resTexture.destroy();
    fenceSrc.destroy();
    fenceDst.destroy();
  };
}
