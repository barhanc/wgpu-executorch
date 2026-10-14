/**
 * Per-frame WebGPU loop for the YOLO26-N detection demo: a record-once
 * pipeline (preprocess -> model -> compact -> draw boxes) replayed every
 * camera frame.
 */

import { WgpuExecutionContext, type Program } from '../../../../src';
import {
  clearDetections,
  compactDetections,
  drawDetections,
  MAX_DETECTIONS,
  preprocess,
} from './shaders';

const size = [640, 640];

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
 * @param canvas The canvas element to render the graded video output to.
 * @param onMetrics Optional callback invoked with FPS and total GPU latency
 * in ms.
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

  // Crop + scale the centre square on the CPU: copyExternalImageToTexture cannot scale.
  const cropCanvas = new OffscreenCanvas(size[0], size[1]);
  const cropCtx = cropCanvas.getContext('2d')!;

  // Load model program descriptor and weights
  const base = import.meta.env.BASE_URL;
  const [programRes, weightsRes] = await Promise.all([
    fetch(`${base}assets/yolo26n/program.json`),
    fetch(`${base}assets/yolo26n/weights.bin`),
  ]);

  if (!programRes.ok || !weightsRes.ok) {
    throw new Error('failed to fetch yolo26n model artifacts');
  }

  const program = (await programRes.json()) as Program;
  const weights = await weightsRes.arrayBuffer();

  // Record the full pipeline once: model -> compaction -> draw.
  const ctx = new WgpuExecutionContext(device);
  const inp = ctx.tensor('float32', [1, 3, ...size]);
  const out = ctx.tensor('float32', [1, 84, 8400]);
  const counterBuffer = ctx.storageBuffer(16);
  const detsBuffer = ctx.storageBuffer(MAX_DETECTIONS * 6 * 4);
  ctx
    .recordShader(preprocess, [vidTexture, inp])
    .recordProgram(program, weights, [inp, out])
    .recordShader(clearDetections, [counterBuffer])
    .recordShader(compactDetections, [out, counterBuffer, detsBuffer])
    .recordShader(drawDetections, [vidTexture, resTexture, counterBuffer, detsBuffer]);

  let running = true;
  let startTime = 0;
  let frameCount = 0;
  let smoothedMs = 0;

  const onFrame = (now: number): void => {
    if (!running) return;

    // Cumulative-average FPS (frames / elapsed), steady despite callback jitter.
    if (startTime === 0) {
      startTime = now;
    }
    frameCount++;
    const elapsed = (now - startTime) / 1000;
    const fps = elapsed > 0 ? frameCount / elapsed : 0;

    // Crop, scale and upload the latest camera frame.
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const sx = Math.floor((video.videoWidth - minDim) / 2);
    const sy = Math.floor((video.videoHeight - minDim) / 2);

    cropCtx.drawImage(video, sx, sy, minDim, minDim, 0, 0, size[0], size[1]);
    device.queue.copyExternalImageToTexture({ source: cropCanvas }, { texture: vidTexture }, size);

    // Re-encode and submit recorded compute commands
    ctx.submit();

    // Blit the result texture to the canvas
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToTexture(
      { texture: resTexture },
      { texture: canvasCtx.getCurrentTexture() },
      size
    );
    device.queue.submit([encoder.finish()]);

    // Async GPU latency tracking; barriers settle in submission order (WebGPU spec).
    if (onMetrics) {
      device.queue.onSubmittedWorkDone().then(() => {
        if (!running) return;
        const ms = performance.now() - now;
        smoothedMs = smoothedMs ? smoothedMs * 0.8 + ms * 0.2 : ms;
        onMetrics(Math.round(fps), smoothedMs);
      });
    }

    onNextVideoFrame(video, onFrame);
  };

  onNextVideoFrame(video, onFrame);

  return () => {
    running = false;
    ctx.destroy();
    vidTexture.destroy();
    resTexture.destroy();
  };
}
