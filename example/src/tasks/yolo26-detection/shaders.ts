/**
 * Compute shaders for the YOLO26-N detection pipeline.
 *
 * The per-frame pipeline is fully GPU-resident (recorded once, replayed per
 * frame): preprocess (video -> input tensor), the ExecuTorch model program,
 * then three postprocess kernels — `clear` resets the detection counter,
 * `compact` scans the raw predictions into a small detection buffer, and
 * `draw` grades the video and rasterizes the bounding boxes.
 */

import type { Shader, Tensor } from '../../../../src';
import { createComputeBundle } from '../../../../src';

/** Minimum class score for a detection to be recorded. */
export const CONF_THRESHOLD = 0.35;

/** At most this many detections are recorded per frame. */
export const MAX_DETECTIONS = 64;

/**
 * Preprocess compute shader: reads normalized [0, 1] RGB texels from the video
 * texture and unpacks them into planar NCHW format in the model input tensor
 * (YOLO26 consumes pixels scaled to [0, 1], i.e. uint8 / 255).
 */
export const preprocess: Shader<[GPUTexture, Tensor]> = {
  name: 'yolo.preprocess',
  code: /* wgsl */ `
    @group(0) @binding(0) var vid_tex: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let dims = textureDimensions(vid_tex);
      if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
      }

      // Convert interleaved RGBA texel to planar NCHW channels
      let pixel = textureLoad(vid_tex, vec2<i32>(gid.xy), 0);
      let hw = dims.x * dims.y;
      let offset = gid.y * dims.x + gid.x;

      output[offset] = pixel.r;
      output[hw + offset] = pixel.g;
      output[2u * hw + offset] = pixel.b;
    }
  `,
  recordIn(ctx, [vidTexture, inp]) {
    const bundle = createComputeBundle(ctx.device, preprocess.code as string, [
      { binding: 0, resource: vidTexture.createView() },
      { binding: 1, buffer: inp.buffer },
    ]);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(vidTexture.width / 16),
      workgroupCountY: Math.ceil(vidTexture.height / 16),
    });
  },
};

/**
 * Counter reset compute shader: zeroes the detection counter so the compaction
 * kernel can claim slots with atomic increments. One thread.
 */
export const clearDetections: Shader<[GPUBuffer]> = {
  name: 'yolo.clear_detections',
  code: /* wgsl */ `
    @group(0) @binding(0) var<storage, read_write> counter: array<atomic<u32>>;

    @compute @workgroup_size(1)
    fn main() {
      atomicStore(&counter[0], 0u);
    }
  `,
  recordIn(ctx, [counterBuffer]) {
    const bundle = createComputeBundle(ctx.device, clearDetections.code as string, [
      { binding: 0, buffer: counterBuffer },
    ]);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: 1,
      workgroupCountY: 1,
    });
  },
};

/**
 * Detection compaction compute shader: one thread per prediction anchor. Each
 * thread scans the 80 class scores of its anchor, and anchors above the
 * confidence threshold atomically claim a slot in the detection buffer as
 * `[x1, y1, x2, y2, score, class]`.
 */
export const compactDetections: Shader<[Tensor, GPUBuffer, GPUBuffer]> = {
  name: 'yolo.compact_detections',
  code: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> preds: array<f32>;
    @group(0) @binding(1) var<storage, read_write> counter: array<atomic<u32>>;
    @group(0) @binding(2) var<storage, read_write> dets: array<f32>;

    override wg_size: u32 = 256;
    override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill
    override conf: f32 = 0.5;
    override max_det: u32 = 64;
    override num_anchors: u32 = 8400;
    override num_classes: u32 = 80;

    @compute @workgroup_size(wg_size)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let a = gid.y * stride_x + gid.x;
      if (a >= num_anchors) {
        return;
      }

      // Best class over the anchor's score channel slice
      var best: f32 = 0.0;
      var best_class: u32 = 0u;
      for (var c: u32 = 0u; c < num_classes; c = c + 1u) {
        let s = preds[(4u + c) * num_anchors + a];
        if (s > best) {
          best = s;
          best_class = c;
        }
      }
      if (best < conf) {
        return;
      }

      let slot = atomicAdd(&counter[0], 1u);
      if (slot >= max_det) {
        return;
      }
      let b = slot * 6u;
      dets[b] = preds[a];
      dets[b + 1u] = preds[num_anchors + a];
      dets[b + 2u] = preds[2u * num_anchors + a];
      dets[b + 3u] = preds[3u * num_anchors + a];
      dets[b + 4u] = best;
      dets[b + 5u] = f32(best_class);
    }
  `,
  recordIn(ctx, [preds, counterBuffer, detsBuffer]) {
    if (preds.dtype !== 'float32') {
      throw new Error('yolo.compact_detections: predictions must be float32');
    }

    const wgSize = 256;
    const numAnchors = preds.shape[2];
    const numClasses = preds.shape[1] - 4;
    const totalWorkgroups = Math.ceil(numAnchors / wgSize);
    const workgroupCountX = Math.min(totalWorkgroups, 65535);

    /* eslint-disable camelcase, @typescript-eslint/naming-convention -- WGSL override names */
    const constants = {
      wg_size: wgSize,
      stride_x: workgroupCountX * wgSize,
      conf: CONF_THRESHOLD,
      max_det: MAX_DETECTIONS,
      num_anchors: numAnchors,
      num_classes: numClasses,
    };
    /* eslint-enable camelcase, @typescript-eslint/naming-convention */

    const bundle = createComputeBundle(
      ctx.device,
      compactDetections.code as string,
      [
        { binding: 0, buffer: preds.buffer },
        { binding: 1, buffer: counterBuffer },
        { binding: 2, buffer: detsBuffer },
      ],
      constants
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX,
      workgroupCountY: Math.ceil(totalWorkgroups / 65535),
    });
  },
};

/**
 * Detection draw compute shader: grades the camera frame and outlines the
 * compacted detections as anti-aliased, per-class-colored bounding boxes.
 */
export const drawDetections: Shader<[GPUTexture, GPUTexture, GPUBuffer, GPUBuffer]> = {
  name: 'yolo.draw_detections',
  code: /* wgsl */ `
    @group(0) @binding(0) var vid_tex: texture_2d<f32>;
    @group(0) @binding(1) var res_tex: texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(2) var<storage, read> counter: array<u32>;
    @group(0) @binding(3) var<storage, read> dets: array<f32>;

    override max_det: u32 = 64;

    fn hsl_f(n: f32, h: f32, s: f32, l: f32) -> f32 {
      let a = s * min(l, 1.0 - l);
      let k = (n + h * 12.0) % 12.0;
      return l - a * clamp(min(k - 3.0, 9.0 - k), -1.0, 1.0);
    }

    // Distinct per-class hue
    fn class_rgb(cls: u32) -> vec3<f32> {
      let h = f32((cls * 47u) % 360u) / 360.0;
      return vec3<f32>(
        hsl_f(0.0, h, 0.85, 0.58),
        hsl_f(8.0, h, 0.85, 0.58),
        hsl_f(4.0, h, 0.85, 0.58)
      );
    }

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let dims = textureDimensions(vid_tex);
      if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
      }

      // Graded video base: mild contrast, warm lift, vignette
      let pixel = textureLoad(vid_tex, vec2<i32>(gid.xy), 0);
      var color = clamp((pixel.rgb - 0.5) * 1.08 + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
      color = color + vec3<f32>(0.015, 0.012, 0.0);
      let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
      let d = distance(uv, vec2<f32>(0.5, 0.5));
      let vignette = smoothstep(0.85, 0.35, d);
      color = color * mix(0.55, 1.0, vignette);

      let px = f32(gid.x) + 0.5;
      let py = f32(gid.y) + 0.5;
      let count = min(counter[0], max_det);

      for (var i: u32 = 0u; i < count; i = i + 1u) {
        let b = i * 6u;
        let x1 = dets[b];
        let y1 = dets[b + 1u];
        let x2 = dets[b + 2u];
        let y2 = dets[b + 3u];
        let cls = u32(dets[b + 5u] + 0.5);

        // Anti-aliased hollow rect: d < 0 inside, 0 on the border
        let dx = max(x1 - px, px - x2);
        let dy = max(y1 - py, py - y2);
        let edge = max(dx, dy);
        let cov = 1.0 - smoothstep(0.5, 1.5, abs(edge));
        if (cov > 0.0) {
          color = mix(color, class_rgb(cls), cov);
        }
      }

      textureStore(res_tex, vec2<i32>(gid.xy), vec4<f32>(color, 1.0));
    }
  `,
  recordIn(ctx, [vidTexture, resTexture, counterBuffer, detsBuffer]) {
    /* eslint-disable camelcase, @typescript-eslint/naming-convention -- WGSL override names */
    const constants = { max_det: MAX_DETECTIONS };
    /* eslint-enable camelcase, @typescript-eslint/naming-convention */

    const bundle = createComputeBundle(
      ctx.device,
      drawDetections.code as string,
      [
        { binding: 0, resource: vidTexture.createView() },
        { binding: 1, resource: resTexture.createView() },
        { binding: 2, buffer: counterBuffer },
        { binding: 3, buffer: detsBuffer },
      ],
      constants
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(vidTexture.width / 16),
      workgroupCountY: Math.ceil(vidTexture.height / 16),
    });
  },
};
