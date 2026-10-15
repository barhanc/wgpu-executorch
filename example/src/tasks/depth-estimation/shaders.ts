/**
 * Preprocess and postprocess compute shaders for the depth-estimation pipeline:
 * ImageNet-normalized NCHW input, min/max-normalized grayscale depth output.
 */

import type { Shader, Tensor } from '../../../../src';
import { createComputeBundle } from '../../../../src';

/**
 * Preprocess compute shader: reads normalized [0, 1] RGB texels from the video
 * texture and writes ImageNet-normalized planar NCHW into the model input
 * tensor ((v - mean) / std, matching the DepthART export).
 */
export const preprocess: Shader<[GPUTexture, Tensor]> = {
  name: 'depth.preprocess',
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

      output[offset] = (pixel.r - 0.485) / 0.229;
      output[hw + offset] = (pixel.g - 0.456) / 0.224;
      output[2u * hw + offset] = (pixel.b - 0.406) / 0.225;
    }
  `,
  recordIn(ctx, [vidTexture, inp]) {
    // Bind video texture and model input buffer
    const bundle = createComputeBundle(ctx.device, preprocess.code as string, [
      { binding: 0, resource: vidTexture.createView() },
      { binding: 1, buffer: inp.buffer },
    ]);

    // Dispatch 16x16 workgroups over the image
    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(vidTexture.width / 16),
      workgroupCountY: Math.ceil(vidTexture.height / 16),
    });
  },
};

/**
 * Clear compute shader: resets the depth min/max statistics buffer to the
 * identity keys (largest / smallest ordered float) before each reduction.
 */
export const clearStats: Shader<[GPUBuffer]> = {
  name: 'depth.clear_stats',
  code: /* wgsl */ `
    @group(0) @binding(0) var<storage, read_write> stats: array<u32>;

    @compute @workgroup_size(1)
    fn main() {
      stats[0] = 0xffffffffu;
      stats[1] = 0x00000000u;
    }
  `,
  recordIn(ctx, [statsBuffer]) {
    const bundle = createComputeBundle(ctx.device, clearStats.code as string, [
      { binding: 0, buffer: statsBuffer },
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
 * Stats compute shader: min/max reduction over the depth map into the
 * statistics buffer. Floats are compared as total-order u32 keys so the
 * result is correct for any finite value. Each workgroup reduces its tile in
 * shared memory and folds one min/max pair into the atomics.
 */
export const depthStats: Shader<[Tensor, GPUBuffer]> = {
  name: 'depth.stats',
  code: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> depth: array<f32>;
    @group(0) @binding(1) var<storage, read_write> stats: array<atomic<u32>>;

    override wg_size: u32 = 256;

    var<workgroup> tile_min: array<u32, wg_size>;
    var<workgroup> tile_max: array<u32, wg_size>;

    // Total-order float keys: larger key <=> larger value
    fn key(v: f32) -> u32 {
      let bits = bitcast<u32>(v);
      return select(bits ^ 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
    }

    @compute @workgroup_size(wg_size)
    fn main(
      @builtin(global_invocation_id) gid: vec3<u32>,
      @builtin(local_invocation_id) lid: vec3<u32>,
      @builtin(num_workgroups) num_workgroups: vec3<u32>) {
      let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
      var k_min = 0xffffffffu;
      var k_max = 0x00000000u;
      if (idx < arrayLength(&depth)) {
        let k = key(depth[idx]);
        k_min = k;
        k_max = k;
      }

      tile_min[lid.x] = k_min;
      tile_max[lid.x] = k_max;
      workgroupBarrier();

      for (var stride = wg_size / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
          tile_min[lid.x] = min(tile_min[lid.x], tile_min[lid.x + stride]);
          tile_max[lid.x] = max(tile_max[lid.x], tile_max[lid.x + stride]);
        }
        workgroupBarrier();
      }

      if (lid.x == 0u) {
        atomicMin(&stats[0], tile_min[0]);
        atomicMax(&stats[1], tile_max[0]);
      }
    }
  `,
  recordIn(ctx, [out, statsBuffer]) {
    const wgSize = 256;
    const totalWorkgroups = Math.ceil(out.numel / wgSize);

    const bundle = createComputeBundle(
      ctx.device,
      depthStats.code as string,
      [
        { binding: 0, buffer: out.buffer },
        { binding: 1, buffer: statsBuffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { wg_size: wgSize }
    );

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.min(totalWorkgroups, 65535),
      workgroupCountY: Math.ceil(totalWorkgroups / 65535),
    });
  },
};

/**
 * Draw compute shader: normalizes the depth map with the per-frame min/max
 * and renders it grayscale into the result storage texture — smaller depth
 * (nearer) is brighter.
 */
export const drawDepth: Shader<[Tensor, GPUBuffer, GPUTexture]> = {
  name: 'depth.draw',
  code: /* wgsl */ `
    @group(0) @binding(0) var<storage, read> depth: array<f32>;
    @group(0) @binding(1) var<storage, read> stats: array<u32>;
    @group(0) @binding(2) var res_tex: texture_storage_2d<rgba8unorm, write>;

    // Inverse of the total-order key mapping in depth.stats
    fn unkey(k: u32) -> f32 {
      let bits = select(k ^ 0x80000000u, ~k, (k & 0x80000000u) == 0u);
      return bitcast<f32>(bits);
    }

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let dims = textureDimensions(res_tex);
      if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
      }

      let lo = unkey(stats[0]);
      let hi = unkey(stats[1]);
      let offset = gid.y * dims.x + gid.x;
      let t = clamp((depth[offset] - lo) / max(hi - lo, 1e-6), 0.0, 1.0);

      let gray = 1.0 - t;
      textureStore(res_tex, vec2<i32>(gid.xy), vec4<f32>(gray, gray, gray, 1.0));
    }
  `,
  recordIn(ctx, [out, statsBuffer, resTexture]) {
    const bundle = createComputeBundle(ctx.device, drawDepth.code as string, [
      { binding: 0, buffer: out.buffer },
      { binding: 1, buffer: statsBuffer },
      { binding: 2, resource: resTexture.createView() },
    ]);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(resTexture.width / 16),
      workgroupCountY: Math.ceil(resTexture.height / 16),
    });
  },
};
