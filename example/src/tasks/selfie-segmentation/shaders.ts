/**
 * Preprocess and postprocess compute shaders for the selfie segmentation pipeline.
 */

import type { Shader, Tensor } from '../../../../src';
import { createComputeBundle } from '../../../../src';

/**
 * Preprocess compute shader: reads normalized [0, 1] RGB texels from the video
 * texture and unpacks them into planar NCHW format in the model input tensor.
 */
export const preprocess: Shader<[GPUTexture, Tensor]> = {
  name: 'selfie.preprocess',
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
 * Postprocess compute shader: blends the segmented person mask over a darkened,
 * semi-transparent background to render into the result storage texture.
 */
export const postprocess: Shader<[GPUTexture, Tensor, GPUTexture]> = {
  name: 'selfie.postprocess',
  code: /* wgsl */ `
    @group(0) @binding(0) var vid_tex: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read> mask: array<f32>;
    @group(0) @binding(2) var res_tex: texture_storage_2d<rgba8unorm, write>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let dims = textureDimensions(vid_tex);
      if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
      }

      let offset = gid.y * dims.x + gid.x;
      let m = clamp(mask[offset], 0.0, 1.0);

      // Very dark background with slight transparency (premultiplied alpha)
      let bg_alpha = 0.8;
      let bg_rgb = vec3<f32>(0.02, 0.02, 0.03) * bg_alpha;
      let bg = vec4<f32>(bg_rgb, bg_alpha);

      // Solid white fill for the person
      let fg = vec4<f32>(1.0, 1.0, 1.0, 1.0);

      let color = mix(bg, fg, m);
      textureStore(res_tex, vec2<i32>(gid.xy), color);
    }
  `,
  recordIn(ctx, [vidTexture, out, resTexture]) {
    // Bind video texture, model output mask, and result storage texture
    const bundle = createComputeBundle(ctx.device, postprocess.code as string, [
      { binding: 0, resource: vidTexture.createView() },
      { binding: 1, buffer: out.buffer },
      { binding: 2, resource: resTexture.createView() },
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
 * Preprocess compute shader for FP16: reads normalized [0, 1] RGB texels from the video
 * texture and unpacks them into planar NCHW f16 elements.
 */
export const preprocessF16: Shader<[GPUTexture, Tensor]> = {
  name: 'selfie.preprocess.f16',
  code: /* wgsl */ `
    enable f16;

    @group(0) @binding(0) var vid_tex: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read_write> output: array<f16>;

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

      output[offset] = f16(pixel.r);
      output[hw + offset] = f16(pixel.g);
      output[2u * hw + offset] = f16(pixel.b);
    }
  `,
  recordIn(ctx, [vidTexture, inp]) {
    const bundle = createComputeBundle(ctx.device, preprocessF16.code as string, [
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
 * Postprocess compute shader for FP16 mask: blends segmented mask with darkened background.
 */
export const postprocessF16: Shader<[GPUTexture, Tensor, GPUTexture]> = {
  name: 'selfie.postprocess.f16',
  code: /* wgsl */ `
    enable f16;

    @group(0) @binding(0) var vid_tex: texture_2d<f32>;
    @group(0) @binding(1) var<storage, read> mask: array<f16>;
    @group(0) @binding(2) var res_tex: texture_storage_2d<rgba8unorm, write>;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let dims = textureDimensions(vid_tex);
      if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
      }

      let offset = gid.y * dims.x + gid.x;
      let m = clamp(f32(mask[offset]), 0.0, 1.0);

      // Very dark background with slight transparency (premultiplied alpha)
      let bg_alpha = 0.8;
      let bg_rgb = vec3<f32>(0.02, 0.02, 0.03) * bg_alpha;
      let bg = vec4<f32>(bg_rgb, bg_alpha);

      // Solid white fill for the person
      let fg = vec4<f32>(1.0, 1.0, 1.0, 1.0);

      let color = mix(bg, fg, m);
      textureStore(res_tex, vec2<i32>(gid.xy), color);
    }
  `,
  recordIn(ctx, [vidTexture, out, resTexture]) {
    const bundle = createComputeBundle(ctx.device, postprocessF16.code as string, [
      { binding: 0, resource: vidTexture.createView() },
      { binding: 1, buffer: out.buffer },
      { binding: 2, resource: resTexture.createView() },
    ]);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: Math.ceil(vidTexture.width / 16),
      workgroupCountY: Math.ceil(vidTexture.height / 16),
    });
  },
};
