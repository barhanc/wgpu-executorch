/**
 * WGSL compute shader for NCHW to HWC4 layout conversion (FP16).
 * Converts planar NCHW activations [B, C, H, W] to packed channels-last [B, H, W, C/4] (vec4<f16>).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_TO_HWC4_F16 = /* wgsl */ `
enable f16;

struct Params {
  B: u32,
  C: u32,
  H: u32,
  W: u32,
  C4: u32,
  stride_x: u32,
  pad1: u32,
  pad2: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.stride_x + gid.x;
  let total = params.B * params.H * params.W * params.C4;
  if (idx >= total) {
    return;
  }

  let c4 = idx % params.C4;
  let rem0 = idx / params.C4;
  let w = rem0 % params.W;
  let rem1 = rem0 / params.W;
  let h = rem1 % params.H;
  let b = rem1 / params.H;

  let p = params.H * params.W;
  let px = h * params.W + w;
  let c0 = c4 * 4u;
  let b_base = b * params.C * p;

  var v = vec4<f16>(0.0h);
  if (c0 + 0u < params.C) { v.x = x[b_base + (c0 + 0u) * p + px]; }
  if (c0 + 1u < params.C) { v.y = x[b_base + (c0 + 1u) * p + px]; }
  if (c0 + 2u < params.C) { v.z = x[b_base + (c0 + 2u) * p + px]; }
  if (c0 + 3u < params.C) { v.w = x[b_base + (c0 + 3u) * p + px]; }

  out[idx] = v;
}
`;
