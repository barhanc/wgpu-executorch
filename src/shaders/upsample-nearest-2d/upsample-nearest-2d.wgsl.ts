/**
 * WGSL compute shader for nearest-neighbour 2D upsampling (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_UPSAMPLE_NEAREST2D = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  _p0: u32,
  _p1: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.C * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let c = (i / (params.OW * params.OH)) % params.C;
  let n = i / (params.OW * params.OH * params.C);

  let ih = (oh * params.IH) / params.OH;
  let iw = (ow * params.IW) / params.OW;
  let in_idx = ((n * params.C + c) * params.IH + ih) * params.IW + iw;
  out[i] = inp[in_idx];
}
`;
