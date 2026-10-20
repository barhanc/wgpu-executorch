/**
 * WGSL compute shader for nearest-neighbour 2D upsampling using 2D spatial grid (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_UPSAMPLE_NEAREST2D_SPATIAL = /* wgsl */ `
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

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let ow = gid.x;
  let oh = gid.y;
  let c_idx = wg_id.z;

  if (ow >= params.OW || oh >= params.OH) {
    return;
  }

  let ih = (oh * params.IH) / params.OH;
  let iw = (ow * params.IW) / params.OW;
  let in_idx = (c_idx * params.IH + ih) * params.IW + iw;
  let out_idx = (c_idx * params.OH + oh) * params.OW + ow;
  out[out_idx] = inp[in_idx];
}
`;
