/**
 * WGSL compute shader for spatial constant padding using 2D grid (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONSTANT_PAD_ND_SPATIAL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params {
  out_dims: vec4<u32>,
  in_dims: vec4<u32>,
  left: vec4<u32>,
  out_numel: u32,
  value: f32,
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
  let slice = wg_id.z;

  if (ow >= params.out_dims.w || oh >= params.out_dims.z) {
    return;
  }

  let iw = ow - params.left.w;
  let ih = oh - params.left.z;

  let out_idx = (slice * params.out_dims.z + oh) * params.out_dims.w + ow;

  let in_w = ow >= params.left.w && iw < params.in_dims.w;
  let in_h = oh >= params.left.z && ih < params.in_dims.z;

  if (in_w && in_h) {
    let in_idx = (slice * params.in_dims.z + ih) * params.in_dims.w + iw;
    out[out_idx] = inp[in_idx];
  } else {
    out[out_idx] = params.value;
  }
}
`;
