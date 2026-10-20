/**
 * WGSL compute shader for constant padding on up to 4D tensors (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONSTANT_PAD_ND = /* wgsl */ `
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

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let i = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (i >= params.out_numel) {
    return;
  }

  var rem = i;
  let o3 = rem % params.out_dims.w;
  rem = rem / params.out_dims.w;
  let o2 = rem % params.out_dims.z;
  rem = rem / params.out_dims.z;
  let o1 = rem % params.out_dims.y;
  rem = rem / params.out_dims.y;
  let o0 = rem % params.out_dims.x;

  let c0 = o0 - params.left.x;
  let c1 = o1 - params.left.y;
  let c2 = o2 - params.left.z;
  let c3 = o3 - params.left.w;

  let in0 = o0 >= params.left.x && c0 < params.in_dims.x;
  let in1 = o1 >= params.left.y && c1 < params.in_dims.y;
  let in2 = o2 >= params.left.z && c2 < params.in_dims.z;
  let in3 = o3 >= params.left.w && c3 < params.in_dims.w;

  if (in0 && in1 && in2 && in3) {
    let in_idx =
        ((c0 * params.in_dims.y + c1) * params.in_dims.z + c2) * params.in_dims.w
        + c3;
    out[i] = inp[in_idx];
  } else {
    out[i] = params.value;
  }
}
`;
