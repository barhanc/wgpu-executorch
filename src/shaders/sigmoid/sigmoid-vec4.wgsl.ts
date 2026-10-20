/**
 * WGSL compute shader for Sigmoid activation (vec4<f32> 128-bit vectorized).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_SIGMOID_VEC4 = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-input[idx]));
}
`;
