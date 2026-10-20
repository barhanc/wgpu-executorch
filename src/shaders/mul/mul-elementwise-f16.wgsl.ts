/**
 * WGSL compute shader for contiguous 1D FP16 elementwise multiplication.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_MUL_ELEMENTWISE_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> input1: array<f16>;
@group(0) @binding(1) var<storage, read> input2: array<f16>;
@group(0) @binding(2) var<storage, read_write> output: array<f16>;

struct Params {
  num_elements: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.num_elements) {
    return;
  }
  output[idx] = input1[idx] * input2[idx];
}
`;
