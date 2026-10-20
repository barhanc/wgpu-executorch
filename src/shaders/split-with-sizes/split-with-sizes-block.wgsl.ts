/**
 * WGSL compute shader for contiguous block-gather tensor slicing (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_SPLIT_WITH_SIZES_BLOCK = /* wgsl */ `
struct BlockParams {
  c_in: u32,
  c_out: u32,
  inner_size: u32,
  start: u32,
  total_elements: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: BlockParams;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.total_elements) {
    return;
  }
  let inner_idx = idx % params.inner_size;
  let slice_idx = idx / params.inner_size;
  let c = slice_idx % params.c_out;
  let b = slice_idx / params.c_out;

  let in_idx = (b * params.c_in + params.start + c) * params.inner_size + inner_idx;
  output[idx] = input[in_idx];
}
`;
