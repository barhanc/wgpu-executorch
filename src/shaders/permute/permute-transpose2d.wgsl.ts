/**
 * WGSL compute shader for 2D matrix transpose using shared-memory tile (16x16 with 17-stride, f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_PERMUTE_TRANSPOSE2D = /* wgsl */ `
struct Transpose2dParams {
  rows: u32,
  cols: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Transpose2dParams;

var<workgroup> tile: array<array<f32, 17>, 16>;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let x = wg_id.x * 16u + local_id.x;
  let y = wg_id.y * 16u + local_id.y;

  if (x < params.cols && y < params.rows) {
    tile[local_id.y][local_id.x] = input[y * params.cols + x];
  }
  workgroupBarrier();

  let out_x = wg_id.y * 16u + local_id.x;
  let out_y = wg_id.x * 16u + local_id.y;

  if (out_x < params.rows && out_y < params.cols) {
    output[out_y * params.rows + out_x] = tile[local_id.x][local_id.y];
  }
}
`;
