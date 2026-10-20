import { TENSOR_META_WGSL } from '../utils/meta';

/**
 * WGSL compute shader for general tensor slicing along one dimension (f16).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_SPLIT_WITH_SIZES_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;

${TENSOR_META_WGSL}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  dim: u32,
  start: u32,
  step: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let out_bufi = gid.x;
  if (out_bufi >= out_meta.numel) {
    return;
  }

  var rem = out_bufi;
  var in_bufi: u32 = 0u;
  for (var d: u32 = 0u; d < out_meta.ndim; d = d + 1u) {
    let coord = rem / out_meta.strides[d >> 2u][d & 3u];
    rem = rem % out_meta.strides[d >> 2u][d & 3u];
    var in_coord = coord;
    if (d == params.dim) {
      in_coord = params.start + coord * params.step;
    }
    in_bufi = in_bufi + in_coord * in_meta.strides[d >> 2u][d & 3u];
  }
  output[out_bufi] = input[in_bufi];
}
`;
