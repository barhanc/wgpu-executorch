import { TENSOR_META_WGSL } from '../utils/meta';

/**
 * WGSL compute shader for general tensor concatenation (f16).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CAT_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;

${TENSOR_META_WGSL}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  concat_dim: u32,
  off_k: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let in_bufi = gid.x;
  if (in_bufi >= in_meta.numel) {
    return;
  }

  var rem = in_bufi;
  var out_bufi: u32 = 0u;
  for (var d: u32 = 0u; d < in_meta.ndim; d = d + 1u) {
    let coord = rem / in_meta.strides[d >> 2u][d & 3u];
    rem = rem % in_meta.strides[d >> 2u][d & 3u];
    var out_coord = coord;
    if (d == params.concat_dim) {
      out_coord = coord + params.off_k;
    }
    out_bufi = out_bufi + out_coord * out_meta.strides[d >> 2u][d & 3u];
  }
  output[out_bufi] = input[in_bufi];
}
`;
