/**
 * WGSL compute shader for vectorized contiguous last-dimension softmax (vec4<f16>).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_SOFTMAX_LAST_DIM_VEC4_F16 = /* wgsl */ `
enable f16;

struct Params {
  outer_: u32,
  r_vecs: u32,
  inner_: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let row = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (row >= params.outer_) {
    return;
  }
  let base = row * params.r_vecs;
  var mx: f16 = -65504.0h;
  var ssum: f16 = 0.0h;
  for (var r: u32 = 0u; r < params.r_vecs; r = r + 1u) {
    let v = inp[base + r];
    let row_max = max(max(v.x, v.y), max(v.z, v.w));
    if (row_max > mx) {
      ssum = ssum * exp(mx - row_max);
      mx = row_max;
    }
    let e = exp(v - vec4<f16>(mx));
    ssum = ssum + e.x + e.y + e.z + e.w;
  }
  let inv_ssum = 1.0h / ssum;
  for (var r: u32 = 0u; r < params.r_vecs; r = r + 1u) {
    let idx = base + r;
    out[idx] = exp(inp[idx] - vec4<f16>(mx)) * inv_ssum;
  }
}
`;
