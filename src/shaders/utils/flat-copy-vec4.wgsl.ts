/**
 * WGSL compute shader for flat byte copy using 128-bit vec4<u32> loads and stores.
 * Coalesced, high-throughput memory streaming within the active compute pass.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_FLAT_COPY_VEC4 = /* wgsl */ `
@group(0) @binding(0) var<storage, read> src: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<u32>>;

struct Params {
  count: u32,
}
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>
) {
  let idx = gid.y * 65535u * wg_size + gid.x;
  if (idx < params.count) {
    dst[idx] = src[idx];
  }
}
`;
