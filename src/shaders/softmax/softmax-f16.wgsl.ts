/**
 * WGSL compute shader for general dimension softmax (f16).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_SOFTMAX_F16 = /* wgsl */ `
enable f16;

struct Params {
  outer_: u32,
  r_: u32,
  inner_: u32,
  pad_: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = gid.x;
  let lines = params.outer_ * params.inner_;
  if (t >= lines) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;

  var mx: f16 = -65504.0h;
  var ssum: f16 = 0.0h;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let v = inp[base + r * params.inner_];
    if (v > mx) {
      ssum = ssum * exp(mx - v);
      mx = v;
    }
    ssum = ssum + exp(v - mx);
  }
  let inv_ssum = 1.0h / ssum;
  for (var r: u32 = 0u; r < params.r_; r = r + 1u) {
    let idx = base + r * params.inner_;
    out[idx] = exp(inp[idx] - mx) * inv_ssum;
  }
}
`;
