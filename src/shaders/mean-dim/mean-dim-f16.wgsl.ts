/**
 * WGSL compute shader for dimension reduction mean/sum (f16).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_MEAN_DIM_F16 = /* wgsl */ `
enable f16;

struct Params {
  outer_: u32,
  r_: u32,
  inner_: u32,
  is_mean: u32,
};

@group(0) @binding(0) var<storage, read> inp: array<f16>;
@group(0) @binding(1) var<storage, read_write> out: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;

override wg_size: u32 = 256;

var<workgroup> partials: array<f16, 256>;

@compute @workgroup_size(wg_size)
fn main(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>) {
  let t = wid.x + wid.y * num_workgroups.x;
  let outs = params.outer_ * params.inner_;
  if (t >= outs) {
    return;
  }
  let oo = t / params.inner_;
  let ii = t % params.inner_;
  let base = oo * params.r_ * params.inner_ + ii;

  var acc: f16 = 0.0h;
  var k: u32 = lid.x;
  while (k < params.r_) {
    acc = acc + inp[base + k * params.inner_];
    k = k + wg_size;
  }
  partials[lid.x] = acc;

  for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
    workgroupBarrier();
    if (lid.x < stride && lid.x + stride < wg_size) {
      partials[lid.x] = partials[lid.x] + partials[lid.x + stride];
    }
  }
  workgroupBarrier();

  if (lid.x == 0u) {
    var s: f16 = partials[0];
    if (params.is_mean == 1u) {
      s = s / f16(params.r_);
    }
    out[t] = s;
  }
}
`;
