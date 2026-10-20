/**
 * WGSL compute shader for 2D max pooling with 2D spatial grid (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_MAX_POOL2D_SPATIAL = /* wgsl */ `
struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  kH: u32,
  kW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  _p0: u32,
  _p1: u32,
}

@group(0) @binding(0) var<storage, read_write> out_vals: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> out_idx: array<i32>;

override write_indices: u32 = 0u;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let ow = gid.x;
  let oh = gid.y;
  let c = wg_id.z % params.C;
  let n = wg_id.z / params.C;

  if (ow >= params.OW || oh >= params.OH || n >= params.N) {
    return;
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_c_base = (n * params.C + c) * params.IH;

  var best: f32 = -3.4e38;
  var best_idx: i32 = 0;
  for (var kh: u32 = 0u; kh < params.kH; kh = kh + 1u) {
    let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    let in_row = (in_c_base + u32(ih)) * params.IW;
    for (var kw: u32 = 0u; kw < params.kW; kw = kw + 1u) {
      let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      let v = inp[in_row + u32(iw)];
      if (v > best) {
        best = v;
        best_idx = ih * iW + iw;
      }
    }
  }

  let out_pos = ((n * params.C + c) * params.OH + oh) * params.OW + ow;
  out_vals[out_pos] = best;
  if (write_indices != 0u) {
    out_idx[out_pos] = best_idx;
  }
}
`;
