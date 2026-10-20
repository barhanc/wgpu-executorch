/**
 * WGSL compute shader for 2D max pooling with 1D-flattened grid (f16).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_MAX_POOL2D_F16 = /* wgsl */ `
enable f16;

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

@group(0) @binding(0) var<storage, read_write> out_vals: array<f16>;
@group(0) @binding(1) var<storage, read> inp: array<f16>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> out_idx: array<i32>;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u;
override write_indices: u32 = 0u;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.N * params.C * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let c = (i / (params.OW * params.OH)) % params.C;
  let n = i / (params.OW * params.OH * params.C);

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_c_base = (n * params.C + c) * params.IH;

  var best: f16 = -65504.0h;
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
  out_vals[i] = best;
  if (write_indices != 0u) {
    out_idx[i] = best_idx;
  }
}
`;
