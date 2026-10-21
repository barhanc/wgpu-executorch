/**
 * Direct 2D convolution compute shader (fallback for general grouped cases) in float16.
 * Requires WebGPU shader-f16 feature.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> out: array<f16>;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<f16>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;

struct Params {
  B: u32,
  IC: u32,
  IH: u32,
  IW: u32,
  OC: u32,
  OH: u32,
  OW: u32,
  KH: u32,
  KW: u32,
  sH: u32,
  sW: u32,
  pH: u32,
  pW: u32,
  dH: u32,
  dW: u32,
  groups: u32,
  has_bias: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 256;
override stride_x: u32 = 4294967295u;

override fuse_silu: bool = false;

// Preserve the convolution output and sigmoid's storage precision at the fusion boundary.
fn conv_epilogue(value: f16) -> f16 {
  if (!fuse_silu) { return value; }
  let stored = f16(quantizeToF16(f32(value)));
  let gate = f16(quantizeToF16(f32(f16(1.0) / (f16(1.0) + exp(-stored)))));
  return stored * gate;
}

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let total = params.B * params.OC * params.OH * params.OW;
  let i = gid.y * stride_x + gid.x;
  if (i >= total) {
    return;
  }
  let ow = i % params.OW;
  let oh = (i / params.OW) % params.OH;
  let oc = (i / (params.OW * params.OH)) % params.OC;
  let b = i / (params.OW * params.OH * params.OC);

  let icpg = params.IC / params.groups;
  let ocpg = params.OC / params.groups;
  let g = oc / ocpg;
  let ic0 = g * icpg;

  var acc: f16 = 0.0h;
  if (params.has_bias != 0u) {
    acc = bias[oc];
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
    let ic = ic0 + icg;
    let in_c_base = (b * params.IC + ic) * params.IH;
    let w_c_base = (oc * icpg + icg) * params.KH;
    for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
      let ih = i32(oh) * i32(params.sH) - i32(params.pH) + i32(kh) * i32(params.dH);
      if (ih < 0 || ih >= iH) {
        continue;
      }
      let in_row = (in_c_base + u32(ih)) * params.IW;
      let w_row = (w_c_base + kh) * params.KW;
      for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
        let iw = i32(ow) * i32(params.sW) - i32(params.pW) + i32(kw) * i32(params.dW);
        if (iw < 0 || iw >= iW) {
          continue;
        }
        acc = acc + input[in_row + u32(iw)] * weight[w_row + kw];
      }
    }
  }
  out[i] = conv_epilogue(acc);
}
`;
