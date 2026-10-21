/**
 * WGSL compute shader for float16 transposed 2D convolution (gather form).
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/et_vk_conv2d/conv_transpose2d.wgsl)
 * Adapted for WebGPU float16 (shader-f16).
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 * Project: https://github.com/barhanc/wgpu-executorch
 */
export const SHADER_CONV_TRANSPOSE2D_F16 = /* wgsl */ `
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
override stride_x: u32 = 4294967295u; // = count_x * wg_size; set by host for 2D-spill

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

  let icpg = params.IC / params.groups; // input channels per group
  let ocpg = params.OC / params.groups; // output channels per group
  let g = oc / ocpg;
  let ic0 = g * icpg;
  let oc_in_g = oc % ocpg;

  var acc: f32 = 0.0;
  if (params.has_bias != 0u) {
    acc = f32(bias[oc]);
  }

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  for (var kh: u32 = 0u; kh < params.KH; kh = kh + 1u) {
    let num_h = i32(oh) + i32(params.pH) - i32(kh) * i32(params.dH);
    if (num_h % i32(params.sH) != 0) {
      continue;
    }
    let ih = num_h / i32(params.sH);
    if (ih < 0 || ih >= iH) {
      continue;
    }
    for (var kw: u32 = 0u; kw < params.KW; kw = kw + 1u) {
      let num_w = i32(ow) + i32(params.pW) - i32(kw) * i32(params.dW);
      if (num_w % i32(params.sW) != 0) {
        continue;
      }
      let iw = num_w / i32(params.sW);
      if (iw < 0 || iw >= iW) {
        continue;
      }
      for (var icg: u32 = 0u; icg < icpg; icg = icg + 1u) {
        let ic = ic0 + icg;
        let in_idx = ((b * params.IC + ic) * params.IH + u32(ih)) * params.IW + u32(iw);
        let w_idx = ((ic * ocpg + oc_in_g) * params.KH + kh) * params.KW + kw;
        acc = acc + f32(input[in_idx]) * f32(weight[w_idx]);
      }
    }
  }
  out[i] = conv_epilogue(f16(acc));
}
`;
