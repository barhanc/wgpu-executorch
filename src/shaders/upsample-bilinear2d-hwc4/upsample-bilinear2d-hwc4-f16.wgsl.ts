/**
 * WGSL compute shader for 2D bilinear upsampling in packed HWC4 layout (FP16).
 * Vectorized interpolation across 4 channels concurrently with exact PyTorch coordinate math.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_UPSAMPLE_BILINEAR2D_HWC4_F16 = /* wgsl */ `
enable f16;

struct Params {
  B: u32,
  C4: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  align_corners: u32,
  stride_x: u32,
};

@group(0) @binding(0) var<storage, read> x: array<vec4<f16>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f16>>;
@group(0) @binding(2) var<uniform> params: Params;

fn src_index(dst: u32, insz: u32, outsz: u32, align: u32) -> f32 {
  if (align == 1u) {
    if (outsz <= 1u) {
      return 0.0;
    }
    return f32(dst) * f32(insz - 1u) / f32(outsz - 1u);
  }
  return (f32(dst) + 0.5) * f32(insz) / f32(outsz) - 0.5;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.y * params.stride_x + gid.x;
  let total_pixels = params.B * params.OH * params.OW;
  let total = total_pixels * params.C4;
  if (idx >= total) {
    return;
  }

  let c4 = idx % params.C4;
  let rem0 = idx / params.C4;
  let ow = rem0 % params.OW;
  let rem1 = rem0 / params.OW;
  let oh = rem1 % params.OH;
  let b = rem1 / params.OH;

  let sh = src_index(oh, params.IH, params.OH, params.align_corners);
  let sw = src_index(ow, params.IW, params.OW, params.align_corners);

  let h0 = i32(floor(sh));
  let w0 = i32(floor(sw));
  let lh = f16(sh - f32(h0));
  let lw = f16(sw - f32(w0));

  let ih_max = i32(params.IH) - 1;
  let iw_max = i32(params.IW) - 1;
  let h0c = u32(clamp(h0, 0, ih_max));
  let h1c = u32(clamp(h0 + 1, 0, ih_max));
  let w0c = u32(clamp(w0, 0, iw_max));
  let w1c = u32(clamp(w0 + 1, 0, iw_max));

  let in_b = b * (params.IH * params.IW) * params.C4;
  let v00 = x[in_b + (h0c * params.IW + w0c) * params.C4 + c4];
  let v01 = x[in_b + (h0c * params.IW + w1c) * params.C4 + c4];
  let v10 = x[in_b + (h1c * params.IW + w0c) * params.C4 + c4];
  let v11 = x[in_b + (h1c * params.IW + w1c) * params.C4 + c4];

  let top = v00 * (1.0h - lw) + v01 * lw;
  let bot = v10 * (1.0h - lw) + v11 * lw;
  out[idx] = top * (1.0h - lh) + bot * lh;
}
`;
