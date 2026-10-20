/**
 * WGSL compute shader for bilinear 2D upsampling using 2D spatial grid (f32).
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_UPSAMPLE_BILINEAR2D_SPATIAL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> inp: array<f32>;

struct Params {
  N: u32,
  C: u32,
  IH: u32,
  IW: u32,
  OH: u32,
  OW: u32,
  align_corners: u32,
  _p0: u32,
}
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

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let ow = gid.x;
  let oh = gid.y;
  let c_idx = wg_id.z;

  if (ow >= params.OW || oh >= params.OH) {
    return;
  }

  let sh = src_index(oh, params.IH, params.OH, params.align_corners);
  let sw = src_index(ow, params.IW, params.OW, params.align_corners);

  let h0 = i32(floor(sh));
  let w0 = i32(floor(sw));
  let lh = sh - f32(h0);
  let lw = sw - f32(w0);

  let ih_max = i32(params.IH) - 1;
  let iw_max = i32(params.IW) - 1;
  let h0c = u32(clamp(h0, 0, ih_max));
  let h1c = u32(clamp(h0 + 1, 0, ih_max));
  let w0c = u32(clamp(w0, 0, iw_max));
  let w1c = u32(clamp(w0 + 1, 0, iw_max));

  let base = c_idx * params.IH;
  let r0 = (base + h0c) * params.IW;
  let r1 = (base + h1c) * params.IW;
  let v00 = inp[r0 + w0c];
  let v01 = inp[r0 + w1c];
  let v10 = inp[r1 + w0c];
  let v11 = inp[r1 + w1c];

  let top = v00 + (v01 - v00) * lw;
  let bot = v10 + (v11 - v10) * lw;
  let out_pos = (c_idx * params.OH + oh) * params.OW + ow;
  out[out_pos] = top + (bot - top) * lh;
}
`;
