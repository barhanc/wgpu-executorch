/**
 * Experimental Winograd F(2x2,3x3) / F(4x4,3x3) convolution transform.
 * Algorithm: https://arxiv.org/abs/1509.09308 (Lavin and Gray).
 * Source: https://github.com/barhanc/wgpu-executorch
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_WINOGRAD_INPUT_F16 = /* wgsl */ `
enable f16;
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> transformed: array<f32>;
struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, PH: u32,
  PW: u32, TH: u32, TW: u32, T: u32,
  has_bias: u32, pad0: u32, minimum: f32, maximum: f32,
}
@group(0) @binding(2) var<uniform> p: Params;
override tile_size: u32 = 2u;

fn transform(d: array<f32, 6>) -> array<f32, 6> {
  if (tile_size == 2u) {
    return array<f32, 6>(d[0]-d[2], d[1]+d[2], d[2]-d[1], d[1]-d[3], 0.0, 0.0);
  }
  return array<f32, 6>(
    4.0*d[0]-5.0*d[2]+d[4],
    -4.0*d[1]-4.0*d[2]+d[3]+d[4],
    4.0*d[1]-4.0*d[2]-d[3]+d[4],
    -2.0*d[1]-d[2]+2.0*d[3]+d[4],
    2.0*d[1]-d[2]-2.0*d[3]+d[4],
    4.0*d[1]-5.0*d[3]+d[5]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tile = gid.x;
  let ic = gid.y;
  if (tile >= p.T) { return; }
  let alpha = tile_size + 2u;
  let tiles_per_batch = p.TH * p.TW;
  let batch = tile / tiles_per_batch;
  let y0 = i32(((tile % tiles_per_batch) / p.TW) * tile_size) - i32(p.PH);
  let x0 = i32((tile % p.TW) * tile_size) - i32(p.PW);
  var rows: array<array<f32, 6>, 6>;
  for (var y = 0u; y < alpha; y++) {
    var row: array<f32, 6>;
    for (var x = 0u; x < alpha; x++) {
      let iy = y0 + i32(y);
      let ix = x0 + i32(x);
      if (batch < p.B && iy >= 0 && iy < i32(p.IH) && ix >= 0 && ix < i32(p.IW)) {
        row[x] = f32(input[((batch*p.IC+ic)*p.IH+u32(iy))*p.IW+u32(ix)]);
      }
    }
    rows[y] = transform(row);
  }
  for (var x = 0u; x < alpha; x++) {
    var column: array<f32, 6>;
    for (var y = 0u; y < alpha; y++) { column[y] = rows[y][x]; }
    let values = transform(column);
    for (var y = 0u; y < alpha; y++) {
      transformed[((y*alpha+x)*p.IC+ic)*p.T+tile] = values[y];
    }
  }
}
`;
