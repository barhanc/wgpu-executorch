/**
 * Experimental Winograd F(2x2,3x3) / F(4x4,3x3) convolution transform.
 * Algorithm: https://arxiv.org/abs/1509.09308 (Lavin and Gray).
 * Source: https://github.com/barhanc/wgpu-executorch
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_WINOGRAD_OUTPUT = /* wgsl */ `
@group(0) @binding(0) var<storage, read> products: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, PH: u32,
  PW: u32, TH: u32, TW: u32, T: u32,
  has_bias: u32, pad0: u32, minimum: f32, maximum: f32,
}
@group(0) @binding(2) var<uniform> p: Params;
override tile_size: u32 = 2u;

@group(0) @binding(3) var<storage, read> bias: array<f32>;
fn transform(d: array<f32, 6>) -> array<f32, 4> {
  if (tile_size == 2u) {
    return array<f32, 4>(d[0]+d[1]+d[2], d[1]-d[2]-d[3], 0.0, 0.0);
  }
  return array<f32, 4>(
    d[0]+d[1]+d[2]+d[3]+d[4],
    d[1]-d[2]+2.0*(d[3]-d[4]),
    d[1]+d[2]+4.0*(d[3]+d[4]),
    d[1]-d[2]+8.0*(d[3]-d[4])+d[5]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tile = gid.x;
  let oc = gid.y;
  let tiles_per_batch = p.TH * p.TW;
  if (tile >= p.B * tiles_per_batch) { return; }
  let batch = tile / tiles_per_batch;
  let y0 = ((tile % tiles_per_batch) / p.TW) * tile_size;
  let x0 = (tile % p.TW) * tile_size;
  let alpha = tile_size + 2u;
  var rows: array<array<f32, 4>, 6>;
  for (var y = 0u; y < alpha; y++) {
    var row: array<f32, 6>;
    for (var x = 0u; x < alpha; x++) {
      row[x] = products[((y*alpha+x)*p.OC+oc)*p.T+tile];
    }
    rows[y] = transform(row);
  }
  var bias_value = 0.0;
  if (p.has_bias != 0u) { bias_value = f32(bias[oc]); }
  for (var x = 0u; x < tile_size; x++) {
    var column: array<f32, 6>;
    for (var y = 0u; y < alpha; y++) { column[y] = rows[y][x]; }
    let values = transform(column);
    for (var y = 0u; y < tile_size; y++) {
      if (y0+y < p.OH && x0+x < p.OW) {
        output[((batch*p.OC+oc)*p.OH+y0+y)*p.OW+x0+x] = f32(clamp(values[y]+bias_value, p.minimum, p.maximum));
      }
    }
  }
}
`;
