/**
 * Direct small-channel 3x3 convolution: stage four filters once and reuse them across
 * adjacent spatial outputs. Small reductions avoid implicit-GEMM input staging.
 * Inspired by ruNNtime's small-channel direct convolution, adapted to NCHW.
 * https://github.com/software-mansion/runntime (MIT)
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_DIRECT_SMALL = /* wgsl */ `
struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

const pixels: u32 = 2u;
override input_h: u32 = 1u;
override input_w: u32 = 1u;
override output_h: u32 = 1u;
override output_w: u32 = 1u;

override channels: u32 = 3u;
override filter_count: u32 = 27u;
override stride_h: u32 = 2u;
override stride_w: u32 = 2u;
override pad_h: u32 = 1u;
override pad_w: u32 = 1u;
override dilation_h: u32 = 1u;
override dilation_w: u32 = 1u;

var<workgroup> filters: array<vec4<f32>, filter_count>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_index) tid: u32) {
  let channel = wg.y * 4u;
  for (var k = tid; k < filter_count; k += 64u) {
    var v = vec4<f32>(0.0);
    for (var c = 0u; c < 4u; c++) {
      if (channel + c < params.OC) { v[c] = weight[(channel + c) * filter_count + k]; }
    }
    filters[k] = v;
  }
  workgroupBarrier();
  let spatial = output_h * output_w;
  let n0 = wg.x * (64u * pixels) + tid * pixels;
  let source_base = wg.z * channels * input_h * input_w;
  var acc: array<vec4<f32>, pixels>;
  var row: array<i32, pixels>;
  var col: array<i32, pixels>;
  for (var p = 0u; p < pixels; p++) {
    row[p] = i32((n0 + p) / output_w) * i32(stride_h) - i32(pad_h);
    col[p] = i32((n0 + p) % output_w) * i32(stride_w) - i32(pad_w);
    acc[p] = vec4<f32>(0.0);
  }
  // Two neighboring stride-one stencils share two of their three columns.
  // Keep that four-value row in registers; other geometries use guarded gathers.
  if (channels != 3u && stride_w == 1u && (output_w & 1u) == 0u) {
    for (var ic = 0u; ic < channels; ic++) {
      for (var ky = 0u; ky < 3u; ky++) {
        let iy = row[0] + i32(ky);
        var line = vec4<f32>(0.0);
        for (var j = 0u; j < 4u; j++) {
          let ix = col[0] + i32(j);
          if (n0 < spatial && iy >= 0 && iy < i32(input_h) && ix >= 0 && ix < i32(input_w)) {
            line[j] = input[source_base + ic * input_h * input_w + u32(iy) * input_w + u32(ix)];
          }
        }
        for (var kx = 0u; kx < 3u; kx++) {
          let w = filters[ic * 9u + ky * 3u + kx];
          acc[0] = acc[0] + vec4<f32>(line[kx]) * w;
          acc[1] = acc[1] + vec4<f32>(line[kx + 1u]) * w;
        }
      }
    }
  } else {
    for (var ic = 0u; ic < channels; ic++) {
      for (var ky = 0u; ky < 3u; ky++) {
        for (var kx = 0u; kx < 3u; kx++) {
          let w = filters[ic * 9u + ky * 3u + kx];
          for (var p = 0u; p < pixels; p++) {
            let iy = row[p] + i32(ky * dilation_h);
            let ix = col[p] + i32(kx * dilation_w);
            var x = 0.0;
            if (n0 + p < spatial && iy >= 0 && ix >= 0 && iy < i32(input_h) && ix < i32(input_w)) {
              x = input[source_base + ic * input_h * input_w + u32(iy) * input_w + u32(ix)];
            }
            acc[p] = acc[p] + vec4<f32>(x) * w;
          }
        }
      }
    }
  }
  var bias_value = vec4<f32>(0.0);
  if (params.has_bias != 0u) {
    for (var c = 0u; c < 4u; c++) {
      if (channel + c < params.OC) { bias_value[c] = bias[channel + c]; }
    }
  }
  for (var p = 0u; p < pixels; p++) {
    if (n0 + p < spatial) {
      let value = acc[p] + bias_value;
      for (var c = 0u; c < 4u; c++) {
        if (channel + c < params.OC) {
          out[(wg.z * params.OC + channel + c) * spatial + n0 + p] = value[c];
        }
      }
    }
  }
}
`;
