/**
 * WGSL compute shader for 1x1 2D convolution in packed HWC4 layout (FP32).
 * Pointwise GEMM streaming with 4-pixel register unrolling, zero shared memory, and zero barriers.
 *
 * Source: wgpu-executorch (https://github.com/barhanc/wgpu-executorch)
 * License: MIT (Copyright (c) 2026 Bartosz Hanc <bartosz.hanc02@gmail.com>)
 */
export const SHADER_CONV2D_HWC4_1X1 = /* wgsl */ `
struct Params {
  B: u32,
  IC4: u32,
  H: u32,
  W: u32,
  OC4: u32,
  has_bias: u32,
  stride_x: u32,
  pad0: u32,
};

@group(0) @binding(0) var<storage, read_write> out: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> weights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let thread_id = gid.y * params.stride_x + gid.x;
  let total_pixels = params.B * params.H * params.W;
  let px_blocks = (total_pixels + 3u) / 4u;
  let total_work = px_blocks * params.OC4;
  if (thread_id >= total_work) {
    return;
  }

  let oc4 = thread_id % params.OC4;
  let px_base = (thread_id / params.OC4) * 4u;

  var b_val = vec4<f32>(0.0);
  if (params.has_bias == 1u) {
    b_val = bias[oc4];
  }

  var acc0 = b_val;
  var acc1 = b_val;
  var acc2 = b_val;
  var acc3 = b_val;

  let wt_oc_base = oc4 * params.IC4 * 4u;
  let p0 = px_base + 0u;
  let p1 = px_base + 1u;
  let p2 = px_base + 2u;
  let p3 = px_base + 3u;

  for (var ic4 = 0u; ic4 < params.IC4; ic4++) {
    let wt_base = wt_oc_base + ic4 * 4u;
    let w0 = weights[wt_base + 0u];
    let w1 = weights[wt_base + 1u];
    let w2 = weights[wt_base + 2u];
    let w3 = weights[wt_base + 3u];

    if (p0 < total_pixels) {
      let x0 = x[p0 * params.IC4 + ic4];
      acc0 += w0 * x0.x + w1 * x0.y + w2 * x0.z + w3 * x0.w;
    }
    if (p1 < total_pixels) {
      let x1 = x[p1 * params.IC4 + ic4];
      acc1 += w0 * x1.x + w1 * x1.y + w2 * x1.z + w3 * x1.w;
    }
    if (p2 < total_pixels) {
      let x2 = x[p2 * params.IC4 + ic4];
      acc2 += w0 * x2.x + w1 * x2.y + w2 * x2.z + w3 * x2.w;
    }
    if (p3 < total_pixels) {
      let x3 = x[p3 * params.IC4 + ic4];
      acc3 += w0 * x3.x + w1 * x3.y + w2 * x3.z + w3 * x3.w;
    }
  }

  if (p0 < total_pixels) { out[p0 * params.OC4 + oc4] = acc0; }
  if (p1 < total_pixels) { out[p1 * params.OC4 + oc4] = acc1; }
  if (p2 < total_pixels) { out[p2 * params.OC4 + oc4] = acc2; }
  if (p3 < total_pixels) { out[p3 * params.OC4 + oc4] = acc3; }
}
`;
