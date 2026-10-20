/**
 * WGSL compute shader for float16 fused depthwise convolution with clamp bounds (up to 7x7 kernels).
 * Uses 2D spatial workgroup layout (16x16) and workgroup-shared cached weights and bias.
 * Requires WebGPU shader-f16 feature.
 */
export const SHADER_CONV_WITH_CLAMP_DEPTHWISE_F16 = /* wgsl */ `
enable f16;

@group(0) @binding(0) var<storage, read_write> t_out: array<f16>;
@group(0) @binding(1) var<storage, read> t_x: array<f16>;
@group(0) @binding(2) var<storage, read> t_weight: array<f16>;
@group(0) @binding(3) var<storage, read> t_bias: array<f16>;

struct Params {
  N: u32,
  IC: u32,
  H_in: u32,
  W_in: u32,
  OC: u32,
  H_out: u32,
  W_out: u32,
  Kh: u32,
  Kw: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dil_h: u32,
  dil_w: u32,
  has_bias: u32,
  numel: u32,
  groups: u32,
  ic_per_group: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
  output_min: f32,
  output_max: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> w_tile: array<f16, 49>;
var<workgroup> bias_val: f16;

@compute @workgroup_size(16, 16, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(workgroup_id) wg_id: vec3<u32>
) {
  let oc = wg_id.z % params.OC;
  let b = wg_id.z / params.OC;

  let tid = local_id.y * 16u + local_id.x;
  let khw = params.Kh * params.Kw;
  if (tid < khw) {
    w_tile[tid] = t_weight[oc * khw + tid];
  }
  if (tid == 0u) {
    if (params.has_bias != 0u) {
      bias_val = t_bias[oc];
    } else {
      bias_val = 0.0h;
    }
  }
  workgroupBarrier();

  let ow = gid.x;
  let oh = gid.y;

  if (ow >= params.W_out || oh >= params.H_out || b >= params.N) {
    return;
  }

  var acc = bias_val;
  let ih0 = i32(oh) * i32(params.stride_h) - i32(params.pad_h);
  let iw0 = i32(ow) * i32(params.stride_w) - i32(params.pad_w);
  let iH = i32(params.H_in);
  let iW = i32(params.W_in);
  let in_base = (b * params.IC + oc) * params.H_in;

  for (var kh: u32 = 0u; kh < params.Kh; kh = kh + 1u) {
    let ih = ih0 + i32(kh) * i32(params.dil_h);
    if (ih >= 0 && ih < iH) {
      let row_offset = (in_base + u32(ih)) * params.W_in;
      let w_offset = kh * params.Kw;
      for (var kw: u32 = 0u; kw < params.Kw; kw = kw + 1u) {
        let iw = iw0 + i32(kw) * i32(params.dil_w);
        if (iw >= 0 && iw < iW) {
          acc = acc + t_x[row_offset + u32(iw)] * w_tile[w_offset + kw];
        }
      }
    }
  }

  let out_idx = ((b * params.OC + oc) * params.H_out + oh) * params.W_out + ow;
  t_out[out_idx] = clamp(acc, f16(params.output_min), f16(params.output_max));
}
`;
