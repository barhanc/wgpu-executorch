/**
 * WGSL compute shader for 64x64 register-tiled float16 2D convolution GEMM.
 * Requires WebGPU shader-f16 feature.
 */
export const SHADER_CONV2D_TILE64_F16 = /* wgsl */ `
enable f16;

struct Params {
  B: u32, IC: u32, IH: u32, IW: u32,
  OC: u32, OH: u32, OW: u32, KH: u32, KW: u32,
  sH: u32, sW: u32, pH: u32, pW: u32, dH: u32, dW: u32,
  groups: u32, has_bias: u32, _p0: u32, _p1: u32, _p2: u32,
}
@group(0) @binding(0) var<storage, read_write> out: array<f16>;
@group(0) @binding(1) var<storage, read> input: array<f16>;
@group(0) @binding(2) var<storage, read> weight: array<vec4<f16>>;
@group(0) @binding(3) var<storage, read> bias: array<f16>;
@group(0) @binding(4) var<uniform> params: Params;

// a_sub: 64 output channels x 16 K-elements (4 vec4s)
var<workgroup> a_sub: array<array<vec4<f16>, 4>, 64>;
// b_sub: 16 K-elements x 64 spatial elements (16 vec4s)
var<workgroup> b_sub: array<array<vec4<f16>, 16>, 16>;

@compute @workgroup_size(8, 8, 1)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) tid: u32
) {
  let tile_row0 = wg_id.y * 64u;
  let tile_col0 = wg_id.x * 64u;
  let M = params.OC;
  let N = params.B * params.OH * params.OW;
  let K = params.IC * params.KH * params.KW;
  let K_vecs = K / 4u;
  let khw = params.KH * params.KW;
  let ohw = params.OH * params.OW;

  let iH = i32(params.IH);
  let iW = i32(params.IW);
  let in_hw = params.IH * params.IW;
  let in_w = params.IW;
  let sH = i32(params.sH);
  let sW = i32(params.sW);
  let pH = i32(params.pH);
  let pW = i32(params.pW);
  let dH = i32(params.dH);
  let dW = i32(params.dW);

  var acc0: array<vec4<f16>, 8>;
  var acc1: array<vec4<f16>, 8>;
  for (var i = 0u; i < 8u; i = i + 1u) {
    acc0[i] = vec4<f16>(0.0h);
    acc1[i] = vec4<f16>(0.0h);
  }

  let num_tiles = (K + 15u) / 16u;
  for (var t = 0u; t < num_tiles; t = t + 1u) {
    let k_start = t * 16u;
    let k4_start = t * 4u;

    // 1. Load a_sub (weights): each thread tid (0..63) loads row tid (4 vec4s)
    let a_m = tile_row0 + tid;
    let a_valid = a_m < M;
    let a_base = a_m * K_vecs + k4_start;
    for (var col = 0u; col < 4u; col = col + 1u) {
      if (a_valid && (k4_start + col) < K_vecs) {
        a_sub[tid][col] = weight[a_base + col];
      } else {
        a_sub[tid][col] = vec4<f16>(0.0h);
      }
    }

    // 2. Load b_sub (implicit im2col input): 16 rows x 16 vec4s = 256 vec4s total
    let b_vec_idx = tid << 2u;
    for (var i = 0u; i < 4u; i = i + 1u) {
      let flat_idx = b_vec_idx + i;
      let b_r = flat_idx >> 4u;
      let b_c4 = flat_idx & 15u;
      let kk_idx = k_start + b_r;
      let n_base = tile_col0 + (b_c4 << 2u);

      var b_row = vec4<f16>(0.0h);
      if (kk_idx < K) {
        let k_ic = kk_idx / khw;
        let rem = kk_idx % khw;
        let kh = i32(rem / params.KW) * dH;
        let kw = i32(rem % params.KW) * dW;

        for (var kk = 0u; kk < 4u; kk = kk + 1u) {
          let n = n_base + kk;
          if (n < N) {
            let b_val = n / ohw;
            let sp = n % ohw;
            let oh = sp / params.OW;
            let ow = sp % params.OW;
            let ih = i32(oh) * sH - pH + kh;
            let iw = i32(ow) * sW - pW + kw;
            if (ih >= 0 && ih < iH && iw >= 0 && iw < iW) {
              let ch_idx = b_val * params.IC + k_ic;
              b_row[kk] = input[ch_idx * in_hw + u32(ih) * in_w + u32(iw)];
            }
          }
        }
      }
      b_sub[b_r][b_c4] = b_row;
    }

    workgroupBarrier();

    // 3. Compute phase: 8x8 register accumulation per thread
    let thread_row = local_id.y * 8u;
    let thread_col4 = local_id.x * 2u;

    for (var k4 = 0u; k4 < 4u; k4 = k4 + 1u) {
      let b0_0 = b_sub[k4 * 4u + 0u][thread_col4];
      let b0_1 = b_sub[k4 * 4u + 0u][thread_col4 + 1u];

      let b1_0 = b_sub[k4 * 4u + 1u][thread_col4];
      let b1_1 = b_sub[k4 * 4u + 1u][thread_col4 + 1u];

      let b2_0 = b_sub[k4 * 4u + 2u][thread_col4];
      let b2_1 = b_sub[k4 * 4u + 2u][thread_col4 + 1u];

      let b3_0 = b_sub[k4 * 4u + 3u][thread_col4];
      let b3_1 = b_sub[k4 * 4u + 3u][thread_col4 + 1u];

      for (var ir = 0u; ir < 8u; ir = ir + 1u) {
        let ac = a_sub[thread_row + ir][k4];
        acc0[ir] = acc0[ir] + b0_0 * ac.x + b1_0 * ac.y + b2_0 * ac.z + b3_0 * ac.w;
        acc1[ir] = acc1[ir] + b0_1 * ac.x + b1_1 * ac.y + b2_1 * ac.z + b3_1 * ac.w;
      }
    }

    workgroupBarrier();
  }

  // 4. Write output with optional bias
  let thread_row = local_id.y * 8u;
  let thread_col4 = local_id.x * 2u;
  for (var ir = 0u; ir < 8u; ir = ir + 1u) {
    let m = tile_row0 + thread_row + ir;
    if (m < M) {
      var bias_val = 0.0h;
      if (params.has_bias != 0u) {
        bias_val = bias[m];
      }
      let v0 = acc0[ir] + vec4<f16>(bias_val);
      let v1 = acc1[ir] + vec4<f16>(bias_val);

      let n_base0 = tile_col0 + (thread_col4 << 2u);
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let n = n_base0 + kk;
        if (n < N) {
          let b_idx = n / ohw;
          let sp = n % ohw;
          let oh = sp / params.OW;
          let ow = sp % params.OW;
          out[((b_idx * params.OC + m) * params.OH + oh) * params.OW + ow] = v0[kk];
        }
      }

      let n_base1 = n_base0 + 4u;
      for (var kk = 0u; kk < 4u; kk = kk + 1u) {
        let n = n_base1 + kk;
        if (n < N) {
          let b_idx = n / ohw;
          let sp = n % ohw;
          let oh = sp / params.OW;
          let ow = sp % params.OW;
          out[((b_idx * params.OC + m) * params.OH + oh) * params.OW + ow] = v1[kk];
        }
      }
    }
  }
}
`;
