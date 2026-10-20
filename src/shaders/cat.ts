import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { TENSOR_META_WGSL } from './utils/meta';

const name = 'aten.cat.default';

/**
 * WGSL compute shader for concatenation along one dimension.
 *
 * Source: ExecuTorch (backends/webgpu/runtime/ops/cat/cat.wgsl)
 * GitHub: https://github.com/pytorch/executorch/blob/main/backends/webgpu/runtime/ops/cat/cat.wgsl
 * License: BSD-3-Clause (Copyright (c) Meta Platforms, Inc. and affiliates)
 */
const SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

${TENSOR_META_WGSL}
@group(0) @binding(2) var<uniform> out_meta: TensorMeta;
@group(0) @binding(3) var<uniform> in_meta: TensorMeta;

struct Params {
  concat_dim: u32,
  off_k: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

override wg_size: u32 = 64u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let in_bufi = gid.x;
  if (in_bufi >= in_meta.numel) {
    return;
  }

  // Scatter: in coord -> out coord, concat dim shifted by off_k (Vulkan concat).
  var rem = in_bufi;
  var out_bufi: u32 = 0u;
  for (var d: u32 = 0u; d < in_meta.ndim; d = d + 1u) {
    let coord = rem / in_meta.strides[d >> 2u][d & 3u];
    rem = rem % in_meta.strides[d >> 2u][d & 3u];
    var out_coord = coord;
    if (d == params.concat_dim) {
      out_coord = coord + params.off_k;
    }
    out_bufi = out_bufi + out_coord * out_meta.strides[d >> 2u][d & 3u];
  }
  output[out_bufi] = input[in_bufi];
}
`;

/**
 * Fast block-scatter shader for contiguous tensor concatenation.
 * Eliminates coordinate-unraveling integer divisions across all dimensions.
 */
const SHADER_BLOCK = /* wgsl */ `
struct BlockParams {
  c_in: u32,
  c_out: u32,
  inner_size: u32,
  off_k: u32,
  total_elements: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: BlockParams;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.total_elements) {
    return;
  }
  let inner_idx = idx % params.inner_size;
  let slice_idx = idx / params.inner_size;
  let c = slice_idx % params.c_in;
  let b = slice_idx / params.c_in;

  let out_idx = (b * params.c_out + params.off_k + c) * params.inner_size + inner_idx;
  output[out_idx] = input[idx];
}
`;

/**
 * 128-bit vectorized block-scatter shader for contiguous tensor concatenation
 * when inner_size is divisible by 4.
 */
const SHADER_BLOCK_VEC4 = /* wgsl */ `
struct BlockParams {
  c_in: u32,
  c_out: u32,
  inner_size: u32,
  off_k: u32,
  total_elements: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<storage, read> input: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: BlockParams;

override wg_size: u32 = 256u;

@compute @workgroup_size(wg_size, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let idx = gid.x + gid.y * (num_workgroups.x * wg_size);
  if (idx >= params.total_elements) {
    return;
  }
  let inner_idx = idx % params.inner_size;
  let slice_idx = idx / params.inner_size;
  let c = slice_idx % params.c_in;
  let b = slice_idx / params.c_in;

  let out_idx = (b * params.c_out + params.off_k + c) * params.inner_size + inner_idx;
  output[out_idx] = input[idx];
}
`;

/**
 * Positional argument tuple for `aten::cat.default`:
 * `[tensors, dim, out]`
 */
export type CatArgs = readonly [tensors: readonly Tensor[], dim: number, out: Tensor];

/**
 * Internal dispatch builder for `aten::cat.default`.
 *
 * @param ctx The execution context.
 * @param args Positional argument tuple for the operator.
 */
function recordIn(ctx: WgpuExecutionContext, args: CatArgs): void {
  const [tensors, dimArg, out] = args;

  if (tensors.length === 0) {
    throw new Error(`${name}: empty input list`);
  }

  const ndim = out.shape.length;
  let dim = dimArg;
  if (dim < 0) {
    dim += ndim;
  }
  if (dim < 0 || dim >= ndim) {
    throw new Error(`${name}: dim out of range`);
  }

  let innerSize = 1;
  for (let d = dim + 1; d < ndim; d++) {
    innerSize *= out.shape[d];
  }

  let offK = 0;
  let concatSum = 0;
  const wgSize = 256;

  for (const inp of tensors) {
    if (inp.dtype !== 'float32' || out.dtype !== 'float32') {
      throw new Error(`${name}: Only float32 tensors are currently supported`);
    }
    if (inp.shape.length !== ndim) {
      throw new Error(`${name}: input rank != output rank`);
    }
    for (let d = 0; d < ndim; d++) {
      if (d !== dim && inp.shape[d] !== out.shape[d]) {
        throw new Error(`${name}: non-concat dim ${d} size mismatch`);
      }
    }
    concatSum += inp.shape[dim];

    const cIn = inp.shape[dim];
    const cOut = out.shape[dim];
    const isVec4 = innerSize % 4 === 0;
    const effectiveInner = isVec4 ? innerSize / 4 : innerSize;
    const totalElements = isVec4 ? inp.numel / 4 : inp.numel;

    const paramsBuffer = ctx.uniformBuffer(
      new Uint32Array([cIn, cOut, effectiveInner, offK, totalElements, 0, 0, 0])
    );
    offK += cIn;

    const bundle = createComputeBundle(
      ctx.device,
      isVec4 ? SHADER_BLOCK_VEC4 : SHADER_BLOCK,
      [
        { binding: 0, buffer: inp.buffer },
        { binding: 1, buffer: out.buffer },
        { binding: 2, buffer: paramsBuffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { wg_size: wgSize }
    );

    const totalWorkgroups = Math.ceil(totalElements / wgSize);
    const workgroupCountX = Math.min(totalWorkgroups, 65535);
    const workgroupCountY = Math.ceil(totalWorkgroups / 65535);

    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX,
      workgroupCountY,
    });
  }

  if (concatSum !== out.shape[dim]) {
    throw new Error(`${name}: concat dim sizes do not sum to output`);
  }
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::cat.default`.
 *
 * Concatenates a list of tensors along one dimension. Uses fast block-scatter
 * and 128-bit vectorization for contiguous slices.
 */
export const cat: Shader<CatArgs, typeof name> = {
  name,
  code: { default: SHADER, block: SHADER_BLOCK, blockVec4: SHADER_BLOCK_VEC4 },
  recordIn,
};
