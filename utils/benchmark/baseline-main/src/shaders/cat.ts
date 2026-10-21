import type { Shader } from '../shader';
import type { Tensor } from '../tensor';
import type { WgpuExecutionContext } from '../context';
import { createComputeBundle } from '../command';

import { encodeTensorMeta, TENSOR_META_WGSL } from './utils/meta';

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

  const wgSize = 64;
  const outMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(out));

  let offK = 0;
  let concatSum = 0;
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

    const inMetaBuffer = ctx.uniformBuffer(encodeTensorMeta(inp));
    const paramsBuffer = ctx.uniformBuffer(new Uint32Array([dim, offK, 0, 0]));
    offK += inp.shape[dim];

    const bundle = createComputeBundle(
      ctx.device,
      SHADER,
      [
        { binding: 0, buffer: inp.buffer },
        { binding: 1, buffer: out.buffer },
        { binding: 2, buffer: outMetaBuffer },
        { binding: 3, buffer: inMetaBuffer },
        { binding: 4, buffer: paramsBuffer },
      ],
      // eslint-disable-next-line camelcase, @typescript-eslint/naming-convention
      { wg_size: wgSize }
    );

    const workgroupCount = Math.ceil(inp.numel / wgSize);
    if (workgroupCount > 65535) {
      throw new Error(`${name}: input exceeds dispatch limit`);
    }
    ctx.addDispatch({
      pipeline: bundle.pipeline,
      bindGroup: bundle.bindGroup,
      workgroupCountX: workgroupCount,
      workgroupCountY: 1,
    });
  }

  if (concatSum !== out.shape[dim]) {
    throw new Error(`${name}: concat dim sizes do not sum to output`);
  }
}

/**
 * WebGPU compute kernel for ExecuTorch `aten::cat.default`.
 *
 * Concatenates a list of tensors along one dimension. One dispatch per input
 * tensor scatters it into its disjoint slab of the output buffer.
 */
export const cat: Shader<CatArgs, typeof name> = {
  name,
  code: SHADER,
  recordIn,
};
