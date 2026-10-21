import type { Program, ProgramOp, ProgramTensor, ProgramValue } from '../src/program';

/**
 * Replace eligible dense 3x3 operators with experimental Winograd transforms/BMM.
 * Constant weights are transformed on the CPU once. Scratch buffers are shared by
 * sequential convolutions and are never allocated or transformed in a timed loop.
 *
 * @param source Original program, unmodified.
 * @param weights Original weights, unmodified.
 * @param tile Output tile edge (2 or 4).
 * @param selected Optional chain indices for isolating operator regressions.
 */
export function winogradProgram(
  source: Program,
  weights: ArrayBuffer,
  tile: 2 | 4,
  selected?: readonly number[]
) {
  const values: ProgramValue[] = structuredClone(source.values) as ProgramValue[];
  const chain: ProgramOp[] = [];
  const pools = structuredClone(source.memoryPlan.pools) as { id: number; size: number }[];
  const nextPool = Math.max(-1, ...pools.map((pool) => pool.id)) + 1;
  let scratchInputBytes = 4,
    scratchProductBytes = 4;
  const scratchInput = values.length;
  values.push({ type: 'tensor', dtype: 'float32', shape: [1], memObjId: nextPool });
  const scratchProduct = values.length;
  values.push({ type: 'tensor', dtype: 'float32', shape: [1], memObjId: nextPool + 1 });
  const nullId = values.length;
  values.push({ type: 'null' });
  const g =
    tile === 2
      ? [
          [1, 0, 0],
          [0.5, 0.5, 0.5],
          [0.5, -0.5, 0.5],
          [0, 0, 1],
        ]
      : [
          [0.25, 0, 0],
          [-1 / 6, -1 / 6, -1 / 6],
          [-1 / 6, 1 / 6, -1 / 6],
          [1 / 24, 1 / 12, 1 / 6],
          [1 / 24, -1 / 12, 1 / 6],
          [0, 0, 1],
        ];
  const alpha = tile + 2;
  const planes = alpha * alpha;
  let offset = Math.ceil(weights.byteLength / 4) * 4;
  const additions: { offset: number; data: Float32Array }[] = [];
  const packed = new Map<number, number>();
  const replaced: number[] = [];
  source.chain.forEach((op, index) => {
    const args = op.args;
    const weight = values[args[1]] as ProgramTensor;
    const scalarIs = (arg: number, expected: number | boolean) => {
      const value = values[args[arg]];
      return value.type === 'scalar' && value.value === expected;
    };
    const listIs = (arg: number, expected: number) => {
      const value = values[args[arg]];
      return (
        value.type === 'list' &&
        value.items.length > 0 &&
        value.items.every((item) => item === expected)
      );
    };
    if (
      (op.name !== 'aten.convolution.default' && op.name !== 'et_vk.conv_with_clamp.default') ||
      (selected && !selected.includes(index)) ||
      weight.shape?.[2] !== 3 ||
      weight.shape?.[3] !== 3 ||
      weight.weightsOffset === undefined ||
      !scalarIs(6, false) ||
      !scalarIs(8, 1) ||
      !listIs(3, 1) ||
      !listIs(5, 1) ||
      !listIs(7, 0)
    ) {
      chain.push(op);
      return;
    }
    const input = values[args[0]] as ProgramTensor;
    const output = values[args.at(-1)!] as ProgramTensor;
    const [oc, ic] = weight.shape;
    const n =
      Math.ceil(
        (input.shape[0] * Math.ceil(output.shape[2] / tile) * Math.ceil(output.shape[3] / tile)) / 4
      ) * 4;
    scratchInputBytes = Math.max(scratchInputBytes, planes * ic * n * 4);
    scratchProductBytes = Math.max(scratchProductBytes, planes * oc * n * 4);
    let packedId = packed.get(args[1]);
    if (packedId === undefined) {
      const array = weight.dtype === 'float16' ? Float16Array : Float32Array;
      const original = new array(weights, weight.weightsOffset, oc * ic * 9);
      const transformed = new Float32Array(planes * oc * ic);
      for (let y = 0; y < alpha; y++)
        for (let x = 0; x < alpha; x++)
          for (let o = 0; o < oc; o++)
            for (let i = 0; i < ic; i++) {
              let sum = 0;
              for (let ky = 0; ky < 3; ky++)
                for (let kx = 0; kx < 3; kx++)
                  sum += g[y][ky] * original[(o * ic + i) * 9 + ky * 3 + kx] * g[x][kx];
              transformed[((y * alpha + x) * oc + o) * ic + i] = sum;
            }
      packedId = values.length;
      values.push({
        type: 'tensor',
        dtype: 'float32',
        shape: [planes, oc, ic],
        weightsOffset: offset,
        weightsLength: transformed.byteLength,
      });
      additions.push({ offset, data: transformed });
      offset += transformed.byteLength;
      packed.set(args[1], packedId);
    }
    const config =
      op.name === 'et_vk.conv_with_clamp.default'
        ? args.slice(2, -1)
        : [...args.slice(2, -1), nullId, nullId];
    chain.push({
      name: 'wgpu.winograd_convolution.default',
      args: [args[0], packedId, ...config, scratchInput, scratchProduct, args.at(-1)!],
    });
    replaced.push(index);
  });
  if (!replaced.length) return { program: source, weights, replaced, scratchBytes: 0 };
  values[scratchInput] = {
    type: 'tensor',
    dtype: 'float32',
    shape: [scratchInputBytes / 4],
    memObjId: nextPool,
  };
  values[scratchProduct] = {
    type: 'tensor',
    dtype: 'float32',
    shape: [scratchProductBytes / 4],
    memObjId: nextPool + 1,
  };
  pools.push(
    { id: nextPool, size: scratchInputBytes },
    { id: nextPool + 1, size: scratchProductBytes }
  );
  const transformedWeights = new ArrayBuffer(offset);
  new Uint8Array(transformedWeights).set(new Uint8Array(weights));
  for (const addition of additions)
    new Uint8Array(transformedWeights, addition.offset, addition.data.byteLength).set(
      new Uint8Array(addition.data.buffer)
    );
  return {
    program: { ...source, chain, values, memoryPlan: { pools } } as Program,
    weights: transformedWeights,
    replaced,
    scratchBytes: scratchInputBytes + scratchProductBytes,
  };
}
