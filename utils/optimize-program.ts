import type { Program, ProgramOp, ProgramTensor, ProgramValue } from '../src/program';

type MutableProgram = {
  values: ProgramValue[];
  chain: ProgramOp[];
  inputIds: number[];
  outputIds: number[];
  version: string;
  memoryPlan: { pools: { id: number; size: number }[] };
};

export type OptimizeOptions = {
  aliasCopies?: boolean;
  fuseConvolutionSilu?: boolean;
  movePointwiseBeforeBilinear?: boolean;
};

/**
 * Offline rewrites for immutable, contiguous, single-assignment inference graphs.
 * Rebuilds buffer lifetimes rather than reusing the exporter's obsolete memory plan.
 * Pointwise reordering is algebraically equivalent, but changes floating-point rounding.
 *
 * @param source Original graph; never mutated.
 * @param options Enable individual passes for controlled comparisons.
 */
export function optimizeProgram(source: Program, options: OptimizeOptions = {}) {
  const program = structuredClone(source) as MutableProgram;
  const aliases = new Map<number, number>();
  const stats = { aliasedCopies: 0, reorderedPointwise: 0, fusedSilu: 0 };
  const root = (id: number): number => {
    while (aliases.has(id)) id = aliases.get(id)!;
    return id;
  };
  const tensor = (id: number): ProgramTensor => {
    const value = program.values[id];
    if (value.type !== 'tensor') throw new Error(`Expected tensor at ${id}`);
    return value;
  };
  const tensorIds = (ids: readonly number[]): number[] =>
    ids.flatMap((id) => {
      const value = program.values[id];
      return value.type === 'tensor' ? [id] : value.type === 'tensorlist' ? [...value.items] : [];
    });
  const consumers = new Map<number, number>();
  const writers = new Set<number>();
  for (const op of program.chain) {
    for (const id of tensorIds(op.args.slice(0, -1))) {
      consumers.set(id, (consumers.get(id) ?? 0) + 1);
    }
    for (const id of tensorIds(op.args.slice(-1))) {
      if (
        writers.has(id) ||
        source.inputIds.includes(id) ||
        tensor(id).weightsOffset !== undefined
      ) {
        throw new Error('Optimizer requires immutable, single-assignment tensors');
      }
      writers.add(id);
    }
  }
  for (const id of program.outputIds) consumers.set(id, (consumers.get(id) ?? 0) + 1);

  if (options.movePointwiseBeforeBilinear) {
    for (let i = 0; i + 1 < program.chain.length; i++) {
      const up = program.chain[i];
      const conv = program.chain[i + 1];
      if (up.name !== 'aten.upsample_bilinear2d.vec' || conv.name !== 'aten.convolution.default')
        continue;
      const middle = up.args[4];
      if (conv.args[0] !== middle || consumers.get(middle) !== 1) continue;
      const input = tensor(up.args[0]);
      const weight = tensor(conv.args[1]);
      const output = tensor(conv.args[9]);
      const listIs = (arg: number, expected: number) => {
        const value = program.values[conv.args[arg]];
        return (
          value.type === 'list' &&
          value.items.length > 0 &&
          value.items.every((v) => v === expected)
        );
      };
      const scalarIs = (arg: number, expected: number | boolean) => {
        const value = program.values[conv.args[arg]];
        return value.type === 'scalar' && value.value === expected;
      };
      if (
        input.shape.length !== 4 ||
        weight.shape[2] !== 1 ||
        weight.shape[3] !== 1 ||
        !listIs(3, 1) ||
        !listIs(4, 0) ||
        !listIs(5, 1) ||
        !listIs(7, 0) ||
        !scalarIs(6, false) ||
        !scalarIs(8, 1) ||
        input.shape[2] > output.shape[2] ||
        input.shape[3] > output.shape[3]
      )
        continue;
      const low = program.values.length;
      program.values.push({
        type: 'tensor',
        dtype: output.dtype,
        shape: [input.shape[0], output.shape[1], input.shape[2], input.shape[3]],
      });
      program.chain[i] = { name: conv.name, args: [up.args[0], ...conv.args.slice(1, -1), low] };
      program.chain[i + 1] = { name: up.name, args: [low, ...up.args.slice(1, -1), conv.args[9]] };
      stats.reorderedPointwise++;
      i++;
    }
  }

  if (options.fuseConvolutionSilu) {
    const chain: ProgramOp[] = [];
    for (let i = 0; i < program.chain.length; i++) {
      const conv = program.chain[i];
      const sigmoid = program.chain[i + 1];
      const mul = program.chain[i + 2];
      const out = conv.args[9];
      const gate = sigmoid?.args[1];
      if (
        conv.name === 'aten.convolution.default' &&
        sigmoid?.name === 'aten.sigmoid.default' &&
        mul?.name === 'aten.mul.Tensor' &&
        sigmoid.args[0] === out &&
        ((mul.args[0] === out && mul.args[1] === gate) ||
          (mul.args[1] === out && mul.args[0] === gate)) &&
        consumers.get(out) === 2 &&
        consumers.get(gate) === 1
      ) {
        chain.push({
          name: 'wgpu.convolution_silu.default',
          args: [...conv.args.slice(0, -1), mul.args[2]],
        });
        stats.fusedSilu++;
        i += 2;
      } else chain.push(conv);
    }
    program.chain = chain;
  }

  if (options.aliasCopies) {
    const copyOps = new Set([
      'aten.clone.default',
      'aten.view_copy.default',
      'aten.squeeze_copy.dims',
      'aten.unsqueeze_copy.default',
      'et_vk.prepack.default',
    ]);
    program.chain = program.chain.filter((op) => {
      if (!copyOps.has(op.name)) return true;
      const from = tensor(op.args[0]);
      const destination = op.args[op.args.length - 1];
      const to = tensor(destination);
      const numel = (value: ProgramTensor) => value.shape.reduce((n, size) => n * size, 1);
      if (from.dtype !== to.dtype || numel(from) !== numel(to)) return true;
      const original = tensor(root(op.args[0]));
      if (original.weightsOffset !== undefined) {
        program.values[destination] = {
          ...to,
          weightsOffset: original.weightsOffset,
          weightsLength: original.weightsLength,
        };
      } else {
        aliases.set(destination, root(op.args[0]));
      }
      stats.aliasedCopies++;
      return false;
    });
  }

  // Inclusive intervals prohibit overlapping input/output bindings in one dispatch.
  const intervals = new Map<number, { start: number; end: number; size: number; pool: number }>();
  const touch = (id: number, time: number) => {
    if (tensor(id).weightsOffset !== undefined) return;
    const canonical = root(id);
    const value = tensor(id);
    const size =
      Math.ceil(
        (value.shape.reduce((n, d) => n * d, 1) *
          { float16: 2, float32: 4, int32: 4, float64: 8, int64: 8, int8: 1, uint8: 1, bool: 1 }[
            value.dtype
          ]) /
          4
      ) * 4;
    const interval = intervals.get(canonical);
    if (interval) {
      interval.start = Math.min(interval.start, time);
      interval.end = Math.max(interval.end, time);
      interval.size = Math.max(interval.size, size);
    } else intervals.set(canonical, { start: time, end: time, size, pool: -1 });
  };
  program.inputIds.forEach((id) => touch(id, -1));
  program.chain.forEach((op, i) => tensorIds(op.args).forEach((id) => touch(id, i)));
  program.outputIds.forEach((id) => touch(id, program.chain.length));
  const pools: { id: number; size: number; end: number }[] = [];
  for (const interval of [...intervals.values()].sort(
    (a, b) => a.start - b.start || b.size - a.size
  )) {
    const available = pools.filter((pool) => pool.end < interval.start);
    available.sort((a, b) => Math.max(a.size, interval.size) - Math.max(b.size, interval.size));
    const pool = available[0] ?? { id: pools.length, size: 0, end: -Infinity };
    if (pool.id === pools.length) pools.push(pool);
    pool.size = Math.max(pool.size, interval.size);
    pool.end = interval.end;
    interval.pool = pool.id;
  }
  program.memoryPlan.pools = pools.map(({ id, size }) => ({ id, size }));

  // Drop dead values so unused intermediate tensors do not allocate dedicated buffers.
  const used = new Set([
    ...program.inputIds,
    ...program.outputIds,
    ...program.chain.flatMap((op) => [...op.args]),
  ]);
  for (const id of used) {
    const value = program.values[id];
    if (value.type === 'tensorlist') value.items.forEach((item) => used.add(item));
  }
  const oldIds = [...used].sort((a, b) => a - b);
  const ids = new Map(oldIds.map((id, index) => [id, index]));
  program.values = oldIds.map((id) => {
    const value = program.values[id];
    if (value.type === 'tensorlist')
      return { ...value, items: value.items.map((item) => ids.get(item)!) };
    if (value.type !== 'tensor') return value;
    const { memObjId: previousPool, ...rest } = value;
    void previousPool;
    return value.weightsOffset !== undefined
      ? rest
      : { ...rest, memObjId: intervals.get(root(id))!.pool };
  });
  program.chain = program.chain.map((op) => ({ ...op, args: op.args.map((id) => ids.get(id)!) }));
  program.inputIds = program.inputIds.map((id) => ids.get(id)!);
  program.outputIds = program.outputIds.map((id) => ids.get(id)!);
  return { program: program as Program, stats };
}
