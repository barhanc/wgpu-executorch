import type { Tensor, DType } from './tensor';

// ===========================================================================
// Value types
// ===========================================================================

/**
 * A tensor value in a serialized program — declares a GPU buffer with its
 * shape, element type, and how the buffer is allocated.
 *
 * Every tensor value maps to exactly one `GPUBuffer` at runtime. The buffer
 * source is determined by the optional fields:
 *
 * - **Constant** (`weightsOffset` present): data is loaded from `weights.bin`
 *   into a dedicated buffer during {@link WgpuExecutionContext.recordProgram}.
 *   These are model weights and stay immutable after upload.
 *
 * - **Shared pool** (`memObjId` present): the tensor is allocated inside a
 *   shared memory pool (see {@link ProgramPool}). Multiple tensors in the same
 *   pool share one `GPUBuffer` because their lifetimes don't overlap — the
 *   memory planner guarantees only one is live at a time.
 *
 * - **Dedicated** (neither present): a standalone `GPUBuffer` allocated just
 *   for this value. Used when the memory planner can't find a pool to share.
 */
export type ProgramTensor = {
  readonly type: 'tensor';
  /** Tensor dimensions (e.g., `[1, 3, 520, 520]`). */
  readonly shape: readonly number[];
  /** Element data type. */
  readonly dtype: DType;
  /** Shared memory pool index (absent = dedicated buffer). */
  readonly memObjId?: number;
  /** Byte offset into weights.bin (absent = not a constant). */
  readonly weightsOffset?: number;
  /** Byte length in weights.bin (absent = not a constant). */
  readonly weightsLength?: number;
};

/**
 * A scalar value passed directly to operators — not backed by a GPU buffer.
 *
 * Scalars carry per-operator parameters like `alpha` in `aten.add.Tensor`,
 * clamp bounds in `conv_with_clamp`, or boolean flags. They are resolved at
 * dispatch time and passed to the shader's `recordIn` as plain JS values.
 */
export type ProgramScalar = {
  readonly type: 'scalar';
  readonly value: number | boolean | string;
};

/**
 * An integer list used for operator configuration — not backed by a GPU
 * buffer. Common uses: shape hints for `view_copy` (`[1, -1]`), convolution
 * padding (`[1, 1]`), stride, dilation.
 */
export type ProgramList = {
  readonly type: 'list';
  readonly items: readonly number[];
};

/**
 * A list of tensor value indices used by variadic operators — e.g. the input
 * tensor list of `aten.cat.default` or the output tensor list of
 * `aten.split_with_sizes_copy.default`.
 *
 * Items are indices into the program's {@link Program.values} table and
 * resolve to `Tensor` objects at dispatch time.
 */
export type ProgramTensorList = {
  /** Discriminates the tensor-list variant. */
  readonly type: 'tensorlist';
  /** Value indices of the tensors in this list. */
  readonly items: readonly number[];
};

/**
 * An empty/unused value slot. Reserved by the serializer for optional
 * operator arguments that were omitted (e.g., `memory_format=None` for
 * `clone`). Resolves to `undefined` at dispatch time.
 */
export type ProgramNull = {
  readonly type: 'null';
};

/** Union of all value types in a serialized program. */
export type ProgramValue =
  ProgramTensor | ProgramScalar | ProgramList | ProgramTensorList | ProgramNull;

// ===========================================================================
// Program structure
// ===========================================================================

/**
 * A single operator invocation. `name` is the ExecuTorch operator identifier
 * (e.g., `"aten.add.Tensor"`, `"et_vk.conv_with_clamp.default"`) and `args`
 * are indices into the program's {@link Program.values} table.
 *
 * The operator is dispatched by looking up `name` in the shader registry and
 * calling its `recordIn` with the resolved arguments.
 */
export type ProgramOp = {
  /** ExecuTorch operator identifier. */
  readonly name: string;
  /** Value indices into `Program.values`. */
  readonly args: readonly number[];
};

/**
 * A shared memory pool — a single `GPUBuffer` that multiple tensors use at
 * different points in the computation (their lifetimes don't overlap).
 *
 * Pool size is the maximum byte size of any tensor assigned to it. This is
 * computed at export time by the ExecuTorch memory planner and embedded in
 * the program JSON.
 */
export type ProgramPool = {
  /** Unique pool index (matches {@link ProgramTensor.memObjId}). */
  readonly id: number;
  /** Buffer size in bytes. */
  readonly size: number;
};

/**
 * The serialized program format — the JSON structure produced by
 * `scripts/pte_to_program.py` and consumed by
 * {@link WgpuExecutionContext.recordProgram}.
 *
 * A program is a flat, ordered list of operator calls (the "chain") operating
 * on a table of values (tensors, scalars, lists). Each operator's `args` are
 * value indices — the dependency graph is implicit in the value indices, not
 * stored as pointers. The order of `chain` defines execution order.
 *
 * Example:
 * ```json
 * {
 *   "version": "1",
 *   "chain": [{ "name": "aten.add.Tensor", "args": [0, 1, 2, 3] }],
 *   "values": [
 *     { "type": "tensor", "shape": [2, 3], "dtype": "float32", "memObjId": 1 },
 *     { "type": "tensor", "shape": [1, 3], "dtype": "float32", "memObjId": 1 },
 *     { "type": "scalar", "value": 1.0 },
 *     { "type": "tensor", "shape": [2, 3], "dtype": "float32", "memObjId": 0 }
 *   ],
 *   "inputIds": [0, 1],
 *   "outputIds": [3],
 *   "memoryPlan": { "pools": [{ "id": 0, "size": 24 }, { "id": 1, "size": 36 }] }
 * }
 * ```
 */
export type Program = {
  /** Schema version (currently `"1"`). */
  readonly version: string;
  /** Ordered list of operator invocations — defines execution order. */
  readonly chain: readonly ProgramOp[];
  /** Value table — tensors, scalars, lists, and nulls referenced by `chain`. */
  readonly values: readonly ProgramValue[];
  /** Value indices that are graph inputs (map to `args[0..n]` in `recordProgram`). */
  readonly inputIds: readonly number[];
  /** Value indices that are graph outputs (map to `args[n..]` in `recordProgram`). */
  readonly outputIds: readonly number[];
  /** Memory plan — shared GPU buffer pools for intermediate activations. */
  readonly memoryPlan: {
    readonly pools: readonly ProgramPool[];
  };
};

// ===========================================================================
// Argument resolution
// ===========================================================================

/**
 * Resolves an operator's value-index args into concrete values.
 *
 * For each index in `op.args`, looks up the corresponding entry in `values`
 * and returns:
 * - `Tensor` for tensor values (from the `tensors` map)
 * - `number | boolean | string` for scalar values
 * - `number[]` for list values
 * - `Tensor[]` for tensor-list values (from the `tensors` map)
 * - `undefined` for null values
 *
 * @param op The operator whose args to resolve.
 * @param values The full value table.
 * @param tensors Map from value index to its GPU Tensor view.
 * @returns Resolved args ready to pass to a shader's `recordIn`.
 * @internal
 */
export function resolveArgs(
  op: ProgramOp,
  values: readonly ProgramValue[],
  tensors: Map<number, Tensor>
): unknown[] {
  return op.args.map((idx) => {
    const v = values[idx];
    switch (v.type) {
      case 'tensor': {
        const t = tensors.get(idx);
        if (t === undefined) {
          throw new Error(`program: no tensor for value ${idx} referenced by '${op.name}'`);
        }
        return t;
      }
      case 'scalar':
        return v.value;
      case 'list':
        return [...v.items];
      case 'tensorlist': {
        return v.items.map((idx) => {
          const item = values[idx];
          if (item?.type !== 'tensor') {
            throw new Error(`program: tensorlist item ${idx} is not a tensor (${op.name})`);
          }
          const t = tensors.get(idx);
          if (t === undefined) {
            throw new Error(`program: no tensor for value ${idx} referenced by '${op.name}'`);
          }
          return t;
        });
      }
      case 'null':
        return undefined;
    }
  });
}
