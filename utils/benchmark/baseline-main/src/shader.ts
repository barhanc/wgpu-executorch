import type { WgpuExecutionContext } from './context';

/**
 * Represents a GPU compute shader in the WebGPU execution engine.
 *
 * @typeParam TArgs The tuple schema of arguments accepted by this shader's
 * {@link recordIn} method.
 * @typeParam TName Literal string type of the operator name.
 */
export type Shader<
  TArgs extends readonly unknown[] = readonly unknown[],
  TName extends string = string,
> = {
  /**
   * Canonical PyTorch / ExecuTorch operator identifier.
   */
  readonly name: TName;

  /**
   * Raw WebGPU Shading Language source code, either as a single shader string
   * or a dictionary of named shader variants.
   */
  readonly code: string | Record<string, string>;

  /**
   * Dispatches and records the operator's compute pass into the execution
   * context. Validates input/output tensors, allocates uniform buffers,
   * compiles or retrieves cached pipelines, and records the dispatch.
   *
   * @param ctx The execution context to record dispatches into.
   * @param args Positional arguments for the operator.
   */
  readonly recordIn: (ctx: WgpuExecutionContext, args: TArgs) => void;
};
