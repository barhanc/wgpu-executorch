/**
 * A demo task: a camera-driven WebGPU pipeline with its own page in the
 * gallery. Tasks live in their own directory under `src/tasks/` and only need
 * a `startFrameLoop` implementation plus the page copy.
 */

/** Per-frame metrics callback: smoothed FPS and inference time in ms. */
export type MetricsFn = (fps: number, ms: number) => void;

/**
 * Starts a task's per-frame pipeline for the given camera video.
 * Resolves with a stop function that halts the loop and releases resources.
 */
export type StartFrameLoop = (
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  onMetrics?: MetricsFn
) => Promise<() => void>;

/** Gallery tile + task page configuration. */
export type Task = {
  /** Hash-route segment and tile key (e.g. `yolo26-detection`). */
  readonly id: string;
  /** Tile and page title. */
  readonly title: string;
  /** One-line description shown on the tile and the task page. */
  readonly description: string;
  /** The task's record-once GPU pipeline. */
  readonly startFrameLoop: StartFrameLoop;
};
