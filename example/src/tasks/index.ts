import type { Task } from './types';
import { task as depthEstimation } from './depth-estimation';
import { task as selfieSegmentation } from './selfie-segmentation';
import { task as yolo26Detection } from './yolo26-detection';

/**
 * All demo tasks shown on the gallery tiles, in display order. Adding a demo
 * is one entry here plus a directory under `tasks/`.
 */
export const tasks: readonly Task[] = [selfieSegmentation, yolo26Detection, depthEstimation];
