import type { Task } from '../types';
import { startFrameLoop } from './loop';

export const task: Task = {
  id: 'yolo26-detection',
  title: 'YOLO26-N Detection',
  description: 'NMS-free object detection with GPU-rasterized bounding boxes.',
  startFrameLoop,
};
