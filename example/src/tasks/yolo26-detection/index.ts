import type { Task } from '../types';
import { startFrameLoop } from './loop';

export const task: Task = {
  id: 'yolo26-detection',
  title: 'YOLO26-N Detection',
  tagline: 'yolo26-n · object detection',
  description: 'NMS-free object detection with GPU-rasterized bounding boxes.',
  metricsTip: [
    ['FPS', 'live processing rate: video frames processed per second (camera-paced)'],
    ['MS', 'GPU time per frame; the model alone can do up to 1000 / MS FPS'],
  ],
  startFrameLoop,
};
