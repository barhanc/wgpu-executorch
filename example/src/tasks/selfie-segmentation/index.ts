import type { Task } from '../types';
import { startFrameLoop } from './loop';

export const task: Task = {
  id: 'selfie-segmentation',
  title: 'Selfie Segmentation',
  tagline: 'real-time segmentation',
  description: 'Foreground segmentation on live camera video, composited fully on the GPU.',
  metricsTip: [
    ['FPS', 'live processing rate: video frames processed per second (camera-paced)'],
    ['MS', 'GPU time per frame; the model alone can do up to 1000 / MS FPS'],
  ],
  startFrameLoop,
};
