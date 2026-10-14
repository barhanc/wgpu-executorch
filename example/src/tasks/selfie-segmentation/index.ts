import type { Task } from '../types';
import { startFrameLoop } from './loop';

export const task: Task = {
  id: 'selfie-segmentation',
  title: 'Selfie Segmentation',
  description: 'Foreground segmentation on live camera video, composited fully on the GPU.',
  startFrameLoop,
};
