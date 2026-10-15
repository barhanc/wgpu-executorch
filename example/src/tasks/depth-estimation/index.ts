import type { Task } from '../types';
import { startFrameLoop } from './loop';

export const task: Task = {
  id: 'depth-estimation',
  title: 'Depth Estimation',
  description: 'Monocular relative depth on live camera video, predicted fully on the GPU.',
  startFrameLoop,
};
