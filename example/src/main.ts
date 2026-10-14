/**
 * Hash-routed demo UI: a gallery of tiles at `#/`, one page per task.
 * The UI is plain DOM — the interesting code lives in each task's GPU loop.
 */

import type { Task } from './tasks/types';
import { tasks } from './tasks';

/** Metrics tooltip lines shared by every task: `[token, description]` pairs. */
const METRICS_TIP: readonly (readonly [token: string, description: string])[] = [
  ['FPS', 'live processing rate: video frames processed per second (camera-paced)'],
  ['MS', 'GPU time per frame; the model alone can do up to 1000 / MS FPS'],
];

const app = document.getElementById('app') as HTMLElement;
let cleanup: (() => void) | null = null;

function cloneTemplate(id: string): DocumentFragment {
  const template = document.getElementById(id) as HTMLTemplateElement;
  return template.content.cloneNode(true) as DocumentFragment;
}

function fill(node: ParentNode, selector: string, text: string): void {
  (node.querySelector(selector) as HTMLElement).textContent = text;
}

function renderGallery(): void {
  const node = cloneTemplate('gallery-template');
  const tiles = node.querySelector('.tiles') as HTMLElement;

  for (const task of tasks) {
    const tile = document.createElement('a');
    tile.className = 'tile';
    tile.href = `#/${task.id}`;
    tile.innerHTML =
      '<h2 class="tile-title"></h2><p class="tile-blurb"></p><span class="tile-cta">launch</span>';
    fill(tile, '.tile-title', task.title);
    fill(tile, '.tile-blurb', task.description);
    tiles.append(tile);
  }

  app.replaceChildren(node);
}

function renderTask(task: Task): void {
  const node = cloneTemplate('task-template');
  fill(node, '.task-title', task.title);

  const tip = node.querySelector('.metrics-tip') as HTMLElement;
  for (const [token, description] of METRICS_TIP) {
    const line = document.createElement('p');
    const key = document.createElement('b');
    key.textContent = token;
    line.append(key, ` — ${description}`);
    tip.append(line);
  }

  const video = node.querySelector('video') as HTMLVideoElement;
  const canvas = node.querySelector('canvas') as HTMLCanvasElement;
  const statusDot = node.querySelector('.status-dot') as HTMLElement;
  const statusText = node.querySelector('.status-text') as HTMLElement;
  const metricsValues = node.querySelector('.metrics-values') as HTMLElement;
  const startBtn = node.querySelector('.action') as HTMLButtonElement;

  const setStatus = (text: string, state: 'idle' | 'starting' | 'live' | 'error') => {
    statusText.textContent = text;
    statusDot.classList.toggle('live', state === 'live');
    statusDot.classList.toggle('error', state === 'error');
    startBtn.disabled = state === 'starting' || state === 'live';
    startBtn.setAttribute('aria-label', state === 'live' ? 'Camera running' : 'Start camera');
  };

  startBtn.addEventListener('click', () => {
    void (async () => {
      setStatus('requesting camera', 'starting');
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });
        video.srcObject = stream;
        await video.play();

        setStatus('camera live', 'live');
        const stopLoop = await task.startFrameLoop(video, canvas, (fps, ms) => {
          const fpsStr = String(fps).padStart(2, ' ');
          const msStr = ms.toFixed(1).padStart(4, ' ');
          const modelFps = Math.round(1000 / ms);
          metricsValues.textContent = `${fpsStr} FPS · ${msStr} MS (${modelFps} FPS)`;
        });
        cleanup = () => {
          stopLoop();
          stream.getTracks().forEach((track) => track.stop());
        };
      } catch (error) {
        // eslint-disable-next-line no-console
        console.error(error);
        metricsValues.textContent = '';
        setStatus('pipeline error', 'error');
      }
    })();
  });

  app.replaceChildren(node);
}

function route(): void {
  cleanup?.();
  cleanup = null;

  const id = window.location.hash.replace(/^#\/?/, '');
  const task = tasks.find((t) => t.id === id);
  if (task) {
    renderTask(task);
    document.title = `wgpu inference — ${task.title}`;
  } else {
    renderGallery();
    document.title = 'wgpu inference — demos';
  }
}

window.addEventListener('hashchange', route);
route();
