// Usage: node utils/benchmark/profile-live.mjs [--controlled | --narrow | --hf | --runntime] [--no-timestamps] [results.json]
// Narrow comparison: WGPU_KERNEL_BASELINE=<commit> npm run benchmark:narrow
// Starts its own local Vite server and dedicated headless Chrome profile. It
// never attaches to, closes, or changes an existing browser process/profile.
import { execFileSync, spawn } from 'node:child_process';
import { access, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { profileLivePipeline } from './profile-live.browser.mjs';
import { profilePipelineStages } from './profile-stages.browser.mjs';
import { profileNarrowConvolution } from './profile-narrow.browser.mjs';
import { profileHuggingFace } from './profile-hf.browser.mjs';
import { profileRunntime } from './profile-runntime.browser.mjs';
import { prepareRunntimeAssets } from './runntime/setup.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const controlled = process.argv.includes('--controlled');
const narrow = process.argv.includes('--narrow');
const hf = process.argv.includes('--hf');
const runntime = process.argv.includes('--runntime');
let runntimeAssets;
if (runntime) {
  try {
    await access(join(repo, 'utils/benchmark/runntime/node_modules/runntime/package.json'));
  } catch {
    throw new Error(
      'Install isolated benchmark dependencies: npm ci --prefix utils/benchmark/runntime'
    );
  }
  runntimeAssets = await prepareRunntimeAssets();
}
const baseline =
  process.env.WGPU_KERNEL_BASELINE ??
  execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repo, encoding: 'utf8' }).trim();
let baselineRepo;
const timestamps = !process.argv.includes('--no-timestamps');
if (!controlled && !timestamps) throw new Error('--no-timestamps requires --controlled');
const outputPath = process.argv.slice(2).find((arg) => !arg.startsWith('--'));
const output = resolve(
  outputPath ??
    join(
      repo,
      `utils/benchmark/${runntime ? 'runntime' : hf ? 'hf' : narrow ? 'narrow' : controlled ? 'stage' : 'pipeline'}-results.json`
    )
);
const candidates = process.env.WGPU_CHROME
  ? [process.env.WGPU_CHROME]
  : process.platform === 'darwin'
    ? ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
    : process.platform === 'win32'
      ? [
          join(process.env.PROGRAMFILES ?? '', 'Google/Chrome/Application/chrome.exe'),
          join(process.env.LOCALAPPDATA ?? '', 'Google/Chrome/Application/chrome.exe'),
        ]
      : ['/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/chromium-browser'];
let executable;
for (const candidate of candidates) {
  try {
    await access(candidate);
    executable = candidate;
    break;
  } catch {}
}
if (!executable) throw new Error('Chrome not found; set WGPU_CHROME to its executable path');
const profile = await mkdtemp(join(tmpdir(), 'wgpu-pipeline-chrome-'));
let server;
let browser;
let ws;
const report = {
  baseline,
  method: runntime
    ? 'Published ruNNtime 0.1.0 vs our unchanged graphs; paired pixels-to-CPU-results and separate GPU compute times. See per-case model equivalence and numerical validation.'
    : hf
      ? 'Pinned Hugging Face Conv adaptation against archived baseline; see per-run method and numerical validation.'
      : narrow
        ? 'Compare raw 16-row/32-row convolutions and unchanged full model graphs against archived source. See per-run timing details.'
        : controlled
          ? 'Single-page controlled stage ablation; shuffled paired rounds, same device and model inputs. When enabled, GPU timestamp readback occurs outside wall timer. Additional sustained ABBA blocks compare continuous submissions and video-paced presentation.'
          : 'Instrument actual example loops with CPU clocks and GPU pass timestamps. Synthetic video; no camera permissions. Warmup frames excluded. Configurations run sequentially; model clock variation prevents interpreting total frame differences as presentation speedups.',
  warning: runntime
    ? 'DepthART backbones differ (ours MNV4-S, ruNNtime MobileMamba-S); DepthART is a workload comparison, not an isolated runtime benchmark. YOLO26-N uses the same task/variant and input size.'
    : hf
      ? 'Results are from one adapter/browser. Each variant is paired within a run; do not interpret between-run clock changes as kernel speedups. These adapters force selected templates and do not implement the full HF routing policy.'
      : 'Do not attribute frame-minus-model time entirely to pre/post shaders: CPU handling, GPU copies, presentation, queue/browser scheduling and callback delivery also contribute. Forced detections include a small diagnostic setup dispatch in postprocessing.',
  results: [],
};
try {
  if (narrow || hf) {
    baselineRepo = await mkdtemp(join(tmpdir(), 'wgpu-narrow-baseline-'));
    const archive = execFileSync('git', ['archive', baseline, 'src'], { cwd: repo });
    execFileSync('tar', ['-x', '-C', baselineRepo], { input: archive });
  }
  server = await createServer({
    root: join(repo, 'example'),
    ...(runntime
      ? {
          optimizeDeps: {
            entries: [join(repo, 'utils/benchmark/runntime/entry.mjs')],
            include: [
              join(
                repo,
                'utils/benchmark/runntime/node_modules/runntime/dist/wgpu-benchmark-internals.mjs'
              ),
            ],
          },
        }
      : {}),
    configFile: join(repo, 'example/vite.config.ts'),
    server: {
      host: '127.0.0.1',
      port: 0,
      strictPort: true,
      fs: {
        allow: [
          repo,
          ...(baselineRepo ? [baselineRepo] : []),
          ...(runntimeAssets ? [runntimeAssets.directory] : []),
        ],
      },
    },
    plugins: [
      {
        name: 'live-profile-results',
        configureServer(server) {
          server.middlewares.use('/api/results', async (request, response) => {
            if (request.method !== 'POST') {
              response.writeHead(405).end();
              return;
            }
            try {
              let body = '';
              for await (const chunk of request) body += chunk;
              report.results.push(JSON.parse(body));
              await writeFile(output, JSON.stringify(report, null, 2) + '\n');
              response.writeHead(200).end('ok');
            } catch (error) {
              response.writeHead(500).end(String(error));
            }
          });
        },
      },
    ],
  });
  await server.listen();
  const url = server.resolvedUrls.local[0];
  browser = spawn(
    executable,
    [
      '--headless=new',
      `--user-data-dir=${profile}`,
      '--remote-debugging-port=0',
      '--enable-unsafe-webgpu',
      '--disable-background-networking',
      '--no-first-run',
      '--no-default-browser-check',
      'about:blank',
    ],
    { stdio: ['ignore', 'ignore', 'pipe'] }
  );
  let browserErrors = '';
  browser.stderr.on('data', (data) => {
    browserErrors = (browserErrors + data).slice(-4000);
  });
  let port;
  for (let attempt = 0; attempt < 400; attempt++) {
    if (browser.exitCode !== null) throw new Error(`Owned Chrome exited: ${browserErrors}`);
    try {
      port = (await readFile(join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0];
      break;
    } catch {}
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  if (!port) throw new Error(`Chrome did not start: ${browserErrors}`);
  const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
  const page = targets.find((target) => target.type === 'page');
  ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener('open', resolve, { once: true });
    ws.addEventListener('error', reject, { once: true });
  });
  let sequence = 0;
  const pending = new Map();
  ws.addEventListener('message', (event) => {
    const response = JSON.parse(event.data);
    if (response.id) {
      pending.get(response.id)?.(response);
      pending.delete(response.id);
    }
  });
  const call = (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++sequence;
      pending.set(id, (response) =>
        response.error
          ? reject(new Error(JSON.stringify(response.error)))
          : resolve(response.result)
      );
      ws.send(JSON.stringify({ id, method, params }));
    });
  await call('Runtime.enable');
  const configurations = [];
  for (const task of narrow || hf ? ['yolo26-detection'] : ['depth-estimation', 'yolo26-detection'])
    for (const precision of runntime ? ['float16'] : ['float32', 'float16'])
      configurations.push({ task, precision, presentation: 'current' });
  if (!controlled && !narrow && !hf && !runntime)
    configurations.push({
      task: 'yolo26-detection',
      precision: 'float16',
      presentation: 'current',
      detections: 64,
    });
  if (!controlled && !narrow && !hf && !runntime)
    for (const task of ['depth-estimation', 'yolo26-detection'])
      configurations.push({ task, precision: 'float16', presentation: 'preferred' });
  for (const config of configurations) {
    await call('Page.navigate', { url });
    await new Promise((resolve) => setTimeout(resolve, 1000));
    const options = {
      runntimeAssets,
      reportEndpoint: '/api/results',
      baselineRepo: baselineRepo?.replaceAll('\\', '/'),
      ...config,
      repo: repo.replaceAll('\\', '/'),
      warmups: runntime
        ? Number(process.env.WGPU_BENCH_WARMUPS ?? 12)
        : hf
          ? 8
          : controlled
            ? 8
            : 30,
      samples: runntime
        ? Number(process.env.WGPU_BENCH_SAMPLES ?? 61)
        : hf
          ? 51
          : controlled
            ? 61
            : 80,
      timestamps,
    };
    const response = await call('Runtime.evaluate', {
      expression: `(${(runntime ? profileRunntime : hf ? profileHuggingFace : narrow ? profileNarrowConvolution : controlled ? profilePipelineStages : profileLivePipeline).toString()})(${JSON.stringify(options)})`,
      awaitPromise: true,
      returnByValue: true,
    });
    if (response.exceptionDetails)
      throw new Error(
        response.exceptionDetails.exception?.description ??
          JSON.stringify(response.exceptionDetails)
      );
    const result = response.result.value;
    console.log(
      JSON.stringify({
        results: runntime
          ? result.results.map(
              ({ name, wallMs, gpuComputeMs, runntimeOverOursWall, runntimeOverOursGpu }) => ({
                name,
                wallMs: wallMs.map((x) => x.median),
                gpuComputeMs: gpuComputeMs.map((x) => x.median),
                runntimeOverOursWall,
                runntimeOverOursGpu,
              })
            )
          : result.results?.map(({ samples, ...summary }) => summary),
        task: result.task,
        precision: result.precision,
        presentation: result.presentation,
        detections: result.detections,
        medians: result.medians,
        pairedDeltas: result.pairedDeltas,
        sustained: result.sustained?.map(({ name, medians }) => ({ name, medians })),
      })
    );
  }
  console.log(`Saved ${output}`);
} finally {
  ws?.close();
  if (browser && browser.exitCode === null) {
    const exited = new Promise((resolve) => browser.once('exit', resolve));
    browser.kill('SIGTERM'); // Only the process spawned above, with its unique profile.
    await Promise.race([exited, new Promise((resolve) => setTimeout(resolve, 5000))]);
  }
  await server?.close();
  if (baselineRepo) await rm(baselineRepo, { recursive: true, force: true });
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
