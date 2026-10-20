#!/usr/bin/env node
import { spawn, execSync } from 'node:child_process';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const REPO_ROOT = path.resolve(import.meta.dirname, '..');
const DIST_DIR = path.join(REPO_ROOT, 'dist-benchmark');
const PORT = 5190;
const DEBUG_PORT = 9222;

console.log('1. Building browser benchmark bundle...');
execSync('npx vite build benchmark', { cwd: REPO_ROOT, stdio: 'inherit' });

console.log('2. Starting local HTTP static server on port ' + PORT + '...');
const mimeTypes = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
};

const server = http.createServer((req, res) => {
  let reqPath = req.url.split('?')[0];
  if (reqPath === '/') reqPath = '/index.html';
  const filePath = path.join(DIST_DIR, reqPath);
  if (!fs.existsSync(filePath)) {
    res.writeHead(404);
    res.end('Not found');
    return;
  }
  const ext = path.extname(filePath);
  res.writeHead(200, {
    'Content-Type': mimeTypes[ext] || 'application/octet-stream',
    'Access-Control-Allow-Origin': '*',
  });
  fs.createReadStream(filePath).pipe(res);
});

await new Promise((resolve) => server.listen(PORT, resolve));

console.log('3. Launching host Chrome with WebGPU and CDP...');
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'chrome-wgpu-bench-'));
let winUserDir = tmpDir;
try {
  winUserDir = execSync(`wslpath -w "${tmpDir}"`, { encoding: 'utf8' }).trim().replace(/\\/g, '/');
} catch {}

const chromePath = '/mnt/c/Program Files/Google/Chrome/Application/chrome.exe';
const chromeArgs = [
  '--headless=new',
  `--remote-debugging-port=${DEBUG_PORT}`,
  '--enable-unsafe-webgpu',
  '--use-angle=d3d11',
  '--no-first-run',
  `--user-data-dir=${winUserDir}`,
  `http://localhost:${PORT}`,
];

const chromeProc = spawn(chromePath, chromeArgs, {
  stdio: 'ignore',
  detached: true,
});

function cleanup() {
  try {
    process.kill(-chromeProc.pid);
  } catch {
    try {
      chromeProc.kill();
    } catch {}
  }
  server.close();
  try {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  } catch {}
}

process.on('SIGINT', () => {
  cleanup();
  process.exit(1);
});
process.on('exit', cleanup);

console.log('4. Waiting for Chrome DevTools WebSocket connection...');
let wsUrl = null;
for (let attempt = 0; attempt < 30; attempt++) {
  await new Promise((r) => setTimeout(r, 500));
  try {
    const res = await fetch(`http://localhost:${DEBUG_PORT}/json`);
    const pages = await res.json();
    const target = pages.find((p) => p.type === 'page' && p.url.includes(String(PORT)));
    if (target?.webSocketDebuggerUrl) {
      wsUrl = target.webSocketDebuggerUrl;
      break;
    }
  } catch {}
}

if (!wsUrl) {
  console.error('Failed to connect to Chrome DevTools Protocol.');
  cleanup();
  process.exit(1);
}

console.log('5. Running benchmarks on real GPU...\n');

const ws = new WebSocket(wsUrl);

await new Promise((resolve, reject) => {
  ws.onopen = () => {
    ws.send(JSON.stringify({ id: 1, method: 'Runtime.enable' }));
    ws.send(JSON.stringify({ id: 2, method: 'Console.enable' }));
  };

  const timeout = setTimeout(() => {
    reject(new Error('Benchmark timed out after 60s'));
  }, 60000);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.method === 'Runtime.consoleAPICalled') {
      const text = msg.params.args.map((a) => a.value).join(' ');
      if (text.startsWith('__BENCHMARK_RESULTS__:')) {
        clearTimeout(timeout);
        const data = JSON.parse(text.slice('__BENCHMARK_RESULTS__:'.length));
        resolve(data);
      }
    }
  };

  ws.onerror = (err) => reject(err);
})
  .then((data) => {
    console.log(
      '\n========================================================================================'
    );
    console.log(
      `GPU HARDWARE: ${data.gpu.vendor.toUpperCase()} (${data.gpu.architecture}) - ${data.gpu.device}`
    );
    console.log(`NATIVE FP16:  ${data.gpu.hasF16 ? 'YES (Hardware 16-bit packed ALUs)' : 'NO'}`);
    console.log(
      '========================================================================================\n'
    );

    console.log(
      'Category'.padEnd(22) +
        'Operator / Kernel'.padEnd(36) +
        'Tensor Shape'.padEnd(28) +
        'FP32 (ms)'.padStart(10) +
        'FP16 (ms)'.padStart(10) +
        'Speedup'.padStart(10)
    );
    console.log('-'.repeat(116));

    for (const r of data.results) {
      console.log(
        r.category.padEnd(22) +
          r.name.padEnd(36) +
          r.shape.padEnd(28) +
          r.fp32Ms.toFixed(3).padStart(10) +
          (r.fp16Ms ? r.fp16Ms.toFixed(3) : 'N/A').padStart(10) +
          r.speedup.padStart(10)
      );
    }
    console.log('-'.repeat(116));
  })
  .catch((err) => {
    console.error('Benchmark error:', err);
  })
  .finally(() => {
    cleanup();
    process.exit(0);
  });
