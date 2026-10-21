/* eslint-disable no-console, @typescript-eslint/naming-convention */
import { resolve } from 'node:path';
import fs from 'node:fs';
import { defineConfig, type Plugin } from 'vite';

function benchmarkReporterPlugin(): Plugin {
  return {
    name: 'benchmark-reporter',
    configureServer(server) {
      server.middlewares.use('/api/results', (req, res) => {
        if (req.method === 'POST') {
          let body = '';
          req.on('data', (chunk) => {
            body += chunk;
          });
          req.on('end', () => {
            try {
              const data = JSON.parse(body);
              fs.writeFileSync(
                resolve(import.meta.dirname, '../benchmark-results.json'),
                JSON.stringify(data, null, 2)
              );

              // Print nicely formatted table to console
              console.log(
                '\n====================================================================================================================================='
              );
              console.log(
                `GPU HARDWARE: ${data.gpu.vendor.toUpperCase()} (${data.gpu.architecture}) - ${data.gpu.device}`
              );
              console.log(
                `NATIVE FP16:  ${data.gpu.hasF16 ? 'YES (Hardware 16-bit packed ALUs)' : 'NO'}`
              );
              console.log(
                '=====================================================================================================================================\n'
              );

              console.log(
                'Category'.padEnd(20) +
                  'Operator / Model'.padEnd(30) +
                  'Tensor Shape'.padEnd(26) +
                  'Baseline (ms)'.padStart(14) +
                  'Opt FP32 (ms)'.padStart(14) +
                  'Native FP16'.padStart(14) +
                  'Opt Speedup'.padStart(13) +
                  'FP16 Speedup'.padStart(14)
              );
              console.log('-'.repeat(145));

              for (const r of data.results) {
                console.log(
                  r.category.padEnd(20) +
                    r.name.padEnd(30) +
                    r.shape.padEnd(26) +
                    (r.baselineMs !== undefined && r.baselineMs !== null
                      ? r.baselineMs.toFixed(3)
                      : 'N/A'
                    ).padStart(14) +
                    r.fp32Ms.toFixed(3).padStart(14) +
                    (r.fp16Ms ? r.fp16Ms.toFixed(3) : 'N/A').padStart(14) +
                    (r.speedupFp32 || r.speedup || '1.00x').padStart(13) +
                    (r.speedupFp16 || 'N/A').padStart(14)
                );
              }
              console.log('-'.repeat(145));
              console.log('Saved benchmark results to benchmark-results.json\n');
            } catch (err) {
              console.error('Failed to parse benchmark results:', err);
            }
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'ok' }));
          });
        } else {
          res.writeHead(405);
          res.end();
        }
      });
    },
  };
}

export default defineConfig({
  base: './',
  publicDir: resolve(import.meta.dirname, '../example/public'),
  plugins: [benchmarkReporterPlugin()],
  server: {
    port: 5178,
    host: true,
  },
  build: {
    outDir: resolve(import.meta.dirname, '../dist-benchmark'),
    emptyOutDir: true,
  },
});
