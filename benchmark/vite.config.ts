import { resolve } from 'node:path';
import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  build: {
    outDir: resolve(import.meta.dirname, '../dist-benchmark'),
    emptyOutDir: true,
  },
});
