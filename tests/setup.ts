import { create, globals } from 'webgpu';
import { Float16Array as PolyfillFloat16Array } from '@petamoriken/float16';

// Set up Dawn WebGPU globals before any source file is loaded.
// This must run before src/tensor.ts which uses GPUBufferUsage at module level.
Object.assign(globalThis, globals);

// Polyfill Float16Array for Node runtimes where it is not yet exposed by default (e.g. Node 22).
if (typeof globalThis.Float16Array === 'undefined') {
  (globalThis as Record<string, unknown>).Float16Array = PolyfillFloat16Array;
}

// Create a global navigator with WebGPU for Node.js.
// Use SwiftShader for headless environments (CI) where no GPU is available.
// navigator is read-only on globalThis in Node.js, so use defineProperty.
Object.defineProperty(globalThis, 'navigator', {
  value: { gpu: create(['--enable-unsafe-swiftshader=true']) },
  writable: true,
  configurable: true,
});
