let cached: GPUDevice | null = null;

/**
 * Returns a shared WebGPU device.
 * The device is created once and reused across all tests.
 * Globals are set up by tests/setup.ts before any test runs.
 *
 * Requests the `shader-f16` feature (FP16 kernel variants need it); the
 * feature is required by the FP16 example models.
 */
export async function setupGPU(): Promise<GPUDevice> {
  if (cached) return cached;

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('No suitable GPU adapter found');
  }
  const requiredFeatures: GPUFeatureName[] = [];
  if (adapter.features.has('shader-f16')) {
    requiredFeatures.push('shader-f16');
  }
  cached = await adapter.requestDevice({ requiredFeatures });
  if (!cached.features.has('shader-f16')) {
    throw new Error('shader-f16 feature is required but not available on this adapter');
  }
  return cached;
}
