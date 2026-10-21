#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { Float16Array } from '@petamoriken/float16';

const REPO_ROOT = path.resolve(import.meta.dirname, '..');
const MODELS = ['depthart', 'selfie-segmentation', 'yolo26n'];

for (const model of MODELS) {
  const dir = path.join(REPO_ROOT, 'example/public/assets', model);
  const progPath = path.join(dir, 'program.json');
  const weightsPath = path.join(dir, 'weights.bin');

  if (!fs.existsSync(progPath) || !fs.existsSync(weightsPath)) {
    console.log(`Skipping ${model}: files not found`);
    continue;
  }

  console.log(`Converting ${model} to FP16...`);
  const program = JSON.parse(fs.readFileSync(progPath, 'utf8'));
  const oldWeightsBuf = fs.readFileSync(weightsPath);

  const weightScales = new Map();
  if (model === 'depthart') {
    // The decoder's penultimate ReLU produces values above 65504 on photographs,
    // even though the final depth fits FP16. Rebalance the adjacent convolutions:
    // ReLU(W*x+b)/16, followed by (16*V)*h+c, preserves the FP32 computation and
    // leaves headroom for FP16 activations and partial sums. Powers of two do not
    // change the significand when the scaled weights remain representable.
    const [hidden, output] = program.chain.slice(-2);
    const hiddenOut = hidden.args.at(-1);
    if (
      hidden.name !== 'et_vk.conv_with_clamp.default' ||
      output.name !== 'et_vk.conv_with_clamp.default' ||
      output.args[0] !== hiddenOut ||
      program.values[hidden.args[9]].value !== 0 ||
      program.values[hidden.args[10]].value < 1e30 ||
      program.values[output.args[1]].shape.slice(2).some((dim) => dim !== 1) ||
      program.chain.filter((op) => op.args.slice(0, -1).includes(hiddenOut)).length !== 1
    ) {
      throw new Error('DepthART decoder changed: verify FP16 range balancing before conversion');
    }
    weightScales.set(hidden.args[1], 1 / 16);
    weightScales.set(hidden.args[2], 1 / 16);
    weightScales.set(output.args[1], 16);
  }

  // We will build a new weights buffer
  const newWeightsChunks = [];
  let currentOffset = 0;

  for (let i = 0; i < program.values.length; i++) {
    const v = program.values[i];
    if (v.type !== 'tensor') continue;

    if (v.dtype === 'float32') {
      v.dtype = 'float16';
    }

    if (v.weightsOffset !== undefined && v.weightsLength !== undefined) {
      // Align offset to 4 bytes for WebGPU storage buffers
      const alignedOffset = (currentOffset + 3) & ~3;
      if (alignedOffset > currentOffset) {
        newWeightsChunks.push(Buffer.alloc(alignedOffset - currentOffset));
        currentOffset = alignedOffset;
      }

      const numel = v.weightsLength / 4;
      const f32Slice = new Float32Array(
        oldWeightsBuf.buffer,
        oldWeightsBuf.byteOffset + v.weightsOffset,
        numel
      );

      const scale = weightScales.get(i) ?? 1;
      const f16Slice = new Float16Array(
        scale === 1 ? f32Slice : Float32Array.from(f32Slice, (value) => value * scale)
      );
      const f16Buf = Buffer.from(f16Slice.buffer, f16Slice.byteOffset, f16Slice.byteLength);

      v.weightsOffset = currentOffset;
      v.weightsLength = f16Buf.byteLength;

      newWeightsChunks.push(f16Buf);
      currentOffset += f16Buf.byteLength;
    }
  }

  // Recalculate memory pools for intermediate tensors
  if (program.memoryPlan && program.memoryPlan.pools) {
    const poolSizes = new Map();
    for (const pool of program.memoryPlan.pools) {
      poolSizes.set(pool.id, 0);
    }

    const DTYPE_BYTES = {
      float16: 2,
      float32: 4,
      int32: 4,
      int8: 1,
      uint8: 1,
      bool: 1,
      float64: 8,
      int64: 8,
    };

    for (const v of program.values) {
      if (v.type === 'tensor' && v.memObjId !== undefined) {
        const numel = v.shape.reduce((a, b) => a * b, 1);
        const bytes = numel * (DTYPE_BYTES[v.dtype] || 2);
        const alignedBytes = (bytes + 3) & ~3;
        const cur = poolSizes.get(v.memObjId) || 0;
        if (alignedBytes > cur) {
          poolSizes.set(v.memObjId, alignedBytes);
        }
      }
    }

    for (const pool of program.memoryPlan.pools) {
      pool.size = poolSizes.get(pool.id) || Math.ceil(pool.size / 2);
    }
  }

  const newWeightsBuf = Buffer.concat(newWeightsChunks);
  const outProgPath = path.join(dir, 'program-fp16.json');
  const outWeightsPath = path.join(dir, 'weights-fp16.bin');

  fs.writeFileSync(outProgPath, JSON.stringify(program, null, 2));
  fs.writeFileSync(outWeightsPath, newWeightsBuf);

  console.log(
    `✓ ${model}: weights reduced from ${(oldWeightsBuf.byteLength / 1024 / 1024).toFixed(2)} MB to ${(newWeightsBuf.byteLength / 1024 / 1024).toFixed(2)} MB (50% reduction)`
  );
}

console.log('All models converted to FP16 successfully!');
