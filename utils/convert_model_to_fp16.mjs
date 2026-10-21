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

      const f16Slice = new Float16Array(f32Slice);
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
