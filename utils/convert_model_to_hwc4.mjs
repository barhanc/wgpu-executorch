#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { Float16Array } from '@petamoriken/float16';
import { packConvHwc4F16, packDwHwc4F16, packConvHwc4F32, packDwHwc4F32 } from './pack_hwc4.ts';

const REPO_ROOT = path.resolve(import.meta.dirname, '..');
const MODELS = ['depthart'];

for (const model of MODELS) {
  const dir = path.join(REPO_ROOT, 'example/public/assets', model);

  // Convert both FP16 and FP32
  for (const isF16 of [true, false]) {
    const suffix = isF16 ? '-fp16' : '';
    const progPath = path.join(dir, `program${suffix}.json`);
    const weightsPath = path.join(dir, `weights${suffix}.bin`);

    if (!fs.existsSync(progPath) || !fs.existsSync(weightsPath)) {
      continue;
    }

    console.log(`Converting ${model} (${isF16 ? 'FP16' : 'FP32'}) to HWC4...`);
    const program = JSON.parse(fs.readFileSync(progPath, 'utf8'));
    const oldWeightsBuf = fs.readFileSync(weightsPath);

    // Map tensor IDs to their usages
    const opByWeightId = new Map();
    for (const op of program.chain) {
      if (op.name === 'et_vk.conv_with_clamp.default' || op.name === 'aten.convolution.default') {
        opByWeightId.set(op.args[1], op);
      }
    }

    const newWeightsChunks = [];
    let currentOffset = 0;

    // Pack weights
    for (let i = 0; i < program.values.length; i++) {
      const v = program.values[i];
      if (v.type !== 'tensor') continue;

      if (v.weightsOffset !== undefined && v.weightsLength !== undefined) {
        const alignedOffset = (currentOffset + 3) & ~3;
        if (alignedOffset > currentOffset) {
          newWeightsChunks.push(Buffer.alloc(alignedOffset - currentOffset));
          currentOffset = alignedOffset;
        }

        const oldOffset = v.weightsOffset;
        const oldLen = v.weightsLength;
        const bytesPerElem = isF16 ? 2 : 4;
        const numel = oldLen / bytesPerElem;

        const slice = isF16
          ? new Float16Array(oldWeightsBuf.buffer, oldWeightsBuf.byteOffset + oldOffset, numel)
          : new Float32Array(oldWeightsBuf.buffer, oldWeightsBuf.byteOffset + oldOffset, numel);

        const op = opByWeightId.get(i);
        if (op && v.shape.length === 4) {
          const [oc, ic, kh, kw] = v.shape;
          const groups = program.values[op.args[8]]?.value || 1;
          const isDw = groups === oc && ic === 1;

          let packed;
          if (isF16) {
            packed = isDw
              ? packDwHwc4F16(slice, oc, kh, kw)
              : packConvHwc4F16(slice, oc, ic, kh, kw);
          } else {
            packed = isDw
              ? packDwHwc4F32(slice, oc, kh, kw)
              : packConvHwc4F32(slice, oc, ic, kh, kw);
          }

          const outBuf = Buffer.from(packed.buffer, packed.byteOffset, packed.byteLength);
          v.shape = [packed.length];
          v.weightsOffset = currentOffset;
          v.weightsLength = outBuf.byteLength;
          newWeightsChunks.push(outBuf);
          currentOffset += outBuf.byteLength;
        } else {
          // Bias or non-conv constant: if length < 4, pad to 4 elements for WebGPU buffer binding limits
          let outBuf;
          if (v.shape.length === 1 && v.shape[0] < 4) {
            const padded = new slice.constructor(4);
            padded.set(slice);
            outBuf = Buffer.from(padded.buffer, padded.byteOffset, padded.byteLength);
            v.shape = [4];
          } else {
            outBuf = Buffer.from(
              oldWeightsBuf.buffer,
              oldWeightsBuf.byteOffset + oldOffset,
              oldLen
            );
          }
          v.weightsOffset = currentOffset;
          v.weightsLength = outBuf.byteLength;
          newWeightsChunks.push(outBuf);
          currentOffset += outBuf.byteLength;
        }
      }
    }

    // Convert intermediate tensor shapes to [B, H, W, C]
    const origInputId = program.inputIds[0];
    const origOutputId = program.outputIds[0];

    for (let i = 0; i < program.values.length; i++) {
      const v = program.values[i];
      if (v.type === 'tensor' && v.shape.length === 4 && i !== origInputId && i !== origOutputId) {
        if (v.weightsOffset === undefined) {
          // Intermediate activation tensor: [B, C, H, W] -> [B, H, W, C]
          const [b, c, h, w] = v.shape;
          v.shape = [b, h, w, c];
        }
      }
    }

    // Wrap with to_hwc4 at input and to_chw at output
    const inTensor = program.values[origInputId];
    const outTensor = program.values[origOutputId];

    const [inB, inC, inH, inW] = inTensor.shape;
    const hwc4InShape = [inB, inH, inW, Math.ceil(inC / 4) * 4];
    const hwc4InId = program.values.length;
    program.values.push({
      type: 'tensor',
      shape: hwc4InShape,
      dtype: inTensor.dtype,
    });

    // Replace first op input arg
    program.chain[0].args[0] = hwc4InId;

    // Insert to_hwc4 op at start
    program.chain.unshift({
      name: 'et_vk.to_hwc4.default',
      args: [origInputId, hwc4InId],
    });

    // Last op output: replace with hwc4 intermediate tensor
    const lastOp = program.chain[program.chain.length - 1];
    const [outB, outC, outH, outW] = outTensor.shape;
    const hwc4OutShape = [outB, outH, outW, Math.ceil(outC / 4) * 4];
    const hwc4OutId = program.values.length;
    program.values.push({
      type: 'tensor',
      shape: hwc4OutShape,
      dtype: outTensor.dtype,
    });

    const lastOutArgIdx = lastOp.args.length - 1;
    lastOp.args[lastOutArgIdx] = hwc4OutId;

    // Append to_chw op at end
    program.chain.push({
      name: 'et_vk.to_chw.default',
      args: [hwc4OutId, origOutputId],
    });

    // Update operator names
    for (const op of program.chain) {
      if (op.name === 'et_vk.conv_with_clamp.default') {
        op.name = 'et_vk.conv_with_clamp_hwc4.default';
      } else if (op.name === 'aten.convolution.default') {
        op.name = 'et_vk.convolution_hwc4.default';
      } else if (op.name === 'aten.upsample_bilinear2d.vec') {
        op.name = 'aten.upsample_bilinear2d_hwc4.vec';
      }
    }

    const newWeightsBuf = Buffer.concat(newWeightsChunks);
    const outProgPath = path.join(dir, `program-hwc4${suffix}.json`);
    const outWeightsPath = path.join(dir, `weights-hwc4${suffix}.bin`);

    fs.writeFileSync(outProgPath, JSON.stringify(program, null, 2));
    fs.writeFileSync(outWeightsPath, newWeightsBuf);

    console.log(
      `✓ ${model} (${isF16 ? 'FP16' : 'FP32'}) HWC4: saved ${outProgPath} and ${outWeightsPath}`
    );
  }
}
