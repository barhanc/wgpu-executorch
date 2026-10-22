// Download published weights into an external cache; no model files enter git.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

export async function prepareRunntimeAssets() {
  const directory = process.env.WGPU_RUNNTIME_ASSETS ?? join(tmpdir(), 'wgpu-runntime-assets');
  await mkdir(directory, { recursive: true });
  // Expose native core entry points without editing any operator or shader.
  // The published zoo hides core APIs, although their compiled implementations
  // are present in its bundle. The generated sibling only adds exports.
  const dist = join(dirname(fileURLToPath(import.meta.url)), 'node_modules/runntime/dist');
  const sourceName = 'classifier-ByH6OI6h.mjs';
  const source = await readFile(join(dist, sourceName), 'utf8');
  if (
    createHash('sha256').update(source).digest('hex') !==
    '28c6bfa6a8cf86b84d84f173b36fd4ff82b3766ee29412f4ce5615ffa21d859e'
  )
    throw new Error('Unexpected ruNNtime implementation; reinstall pinned benchmark dependencies');
  const internalPath = join(dist, 'wgpu-benchmark-internals.mjs');
  const exposed = [
    'conv2d',
    'matmul',
    'tensor',
    'gpuExecutor',
    'evalValues',
    'initRunntime',
    'materialized',
    'uploadF16',
    'hwc4Meta',
    'packConvHwc4F16',
    'packDwHwc4F16',
    'f32ArrayToF16Bits',
  ];
  await writeFile(
    internalPath,
    source +
      '\n// Benchmark-only exports; original implementation above is unchanged.\nexport { ' +
      exposed.join(', ') +
      ' };\n'
  );
  const internals = {
    sourceName,
    sha256: createHash('sha256').update(source).digest('hex'),
    exposed,
    change: 'Added named exports only; implementation byte-for-byte unchanged.',
  };
  const provenance = [];
  for (const [model, subpath, sha, expectedHash] of [
    [
      'yolo26',
      'n/f16',
      '7d0ca883e0690966c49efefeea19c5d349dc451b',
      '696d40c76a64924d32931de76ff062066888ecc076fe0d91d6b106d435f04287',
    ],
    [
      'depthart',
      's/f16',
      '436a1accadf4030923e7ef4843381b78ab6be28a',
      '3179481f642c9937f601d3911e4fc85f78668bf7f98eb832b2b4e124b00b0d4d',
    ],
  ]) {
    const repository = `software-mansion/runntime-${model}`;
    const url = `https://huggingface.co/${repository}/resolve/${sha}/${subpath}/model.safetensors`;
    const path = join(directory, `${model}-${sha}.safetensors`);
    let bytes;
    try {
      bytes = await readFile(path);
    } catch {
      console.log(`Downloading ${repository} at ${sha}`);
      const weights = await fetch(url);
      if (!weights.ok) throw new Error(`Weights download: ${weights.status}`);
      bytes = Buffer.from(await weights.arrayBuffer());
      await writeFile(path, bytes);
    }
    if (createHash('sha256').update(bytes).digest('hex') !== expectedHash)
      throw new Error(`Cached weights do not match pinned checksum: ${path}`);
    provenance.push({
      model,
      repository,
      revision: sha,
      url,
      path,
      bytes: bytes.length,
      sha256: createHash('sha256').update(bytes).digest('hex'),
    });
  }
  // Public example photograph, downloaded once and decoded outside all timers.
  const imageUrl =
    'https://raw.githubusercontent.com/ultralytics/ultralytics/eb3a9dad886878ada229611763e025b7cf3b6d03/ultralytics/assets/bus.jpg';
  const imagePath = join(directory, 'bus.jpg');
  let image;
  try {
    image = await readFile(imagePath);
  } catch {
    const response = await fetch(imageUrl);
    if (!response.ok) throw new Error(`Image download: ${response.status}`);
    image = Buffer.from(await response.arrayBuffer());
    await writeFile(imagePath, image);
  }
  if (
    createHash('sha256').update(image).digest('hex') !==
    'c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63'
  )
    throw new Error('Benchmark photo checksum mismatch');
  return {
    directory,
    internals,
    provenance,
    image: {
      url: imageUrl,
      path: imagePath,
      bytes: image.length,
      sha256: createHash('sha256').update(image).digest('hex'),
    },
  };
}
