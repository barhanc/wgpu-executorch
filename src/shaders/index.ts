// Elementwise arithmetic & activations
import { add } from './add';
import { hardswish } from './hardswish';
import { mul } from './mul';
import { relu } from './relu';
import { sigmoid } from './sigmoid';
import { sub } from './sub';

// Linear algebra & reductions
import { bmm } from './bmm';
import { meanDim } from './mean-dim';
import { mm } from './mm';
import { softmax } from './softmax';

// Convolutions & pooling
import { convWithClamp } from './conv-with-clamp';
import { convolution, convolutionSilu } from './convolution';
import { maxPool2d } from './max-pool-2d';

// Resampling
import { upsampleBilinear2d } from './upsample-bilinear2d';
import { upsampleNearest2d } from './upsample-nearest-2d';

// Shape & data movement
import { cat } from './cat';
import { clone } from './clone';
import { constantPadNd } from './constant-pad-nd';
import { permuteCopy } from './permute';
import { prepack } from './prepack';
import { splitWithSizesCopy } from './split-with-sizes';
import { squeezeCopy } from './squeeze-copy';
import { unsqueezeCopy } from './unsqueeze-copy';
import { viewCopy } from './view-copy';

/**
 * Namespace containing all registered WebGPU compute shaders.
 */
// prettier-ignore
export const shaders = {
  // Elementwise arithmetic & activations
  add, sub, mul, sigmoid, relu, hardswish,
  // Linear algebra & reductions
  mm, bmm, meanDim, softmax,
  // Convolutions & pooling
  convolution, convolutionSilu, convWithClamp, maxPool2d,
  // Resampling
  upsampleBilinear2d, upsampleNearest2d,
  // Shape & data movement
  cat, constantPadNd, splitWithSizesCopy, permuteCopy, viewCopy,
  clone, squeezeCopy, unsqueezeCopy, prepack,
} as const;

/**
 * Maps ExecuTorch operator names to their shader implementations.
 *
 * @internal
 */
export const shaderRegistry = Object.fromEntries(
  Object.values(shaders).map((shader) => [shader.name, shader])
);
