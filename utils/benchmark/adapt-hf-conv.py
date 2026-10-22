"""Render pinned Hugging Face Conv templates for the opt-in comparison.

Requires Jinja2 (generation only). Run: python utils/benchmark/adapt-hf-conv.py
The original WGSL arithmetic, staging loops and FP32 accumulation are retained.
Only resource declarations, optional-bias guarding and geometry constants adapt
it to this runtime's existing NCHW buffers. No HF loader or per-frame downloads.
"""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import jinja2

ROOT = Path(__file__).resolve().parent / 'huggingface'
REVISION = '7b19a1ea4fd2abef3203991ca933405f12cfcc62'
FILE = 'conv-1x1-gemm-tiled-reg.wgsl.jinja'
URL = f'https://huggingface.co/kernels/webgpu-kernels/ai.onnx.Conv/resolve/{REVISION}/build/webgpu/{FILE}'
ROOT.mkdir(exist_ok=True)
source = urllib.request.urlopen(URL).read()
assert hashlib.sha256(source).hexdigest() == '7eacb50256ee91107af84b0fe0d4fc22c05b880616b7113ba3ccfced037b9c72'
(ROOT / FILE).write_bytes(source)
(ROOT / 'metadata.json').write_bytes(urllib.request.urlopen(URL.rsplit('/', 1)[0] + '/metadata.json').read())
(ROOT / 'LICENSE').write_bytes(urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt').read())
(ROOT / 'NOTICE').write_text(f'''Hugging Face WebGPU Conv adaptation
Upstream: webgpu-kernels/ai.onnx.Conv
Revision: {REVISION}
Source: {URL}
Upstream shader: Apache-2.0 (see LICENSE and upstream metadata.json).
Integration code: MIT, https://github.com/barhanc/wgpu-executorch
Changes: bind existing buffers; guard optional bias via uniform; expose shape
geometry as pipeline constants. Shader load/compute/store loops are retained.
Generated variants use published 64x64/32x64, 8x4-per-thread register layouts,
K=16/32 and carried implicit gathering (IMPLICIT_TILED_GATHER_MODE=1), no split-K or activation/residual fusion. This is not the entire HF
loader or route-selection policy. The separately rendered subgroup-matrix
adaptation is from conv-1x1-subgroup-matrix.wgsl.jinja at the same revision:
aligned 1x1 and implicit 3x3, f32 accumulation, 8x8 hardware matrices,
32/64-row output tiles, 32-lane subgroups; no split-K or fused activation.
Original WGSL templates are included unmodified; metadata JSON is reformatted. Generated derivatives
change bindings, optional-bias guarding and literal shapes into pipeline constants.
Manifest: https://huggingface.co/kernels/webgpu-kernels/ai.onnx.Conv/resolve/{REVISION}/build/webgpu/manifest.json
''')
template = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(source.decode())
variants = {}
for half in [False, True]:
    scalar = 'f16' if half else 'f32'
    declarations = ('enable f16;\n' if half else '') + f'''
@group(0) @binding(0) var<storage, read_write> y: array<{scalar}>;
@group(0) @binding(1) var<storage, read> xm: array<{scalar}>;
@group(0) @binding(2) var<storage, read> w: array<{scalar}>;
@group(0) @binding(3) var<storage, read> bias: array<{scalar}>;
struct Params {{ M: u32, K: u32, N: u32, has_bias: u32 }}
@group(0) @binding(4) var<uniform> params: Params;
'''
    for implicit in [False, True]:
        for rows in ([64, 32] if implicit else [64]):
            for reduction in [16, 32]:
                code = template.render(env={'wgsl': {'resourceDeclarations': declarations}},
                    T=scalar, usesF16=half, hasBias=True, implicitIm2col=implicit,
                    tunables={'IMPLICIT_TILED_GATHER_MODE': 1}, gemmKTile=reduction, gemmMTile=rows, gemmThreadRows=8,
                    convKernelH=3, convKernelW=3, convStrideH=1, convStrideW=1,
                    convDilationH=1, convDilationW=1, convPadTop=1, convPadLeft=1,
                    convInH=1, convInW=1, convOutW=1, convInChannels=1)
                assert code.count('v = v + f32(bias[m]);') == 1
                code = code.replace('v = v + f32(bias[m]);',
                    'if (params.has_bias != 0u) { v = v + f32(bias[m]); }')
                if implicit:
                    for name in ['STRIDE_H', 'STRIDE_W', 'DILATION_H', 'DILATION_W',
                                 'PAD_TOP', 'PAD_LEFT', 'IN_H', 'IN_W', 'OUT_W', 'IN_CHANNELS']:
                        code, count = re.subn(rf'const CONV_{name}:', f'override CONV_{name}:', code)
                        assert count == 1
                header = f'// Adapted from {URL}\n// Upstream Apache-2.0; integration MIT (https://github.com/barhanc/wgpu-executorch).\n'
                variants[f'{"3x3" if implicit else "1x1"}-{scalar}-k{reduction}-m{rows}'] = header + code
(ROOT / 'conv-variants.json').write_text(json.dumps({'upstream': URL, 'revision': REVISION,
    'sha256': hashlib.sha256(source).hexdigest(), 'variants': variants}, indent=2) + '\n')
print(f'Rendered {len(variants)} variants from {REVISION}')
