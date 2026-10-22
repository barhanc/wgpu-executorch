"""Render HF's aligned 1x1 and implicit 3x3 subgroup-matrix route. Requires Jinja2 for generation."""
from pathlib import Path
import hashlib
import json
import re
import urllib.request
import jinja2

ROOT = Path(__file__).resolve().parent / 'huggingface'
REVISION = '7b19a1ea4fd2abef3203991ca933405f12cfcc62'
FILE = 'conv-1x1-subgroup-matrix.wgsl.jinja'
URL = f'https://huggingface.co/kernels/webgpu-kernels/ai.onnx.Conv/resolve/{REVISION}/build/webgpu/{FILE}'
source = urllib.request.urlopen(URL).read()
assert hashlib.sha256(source).hexdigest() == 'fccfc351a2c3936ac10a88cbc62798431421a4c7a2efd9afe22334b21c096e5f'
(ROOT / FILE).write_bytes(source)
template = jinja2.Environment(undefined=jinja2.StrictUndefined).from_string(source.decode())
variants = {}
for half in [False, True]:
    scalar = 'f16' if half else 'f32'
    for implicit in [False, True]:
        for direct in [False, True]:
            rows = 64 if not half and direct else 32
            declarations = ('enable f16;\n' if half else '') + f'''
    @group(0) @binding(0) var<storage, read_write> y: array<{scalar}>;
    @group(0) @binding(1) var<storage, read> xm: array<{scalar}>;
    @group(0) @binding(2) var<storage, read> w: array<{scalar}>;
    @group(0) @binding(3) var<storage, read> bias: array<{scalar}>;
    struct Params {{ has_bias: u32 }}
    @group(0) @binding(4) var<uniform> params: Params;
    '''
            code = template.render(env={'wgsl': {'resourceDeclarations': declarations}},
                T=scalar, usesF16=half, fScalar=scalar, hasBias=True, padded=implicit, implicitIm2col=implicit, kPadded=32,
                convKernelH=3, convKernelW=3, convStrideH=1, convStrideW=1,
                convDilationH=1, convDilationW=1, convPadTop=1, convPadLeft=1,
                convInH=1, convInW=1, convOutW=1, convInChannels=1,
                pinSubgroupSize32=True, directMatrixInputs=direct and not implicit,
                directMatrixStore=not half and direct, tileRows=rows,
                workgroupThreads=256 if rows == 64 else 128,
                subgroupRows=4 if rows == 64 else 2, M=1, K=32, N=64)
            code = code.replace('let bv = f32(bias[row_base + row]);',
                'var bv = 0.0; if (params.has_bias != 0u) { bv = f32(bias[row_base + row]); }')
            code = code.replace('raw += f32(bias[global_row]);',
                'if (params.has_bias != 0u) { raw += f32(bias[global_row]); }')
            for name in ['M', 'K', 'N', 'B_BATCH_STRIDE', 'C_BATCH_STRIDE']:
                code, count = re.subn(rf'const {name}:', f'override {name}:', code)
                assert count == 1
            if implicit:
                for name in ['K_LOOP', 'CONV_STRIDE_H', 'CONV_STRIDE_W', 'CONV_DILATION_H', 'CONV_DILATION_W', 'CONV_PAD_TOP', 'CONV_PAD_LEFT', 'CONV_IN_H', 'CONV_IN_W', 'CONV_OUT_W']:
                    code, count = re.subn(rf'const {name}:', f'override {name}:', code)
                    assert count == 1
                code = code.replace('1u * u32(CONV_IN_H)', 'CONV_IN_CHANNELS * u32(CONV_IN_H)')
                code = code.replace('override M:', 'override CONV_IN_CHANNELS: u32 = 1u;\noverride M:')
            header = f'// Adapted from {URL}\n// Apache-2.0; see LICENSE. Integration MIT, https://github.com/barhanc/wgpu-executorch\n'
            variants[f'{"3x3" if implicit else "1x1"}-{scalar}-{"direct" if direct else "shared"}'] = header + code
(ROOT / 'matrix-variants.json').write_text(json.dumps({'upstream': URL, 'revision': REVISION,
    'sha256': hashlib.sha256(source).hexdigest(), 'variants': variants}, indent=2) + '\n')
print(f'Rendered {len(variants)} subgroup-matrix variants')
