#!/usr/bin/env python3
"""
Export DepthART relative depth (MobileNetV4-S) to program.json + weights.bin
for the WebGPU inference runtime.

Pipeline:
    depthart_relative_mnv4s_448.pth (Hugging Face) -> .pte (WebGPU partitioner)
    -> program.json + weights.bin

The exported graph is the raw affine-invariant depth model: [1, 3, 448, 448]
ImageNet-normalized RGB in, [1, 1, 448, 448] relative depth out (smaller =
nearer). The MobileNetV4 variants use only standard operators, so the whole
graph lowers into the WebGPU partitioner.

Usage:
    python utils/export_depthart.py                   # default paths
    python utils/export_depthart.py -c other.pth -o out/

Setup (beyond utils/pte_to_program.py requirements):
    git clone https://github.com/xuefeng-cvr/DepthART third-party/depthart
    # checkpoint: huggingface.co/Fengxue93/DepthART, relative/..._mnv4s_448.pth
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pte_to_program as p2p  # noqa: E402 (sets up executorch sys.path)

# Default paths
DEPTHART_ROOT = REPO_ROOT / "third-party" / "depthart"
DEFAULT_CHECKPOINT = REPO_ROOT / "utils" / "build" / "depthart_relative_mnv4s_448.pth"
DEFAULT_PTE = REPO_ROOT / "utils" / "build" / "depthart_webgpu_fp32.pte"
DEFAULT_OUT = REPO_ROOT / "example" / "public" / "assets" / "depthart"

ENCODER = "MNV4-S"
IMGSZ = (448, 448)
BATCH = 1

# Deterministic input pattern shared with the runtime regression test
# (tests/program/depthart.test.ts): value[i] = (i % 251) / 250 - 0.5.
INPUT_PATTERN_MOD = 251


def make_reference_input() -> torch.Tensor:
    """Builds the deterministic regression-test input tensor."""
    n = BATCH * 3 * IMGSZ[0] * IMGSZ[1]
    vals = [((i % INPUT_PATTERN_MOD) / 250.0) - 0.5 for i in range(n)]
    return torch.tensor(vals, dtype=torch.float32).view(BATCH, 3, *IMGSZ)


def build_export_model(checkpoint: Path) -> torch.nn.Module:
    """Load the DepthART MobileNetV4-S relative-depth model (eval, fp32)."""
    relative_root = DEPTHART_ROOT / "relative"
    if not (relative_root / "models.py").is_file():
        raise FileNotFoundError(f"DepthART sources not found at {relative_root}")
    sys.path.insert(0, str(relative_root))
    from models import load_relative_model

    model, _ = load_relative_model(ENCODER, checkpoint, "cpu")
    model.eval()
    model.float()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DepthART MNV4-S to WebGPU inference program artifacts"
    )
    parser.add_argument(
        "-c", "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="depthart .pth path"
    )
    parser.add_argument("--pte", type=Path, default=DEFAULT_PTE, help="output .pte path")
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUT, help="program.json/weights.bin dir"
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        metavar="DIR",
        help="also write a reference-output.f32 ground truth (from the PyTorch model) into DIR",
    )
    args = parser.parse_args()

    from executorch.backends.webgpu.partitioner import WebGPUPartitioner
    from executorch.exir import to_edge_transform_and_lower

    model = build_export_model(args.checkpoint)

    im = torch.zeros(BATCH, 3, *IMGSZ)
    with torch.no_grad():
        out_shape = tuple(model(im).shape)
    print(f"Model output shape: {out_shape}")

    print("Exporting with torch.export...")
    exported_program = torch.export.export(model, (im,))

    print("Lowering to WebGPU via partitioner...")
    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[WebGPUPartitioner()],
    ).to_executorch()

    args.pte.parent.mkdir(parents=True, exist_ok=True)
    args.pte.write_bytes(et_program.buffer)
    print(f"  Wrote {len(et_program.buffer)} bytes -> {args.pte}")

    print(f"\nParsing {args.pte}...")
    pte_bytes = args.pte.read_bytes()
    vk_graph, raw_constant_bytes, named_data_map = p2p.extract_vk_graph(pte_bytes)

    print(f"  Operators:  {len(vk_graph.chain)}")
    print(f"  Values:     {len(vk_graph.values)}")
    print(f"  Constants:  {len(vk_graph.constants)}")
    print(f"  Inputs:     {vk_graph.input_ids}")
    print(f"  Outputs:    {vk_graph.output_ids}")

    print("\nOp histogram:")
    for op, count in Counter(op.name for op in vk_graph.chain).most_common():
        print(f"  {count:4d}  {op}")

    print("\nConverting to program.json + weights.bin...")
    program, weights_bytes = p2p.convert_vk_graph_to_program(
        vk_graph, raw_constant_bytes, named_data_map
    )
    p2p._validate_program(program, weights_bytes)
    print("  Validation passed.")

    args.output.mkdir(parents=True, exist_ok=True)
    json_path = args.output / "program.json"
    weights_path = args.output / "weights.bin"
    json_path.write_text(p2p.json.dumps(program, indent=2))
    weights_path.write_bytes(weights_bytes)
    print(f"\n  Wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(f"  Wrote {weights_path} ({len(weights_bytes)} bytes)")

    if args.reference is not None:
        args.reference.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            reference = model(make_reference_input())
        ref_path = args.reference / "reference-output.f32"
        ref_path.write_bytes(reference.contiguous().numpy().tobytes())
        print(f"  Wrote {ref_path} ({ref_path.stat().st_size} bytes, shape {tuple(reference.shape)})")


if __name__ == "__main__":
    main()
