#!/usr/bin/env python3
"""
Export Ultralytics YOLO26-N to program.json + weights.bin for the WebGPU inference runtime.

Pipeline:
    yolo26n.pt (ultralytics) -> .pte (WebGPU partitioner) -> program.json + weights.bin

The exported graph stops at the raw decoded predictions of shape
[batch, 4 + nc, num_anchors] (end-to-end one-to-one head, xyxy boxes in
input-pixel scale, sigmoid class scores). The top-k selection normally baked
into `Detect.postprocess` uses aten.topk/aten.gather which the WebGPU
partitioner does not support, so it is left to the consumer (the JS side of
the demo scans the 8400 anchors and thresholds the scores).

Usage:
    python utils/export_yolo26n.py                    # default paths
    python utils/export_yolo26n.py -w other.pt -o out/

Setup (beyond utils/pte_to_program.py requirements):
    pip install ultralytics
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pte_to_program as p2p  # noqa: E402 (sets up executorch sys.path)

# Default paths
DEFAULT_WEIGHTS = REPO_ROOT / "utils" / "build" / "yolo26n.pt"
DEFAULT_PTE = REPO_ROOT / "utils" / "build" / "yolo26n_webgpu_fp32.pte"
DEFAULT_OUT = REPO_ROOT / "example" / "public" / "assets" / "yolo26n"

IMGSZ = (640, 640)
BATCH = 1
MAX_DET = 300

# Deterministic input pattern shared with the runtime regression test
# (tests/program/yolo26n.test.ts): value[i] = (i % 251) / 250 - 0.5.
INPUT_PATTERN_MOD = 251


def make_reference_input() -> torch.Tensor:
    """Builds the deterministic regression-test input tensor."""
    n = BATCH * 3 * IMGSZ[0] * IMGSZ[1]
    vals = [((i % INPUT_PATTERN_MOD) / 250.0) - 0.5 for i in range(n)]
    return torch.tensor(vals, dtype=torch.float32).view(BATCH, 3, *IMGSZ)


class RawDetectHead(nn.Module):
    """Detect head wrapper returning raw decoded predictions.

    Replaces `Detect.forward`'s export path (decode -> permute -> top-k postprocess)
    with just the decode step, keeping the graph fully within WebGPU-supported ops
    and the output contiguous [batch, 4 + nc, num_anchors] with [x1, y1, x2, y2,
    class scores...] along dim 1.
    """

    def __init__(self, detect: nn.Module):
        super().__init__()
        self.detect = detect

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        detect = self.detect
        preds = detect.forward_head(feats, **detect.one2one)
        return detect._inference(preds)


def build_export_model(weights: Path) -> nn.Module:
    """Load and prepare the YOLO26-N model for export (mirrors ultralytics Exporter)."""
    from ultralytics import YOLO

    if not weights.is_file():
        weights.parent.mkdir(parents=True, exist_ok=True)
        # YOLO() auto-downloads to its cache; fetch then move into place
        auto = YOLO("yolo26n.pt")
        shutil.move(next(Path.cwd().glob("yolo26n.pt")), weights)
    yolo = YOLO(str(weights))

    model = yolo.model  # DetectionModel
    model.eval()
    model.float()
    # Select the NMS-free one-to-one head (true YOLO26 inference path): xyxy
    # output boxes and one anchor per object, so no NMS is needed downstream.
    # Must be set before fuse(), which prunes the unused one-to-many head.
    model.end2end = True
    model.fuse(imgsz=IMGSZ)  # conv+bn fusion; prunes the one-to-many head

    head = model.model[-1]
    # Mirror ultralytics engine/exporter.py model prep for Detect heads
    head.dynamic = False
    head.export = True
    head.format = "webgpu"
    head.agnostic_nms = False
    head.shape = None
    available = sum((IMGSZ[0] // s) * (IMGSZ[1] // s) for s in model.stride.tolist())
    head.max_det = min(MAX_DET, available)

    # Swap the head for a raw-output wrapper (skip top-k/gather postprocess)
    wrapper = RawDetectHead(head)
    wrapper.f = head.f  # from-layer wiring read by BaseModel._predict_once
    wrapper.i = head.i
    model.model[-1] = wrapper
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO26-N to WebGPU inference program artifacts")
    parser.add_argument("-w", "--weights", type=Path, default=DEFAULT_WEIGHTS, help="yolo26n.pt path")
    parser.add_argument("--pte", type=Path, default=DEFAULT_PTE, help="output .pte path")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT, help="program.json/weights.bin dir")
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

    model = build_export_model(args.weights)

    im = torch.zeros(BATCH, 3, *IMGSZ)
    with torch.no_grad():
        model(im)  # dry run (caches anchors for static shapes)
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
    program, weights_bytes = p2p.convert_vk_graph_to_program(vk_graph, raw_constant_bytes, named_data_map)
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
