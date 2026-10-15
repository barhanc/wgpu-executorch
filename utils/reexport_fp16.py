#!/usr/bin/env python3
"""
Reexport a program.json + weights.bin pair in FP16 precision.

The example models are exported through the ExecuTorch WebGPU path in FP32
(see export_yolo26n.py / export_depthart.py; selfie-segmentation ships as a
converted artifact only). This tool reexports such a converted program in
FP16: every float32 tensor (weights and activations) becomes float16, the
weights.bin is rebuilt with IEEE binary16 values, and the memory pool plan is
recomputed for the halved buffers. The operator graph itself is unchanged.

It can also regenerate the FP16 regression-test ground truth for the models
whose PyTorch source is available (YOLO26-N, DepthART): the fused model is
cast to half and run on the deterministic test input, mirroring what the
GPU runtime computes (f16 storage, f32 accumulation).

Usage:
    # Convert artifacts in place
    python utils/reexport_fp16.py example/public/assets/yolo26n

    # Convert to a different directory
    python utils/reexport_fp16.py example/public/assets/depthart -o out/

    # Regenerate a test ground truth (yolo26n | depthart)
    python utils/reexport_fp16.py --reference yolo26n tests/fixtures/yolo26n

Setup (reference generation only):
    Same environment as utils/pte_to_program.py, plus the model sources:
    ultralytics for yolo26n, third-party/depthart for depthart.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bytes per element, mirroring DTYPE_BYTESIZE in src/tensor.ts
DTYPE_BYTES: Dict[str, int] = {
    "bool": 1,
    "uint8": 1,
    "int8": 1,
    "int32": 4,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int64": 8,
}


def cast_program_to_fp16(program: Dict[str, Any], weights: bytes) -> Tuple[Dict[str, Any], bytes]:
    """
    Downcast every float32 tensor of a converted program to float16.

    Constant tensors are re-encoded in weights.bin (IEEE binary16, little
    endian, round-to-nearest-even) with rebuilt offsets/lengths; activation
    tensors only change dtype; memory pools are recomputed for the halved
    element sizes.

    Returns:
        (program_dict, weights_bytes)

    Raises:
        ValueError: If the program is inconsistent or a constant is not f32 data.
    """
    values = program["values"]
    out_values = [dict(v) for v in values]

    chunks = []
    weights_so_far = 0

    for i, v in enumerate(values):
        if v["type"] != "tensor" or v["dtype"] != "float32":
            continue

        out = out_values[i]
        out["dtype"] = "float16"

        if "weightsOffset" in v:
            start, length = v["weightsOffset"], v["weightsLength"]
            end = start + length
            if end > len(weights):
                raise ValueError(f"values[{i}] weights [{start}:{end}] exceeds weights.bin size")
            if length % 4 != 0:
                raise ValueError(f"values[{i}] float32 constant length {length} not 4-byte aligned")

            data = np.frombuffer(weights[start:end], dtype="<f4")
            if not np.all(np.isfinite(data)):
                raise ValueError(f"values[{i}] constant contains non-finite float32 values")
            half = data.astype(np.float16)
            if np.any(np.isinf(half) & np.isfinite(data)):
                raise ValueError(f"values[{i}] constant overflows float16")

            chunk = half.astype("<f2").tobytes()
            out["weightsOffset"] = weights_so_far
            out["weightsLength"] = len(chunk)
            chunks.append(chunk)
            weights_so_far += len(chunk)
        elif "weightsLength" in v:
            raise ValueError(f"values[{i}] has weightsLength without weightsOffset")

    # Re-emit non-float32 constants unchanged (kept in value order)
    for i, v in enumerate(values):
        if v["type"] != "tensor" or v["dtype"] == "float32" or "weightsOffset" not in v:
            continue
        start, length = v["weightsOffset"], v["weightsLength"]
        if start + length > len(weights):
            raise ValueError(f"values[{i}] weights [{start}:{start + length}] exceeds weights.bin size")
        out_values[i]["weightsOffset"] = weights_so_far
        out_values[i]["weightsLength"] = length
        chunks.append(weights[start : start + length])
        weights_so_far += length

    new_weights = b"".join(chunks)

    # Recompute memory pools: max tensor bytes per mem_obj_id
    pool_sizes: Dict[int, int] = {}
    for v in out_values:
        if v["type"] != "tensor":
            continue
        if "weightsOffset" in v or "memObjId" not in v:
            continue
        numel = 1
        for d in v["shape"]:
            numel *= d
        tensor_bytes = numel * DTYPE_BYTES[v["dtype"]]
        pool_id = v["memObjId"]
        pool_sizes[pool_id] = max(pool_sizes.get(pool_id, 0), tensor_bytes)

    out_program = dict(program)
    out_program["values"] = out_values
    out_program["memoryPlan"] = {
        "pools": [{"id": pid, "size": size} for pid, size in sorted(pool_sizes.items())]
    }

    _validate_program(out_program, new_weights)
    return out_program, new_weights


def _validate_program(program: Dict[str, Any], weights_bytes: bytes) -> None:
    """Validate converted-program consistency (mirrors pte_to_program checks)."""
    values = program["values"]
    chain = program["chain"]

    total_declared = sum(v.get("weightsLength", 0) for v in values if v["type"] == "tensor")
    if total_declared != len(weights_bytes):
        raise ValueError(f"Weights size mismatch: declared {total_declared}, actual {len(weights_bytes)}")

    for ci, op in enumerate(chain):
        for arg in op["args"]:
            if isinstance(arg, int) and (arg < 0 or arg >= len(values)):
                raise ValueError(f"chain[{ci}] '{op['name']}' references values[{arg}] out of range")

    for label, ids in [("inputIds", program["inputIds"]), ("outputIds", program["outputIds"])]:
        for idx in ids:
            if idx < 0 or idx >= len(values):
                raise ValueError(f"{label} references values[{idx}] out of range")
            if values[idx]["type"] != "tensor":
                raise ValueError(f"{label} references values[{idx}] of type '{values[idx]['type']}'")

    pool_ids = {p["id"] for p in program["memoryPlan"]["pools"]}
    for i, v in enumerate(values):
        if v["type"] != "tensor":
            continue
        if v["dtype"] not in DTYPE_BYTES:
            raise ValueError(f"values[{i}] unknown dtype '{v['dtype']}'")
        if "memObjId" in v and v["memObjId"] not in pool_ids:
            raise ValueError(f"values[{i}] memObjId={v['memObjId']} has no matching pool")
        if "weightsOffset" in v:
            end = v["weightsOffset"] + v["weightsLength"]
            if end > len(weights_bytes):
                raise ValueError(f"values[{i}] weights [{v['weightsOffset']}:{end}] out of bounds")
            numel = 1
            for d in v["shape"]:
                numel *= d
            expected = numel * DTYPE_BYTES[v["dtype"]]
            if v["weightsLength"] != expected:
                raise ValueError(
                    f"values[{i}] weightsLength {v['weightsLength']} != {expected} "
                    f"({numel} x {v['dtype']})"
                )


def reexport_dir(src: Path, dst: Path) -> None:
    """Convert a program.json + weights.bin directory to FP16 artifacts."""
    program = json.loads((src / "program.json").read_text())
    weights = (src / "weights.bin").read_bytes()

    before = sum(1 for v in program["values"] if v["type"] == "tensor" and v["dtype"] == "float32")
    program, weights = cast_program_to_fp16(program, weights)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "program.json").write_text(json.dumps(program, indent=2))
    (dst / "weights.bin").write_bytes(weights)
    print(f"  {src} -> {dst}: {before} tensors downcast, {len(weights)} weight bytes")


# ---------------------------------------------------------------------------
# Test ground-truth generation (models with available PyTorch sources)
# ---------------------------------------------------------------------------


def write_reference(name: str, out_dir: Path) -> None:
    """
    Regenerate reference-output.f32 for the FP16 regression tests: the fused
    model is cast to half and evaluated on the deterministic test input
    (value[i] = (i % 251) / 250 - 0.5), stored as float32 bytes.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch

    if name == "yolo26n":
        import export_yolo26n as ex
    elif name == "depthart":
        import export_depthart as ex
    else:
        raise ValueError(f"unknown model '{name}' (expected yolo26n or depthart)")

    model = ex.build_export_model(ex.DEFAULT_WEIGHTS if name == "yolo26n" else ex.DEFAULT_CHECKPOINT)
    model.half()
    with torch.no_grad():
        out = model(ex.make_reference_input().half()).float()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reference-output.f32"
    path.write_bytes(out.contiguous().numpy().tobytes())
    print(f"  wrote {path} ({path.stat().st_size} bytes, shape {tuple(out.shape)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reexport program.json + weights.bin artifacts in FP16 precision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        help="directories containing program.json + weights.bin to convert",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output directory (default: convert in place; only valid with one input dir)",
    )
    parser.add_argument(
        "--reference",
        nargs=2,
        metavar=("MODEL", "OUT_DIR"),
        default=None,
        help="write the FP16 regression ground truth for MODEL (yolo26n|depthart) into OUT_DIR",
    )
    args = parser.parse_args()

    if args.reference is not None:
        if args.dirs:
            parser.error("--reference cannot be combined with input directories")
        name, out_dir = args.reference
        print(f"Generating FP16 reference for {name}...")
        write_reference(name, Path(out_dir))
        return

    if not args.dirs:
        parser.error("provide at least one model directory or use --reference")
    if args.output is not None and len(args.dirs) > 1:
        parser.error("-o/--output only works with a single input directory")

    for src in args.dirs:
        if not (src / "program.json").is_file() or not (src / "weights.bin").is_file():
            print(f"Error: {src} does not contain program.json + weights.bin", file=sys.stderr)
            sys.exit(1)
        dst = args.output if args.output is not None else src
        reexport_dir(src, dst)


if __name__ == "__main__":
    main()
