#!/usr/bin/env python3
"""
Convert a PyTorch ExecuTorch .pte file (exported for WebGPU) into a
program.json + weights.bin pair for the WebGPU inference runtime.

Usage:
    # Convert an existing .pte
    python utils/pte_to_program.py model_webgpu.pte -o output_dir/

    # Export a demo model first, then convert it
    python utils/pte_to_program.py --export-demo -o utils/build/

    # Print program.json to stdout after conversion
    python utils/pte_to_program.py model_webgpu.pte --print-json

Setup:
    1. Clone executorch into third-party/:
       git clone --depth 1 https://github.com/pytorch/executorch.git third-party/executorch

    2. Create a venv and install Python dependencies:
       python3.12 -m venv .venv
       source .venv/bin/activate
       pip install torch flatbuffers ruamel.yaml tabulate

    3. Copy flatbuffer schemas (normally done by CMake build):
       cp third-party/executorch/schema/program.fbs    third-party/executorch/exir/_serialize/
       cp third-party/executorch/schema/scalar_type.fbs third-party/executorch/exir/_serialize/

    4. Ensure flatc is on PATH (needed for VkGraph decompilation):
       brew install flatbuffers   # macOS

Outputs:
    program.json   — operator chain, value table, memory plan
    weights.bin    — raw constant tensor data (offsets referenced by program.json)
"""

import argparse
import shutil
import json
import sys

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
ET_ROOT = REPO_ROOT / "third-party"


def _check_environment() -> None:
    """Validate that the environment is ready for conversion."""
    errors: List[str] = []

    if not (ET_ROOT / "executorch").is_dir():
        errors.append(
            f"executorch not found at {ET_ROOT / 'executorch'}\n"
            "  Run: git clone --depth 1 https://github.com/pytorch/executorch.git "
            f"{ET_ROOT / 'executorch'}"
        )

    if shutil.which("flatc") is None:
        errors.append(
            "flatc not found on PATH (needed for VkGraph flatbuffer decompilation)\n"
            "  Run: brew install flatbuffers   (macOS)\n"
            "  Run: apt install flatbuffers-compiler   (Debian/Ubuntu)"
        )

    _serialize_dir = ET_ROOT / "executorch" / "exir" / "_serialize"
    _schema_dir = ET_ROOT / "executorch" / "schema"
    for fname in ("program.fbs", "scalar_type.fbs"):
        if not (_serialize_dir / fname).is_file():
            src = _schema_dir / fname
            if src.is_file():
                # Auto-copy from schema/ (normally done by CMake build)
                shutil.copy2(src, _serialize_dir / fname)
                print(f"  Auto-copied {fname} to {_serialize_dir}")
            else:
                errors.append(f"Missing schema file: {_serialize_dir / fname}\n" f"  Expected source at: {src}")

    if errors:
        print("Environment check failed:\n", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}\n", file=sys.stderr)
        sys.exit(1)


_check_environment()
sys.path.insert(0, str(ET_ROOT))

# ---------------------------------------------------------------------------
# ExecuTorch imports (must come after sys.path setup)
# ---------------------------------------------------------------------------

from executorch.backends.vulkan.serialization.vulkan_graph_schema import (  # noqa: E402
    Bool,
    BoolList,
    Double,
    DoubleList,
    Int,
    IntList,
    Null,
    String,
    SymInt,
    ValueList,
    VkBytes,
    VkGraph,
    VkTensor,
    VkDataType,
)
from executorch.backends.vulkan.serialization.vulkan_graph_serialize import (  # noqa: E402
    VulkanDelegateHeader,
    extract_vk_flatbuffer,
    flatbuffer_to_vk_graph,
)
from executorch.exir._serialize._cord import FileBackedData  # noqa: E402
from executorch.exir._serialize._program import deserialize_pte_binary  # noqa: E402
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from executorch.exir.schema import DataLocation  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NAMED_DATA_OFFSET = 2**64 - 1  # UINT64_MAX sentinel for named-data constants

VK_DTYPE_MAP: Dict[VkDataType, str] = {
    VkDataType.BOOL: "bool",
    VkDataType.UINT8: "uint8",
    VkDataType.INT8: "int8",
    VkDataType.INT32: "int32",
    VkDataType.FLOAT16: "float16",
    VkDataType.FLOAT32: "float32",
    VkDataType.FLOAT64: "float64",
    VkDataType.INT64: "int64",
}

VK_DTYPE_BYTES: Dict[VkDataType, int] = {
    VkDataType.BOOL: 1,
    VkDataType.UINT8: 1,
    VkDataType.INT8: 1,
    VkDataType.INT32: 4,
    VkDataType.FLOAT16: 2,
    VkDataType.FLOAT32: 4,
    VkDataType.FLOAT64: 8,
    VkDataType.INT64: 8,
}


# ---------------------------------------------------------------------------
# Step 1: Parse .pte and extract VkGraph
# ---------------------------------------------------------------------------


def extract_vk_graph(
    pte_bytes: bytes,
    method: Optional[str] = None,
) -> Tuple[VkGraph, bytes, Optional[Dict[str, bytes]]]:
    """
    Parse a .pte file and extract the VkGraph delegate + constant data.

    Args:
        pte_bytes: Raw .pte file bytes.
        method: Method name to extract (e.g. "forward"). If None and the
            .pte contains exactly one method, uses it. If None and multiple
            methods exist, uses the first one with a warning.

    Returns:
        (vk_graph, raw_constant_bytes, named_data_map)

    Raises:
        RuntimeError: If no Vulkan/WebGPU delegate is found, or the
            requested method doesn't exist.
        ValueError:   If the delegate blob header is invalid.
    """
    pte = deserialize_pte_binary(pte_bytes)
    program = pte.program

    # Select the execution plan (method)
    plans = program.execution_plan
    names = [ep.name for ep in plans]

    if method is not None:
        matches = [ep for ep in plans if ep.name == method]
        if not matches:
            raise RuntimeError(f"Method '{method}' not found. Available methods: {names}")
        ep = matches[0]
    elif len(plans) == 1:
        ep = plans[0]
    else:
        print(f"  WARNING: multiple methods found: {names}", file=sys.stderr)
        print(f"  Using first method '{names[0]}'. Use --method to select.", file=sys.stderr)
        ep = plans[0]
    delegate = None
    for d in ep.delegates:
        if "vulkan" in d.id.lower() or "webgpu" in d.id.lower():
            delegate = d
            break
    if delegate is None:
        raise RuntimeError(
            "No Vulkan/WebGPU delegate found in .pte. " f"Available delegates: {[d.id for d in ep.delegates]}"
        )

    # Extract the delegate blob (VH00 + VK00 flatbuffer + raw bytes)
    ref = delegate.processed
    if ref.location == DataLocation.INLINE:
        delegate_blob = bytes(program.backend_delegate_data[ref.index].data)
    elif ref.location == DataLocation.SEGMENT:
        seg = program.segments[ref.index]
        delegate_blob = pte_bytes[seg.offset : seg.offset + seg.size]
    else:
        raise ValueError(f"Unknown DataLocation: {ref.location}")

    # Parse VH00 header to get raw constant bytes
    header = VulkanDelegateHeader.from_bytes(delegate_blob[: VulkanDelegateHeader.EXPECTED_LENGTH])
    raw_constant_bytes = delegate_blob[header.bytes_offset : header.bytes_offset + header.bytes_size]

    # Extract and deserialize the VK00 flatbuffer -> VkGraph
    flatbuffer_bytes = extract_vk_flatbuffer(delegate_blob)
    vk_graph = flatbuffer_to_vk_graph(flatbuffer_bytes)

    # Build named data map if available
    named_data_map: Optional[Dict[str, bytes]] = None
    if pte.named_data and pte.named_data.pte_data:
        named_data_map = {}
        for key, entry in pte.named_data.pte_data.items():
            buf = pte.named_data.buffers[entry.buffer_index]
            if isinstance(buf, FileBackedData):
                named_data_map[key] = buf.to_bytes()
            else:
                named_data_map[key] = bytes(buf)

    return vk_graph, raw_constant_bytes, named_data_map


# ---------------------------------------------------------------------------
# Step 2: Convert VkGraph -> program.json + weights.bin
# ---------------------------------------------------------------------------


def _find_chain_refs(value_index: int, chain: List[Dict[str, Any]]) -> str:
    """Return a human-readable description of which chain ops reference a value."""
    refs = []
    for i, op in enumerate(chain):
        if value_index in op["args"]:
            refs.append(f"chain[{i}] '{op['name']}'")
    return ", ".join(refs) if refs else "(unreferenced)"


def _extract_constant_bytes(
    const: VkBytes,
    raw_constant_bytes: bytes,
    named_data_map: Optional[Dict[str, bytes]],
) -> bytes:
    """
    Extract raw bytes for a constant from the delegate blob or named data.

    Raises:
        ValueError: If the constant source is unknown or data is missing.
    """
    if const.named_key:
        if named_data_map and const.named_key in named_data_map:
            return named_data_map[const.named_key][: const.length]
        raise ValueError(
            f"Named constant '{const.named_key}' not found in named data store. "
            f"Available keys: {list(named_data_map.keys()) if named_data_map else 'none'}"
        )
    elif const.offset == NAMED_DATA_OFFSET:
        raise ValueError(
            "Constant has UINT64_MAX offset sentinel but empty named_key. "
            "This indicates a corrupt or unsupported .pte."
        )
    else:
        end = const.offset + const.length
        if end > len(raw_constant_bytes):
            raise ValueError(
                f"Constant [{const.offset}:{end}] exceeds raw data size " f"({len(raw_constant_bytes)} bytes)"
            )
        return raw_constant_bytes[const.offset : end]


def convert_vk_graph_to_program(
    vk_graph: VkGraph,
    raw_constant_bytes: bytes,
    named_data_map: Optional[Dict[str, bytes]],
) -> Tuple[Dict[str, Any], bytes]:
    """
    Convert a VkGraph to our program.json schema + weights binary.

    Returns:
        (program_dict, weights_bytes)

    Raises:
        TypeError:  If a VkValue type is not recognized.
        ValueError: If constant data extraction fails.
    """
    # Build chain first (needed for error messages)
    chain: List[Dict[str, Any]] = []
    for op in vk_graph.chain:
        chain.append({"name": op.name, "args": list(op.args)})

    weights_chunks: List[bytes] = []
    weights_bytes_so_far = 0
    values: List[Dict[str, Any]] = []

    for i, vk_val in enumerate(vk_graph.values):
        val = vk_val.value

        # --- Null ---
        if isinstance(val, Null):
            values.append({"type": "null"})

        # --- SymInt (symbolic integer, treated as scalar) ---
        elif isinstance(val, SymInt):
            values.append({"type": "scalar", "value": val.value})

        # --- Scalar types ---
        elif isinstance(val, Int):
            values.append({"type": "scalar", "value": val.int_val})
        elif isinstance(val, Double):
            values.append({"type": "scalar", "value": val.double_val})
        elif isinstance(val, Bool):
            values.append({"type": "scalar", "value": val.bool_val})
        elif isinstance(val, String):
            values.append({"type": "scalar", "value": val.string_val})

        # --- List types ---
        elif isinstance(val, (IntList, DoubleList, BoolList)):
            values.append({"type": "list", "items": list(val.items)})
        elif isinstance(val, ValueList):
            # List of value indices (e.g. cat inputs, split outputs) — resolved
            # to Tensor[] by the runtime
            values.append({"type": "tensorlist", "items": list(val.items)})

        # --- Tensor ---
        elif isinstance(val, VkTensor):
            entry: Dict[str, Any] = {
                "type": "tensor",
                "shape": list(val.dims),
                "dtype": VK_DTYPE_MAP.get(val.datatype, f"unknown_{val.datatype}"),
            }
            if val.mem_obj_id >= 0:
                entry["memObjId"] = val.mem_obj_id

            if val.constant_id >= 0:
                const = vk_graph.constants[val.constant_id]
                const_bytes = _extract_constant_bytes(const, raw_constant_bytes, named_data_map)
                entry["weightsOffset"] = weights_bytes_so_far
                entry["weightsLength"] = len(const_bytes)
                weights_chunks.append(const_bytes)
                weights_bytes_so_far += len(const_bytes)

            values.append(entry)

        # --- Unknown type: fail hard ---
        else:
            type_name = type(val).__name__
            refs = _find_chain_refs(i, chain)
            raise TypeError(
                f"Unsupported VkValue type '{type_name}' at values[{i}].\n"
                f"  Referenced by: {refs}\n"
                f"  Add support in convert_vk_graph_to_program()."
            )

    # Compute memory plan
    pools = _compute_memory_pools(vk_graph)

    program = {
        "version": "1",
        "chain": chain,
        "values": values,
        "inputIds": list(vk_graph.input_ids),
        "outputIds": list(vk_graph.output_ids),
        "memoryPlan": {"pools": pools},
    }

    weights_bytes = b"".join(weights_chunks)
    return program, weights_bytes


def _compute_memory_pools(vk_graph: VkGraph) -> List[Dict[str, Any]]:
    """
    Compute memory pool sizes from VkTensor mem_obj_id assignments.

    Tensors with the same mem_obj_id can share a GPU buffer (non-overlapping
    lifetimes). Pool size = max tensor byte size within the group.
    """
    pool_sizes: Dict[int, int] = {}

    for vk_val in vk_graph.values:
        val = vk_val.value
        if not isinstance(val, VkTensor):
            continue
        if val.constant_id >= 0:
            continue  # Constants have dedicated buffers
        if val.mem_obj_id < 0:
            continue  # Dedicated allocation

        elem_bytes = VK_DTYPE_BYTES.get(val.datatype, 4)
        tensor_bytes = 1
        for d in val.dims:
            tensor_bytes *= d
        tensor_bytes *= elem_bytes

        pool_id = val.mem_obj_id
        pool_sizes[pool_id] = max(pool_sizes.get(pool_id, 0), tensor_bytes)

    return [{"id": pid, "size": size} for pid, size in sorted(pool_sizes.items())]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_program(program: Dict[str, Any], weights_bytes: bytes) -> None:
    """
    Validate the converted program for consistency.

    Raises:
        ValueError: If any consistency check fails.
    """
    values = program["values"]
    chain = program["chain"]

    # 1. Weights size must match sum of all weightsLength
    total_declared = sum(v.get("weightsLength", 0) for v in values if v["type"] == "tensor")
    if total_declared != len(weights_bytes):
        raise ValueError(
            f"Weights size mismatch: declared {total_declared} bytes, " f"actual {len(weights_bytes)} bytes"
        )

    # 2. All chain args that are ints must reference valid values
    for ci, op in enumerate(chain):
        for arg in op["args"]:
            if isinstance(arg, int) and (arg < 0 or arg >= len(values)):
                raise ValueError(
                    f"chain[{ci}] '{op['name']}' references values[{arg}] " f"but only {len(values)} values exist"
                )

    # 3. inputIds and outputIds must point to tensor values
    for label, ids in [("inputIds", program["inputIds"]), ("outputIds", program["outputIds"])]:
        for idx in ids:
            if idx < 0 or idx >= len(values):
                raise ValueError(f"{label} references values[{idx}] out of range")
            if values[idx]["type"] != "tensor":
                raise ValueError(
                    f"{label} references values[{idx}] of type " f"'{values[idx]['type']}', expected 'tensor'"
                )

    # 4. All memObjIds must appear in memoryPlan.pools
    pool_ids = {p["id"] for p in program["memoryPlan"]["pools"]}
    for i, v in enumerate(values):
        if v["type"] == "tensor" and "memObjId" in v:
            if v["memObjId"] not in pool_ids:
                raise ValueError(
                    f"values[{i}] has memObjId={v['memObjId']} "
                    f"but no matching pool in memoryPlan. "
                    f"Available pools: {sorted(pool_ids)}"
                )

    # 5. Weights offsets must be within bounds
    for i, v in enumerate(values):
        if v["type"] == "tensor" and "weightsOffset" in v:
            end = v["weightsOffset"] + v["weightsLength"]
            if end > len(weights_bytes):
                raise ValueError(
                    f"values[{i}] weights [{v['weightsOffset']}:{end}] "
                    f"exceeds weights.bin size ({len(weights_bytes)})"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _export_demo(output_dir: Path) -> Path:
    """Export a demo model (conv2d → relu → flatten → linear) to .pte."""

    # === DEMO EXPORT (not part of the conversion tool) ===

    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.linear = nn.Linear(16 * 8 * 8, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x

    from executorch.backends.webgpu.partitioner import WebGPUPartitioner

    model = DemoModel().eval()
    example_inputs = (torch.randn(1, 3, 8, 8),)

    print("Exporting demo model with torch.export...")
    exported_program = torch.export.export(model, example_inputs)

    print("Lowering to WebGPU via partitioner...")
    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[WebGPUPartitioner()],
    ).to_executorch()

    pte_path = output_dir / "model.pte"
    pte_path.write_bytes(et_program.buffer)
    print(f"  Wrote {len(et_program.buffer)} bytes -> {pte_path}")

    # === END DEMO EXPORT ===

    return pte_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ExecuTorch .pte (WebGPU) to program.json + weights.bin",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="path to .pte file (omit if using --export-demo)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output directory (default: same directory as input)",
    )
    parser.add_argument(
        "--export-demo",
        action="store_true",
        help="export a demo model to .pte first, then convert it",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="method name to extract (default: first/only method)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="print program.json to stdout after conversion",
    )
    args = parser.parse_args()

    if args.export_demo:
        output_dir = args.output or (REPO_ROOT / "utils" / "build")
        output_dir.mkdir(parents=True, exist_ok=True)
        pte_path = _export_demo(output_dir)
    elif args.input:
        pte_path = Path(args.input)
        if not pte_path.is_file():
            print(f"Error: {pte_path} not found", file=sys.stderr)
            sys.exit(1)
        output_dir = args.output or pte_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        parser.error("Provide an input .pte file or use --export-demo")

    json_path = output_dir / "program.json"
    weights_path = output_dir / "weights.bin"

    # Step 1: Parse .pte
    print(f"\nParsing {pte_path}...")
    pte_bytes = pte_path.read_bytes()
    vk_graph, raw_constant_bytes, named_data_map = extract_vk_graph(pte_bytes, method=args.method)

    print(f"  Operators:  {len(vk_graph.chain)}")
    print(f"  Values:     {len(vk_graph.values)}")
    print(f"  Constants:  {len(vk_graph.constants)}")
    print(f"  Inputs:     {vk_graph.input_ids}")
    print(f"  Outputs:    {vk_graph.output_ids}")

    # Step 2: Convert
    print("\nConverting to program.json + weights.bin...")
    program, weights_bytes = convert_vk_graph_to_program(vk_graph, raw_constant_bytes, named_data_map)

    # Step 3: Validate
    _validate_program(program, weights_bytes)
    print("  Validation passed.")

    # Write outputs
    json_path.write_text(json.dumps(program, indent=2))
    weights_path.write_bytes(weights_bytes)
    print(f"\n  Wrote {json_path} ({json_path.stat().st_size} bytes)")
    print(f"  Wrote {weights_path} ({len(weights_bytes)} bytes)")

    if args.print_json:
        print("\n" + json.dumps(program, indent=2))


if __name__ == "__main__":
    main()
