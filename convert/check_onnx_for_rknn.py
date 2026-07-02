#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight-check ONNX graph compatibility for RKNN conversion."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("rtmo/rtmo-s.onnx"),
        help="Input ONNX model path",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="input",
        help="Expected input tensor name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")

    model = onnx.load(str(args.model))
    input_names = [node.name for node in model.graph.input]
    if args.input_name not in input_names:
        raise ValueError(
            f"Input '{args.input_name}' not found. Available inputs: {input_names}"
        )

    input_node = next(node for node in model.graph.input if node.name == args.input_name)
    dims = [dim.dim_param or dim.dim_value for dim in input_node.type.tensor_type.shape.dim]
    nms_nodes = [node.name or "(unnamed)" for node in model.graph.node if node.op_type == "NonMaxSuppression"]

    print(f"[INFO] model={args.model}")
    print(f"[INFO] input={args.input_name} shape={dims}")
    print(f"[INFO] NonMaxSuppression nodes: {len(nms_nodes)}")
    if nms_nodes:
        print(f"[WARN] NMS node names: {', '.join(nms_nodes)}")
        print(
            "[WARN] This model is likely to fail in RKNN build if NMS depends on runtime tensors.\n"
            "       Prefer ONNX exported without postprocess NMS and run NMS in Python/C++."
        )
    else:
        print("[INFO] No NMS node found in graph (better for RKNN conversion).")


if __name__ == "__main__":
    main()
