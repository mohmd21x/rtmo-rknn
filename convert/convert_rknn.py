#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import onnx
from rknn.api import RKNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RTMO ONNX model to RKNN")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("convert/no_nms_onnx/rtmo-s-no-nms.onnx"),
        help="Input ONNX model path (should be a no-NMS model from export_no_nms.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("convert/models/rtmo-s.fp16.rknn"),
        help="Output RKNN model path",
    )
    parser.add_argument(
        "--quant",
        type=str,
        choices=["fp16", "int8"],
        default="fp16",
        help="Conversion mode: fp16 or int8 quantization",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="rk3588",
        help="Rockchip target platform",
    )
    parser.add_argument(
        "--sample_data",
        type=Path,
        default=Path("sample-data"),
        help="Directory containing calibration images for INT8",
    )
    parser.add_argument(
        "--input_name",
        type=str,
        default="input",
        help="ONNX input tensor name",
    )
    parser.add_argument(
        "--input_hw",
        type=str,
        default="640,640",
        help="Static input size as H,W (e.g. 640,640)",
    )
    parser.add_argument(
        "--keep_float_io",
        action="store_true",
        default=False,
        help=(
            "Keep model outputs as float32 (sets output_optimize=0 in rknn.config). "
            "Prevents the compiler changing output dtypes from float32 to int8. "
            "Useful when the board runtime auto-dequantization causes issues."
        ),
    )
    return parser.parse_args()


def parse_input_hw(value: str) -> List[int]:
    try:
        h_str, w_str = [v.strip() for v in value.split(",")]
        h = int(h_str)
        w = int(w_str)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid --input_hw '{value}'. Expected format H,W such as 640,640."
        ) from exc
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid --input_hw '{value}'. H and W must be > 0.")
    return [h, w]


def inspect_onnx(model_path: Path, input_name: str) -> Tuple[List[str], List[object]]:
    model = onnx.load(str(model_path))
    nms_nodes = [node.name or "(unnamed)" for node in model.graph.node if node.op_type == "NonMaxSuppression"]
    input_nodes = [node for node in model.graph.input if node.name == input_name]
    if not input_nodes:
        available_inputs = [node.name for node in model.graph.input]
        raise ValueError(
            f"Input '{input_name}' not found in ONNX graph. Available inputs: {available_inputs}"
        )
    return nms_nodes, input_nodes[0].type.tensor_type.shape.dim


def load_onnx_static_nchw(
    rknn: RKNN, model_path: Path, input_name: str, input_h: int, input_w: int
) -> int:
    print("[INFO] ONNX load attempt: fixed static NCHW [1,3,H,W]")
    return rknn.load_onnx(
        model=str(model_path),
        inputs=[input_name],
        input_size_list=[[1, 3, input_h, input_w]],
    )


def collect_calibration_images(sample_data_dir: Path, limit: int = 4) -> List[Path]:
    if not sample_data_dir.exists():
        raise FileNotFoundError(f"Sample data directory not found: {sample_data_dir}")

    image_files = sorted(
        [
            p
            for p in sample_data_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    if len(image_files) < limit:
        raise ValueError(
            f"INT8 quantization needs at least {limit} images, found {len(image_files)} "
            f"in {sample_data_dir}"
        )

    return image_files[:limit]


def write_dataset_txt(dataset_path: Path, image_paths: List[Path]) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p.resolve()) for p in image_paths]
    dataset_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_h, input_w = parse_input_hw(args.input_hw)

    if not args.model.exists():
        raise FileNotFoundError(f"ONNX model not found: {args.model}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    nms_nodes, input_dims = inspect_onnx(args.model, args.input_name)
    dim_debug = [dim.dim_param or dim.dim_value for dim in input_dims]
    print(f"[INFO] ONNX input '{args.input_name}' dims: {dim_debug}")
    if nms_nodes:
        nms_list = ", ".join(nms_nodes)
        raise RuntimeError(
            "Detected NonMaxSuppression node(s) in ONNX graph: "
            f"{nms_list}\n"
            "RKNN build fails on dynamic NMS graphs.\n"
            "Run convert/export_no_nms.py first to produce an NMS-free ONNX, "
            "then convert that file with this script."
        )

    rknn = RKNN(verbose=True)
    try:
        print(f"[INFO] Configuring RKNN for target={args.target}")
        config_kwargs: dict = dict(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target,
        )
        if args.keep_float_io:
            # output_optimize=0 prevents the compiler from changing output dtypes
            # from float32 to int8 for "performance". Input stays uint8 NHWC (raw
            # image) because mean/std normalization is baked into the RKNN model.
            config_kwargs["output_optimize"] = False
            print("[INFO] --keep_float_io: output_optimize=0, outputs will stay float32")
        ret = rknn.config(**config_kwargs)
        if ret != 0:
            raise RuntimeError(f"RKNN.config failed with ret={ret}")

        print(
            "[INFO] Loading ONNX model: "
            f"{args.model} (input={args.input_name}, target-shape=1x3x{input_h}x{input_w})"
        )
        ret = load_onnx_static_nchw(
            rknn=rknn,
            model_path=args.model,
            input_name=args.input_name,
            input_h=input_h,
            input_w=input_w,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN.load_onnx failed with ret={ret}")
        print("[INFO] ONNX loaded using mode: fixed static NCHW [1,3,H,W]")

        if args.quant == "int8":
            dataset_path = args.output.parent / "dataset.txt"
            image_paths = collect_calibration_images(args.sample_data, limit=4)
            write_dataset_txt(dataset_path, image_paths)
            print(f"[INFO] Wrote INT8 dataset file: {dataset_path}")
            ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        else:
            ret = rknn.build(do_quantization=False)

        if ret != 0:
            raise RuntimeError(f"RKNN.build failed with ret={ret}")

        print(f"[INFO] Exporting RKNN model: {args.output}")
        ret = rknn.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError(f"RKNN.export_rknn failed with ret={ret}")

        print("[INFO] Conversion completed successfully.")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
