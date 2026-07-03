#!/usr/bin/env python3
import argparse
import re
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import onnx
from rknn.api import RKNN


def _parse_quantize_parameter_names(cfg_text: str) -> List[str]:
    """Top-level quantize_parameters keys (4-space indent, not 8-space fields)."""
    names: List[str] = []
    for line in cfg_text.splitlines():
        if line.startswith("        "):
            continue
        match = re.match(r"^    ([^\s].+):\s*$", line)
        if match:
            names.append(match.group(1))
    return names


def discover_hybrid_fp16_layers(cfg_text: str) -> List[str]:
    """Keep the full bbox/prior decode chain in fp16 (INT8 breaks on RK3588 NPU)."""
    names = _parse_quantize_parameter_names(cfg_text)
    start_names = ("flatten_bbox_preds-rs_tp", "flatten_bbox_preds-rs")
    start = -1
    for key in start_names:
        if key in names:
            start = names.index(key)
            break
    if start < 0:
        raise RuntimeError(
            "flatten_bbox_preds-rs(_tp) not found in hybrid quantization cfg."
        )
    try:
        end = names.index("bboxes")
    except ValueError as exc:
        raise RuntimeError("bboxes output not found in hybrid quantization cfg.") from exc
    layer_names = names[start : end + 1]
    for extra in ("priors-rs", "priors"):
        if extra in names and extra not in layer_names:
            layer_names.append(extra)
    return layer_names


def patch_hybrid_quant_cfg(cfg_text: str) -> str:
    layer_names = discover_hybrid_fp16_layers(cfg_text)
    print(f"[INFO] Hybrid fp16 layers ({len(layer_names)}): {', '.join(layer_names)}")
    custom = "custom_quantize_layers:\n" + "".join(
        f"    {name}: float16\n" for name in layer_names
    )
    return re.sub(r"custom_quantize_layers:\s*\{\}", custom, cfg_text, count=1)


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
        "--int8_mode",
        type=str,
        choices=["hybrid", "plain"],
        default="hybrid",
        help=(
            "INT8 strategy: hybrid keeps bbox/prior heads in fp16 (recommended for "
            "RK3588); plain is full INT8 (bbox accuracy often breaks on NPU)."
        ),
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
    nms_nodes = [
        node.name or "(unnamed)"
        for node in model.graph.node
        if node.op_type == "NonMaxSuppression"
    ]
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


def letterbox_image(img: np.ndarray, input_h: int, input_w: int) -> np.ndarray:
    padded = np.ones((input_h, input_w, 3), dtype=np.uint8) * 114
    ratio = min(input_h / img.shape[0], input_w / img.shape[1])
    resized = cv2.resize(
        img,
        (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded


def build_letterbox_calibration_dataset(
    sample_data_dir: Path,
    work_dir: Path,
    input_h: int,
    input_w: int,
) -> Path:
    """Write 640×640 letterboxed JPEGs + dataset.txt (matches runtime preprocess)."""
    calib_dir = work_dir / "calib_letterbox"
    calib_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_calibration_images(sample_data_dir)
    out_paths: List[Path] = []
    for src in image_paths:
        img = cv2.imread(str(src))
        if img is None:
            raise RuntimeError(f"Failed to read calibration image: {src}")
        dst = calib_dir / f"{src.stem}_{input_h}x{input_w}.jpg"
        cv2.imwrite(str(dst), letterbox_image(img, input_h, input_w))
        out_paths.append(dst)
    dataset_path = work_dir / "dataset.txt"
    dataset_path.write_text(
        "\n".join(str(p.resolve()) for p in out_paths) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Letterbox calibration dataset ({len(out_paths)} images): {dataset_path}")
    return dataset_path


def rknn_config_kwargs(target: str, keep_float_io: bool, for_int8: bool) -> dict:
    kwargs: dict = dict(
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
        target_platform=target,
    )
    if keep_float_io or for_int8:
        kwargs["output_optimize"] = False
    if for_int8:
        kwargs["quantized_algorithm"] = "mmse"
    return kwargs


def relocate_hybrid_artifacts(model_stem: str, work_dir: Path) -> Tuple[Path, Path, Path]:
    """Move hybrid step1 artifacts from CWD into work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for suffix in (".quantization.cfg", ".model", ".data"):
        name = f"{model_stem}{suffix}"
        src = Path(name)
        dst = work_dir / name
        if src.exists():
            if dst.exists():
                dst.unlink()
            shutil.move(str(src), str(dst))
        if not dst.exists():
            raise FileNotFoundError(
                f"Hybrid quantization artifact not found: {dst}. "
                "hybrid_quantization_step1 may have failed."
            )
        paths.append(dst)
    return paths[0], paths[1], paths[2]


def build_int8_hybrid(
    args: argparse.Namespace,
    input_h: int,
    input_w: int,
) -> None:
    model_stem = args.model.stem
    work_dir = args.output.parent
    dataset_path = build_letterbox_calibration_dataset(
        args.sample_data, work_dir, input_h, input_w
    )

    print(f"[INFO] INT8 hybrid quantization for {args.model.name}")
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(**rknn_config_kwargs(args.target, args.keep_float_io, for_int8=True))
        if ret != 0:
            raise RuntimeError(f"RKNN.config failed with ret={ret}")
        ret = load_onnx_static_nchw(
            rknn, args.model, args.input_name, input_h, input_w
        )
        if ret != 0:
            raise RuntimeError(f"RKNN.load_onnx failed with ret={ret}")
        ret = rknn.hybrid_quantization_step1(
            dataset=str(dataset_path),
            proposal=False,
        )
        if ret != 0:
            raise RuntimeError(f"hybrid_quantization_step1 failed with ret={ret}")
    finally:
        rknn.release()

    cfg_path, model_path, data_path = relocate_hybrid_artifacts(model_stem, work_dir)
    patched_cfg = work_dir / f"{model_stem}.hybrid.cfg"
    patched_cfg.write_text(patch_hybrid_quant_cfg(cfg_path.read_text(encoding="utf-8")))
    print(f"[INFO] Hybrid cfg (bbox fp16 layers): {patched_cfg}")

    rknn2 = RKNN(verbose=True)
    try:
        ret = rknn2.hybrid_quantization_step2(
            model_input=str(model_path),
            data_input=str(data_path),
            model_quantization_cfg=str(patched_cfg),
        )
        if ret != 0:
            raise RuntimeError(f"hybrid_quantization_step2 failed with ret={ret}")
        print(f"[INFO] Exporting RKNN model: {args.output}")
        ret = rknn2.export_rknn(str(args.output))
        if ret != 0:
            raise RuntimeError(f"RKNN.export_rknn failed with ret={ret}")
    finally:
        rknn2.release()


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

    if args.quant == "int8" and args.int8_mode == "hybrid":
        build_int8_hybrid(args, input_h, input_w)
        print("[INFO] Hybrid INT8 conversion completed successfully.")
        return

    rknn = RKNN(verbose=True)
    try:
        print(f"[INFO] Configuring RKNN for target={args.target}")
        config_kwargs = rknn_config_kwargs(
            args.target, args.keep_float_io, for_int8=(args.quant == "int8")
        )
        if config_kwargs.get("output_optimize") is False:
            print("[INFO] output_optimize=0, outputs stay float32")
        if args.quant == "int8":
            print("[INFO] INT8 plain mode (quantized_algorithm=mmse)")
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
            dataset_path = build_letterbox_calibration_dataset(
                args.sample_data, args.output.parent, input_h, input_w
            )
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
