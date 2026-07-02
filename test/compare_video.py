#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "rtmo") not in sys.path:
    sys.path.insert(0, str(ROOT / "rtmo"))

from rtmo_gpu import RTMO_GPU, draw_bbox, draw_skeleton  # noqa: E402
from rtmo_rknn import RTMO_RKNN  # noqa: E402


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ONNX and RKNN RTMO inference side-by-side on video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / 'rtmo/video/Oldest video ever - 1888 [tc-L9_4jGc4].mp4',
        help="Input video path",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "rtmo/rtmo-s.onnx",
        help="ONNX model path",
    )
    parser.add_argument(
        "--rknn",
        type=Path,
        default=ROOT / "convert/models/rtmo-s.fp16.rknn",
        help="RKNN model path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "test/comparison_output.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display side-by-side result while writing output",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="rk3588",
        help="RKNN target platform (used for on-device mode)",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="RKNN device id for on-device mode",
    )
    parser.add_argument(
        "--use_simulator",
        type=str2bool,
        default=True,
        help="Use rknn-toolkit2 simulator on PC (true/false)",
    )
    parser.add_argument(
        "--onnx_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNXRuntime device",
    )
    return parser.parse_args()


def annotate_panel(
    frame: Any,
    bboxes,
    bbox_scores,
    keypoints,
    kpt_scores,
    label: str,
    fps: float,
) -> Any:
    panel = frame.copy()
    if len(keypoints) > 0:
        panel = draw_skeleton(panel, keypoints, kpt_scores)
    if len(bboxes) > 0:
        panel = draw_bbox(panel, bboxes, bbox_scores)
    cv2.putText(
        panel,
        f"{label} | FPS: {fps:.2f}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def infer_onnx(model: RTMO_GPU, frame) -> Tuple[Any, float]:
    start = time.perf_counter()
    bboxes, bbox_scores, keypoints, kpt_scores = model(frame)
    elapsed = time.perf_counter() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    panel = annotate_panel(
        frame, bboxes, bbox_scores, keypoints, kpt_scores, "ONNX", fps
    )
    return panel, fps


def infer_rknn(model: RTMO_RKNN, frame) -> Tuple[Any, float]:
    start = time.perf_counter()
    _, bboxes, bbox_scores, keypoints, kpt_scores = model(frame)
    elapsed = time.perf_counter() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    panel = annotate_panel(
        frame, bboxes, bbox_scores, keypoints, kpt_scores, "RKNN", fps
    )
    return panel, fps


def main() -> None:
    args = parse_args()
    for path_value, label in ((args.video, "video"), (args.onnx, "onnx"), (args.rknn, "rknn")):
        if not path_value.exists():
            if label == "rknn":
                raise FileNotFoundError(
                    f"{label} path not found: {path_value}\n"
                    "Create it first, for example:\n"
                    "  python convert/check_onnx_for_rknn.py --model rtmo/rtmo-s.onnx\n"
                    "  python convert/convert_rknn.py --model rtmo/rtmo-s.onnx "
                    "--output convert/models/rtmo-s.fp16.rknn --quant fp16\n"
                    "If conversion is blocked by NonMaxSuppression, use an ONNX exported without embedded NMS."
                )
            raise FileNotFoundError(f"{label} path not found: {path_value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width * 2, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output}")

    onnx_model = RTMO_GPU(model=str(args.onnx), device=args.onnx_device)
    rknn_model = RTMO_RKNN(
        model_path=str(args.rknn),
        target=args.target,
        device_id=args.device_id,
        use_simulator=args.use_simulator,
    )

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            onnx_panel, _ = infer_onnx(onnx_model, frame)
            rknn_panel, _ = infer_rknn(rknn_model, frame)
            combined = cv2.hconcat([onnx_panel, rknn_panel])

            writer.write(combined)
            frame_count += 1

            if args.display:
                cv2.imshow("ONNX vs RKNN", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        rknn_model.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Saved {frame_count} frames to {args.output}")


if __name__ == "__main__":
    main()
