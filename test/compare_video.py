#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple

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
        help="Use rknn-toolkit2 simulator on PC (true/false). Ignored if --rknn_backend is set.",
    )
    parser.add_argument(
        "--rknn_backend",
        type=str,
        choices=["simulator", "onnx", "device"],
        default=None,
        help=(
            "RKNN inference backend: simulator (PC, slow/inaccurate), "
            "onnx (PC, no-NMS ONNX + CPU decoder — best for visual compare), "
            "device (RK3588 NPU via rknnlite2)"
        ),
    )
    parser.add_argument(
        "--onnx_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="ONNXRuntime device",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=None,
        help="Final RKNN detection score threshold after NMS (default 0.3, matches ONNX rtmo_gpu)",
    )
    parser.add_argument(
        "--kpt_thr",
        type=float,
        default=0.1,
        help="Keypoint visibility threshold for drawing skeletons (default 0.1)",
    )
    parser.add_argument(
        "--count_csv",
        type=Path,
        default=None,
        help=(
            "Per-frame bbox count CSV (frame, onnx_dets, rknn_dets, delta). "
            "Default: <output_stem>_det_counts.csv"
        ),
    )
    return parser.parse_args()


def _rknn_display_label(backend: str) -> str:
    if backend == "onnx":
        return "RKNN (no-nms pipeline)"
    if backend == "device":
        return "RKNN (NPU)"
    return f"RKNN ({backend})"


def annotate_panel(
    frame: Any,
    bboxes,
    bbox_scores,
    keypoints,
    kpt_scores,
    label: str,
    inference_ms: float,
    kpt_thr: float = 0.1,
) -> Any:
    panel = frame.copy()
    if len(keypoints) > 0:
        panel = draw_skeleton(panel, keypoints, kpt_scores, kpt_thr=kpt_thr)
    if len(bboxes) > 0:
        panel = draw_bbox(panel, bboxes, bbox_scores)
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    line1 = f"{label} | {inference_ms:.1f} ms | {fps:.1f} FPS | dets: {len(bboxes)}"
    cv2.putText(
        panel,
        line1,
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def infer_onnx(
    model: RTMO_GPU, frame, kpt_thr: float
) -> Tuple[Any, float, float, int]:
    start = time.perf_counter()
    bboxes, bbox_scores, keypoints, kpt_scores = model(frame)
    inference_ms = (time.perf_counter() - start) * 1000.0
    panel = annotate_panel(
        frame,
        bboxes,
        bbox_scores,
        keypoints,
        kpt_scores,
        "FULL ONNX",
        inference_ms,
        kpt_thr,
    )
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    return panel, inference_ms, fps, len(bboxes)


def infer_rknn(
    model: RTMO_RKNN, frame, kpt_thr: float, panel_label: str = "RKNN"
) -> Tuple[Any, float, float, int]:
    start = time.perf_counter()
    _, bboxes, bbox_scores, keypoints, kpt_scores = model(frame)
    inference_ms = (time.perf_counter() - start) * 1000.0
    panel = annotate_panel(
        frame,
        bboxes,
        bbox_scores,
        keypoints,
        kpt_scores,
        panel_label,
        inference_ms,
        kpt_thr,
    )
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    return panel, inference_ms, fps, len(bboxes)


def write_det_count_csv(
    path: Path,
    onnx_counts: List[int],
    rknn_counts: List[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "onnx_dets", "rknn_dets", "delta"])
        for frame_idx, (onnx_n, rknn_n) in enumerate(zip(onnx_counts, rknn_counts)):
            writer.writerow([frame_idx, onnx_n, rknn_n, rknn_n - onnx_n])


def print_speed_summary(
    onnx_ms: List[float],
    rknn_ms: List[float],
    rknn_label: str,
) -> None:
    if not onnx_ms or not rknn_ms:
        return
    onnx_mean = sum(onnx_ms) / len(onnx_ms)
    rknn_mean = sum(rknn_ms) / len(rknn_ms)
    print("\n=== Inference speed (same video) ===")
    print(
        f"FULL ONNX: mean {onnx_mean:.2f} ms "
        f"({1000.0 / onnx_mean:.2f} FPS) | "
        f"min {min(onnx_ms):.2f} ms | max {max(onnx_ms):.2f} ms"
    )
    print(
        f"{rknn_label}: mean {rknn_mean:.2f} ms "
        f"({1000.0 / rknn_mean:.2f} FPS) | "
        f"min {min(rknn_ms):.2f} ms | max {max(rknn_ms):.2f} ms"
    )
    if rknn_mean > 0:
        print(f"Speed ratio (ONNX / {rknn_label}): {onnx_mean / rknn_mean:.2f}x")


def print_det_count_summary(
    onnx_counts: List[int],
    rknn_counts: List[int],
    rknn_label: str,
) -> None:
    n = len(onnx_counts)
    if n == 0:
        return

    onnx_arr = onnx_counts
    rknn_arr = rknn_counts
    match_frames = sum(1 for o, r in zip(onnx_arr, rknn_arr) if o == r)
    onnx_detect_frames = sum(1 for c in onnx_arr if c > 0)
    rknn_detect_frames = sum(1 for c in rknn_arr if c > 0)
    both_detect_frames = sum(1 for o, r in zip(onnx_arr, rknn_arr) if o > 0 and r > 0)

    print("\n=== Bbox count summary (same video) ===")
    print(f"Frames processed: {n}")
    print(f"Frames with detections: ONNX={onnx_detect_frames}, {rknn_label}={rknn_detect_frames}")
    print(f"Frames with both >0 detections: {both_detect_frames}")
    print(f"Frames with matching counts: {match_frames} ({100.0 * match_frames / n:.1f}%)")
    print(
        f"ONNX  count/frame: min={min(onnx_arr)}, max={max(onnx_arr)}, "
        f"mean={sum(onnx_arr) / n:.2f}, total={sum(onnx_arr)}"
    )
    print(
        f"{rknn_label} count/frame: min={min(rknn_arr)}, max={max(rknn_arr)}, "
        f"mean={sum(rknn_arr) / n:.2f}, total={sum(rknn_arr)}"
    )

    mismatches = [
        (i, o, r)
        for i, (o, r) in enumerate(zip(onnx_arr, rknn_arr))
        if o != r
    ]
    if mismatches:
        print(f"Mismatched frames: {len(mismatches)}")
        preview = mismatches[:10]
        print("  frame | onnx | rknn | delta")
        for frame_idx, o, r in preview:
            print(f"  {frame_idx:5d} | {o:4d} | {r:4d} | {r - o:+4d}")
        if len(mismatches) > len(preview):
            print(f"  ... and {len(mismatches) - len(preview)} more")


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
    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else 0.3
    )
    if args.rknn_backend is None:
        rknn_backend = "simulator" if args.use_simulator else "device"
    else:
        rknn_backend = args.rknn_backend
    rknn_model = RTMO_RKNN(
        model_path=str(args.rknn),
        target=args.target,
        device_id=args.device_id,
        backend=rknn_backend,
        score_threshold=score_threshold,
    )
    print(
        f"[INFO] RKNN backend={rknn_backend}, score_threshold={score_threshold}, "
        f"kpt_thr={args.kpt_thr}"
    )
    rknn_label = _rknn_display_label(rknn_backend)
    if score_threshold < 0.3:
        print(
            "[WARN] score_threshold < 0.3: RKNN will keep extra low-score boxes. "
            "Full ONNX (rtmo_gpu) filters detections with score > 0.3. "
            "Use --score_threshold 0.3 for a fair bbox-count comparison."
        )

    frame_count = 0
    onnx_det_counts: List[int] = []
    rknn_det_counts: List[int] = []
    onnx_ms_values: List[float] = []
    rknn_ms_values: List[float] = []
    count_csv = (
        args.count_csv
        if args.count_csv is not None
        else args.output.with_name(f"{args.output.stem}_det_counts.csv")
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            onnx_panel, onnx_ms, onnx_fps, onnx_n = infer_onnx(
                onnx_model, frame, args.kpt_thr
            )
            rknn_panel, rknn_ms, rknn_fps, rknn_n = infer_rknn(
                rknn_model, frame, args.kpt_thr, panel_label=rknn_label
            )
            onnx_det_counts.append(onnx_n)
            rknn_det_counts.append(rknn_n)
            onnx_ms_values.append(onnx_ms)
            rknn_ms_values.append(rknn_ms)

            if frame_count == 0:
                print(
                    f"[INFO] frame 0: FULL ONNX {onnx_ms:.1f} ms ({onnx_fps:.1f} FPS), "
                    f"dets={onnx_n} | {rknn_label} {rknn_ms:.1f} ms ({rknn_fps:.1f} FPS), "
                    f"dets={rknn_n}"
                )
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
    write_det_count_csv(count_csv, onnx_det_counts, rknn_det_counts)
    print(f"[INFO] Saved per-frame bbox counts to {count_csv}")
    print_speed_summary(onnx_ms_values, rknn_ms_values, rknn_label)
    print_det_count_summary(onnx_det_counts, rknn_det_counts, rknn_label)


if __name__ == "__main__":
    main()
