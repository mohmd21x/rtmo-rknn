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

from rtmo_rknn import RTMO_RKNN  # noqa: E402


def _import_rtmo_gpu():
    from rtmo_gpu import RTMO_GPU, draw_bbox, draw_skeleton  # noqa: E402

    return RTMO_GPU, draw_bbox, draw_skeleton


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
        "--rknn_only",
        action="store_true",
        help=(
            "Run RKNN only (required on RK3588 — full ONNX uses mmdeploy NMS ops and "
            "often segfaults on the board)"
        ),
    )
    parser.add_argument(
        "--with_onnx",
        action="store_true",
        help="Force side-by-side full ONNX on RK3588 (not recommended; may crash)",
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


def _draw_simple_bbox(panel, bboxes, bbox_scores) -> Any:
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if bbox_scores is not None and len(bbox_scores) > i:
            cv2.putText(
                panel,
                f"{float(bbox_scores[i]):.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return panel


def _draw_simple_skeleton(panel, keypoints, kpt_scores, kpt_thr: float) -> Any:
    if len(keypoints) == 0:
        return panel
    kpts = keypoints.reshape(-1, keypoints.shape[-2], keypoints.shape[-1])
    scores = kpt_scores.reshape(-1, kpt_scores.shape[-1])
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    )
    for person_kpts, person_scores in zip(kpts, scores):
        for x, y in person_kpts:
            if x > 0 or y > 0:
                cv2.circle(panel, (int(x), int(y)), 2, (0, 255, 255), -1)
        for a, b in edges:
            if person_scores[a] < kpt_thr or person_scores[b] < kpt_thr:
                continue
            pa = (int(person_kpts[a][0]), int(person_kpts[a][1]))
            pb = (int(person_kpts[b][0]), int(person_kpts[b][1]))
            cv2.line(panel, pa, pb, (255, 128, 0), 2)
    return panel


def annotate_panel(
    frame: Any,
    bboxes,
    bbox_scores,
    keypoints,
    kpt_scores,
    label: str,
    inference_ms: float,
    kpt_thr: float = 0.1,
    draw_bbox_fn=None,
    draw_skeleton_fn=None,
    simple_draw: bool = False,
) -> Any:
    panel = frame.copy()
    if simple_draw:
        if len(keypoints) > 0:
            panel = _draw_simple_skeleton(panel, keypoints, kpt_scores, kpt_thr)
        if len(bboxes) > 0:
            panel = _draw_simple_bbox(panel, bboxes, bbox_scores)
    else:
        if draw_bbox_fn is None or draw_skeleton_fn is None:
            _, draw_bbox_fn, draw_skeleton_fn = _import_rtmo_gpu()
        if len(keypoints) > 0:
            panel = draw_skeleton_fn(panel, keypoints, kpt_scores, kpt_thr=kpt_thr)
        if len(bboxes) > 0:
            panel = draw_bbox_fn(panel, bboxes, bbox_scores)
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
    model, frame, kpt_thr: float, draw_bbox_fn=None, draw_skeleton_fn=None
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
        draw_bbox_fn=draw_bbox_fn,
        draw_skeleton_fn=draw_skeleton_fn,
    )
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
    return panel, inference_ms, fps, len(bboxes)


def infer_rknn(
    model: RTMO_RKNN,
    frame,
    kpt_thr: float,
    panel_label: str = "RKNN",
    simple_draw: bool = False,
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
        simple_draw=simple_draw,
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
    rknn_only: bool = False,
) -> None:
    if not rknn_ms:
        return
    rknn_mean = sum(rknn_ms) / len(rknn_ms)
    print("\n=== Inference speed (same video) ===")
    if rknn_only:
        print(
            f"{rknn_label}: mean {rknn_mean:.2f} ms "
            f"({1000.0 / rknn_mean:.2f} FPS) | "
            f"min {min(rknn_ms):.2f} ms | max {max(rknn_ms):.2f} ms"
        )
        return
    if not onnx_ms:
        return
    onnx_mean = sum(onnx_ms) / len(onnx_ms)
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
    rknn_only: bool = False,
) -> None:
    n = len(rknn_counts)
    if n == 0:
        return

    rknn_arr = rknn_counts
    rknn_detect_frames = sum(1 for c in rknn_arr if c > 0)

    print("\n=== Bbox count summary (same video) ===")
    print(f"Frames processed: {n}")
    if rknn_only:
        print(f"Frames with detections: {rknn_label}={rknn_detect_frames}")
        print(
            f"{rknn_label} count/frame: min={min(rknn_arr)}, max={max(rknn_arr)}, "
            f"mean={sum(rknn_arr) / n:.2f}, total={sum(rknn_arr)}"
        )
        return

    onnx_arr = onnx_counts
    match_frames = sum(1 for o, r in zip(onnx_arr, rknn_arr) if o == r)
    onnx_detect_frames = sum(1 for c in onnx_arr if c > 0)
    both_detect_frames = sum(1 for o, r in zip(onnx_arr, rknn_arr) if o > 0 and r > 0)

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
    if args.rknn_backend is None:
        rknn_backend = "simulator" if args.use_simulator else "device"
    else:
        rknn_backend = args.rknn_backend

    rknn_only = args.rknn_only or (rknn_backend == "device" and not args.with_onnx)
    if rknn_backend == "device" and not args.rknn_only and not args.with_onnx:
        print(
            "[INFO] RK3588 device backend: skipping full ONNX (use --with_onnx to force "
            "side-by-side; may segfault). Running --rknn_only."
        )

    required_paths = [(args.video, "video"), (args.rknn, "rknn")]
    if not rknn_only:
        required_paths.insert(1, (args.onnx, "onnx"))
    for path_value, label in required_paths:
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

    out_width = width if rknn_only else width * 2
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {args.output}")

    onnx_model = None
    draw_bbox_fn = None
    draw_skeleton_fn = None
    if not rknn_only:
        RTMO_GPU, draw_bbox_fn, draw_skeleton_fn = _import_rtmo_gpu()
        onnx_model = RTMO_GPU(model=str(args.onnx), device=args.onnx_device)

    score_threshold = (
        args.score_threshold
        if args.score_threshold is not None
        else 0.3
    )
    rknn_model = RTMO_RKNN(
        model_path=str(args.rknn),
        target=args.target,
        device_id=args.device_id,
        backend=rknn_backend,
        score_threshold=score_threshold,
    )
    print(
        f"[INFO] RKNN backend={rknn_backend}, rknn_only={rknn_only}, "
        f"score_threshold={score_threshold}, kpt_thr={args.kpt_thr}"
    )
    rknn_label = _rknn_display_label(rknn_backend)
    if not rknn_only and score_threshold < 0.3:
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
    window_title = rknn_label if rknn_only else "ONNX vs RKNN"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if rknn_only:
                rknn_panel, rknn_ms, rknn_fps, rknn_n = infer_rknn(
                    rknn_model,
                    frame,
                    args.kpt_thr,
                    panel_label=rknn_label,
                    simple_draw=True,
                )
                rknn_det_counts.append(rknn_n)
                rknn_ms_values.append(rknn_ms)
                if frame_count == 0:
                    print(
                        f"[INFO] frame 0: {rknn_label} {rknn_ms:.1f} ms "
                        f"({rknn_fps:.1f} FPS), dets={rknn_n}"
                    )
                combined = rknn_panel
            else:
                onnx_panel, onnx_ms, onnx_fps, onnx_n = infer_onnx(
                    onnx_model,
                    frame,
                    args.kpt_thr,
                    draw_bbox_fn=draw_bbox_fn,
                    draw_skeleton_fn=draw_skeleton_fn,
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
                cv2.imshow(window_title, combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        rknn_model.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Saved {frame_count} frames to {args.output}")
    if rknn_only:
        write_det_count_csv(count_csv, [], rknn_det_counts)
    else:
        write_det_count_csv(count_csv, onnx_det_counts, rknn_det_counts)
    print(f"[INFO] Saved per-frame bbox counts to {count_csv}")
    print_speed_summary(onnx_ms_values, rknn_ms_values, rknn_label, rknn_only=rknn_only)
    print_det_count_summary(
        onnx_det_counts, rknn_det_counts, rknn_label, rknn_only=rknn_only
    )


if __name__ == "__main__":
    main()
