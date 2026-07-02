#!/usr/bin/env python3
import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "rtmo") not in sys.path:
    sys.path.insert(0, str(ROOT / "rtmo"))

# coco17 is needed for keypoint names even in rknn-only mode; RTMO_GPU is imported
# lazily inside run_benchmark so that aarch64 boards without CUDA/TRT don't crash
# at import time when --rknn_only is used.
from rtmo_gpu import coco17  # noqa: E402
from rtmo_rknn import RTMO_RKNN, match_detections_by_iou  # noqa: E402


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
        description="Benchmark ONNX vs RKNN RTMO outputs and inference speed on video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "rtmo/video/Oldest video ever - 1888 [tc-L9_4jGc4].mp4",
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
        "--frames",
        type=int,
        default=200,
        help="Maximum number of frames to process (0 means full video)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "benchmark/results.csv",
        help="CSV output path",
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
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Final RKNN score threshold after NMS (default 0.3, matches ONNX; try 0.4-0.5 for INT8)",
    )
    parser.add_argument(
        "--match_iou",
        type=float,
        default=0.3,
        help="Min bbox IoU to pair ONNX/RKNN detections for keypoint comparison",
    )
    parser.add_argument(
        "--rknn_only",
        type=str2bool,
        default=False,
        help=(
            "Skip ONNX inference entirely and only benchmark RKNN speed. "
            "Use on the board when onnxruntime crashes (e.g. no CUDA/TRT libs). "
            "Keypoint accuracy columns will be omitted from the report."
        ),
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


def keypoint_names() -> List[str]:
    return [coco17["keypoint_info"][idx]["name"] for idx in sorted(coco17["keypoint_info"])]


def run_benchmark(args: argparse.Namespace) -> Dict[str, object]:
    paths_to_check = [(args.video, "video"), (args.rknn, "rknn")]
    if not args.rknn_only:
        paths_to_check.insert(1, (args.onnx, "onnx"))

    for path_value, label in paths_to_check:
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

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.frames > 0:
        total_to_process = min(total_video_frames, args.frames) if total_video_frames > 0 else args.frames
    else:
        total_to_process = total_video_frames if total_video_frames > 0 else None

    onnx_model = None
    if not args.rknn_only:
        from rtmo_gpu import RTMO_GPU  # deferred: avoid crashing on aarch64 without CUDA/TRT
        onnx_model = RTMO_GPU(model=str(args.onnx), device=args.onnx_device)

    rknn_model = RTMO_RKNN(
        model_path=str(args.rknn),
        target=args.target,
        device_id=args.device_id,
        use_simulator=args.use_simulator,
        score_threshold=args.score_threshold,
    )

    kpt_names = keypoint_names()
    num_kpts = len(kpt_names)

    l2_sum = np.zeros(num_kpts, dtype=np.float64)
    mae_sum = np.zeros(num_kpts, dtype=np.float64)
    sample_count = np.zeros(num_kpts, dtype=np.int64)

    onnx_ms_values: List[float] = []
    rknn_ms_values: List[float] = []
    processed_frames = 0
    valid_frames = 0
    matched_pairs = 0
    iou_sum = 0.0
    onnx_detect_frames = 0
    rknn_detect_frames = 0
    onnx_det_counts: List[int] = []
    rknn_det_counts: List[int] = []
    count_match_frames = 0

    progress_total = total_to_process if total_to_process is not None else 0
    progress = tqdm(total=progress_total, desc="Benchmarking", unit="frame", dynamic_ncols=True)

    try:
        while True:
            if total_to_process is not None and processed_frames >= total_to_process:
                break

            ok, frame = cap.read()
            if not ok:
                break

            if onnx_model is not None:
                onnx_start = time.perf_counter()
                onnx_bboxes, _, onnx_keypoints, _ = onnx_model(frame)
                onnx_ms = (time.perf_counter() - onnx_start) * 1000.0
                onnx_ms_values.append(onnx_ms)
            else:
                onnx_bboxes = []
                onnx_keypoints = []

            rknn_start = time.perf_counter()
            _, rknn_bboxes, _, rknn_keypoints, _ = rknn_model(frame)
            rknn_ms = (time.perf_counter() - rknn_start) * 1000.0
            rknn_ms_values.append(rknn_ms)

            rknn_people = len(rknn_bboxes)
            rknn_det_counts.append(rknn_people)

            if len(rknn_bboxes) > 0:
                rknn_detect_frames += 1

            if onnx_model is not None:
                onnx_people = len(onnx_bboxes)
                onnx_det_counts.append(onnx_people)

                if onnx_people > 0:
                    onnx_detect_frames += 1
                if onnx_people == rknn_people:
                    count_match_frames += 1

                if onnx_people > 0 and rknn_people > 0:
                    pairs = match_detections_by_iou(
                        onnx_bboxes, rknn_bboxes, iou_threshold=args.match_iou
                    )
                    if pairs:
                        valid_frames += 1
                    for onnx_idx, rknn_idx, pair_iou in pairs:
                        matched_pairs += 1
                        iou_sum += pair_iou
                        onnx_kpts = np.asarray(onnx_keypoints[onnx_idx], dtype=np.float64)
                        rknn_kpts = np.asarray(rknn_keypoints[rknn_idx], dtype=np.float64)
                        diffs = onnx_kpts - rknn_kpts
                        l2 = np.linalg.norm(diffs, axis=1)
                        mae = np.mean(np.abs(diffs), axis=1)
                        l2_sum += l2
                        mae_sum += mae
                        sample_count += 1

            processed_frames += 1
            progress.update(1)
    finally:
        progress.close()
        cap.release()
        rknn_model.release()
        cv2.destroyAllWindows()

    onnx_ms_mean = float(np.mean(onnx_ms_values)) if onnx_ms_values else 0.0
    rknn_ms_mean = float(np.mean(rknn_ms_values)) if rknn_ms_values else 0.0

    metrics_rows = []
    for idx, name in enumerate(kpt_names):
        count = int(sample_count[idx])
        if count > 0:
            mean_l2 = float(l2_sum[idx] / count)
            mean_mae = float(mae_sum[idx] / count)
        else:
            mean_l2 = float("nan")
            mean_mae = float("nan")
        metrics_rows.append((name, mean_l2, mean_mae))

    return {
        "metrics_rows": metrics_rows,
        "onnx_ms_mean": onnx_ms_mean,
        "rknn_ms_mean": rknn_ms_mean,
        "onnx_fps_mean": (1000.0 / onnx_ms_mean) if onnx_ms_mean > 0 else 0.0,
        "rknn_fps_mean": (1000.0 / rknn_ms_mean) if rknn_ms_mean > 0 else 0.0,
        "processed_frames": processed_frames,
        "valid_frames": valid_frames,
        "matched_pairs": matched_pairs,
        "mean_match_iou": (iou_sum / matched_pairs) if matched_pairs > 0 else 0.0,
        "onnx_detect_frames": onnx_detect_frames,
        "rknn_detect_frames": rknn_detect_frames,
        "onnx_det_counts": onnx_det_counts,
        "rknn_det_counts": rknn_det_counts,
        "onnx_bbox_total": int(sum(onnx_det_counts)),
        "rknn_bbox_total": int(sum(rknn_det_counts)),
        "count_match_frames": count_match_frames,
        "rknn_only": args.rknn_only,
        "use_simulator": args.use_simulator,
        "score_threshold": args.score_threshold,
    }


def write_det_count_csv(
    output_path: Path,
    onnx_counts: List[int],
    rknn_counts: List[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame", "onnx_dets", "rknn_dets", "delta"])
        for frame_idx, (onnx_n, rknn_n) in enumerate(zip(onnx_counts, rknn_counts)):
            writer.writerow([frame_idx, onnx_n, rknn_n, rknn_n - onnx_n])


def write_csv(output_path: Path, results: Dict[str, object]) -> None:
    rknn_only = results.get("rknn_only", False)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        if rknn_only:
            writer.writerow(["rknn_ms", "rknn_fps"])
            writer.writerow([results["rknn_ms_mean"], results["rknn_fps_mean"]])
        else:
            writer.writerow(["keypoint_name", "mean_l2", "mae", "onnx_ms", "rknn_ms"])
            for keypoint_name, mean_l2, mean_mae in results["metrics_rows"]:
                writer.writerow(
                    [keypoint_name, mean_l2, mean_mae, results["onnx_ms_mean"], results["rknn_ms_mean"]]
                )


def print_report(results: Dict[str, object], output_path: Path) -> None:
    rknn_only = results.get("rknn_only", False)

    print("\n=== RKNN Benchmark ===" if rknn_only else "\n=== ONNX vs RKNN Keypoint Benchmark ===")
    print(f"Frames processed: {results['processed_frames']}")

    if not rknn_only:
        print(f"Valid overlap frames: {results['valid_frames']}")
        print(
            f"Frames with detections: ONNX={results['onnx_detect_frames']}, "
            f"RKNN={results['rknn_detect_frames']}"
        )
        onnx_counts = results.get("onnx_det_counts", [])
        rknn_counts = results.get("rknn_det_counts", [])
        if onnx_counts and rknn_counts:
            n = len(onnx_counts)
            print(
                f"Bbox counts/frame: ONNX total={results['onnx_bbox_total']} "
                f"mean={results['onnx_bbox_total'] / n:.2f} "
                f"min={min(onnx_counts)} max={max(onnx_counts)}"
            )
            print(
                f"Bbox counts/frame: RKNN total={results['rknn_bbox_total']} "
                f"mean={results['rknn_bbox_total'] / n:.2f} "
                f"min={min(rknn_counts)} max={max(rknn_counts)}"
            )
            print(
                f"Frames with matching bbox counts: {results['count_match_frames']} "
                f"({100.0 * results['count_match_frames'] / n:.1f}%)"
            )
        print(
            f"Matched detection pairs: {results['matched_pairs']} "
            f"(mean IoU={results['mean_match_iou']:.3f})"
        )
        if results["valid_frames"] == 0:
            if results["rknn_detect_frames"] == 0 and results["use_simulator"]:
                print(
                    "[WARN] RKNN simulator returned 0 detections. "
                    "Try --score_threshold 0.05 on PC simulator."
                )
            elif results["onnx_detect_frames"] == 0:
                print("[WARN] ONNX returned 0 detections on all frames.")
            elif results["rknn_detect_frames"] == 0:
                print("[WARN] RKNN returned 0 detections on all frames.")
            else:
                print(
                    "[WARN] ONNX and RKNN both detected people, but never on the same frame."
                )
        print(
            f"Mean ONNX inference: {results['onnx_ms_mean']:.2f} ms "
            f"({results['onnx_fps_mean']:.2f} FPS)"
        )

    print(
        f"Mean RKNN inference: {results['rknn_ms_mean']:.2f} ms "
        f"({results['rknn_fps_mean']:.2f} FPS)"
    )
    if rknn_only:
        print(f"Frames with RKNN detections: {results['rknn_detect_frames']}")
        if results["rknn_detect_frames"] == 0:
            hint = (
                "Try --score_threshold 0.05 on PC simulator."
                if results.get("use_simulator")
                else "Try lowering --score_threshold (e.g. 0.05)."
            )
            print(f"[WARN] RKNN returned 0 detections on all frames. {hint}")

    if not rknn_only:
        table_rows = []
        for keypoint_name, mean_l2, mean_mae in results["metrics_rows"]:
            table_rows.append(
                [
                    keypoint_name,
                    f"{mean_l2:.4f}" if np.isfinite(mean_l2) else "nan",
                    f"{mean_mae:.4f}" if np.isfinite(mean_mae) else "nan",
                    f"{results['onnx_ms_mean']:.2f}",
                    f"{results['rknn_ms_mean']:.2f}",
                ]
            )
        print(
            tabulate(
                table_rows,
                headers=["keypoint_name", "mean_l2", "mae", "onnx_ms", "rknn_ms"],
                tablefmt="github",
            )
        )

    print(f"\n[INFO] CSV written to: {output_path}")


def main() -> None:
    args = parse_args()
    results = run_benchmark(args)
    write_csv(args.output, results)
    if not results.get("rknn_only") and results.get("onnx_det_counts"):
        count_csv = (
            args.count_csv
            if args.count_csv is not None
            else args.output.with_name(f"{args.output.stem}_det_counts.csv")
        )
        write_det_count_csv(
            count_csv,
            results["onnx_det_counts"],
            results["rknn_det_counts"],
        )
        print(f"[INFO] Per-frame bbox counts written to: {count_csv}")
    print_report(results, args.output)


if __name__ == "__main__":
    main()
