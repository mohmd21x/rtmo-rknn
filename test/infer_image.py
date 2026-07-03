#!/usr/bin/env python3
"""Run RTMO RKNN inference on a single image (RK3588 NPU or PC onnx backend)."""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtmo_rknn import RTMO_RKNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTMO RKNN single-image inference with optional trace logs."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Input image path (BGR)",
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
        default=None,
        help="Annotated output image (default: test/<image_stem>_<rknn_stem>.jpg)",
    )
    parser.add_argument(
        "--rknn_backend",
        type=str,
        choices=["device", "onnx", "simulator"],
        default="device",
        help="RKNN backend (device = RK3588 NPU)",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Detection score threshold after NMS",
    )
    parser.add_argument(
        "--nms_score_threshold",
        type=float,
        default=None,
        help="NMS score threshold (default 0.15, matches ONNX)",
    )
    parser.add_argument(
        "--kpt_thr",
        type=float,
        default=0.1,
        help="Keypoint visibility threshold for drawing",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full pipeline trace (raw outputs, scores, NMS funnel)",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show result window (needs desktop/X11)",
    )
    return parser.parse_args()


def _draw_result(
    image: np.ndarray,
    bboxes,
    bbox_scores,
    keypoints,
    kpt_scores,
    kpt_thr: float,
    label: str,
    ms: float,
) -> np.ndarray:
    panel = image.copy()
    edges = (
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    )
    if len(keypoints) > 0:
        kpts = np.asarray(keypoints).reshape(-1, 17, 2)
        scores = np.asarray(kpt_scores).reshape(-1, 17)
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
    fps = 1000.0 / ms if ms > 0 else 0.0
    cv2.putText(
        panel,
        f"{label} | {ms:.1f} ms | {fps:.1f} FPS | dets: {len(bboxes)}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def main() -> None:
    args = parse_args()
    if os.environ.get("RTMO_RKNN_INPUT", "").strip() and args.rknn_backend == "device":
        print(
            "[ERROR] Unset RTMO_RKNN_INPUT on the board (float32 segfaults). "
            "Run: unset RTMO_RKNN_INPUT"
        )
        sys.exit(1)

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.rknn.exists():
        raise FileNotFoundError(f"RKNN model not found: {args.rknn}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    out_path = args.output
    if out_path is None:
        out_path = ROOT / "test" / f"{args.image.stem}_{args.rknn.stem}.jpg"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label = f"RKNN ({args.rknn_backend})"
    print(f"[INFO] {args.rknn.name} | backend={args.rknn_backend} | {args.image.name}")
    if args.debug:
        print("[INFO] --debug: full pipeline trace enabled")

    model_kwargs = dict(
        model_path=str(args.rknn),
        backend=args.rknn_backend,
        score_threshold=args.score_threshold,
    )
    if args.nms_score_threshold is not None:
        model_kwargs["nms_score_threshold"] = args.nms_score_threshold

    model = RTMO_RKNN(**model_kwargs)
    try:
        start = time.perf_counter()
        if args.debug:
            _, bboxes, scores, keypoints, kpt_scores = model.diagnose(image)
        else:
            _, bboxes, scores, keypoints, kpt_scores = model(image)
        ms = (time.perf_counter() - start) * 1000.0
    finally:
        model.release()

    print(
        f"[OK] {ms:.1f} ms ({1000.0 / ms:.1f} FPS) | dets={len(bboxes)} | "
        f"kpts={tuple(keypoints.shape) if len(keypoints) else (0, 17, 2)}"
    )
    if len(bboxes) == 0:
        print(
            "[WARN] 0 detections. Re-run with --debug to see raw NPU scores. "
            "If max score is low, try --score_threshold 0.15 or check input image."
        )
    for i, score in enumerate(scores):
        print(f"  det {i}: score={float(score):.3f} bbox={bboxes[i].tolist()}")

    result = _draw_result(
        image, bboxes, scores, keypoints, kpt_scores, args.kpt_thr, label, ms
    )
    cv2.imwrite(str(out_path), result)
    print(f"[INFO] Saved {out_path}")

    if args.display:
        cv2.imshow("RTMO RKNN", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
