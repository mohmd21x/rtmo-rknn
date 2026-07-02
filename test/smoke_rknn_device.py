#!/usr/bin/env python3
"""Minimal RKNN NPU smoke test on RK3588 (no ONNX)."""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtmo_rknn import RTMO_RKNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test RKNN on RK3588 NPU.")
    parser.add_argument(
        "--rknn",
        type=Path,
        default=ROOT / "convert/models/rtmo-s.fp16.rknn",
        help="RKNN model path",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT / "rtmo/video/danial-fall-fast-2-1.mp4",
        help="Video frame source (first frame only)",
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.3,
        help="Detection score threshold",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to run (0 = full video)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print RKNN preprocess/inference/postprocess stages",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import os

    if os.environ.get("RTMO_RKNN_INPUT", "").strip():
        print(
            "[ERROR] Unset RTMO_RKNN_INPUT on the board. "
            "float32 input segfaults rknnlite; use default uint8:\n"
            "  python3 test/smoke_rknn_device.py --rknn convert/models/rtmo-m.fp16.rknn --frames 1"
        )
        sys.exit(1)
    if args.debug:
        os.environ["RTMO_RKNN_DEBUG"] = "1"
    if not args.rknn.exists():
        raise FileNotFoundError(f"RKNN model not found: {args.rknn}")
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    print(f"[INFO] Loading {args.rknn.name} on NPU...")
    model = RTMO_RKNN(
        model_path=str(args.rknn),
        backend="device",
        score_threshold=args.score_threshold,
    )

    max_frames = args.frames if args.frames > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        for frame_idx in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            _, bboxes, scores, keypoints, kpt_scores = model(frame)
            ms = (time.perf_counter() - start) * 1000.0
            print(
                f"[OK] frame {frame_idx}: {ms:.1f} ms, "
                f"dets={len(bboxes)}, kpts shape={keypoints.shape}"
            )
    finally:
        cap.release()
        model.release()

    print("[INFO] RKNN device smoke test passed.")


if __name__ == "__main__":
    main()
