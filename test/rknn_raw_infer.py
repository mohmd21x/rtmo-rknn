#!/usr/bin/env python3
"""Bare rknnlite load + uint8 inference (no rtmo_rknn). Run on RK3588 board."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal rknnlite inference to isolate board segfaults."
    )
    parser.add_argument(
        "--rknn",
        type=Path,
        default=ROOT / "convert/models/rtmo-m.fp16.rknn",
        help="RKNN model path",
    )
    return parser.parse_args()


def log_io(rknn, label: str) -> None:
    for method_name in ("get_sdk_version", "get_input_detail", "get_output_detail"):
        if not hasattr(rknn, method_name):
            continue
        try:
            print(f"[INFO] {label} {method_name}: {getattr(rknn, method_name)()}")
        except Exception as exc:
            print(f"[WARN] {label} {method_name}: {exc}")


def main() -> None:
    args = parse_args()
    if not args.rknn.exists():
        raise FileNotFoundError(f"RKNN model not found: {args.rknn}")

    try:
        from rknnlite.api import RKNNLite
    except ImportError as exc:
        raise ImportError("rknnlite2 required on board") from exc

    print(f"[INFO] load {args.rknn}")
    rknn = RKNNLite(verbose=False)
    ret = rknn.load_rknn(str(args.rknn))
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: ret={ret}")

    ret = rknn.init_runtime()
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: ret={ret}")

    log_io(rknn, "after init")

    inp = np.ones((1, 640, 640, 3), dtype=np.uint8) * 114
    print(f"[INFO] inference uint8 NHWC shape={inp.shape} dtype={inp.dtype}")
    outputs = rknn.inference(inputs=[inp], data_type="uint8")
    if outputs is None:
        raise RuntimeError("inference returned None")
    print(f"[OK] {len(outputs)} output tensors:")
    for idx, out in enumerate(outputs):
        print(f"  [{idx}] shape={out.shape} dtype={out.dtype}")

    rknn.release()
    print("[INFO] rknn_raw_infer passed")


if __name__ == "__main__":
    main()
