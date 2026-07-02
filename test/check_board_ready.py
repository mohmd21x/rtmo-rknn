#!/usr/bin/env python3
"""Verify RK3588 deployment artifacts before copying to the board."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_RKNN = [
    "convert/models/rtmo-s.fp16.rknn",
    "convert/models/rtmo-s.int8.rknn",
    "convert/models/rtmo-m.fp16.rknn",
    "convert/models/rtmo-m.int8.rknn",
]
EXPECTED_DECODER = [
    "convert/decoder/rtmo_s_dcc_decoder_params.yml",
    "convert/decoder/rtmo_m_dcc_decoder_params.yml",
]
CORE_FILES = [
    "rtmo_rknn.py",
    "requirements-rk3588.txt",
    "RK3588.md",
    "test/compare_video.py",
    "benchmark/benchmark.py",
]


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / (1024 * 1024):.1f} MB"


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    print("=== RK3588 board readiness check ===\n")

    for rel in CORE_FILES + EXPECTED_DECODER + EXPECTED_RKNN:
        path = ROOT / rel
        if not path.is_file():
            errors.append(f"Missing file: {rel}")
        else:
            print(f"[OK] {rel} ({_mb(path)})")

    try:
        from rtmo_rknn import DCCDecoder, RTMO_RKNN  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Cannot import rtmo_rknn: {exc}")
        DCCDecoder = None  # type: ignore
    else:
        print("[OK] rtmo_rknn imports")

    if DCCDecoder is not None:
        for rel in EXPECTED_DECODER:
            try:
                DCCDecoder(str(ROOT / rel))
                print(f"[OK] decoder loads: {rel}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Decoder failed {rel}: {exc}")

    if importlib.util.find_spec("rknnlite") is None:
        warnings.append(
            "rknnlite2 not installed (expected on PC). "
            "Install Rockchip wheel on RK3588 before --rknn_backend device."
        )
    else:
        print("[OK] rknnlite2 importable")

    if importlib.util.find_spec("cv2") is None:
        errors.append("opencv-python (cv2) not installed")
    else:
        print("[OK] opencv-python")

    if importlib.util.find_spec("numpy") is None:
        errors.append("numpy not installed")
    else:
        print("[OK] numpy")

    print()
    if warnings:
        for msg in warnings:
            print(f"[WARN] {msg}")
        print()

    if errors:
        print("FAILED:")
        for msg in errors:
            print(f"  - {msg}")
        return 1

    print("Ready for RK3588 deploy.")
    print("\nOn board:")
    print("  pip install -r requirements-rk3588.txt")
    print("  pip install rknnlite2-<board-wheel>.whl")
    print("  python test/compare_video.py --rknn_backend device --use_simulator false \\")
    print("    --rknn convert/models/rtmo-m.fp16.rknn --onnx rtmo/rtmo-m.onnx ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
