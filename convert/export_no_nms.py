#!/usr/bin/env python3
"""
Cut rtmo-s / rtmo-m ONNX graphs before the NMS node and write:
  1. <out_dir>/rtmo-{s|m}-no-nms.onnx        — 5-output NMS-free model
  2. <out_dir>/../decoder/rtmo_{s|m}_dcc_decoder_params.yml — GAU decoder weights

Usage (defaults shown):
  python convert/export_no_nms.py --model rtmo/rtmo-s.onnx
  python convert/export_no_nms.py --model rtmo/rtmo-m.onnx
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import onnx.checker
import onnx.shape_inference
import onnx.utils
from onnx import numpy_helper


# ---------------------------------------------------------------------------
# Pre-NMS cut points  (verified by shape-inference on both model files)
# Output shapes after cut, for a 640×640 input:
#   bboxes        (1, 2000, 4)   NMS boxes tensor (score-sorted anchor order)
#   scores        (1, 1, 2000)   NMS scores tensor (matches bboxes)
#   pose_vecs     (1, 2000, C)   raw anchor order — reorder with sort_indices
#   kpt_vis       (1, 2000, 17)  raw anchor order — reorder with sort_indices
#   priors        (1, 2000, 2)   raw anchor order — not used for CPU DCC decode
#   sort_indices  (1, 2000)      topk_inds: raw anchor index for each sorted slot
#
# NOTE: Do NOT cut at y.3 / onnx::Transpose_* — those are misaligned with each
# other.  Use the same tensors fed to NonMaxSuppression (boxes, scores).
# ---------------------------------------------------------------------------
CUT_TENSORS: Dict[str, Dict[str, str]] = {
    "rtmo-s": {
        "bboxes":        "boxes",
        "scores":        "scores",
        "pose_vecs":     "onnx::Shape_1089",
        "kpt_vis":       "onnx::Shape_1117",
        "priors":        "onnx::Add_1125",
        "sort_indices":  "topk_inds",
    },
    "rtmo-m": {
        "bboxes":        "boxes",
        "scores":        "scores",
        "pose_vecs":     "onnx::Shape_1276",
        "kpt_vis":       "onnx::Shape_1304",
        "priors":        "onnx::Add_1312",
        "sort_indices":  "topk_inds",
    },
}

# ---------------------------------------------------------------------------
# DCC decoder initializer names per model variant
#
# Notation:
#   KEY starting with "_" → private helper; not written directly to YAML.
#   The key names without "_" match the YAML field names expected by
#   PoseEstimator.cpp / PoseEstimatorRKNN.cpp.
#
# Weight shapes before any transform (from ONNX graph):
#   pose_to_kpts_weight  rtmo-s (256,2176)  rtmo-m (384,2176)   → transpose
#   gau_uv_weight        both   (128, 640)                       → transpose
#   gau_o_weight         both   (256, 128)                       → transpose
#   x_fc_weight          both   (128, 128)                       → transpose
#   y_fc_weight          both   (128, 128)                       → transpose
# ---------------------------------------------------------------------------
DCC_INIT: Dict[str, Dict[str, str]] = {
    "rtmo-s": {
        "pose_to_kpts_weight": "onnx::MatMul_1868",
        "pose_to_kpts_bias":   "head.dcc.pose_to_kpts.bias",
        "x_fc_weight":         "onnx::MatMul_1876",
        "x_fc_bias":           "head.dcc.x_fc.bias",
        "y_fc_weight":         "onnx::MatMul_1877",
        "y_fc_bias":           "head.dcc.y_fc.bias",
        "gau_uv_weight":       "onnx::MatMul_1874",
        "gau_o_weight":        "onnx::MatMul_1875",
        "gau_ln_g":            "head.dcc.gau.ln.g",
        "gau_res_scale":       "head.dcc.gau.res_scale.scale",
        # gamma / beta helpers (assembled into (2,S) matrices)
        "_gamma_u":            "onnx::Mul_1503",     # (1,1,1,128) → q-gate scale
        "_gamma_v":            "onnx::Mul_1507",     # (1,1,1,128) → k-gate scale
        "_beta_u":             "onnx::Add_1505",     # (1,1,17,128) → q per-kpt bias
        "_beta_v":             "onnx::Add_1509",     # (1,1,17,128) → k per-kpt bias
        # bin and SPE helpers
        "x_bins":              "onnx::Mul_1531",     # (1,1,192)
        "y_bins":              "onnx::Mul_1544",     # (1,1,256)
        "spe_dim_t":           "onnx::Div_1553",     # (1,1,1,64)
        "flatten_priors_640":  "onnx::Gather_1118",  # (2000,2)
        # scalar hyper-params embedded as ONNX constants
        "_sqrt_s":             "onnx::Div_1513",     # scalar ≈ 11.3137
        "_ln_scale":           "onnx::Mul_1487",     # scalar ≈ 0.08839
        "_ln_eps":             "onnx::Clip_1873",    # scalar ≈ 1e-5
    },
    "rtmo-m": {
        "pose_to_kpts_weight": "onnx::MatMul_2103",
        "pose_to_kpts_bias":   "head.dcc.pose_to_kpts.bias",
        "x_fc_weight":         "onnx::MatMul_2111",
        "x_fc_bias":           "head.dcc.x_fc.bias",
        "y_fc_weight":         "onnx::MatMul_2112",
        "y_fc_bias":           "head.dcc.y_fc.bias",
        "gau_uv_weight":       "onnx::MatMul_2109",
        "gau_o_weight":        "onnx::MatMul_2110",
        "gau_ln_g":            "head.dcc.gau.ln.g",
        "gau_res_scale":       "head.dcc.gau.res_scale.scale",
        "_gamma_u":            "onnx::Mul_1690",
        "_gamma_v":            "onnx::Mul_1694",
        "_beta_u":             "onnx::Add_1692",
        "_beta_v":             "onnx::Add_1696",
        "x_bins":              "onnx::Mul_1718",
        "y_bins":              "onnx::Mul_1731",
        "spe_dim_t":           "onnx::Div_1740",
        "flatten_priors_640":  "onnx::Gather_1305",
        "_sqrt_s":             "onnx::Div_1700",
        "_ln_scale":           "onnx::Mul_1674",
        "_ln_eps":             "onnx::Clip_2108",
    },
}

# ---------------------------------------------------------------------------
# OpenCV-YAML writer (no cv2 dependency)
# ---------------------------------------------------------------------------
_VALUES_PER_LINE = 8


def _fmt_float(v: float) -> str:
    """Format a float32 value the way OpenCV does: enough digits, no trailing zeros."""
    s = f"{v:.10g}"
    # Ensure at least one decimal digit so the value parses unambiguously as a
    # float.  Guard against special strings (nan, inf, -inf) which must not
    # have a trailing dot.
    if "." not in s and "e" not in s and s not in ("nan", "inf", "-inf"):
        s += "."
    return s


def _write_mat(lines: List[str], key: str, arr: np.ndarray) -> None:
    """Append an opencv-matrix block for `arr` (1-D or 2-D, float32)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        rows, cols = arr.size, 1
    elif arr.ndim == 2:
        rows, cols = arr.shape
    else:
        raise ValueError(f"_write_mat: only 1-D / 2-D arrays supported, got shape {arr.shape}")
    flat = arr.reshape(-1)
    lines.append(f"{key}: !!opencv-matrix")
    lines.append(f"   rows: {rows}")
    lines.append(f"   cols: {cols}")
    lines.append("   dt: f")
    vals = [_fmt_float(v) for v in flat]
    # Group into lines of _VALUES_PER_LINE
    chunks: List[str] = []
    for i in range(0, len(vals), _VALUES_PER_LINE):
        chunks.append(", ".join(vals[i : i + _VALUES_PER_LINE]))
    if chunks:
        lines.append("   data: [ " + (",\n       ".join(chunks)) + " ]")
    else:
        lines.append("   data: [  ]")


def _write_empty_mat(lines: List[str], key: str) -> None:
    """Write an empty opencv-matrix (rows=0, cols=1) for zero-bias vectors."""
    lines.append(f"{key}: !!opencv-matrix")
    lines.append("   rows: 0")
    lines.append("   cols: 1")
    lines.append("   dt: f")
    lines.append("   data: [  ]")


# ---------------------------------------------------------------------------
# Decoder YAML extraction
# ---------------------------------------------------------------------------

def _get_init(inits: Dict[str, np.ndarray], name: str, context: str) -> np.ndarray:
    if name not in inits:
        raise KeyError(f"[{context}] ONNX initializer '{name}' not found.")
    return inits[name].astype(np.float32)


def extract_decoder_yaml(
    model_path: Path,
    variant: str,
    out_path: Path,
) -> None:
    """Read DCC weights from model_path and write the decoder-params YAML."""
    model = onnx.load(str(model_path))
    inits: Dict[str, np.ndarray] = {
        i.name: numpy_helper.to_array(i).astype(np.float32)
        for i in model.graph.initializer
    }

    dm = DCC_INIT[variant]

    def g(key: str) -> np.ndarray:
        return _get_init(inits, dm[key], f"{variant}/{key}")

    # ── scalar hyper-params ──────────────────────────────────────────────
    sqrt_s   = float(g("_sqrt_s").flat[0])
    ln_scale = float(g("_ln_scale").flat[0])
    ln_eps   = float(g("_ln_eps").flat[0])
    feat_channels = 128   # architecture constant for all RTMO variants
    num_keypoints = 17
    # ONNX stores MatMul weight B in (in, out) order: shape = (in_channels, K*feat).
    # shape[0] therefore gives in_channels directly (256 for -s, 384 for -m).
    in_channels = g("pose_to_kpts_weight").shape[0]

    # ── bins (reshape to column vectors) ────────────────────────────────
    x_bins = g("x_bins").reshape(-1)       # (192,)
    y_bins = g("y_bins").reshape(-1)       # (256,)
    spe_dim_t = g("spe_dim_t").reshape(-1) # (64,)
    num_bins_x = int(x_bins.size)
    num_bins_y = int(y_bins.size)

    # ── flatten_priors ───────────────────────────────────────────────────
    flatten_priors = g("flatten_priors_640")  # (2000, 2)

    # ── weights that need transposing ────────────────────────────────────
    # ONNX stores MatMul weight as (out, in) which is the transpose of the
    # mathematical weight matrix W (in, out). The C++ / Python decoders
    # use the convention: out = in @ W, so W should be (in, out).
    pose_to_kpts_w = g("pose_to_kpts_weight").T   # (256,2176)→(2176,256) or (384,…)
    gau_uv_w       = g("gau_uv_weight").T          # (128,640)→(640,128)
    gau_o_w        = g("gau_o_weight").T            # (256,128)→(128,256)

    # ── weights that need no transposing ─────────────────────────────────
    pose_to_kpts_b = g("pose_to_kpts_bias").reshape(-1, 1)  # (2176,1)
    x_fc_w         = g("x_fc_weight").T                      # (128,128)→(128,128) row-major
    x_fc_b         = g("x_fc_bias").reshape(-1, 1)           # (128,1)
    y_fc_w         = g("y_fc_weight").T                      # (128,128)→(128,128) row-major
    y_fc_b         = g("y_fc_bias").reshape(-1, 1)           # (128,1)
    gau_ln_g       = g("gau_ln_g").reshape(1, 1)             # (1,1)
    gau_res_scale  = g("gau_res_scale").reshape(-1, 1)       # (128,1)

    # ── gau_gamma  (2, S) ────────────────────────────────────────────────
    # gamma_u / gamma_v are per-gate scale factors, broadcast across all
    # keypoints inside the ONNX graph. We stack them as rows.
    gamma_u = g("_gamma_u").reshape(feat_channels)  # (128,)
    gamma_v = g("_gamma_v").reshape(feat_channels)  # (128,)
    gau_gamma = np.stack([gamma_u, gamma_v], axis=0)  # (2, 128)

    # ── gau_beta  (2, S)  and  gau_pos_enc  (K, S) ───────────────────────
    # The ONNX graph stores per-keypoint biases  (1,1,K,S)  for each gate
    # (q-gate uses _beta_u, k-gate uses _beta_v).
    #
    # The C++ decoder reads gau_beta as a shared (2*S,) vector and
    # gau_pos_enc as a per-keypoint (K, S) matrix, then computes:
    #   q[t,i] = base * gamma[i]   + beta[i]     + pos_enc[t,i]
    #   k[t,i] = base * gamma[S+i] + beta[S+i]   + pos_enc[t,i]
    #
    # We decompose the ONNX per-keypoint bias as:
    #   Add_1505[t] = beta_shared_q + pos_enc[t]          (q gate)
    #   Add_1509[t] = beta_shared_k + pos_enc[t]          (k gate)
    #
    # by setting:
    #   beta_shared = mean over keypoints  → gau_beta rows 0 and 1
    #   pos_enc[t]  = Add_1505[t] - beta_shared_q         → gau_pos_enc
    #
    # (The same pos_enc is applied to both q and k in the C++ decoder;
    # this is exact for q and approximate for k, which is a minor
    # simplification given that Add_1505 ≈ Add_1509 in trained models.)
    beta_u_kx = g("_beta_u").reshape(num_keypoints, feat_channels)  # (17, 128)
    beta_v_kx = g("_beta_v").reshape(num_keypoints, feat_channels)  # (17, 128)
    beta_shared_q = beta_u_kx.mean(axis=0)                           # (128,)
    beta_shared_k = beta_v_kx.mean(axis=0)                           # (128,)
    gau_beta     = np.stack([beta_shared_q, beta_shared_k], axis=0)  # (2, 128)
    gau_pos_enc  = beta_u_kx - beta_shared_q                         # (17, 128)

    # ── featmap_strides  (for 640×640 input) ─────────────────────────────
    featmap_strides = np.array([16.0, 32.0], dtype=np.float32).reshape(1, 2)

    # ── build YAML lines ─────────────────────────────────────────────────
    lines: List[str] = []
    lines.append("%YAML:1.0")
    lines.append("---")
    lines.append("version: 1")
    lines.append(f"num_keypoints: {num_keypoints}")
    lines.append(f"in_channels: {in_channels}")
    lines.append(f"feat_channels: {feat_channels}")
    lines.append(f"num_bins_x: {num_bins_x}")
    lines.append(f"num_bins_y: {num_bins_y}")
    lines.append(f"bbox_padding: 1.25")
    lines.append(f"num_featmap_strides: 2")
    _write_mat(lines, "featmap_strides", featmap_strides)
    lines.append(f"gau_pos_enc_mode: add")
    lines.append(f"gau_s: {feat_channels}")
    lines.append(f"gau_e: {feat_channels * 2}")
    lines.append(f"gau_sqrt_s: {sqrt_s!r}")
    lines.append(f"gau_ln_scale: {ln_scale!r}")
    lines.append(f"gau_ln_eps: {ln_eps!r}")
    _write_mat(lines, "x_bins", x_bins.reshape(-1, 1))
    _write_mat(lines, "y_bins", y_bins.reshape(-1, 1))
    _write_mat(lines, "spe_dim_t", spe_dim_t.reshape(-1, 1))
    _write_mat(lines, "gau_pos_enc", gau_pos_enc)
    _write_mat(lines, "pose_to_kpts_weight", pose_to_kpts_w)
    _write_mat(lines, "pose_to_kpts_bias", pose_to_kpts_b)
    _write_mat(lines, "x_fc_weight", x_fc_w)
    _write_mat(lines, "x_fc_bias", x_fc_b)
    _write_mat(lines, "y_fc_weight", y_fc_w)
    _write_mat(lines, "y_fc_bias", y_fc_b)
    _write_mat(lines, "gau_uv_weight", gau_uv_w)
    _write_empty_mat(lines, "gau_uv_bias")
    _write_mat(lines, "gau_o_weight", gau_o_w)
    _write_empty_mat(lines, "gau_o_bias")
    _write_mat(lines, "gau_gamma", gau_gamma)
    _write_mat(lines, "gau_beta", gau_beta)
    _write_mat(lines, "gau_ln_g", gau_ln_g)
    _write_mat(lines, "gau_res_scale", gau_res_scale)
    _write_mat(lines, "flatten_priors_640", flatten_priors)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO]   decoder params → {out_path}")


# ---------------------------------------------------------------------------
# ONNX graph cut
# ---------------------------------------------------------------------------

def export_no_nms(
    model_path: Path,
    out_onnx: Path,
    variant: str,
) -> None:
    """Cut the ONNX graph before NMS and rename outputs to clean names."""
    cuts = CUT_TENSORS[variant]

    # Read input names from the file so we can pass them to extract_model.
    # (We do NOT run infer_shapes here: extract_model reads the file directly,
    # not the in-memory object.  Shape inference is re-run on the extracted
    # subgraph below.)
    print(f"[INFO] Loading {model_path} …")
    model = onnx.load(str(model_path))
    input_names  = [n.name for n in model.graph.input]
    output_names = list(cuts.values())   # original tensor names at cut points

    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_onnx) + ".tmp.onnx"

    print(f"[INFO] Extracting subgraph (cutting before NMS) …")
    # extract_model writes to disk and returns None
    onnx.utils.extract_model(
        str(model_path),
        tmp_path,
        input_names,
        output_names,
    )
    sub = onnx.load(tmp_path)

    # Rename outputs to clean names (bboxes, scores, pose_vecs, kpt_vis, priors)
    clean_names = list(cuts.keys())
    old_to_new  = dict(zip(output_names, clean_names))

    for node in sub.graph.node:
        node.output[:] = [old_to_new.get(o, o) for o in node.output]
        node.input[:]  = [old_to_new.get(i, i) for i in node.input]

    for vi in sub.graph.output:
        if vi.name in old_to_new:
            vi.name = old_to_new[vi.name]

    # Re-run shape inference on the renamed model
    sub = onnx.shape_inference.infer_shapes(sub)

    # Validate
    onnx.checker.check_model(sub)

    os.remove(tmp_path)
    onnx.save(sub, str(out_onnx))

    # Print output shapes as confirmation
    print(f"[INFO]   outputs:")
    for o in sub.graph.output:
        shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in o.type.tensor_type.shape.dim
        ]
        print(f"[INFO]     {o.name:12s}  {shape}")

    nms_remaining = [
        n.name or "(unnamed)"
        for n in sub.graph.node
        if n.op_type == "NonMaxSuppression"
    ]
    if nms_remaining:
        raise RuntimeError(
            f"NMS nodes still present after cut: {nms_remaining}"
        )
    print(f"[INFO]   no NMS nodes — OK")
    print(f"[INFO]   ONNX model   → {out_onnx}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _detect_variant(model_path: Path) -> str:
    stem = model_path.stem.lower()
    for v in CUT_TENSORS:
        if v in stem:
            return v
    raise ValueError(
        f"Cannot auto-detect model variant from filename '{model_path.name}'. "
        f"Expected one of: {list(CUT_TENSORS.keys())}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export NMS-free ONNX and DCC decoder YAML for RTMO models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Input ONNX model path (rtmo-s.onnx or rtmo-m.onnx).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output ONNX path. Defaults to "
            "convert/no_nms_onnx/<stem>-no-nms.onnx."
        ),
    )
    p.add_argument(
        "--decoder_out",
        type=Path,
        default=None,
        help=(
            "Output decoder YAML path. Defaults to "
            "convert/decoder/<variant>_dcc_decoder_params.yml."
        ),
    )
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=list(CUT_TENSORS.keys()),
        help="Model variant. Auto-detected from filename if omitted.",
    )
    p.add_argument(
        "--skip_decoder",
        action="store_true",
        help="Skip decoder YAML extraction (ONNX cut only).",
    )
    p.add_argument(
        "--skip_onnx",
        action="store_true",
        help="Skip ONNX cut (decoder YAML only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        sys.exit(f"[ERROR] Model not found: {args.model}")

    variant = args.variant or _detect_variant(args.model)
    print(f"[INFO] Model variant : {variant}")
    print(f"[INFO] Source ONNX   : {args.model}")

    # Resolve output paths
    script_dir = Path(__file__).parent
    repo_root  = script_dir.parent

    if args.output is None:
        out_onnx = repo_root / "convert" / "no_nms_onnx" / f"{args.model.stem}-no-nms.onnx"
    else:
        out_onnx = args.output

    tag = variant.replace("-", "_")   # "rtmo-s" → "rtmo_s"
    if args.decoder_out is None:
        decoder_out = repo_root / "convert" / "decoder" / f"{tag}_dcc_decoder_params.yml"
    else:
        decoder_out = args.decoder_out

    # --- Step 1: graph cut ---
    if not args.skip_onnx:
        export_no_nms(args.model, out_onnx, variant)
    else:
        print("[INFO] Skipping ONNX graph cut (--skip_onnx).")

    # --- Step 2: decoder YAML ---
    if not args.skip_decoder:
        print(f"[INFO] Extracting DCC decoder params …")
        extract_decoder_yaml(args.model, variant, decoder_out)
    else:
        print("[INFO] Skipping decoder YAML (--skip_decoder).")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
