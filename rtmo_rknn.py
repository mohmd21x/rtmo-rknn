import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

NUM_KEYPOINTS = 17
# Embedded ONNX NonMaxSuppression uses 0.15; rtmo_gpu.postprocess keeps score > 0.3.
ONNX_NMS_SCORE_THRESHOLD = 0.15
ONNX_OUTPUT_SCORE_THRESHOLD = 0.3
_SCALAR_RE = re.compile(r"^([A-Za-z0-9_]+):\s*(.+?)\s*$")
_MATRIX_HEADER_RE = re.compile(r"^([A-Za-z0-9_]+):\s*!!opencv-matrix\s*$")


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _softmax_stable(v: np.ndarray) -> np.ndarray:
    if v.size == 0:
        return v
    out = v.astype(np.float32, copy=True)
    out -= out.max()
    np.exp(out, out=out)
    s = out.sum()
    if s > 1e-12:
        out /= s
    return out


def _linear_forward(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray],
    out_dim: int,
) -> np.ndarray:
    """Row-major weight layout: weight shape (out_dim, in_dim)."""
    out = np.zeros(out_dim, dtype=np.float32)
    in_dim = x.shape[0]
    for o in range(out_dim):
        v = 0.0 if bias is None or bias.size == 0 else float(bias[o])
        out[o] = v + float(np.dot(weight[o * in_dim : (o + 1) * in_dim], x))
    return out


def _parse_scalar(value: str):
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_opencv_matrix_block(lines: List[str], start_idx: int) -> Tuple[np.ndarray, int]:
    rows = cols = 0
    data_tokens: List[str] = []
    idx = start_idx
    while idx < len(lines):
        raw = lines[idx]
        line = raw.strip()
        if not line:
            idx += 1
            continue
        if not raw[:1].isspace():
            break
        if line.startswith("rows:"):
            rows = int(line.split(":", 1)[1].strip())
        elif line.startswith("cols:"):
            cols = int(line.split(":", 1)[1].strip())
        elif line.startswith("data:"):
            data_part = line.split(":", 1)[1].strip()
            if data_part.startswith("["):
                data_tokens.append(data_part)
                while not data_tokens[-1].endswith("]") and idx + 1 < len(lines):
                    idx += 1
                    next_raw = lines[idx]
                    if not next_raw[:1].isspace():
                        break
                    data_tokens.append(next_raw.strip())
        idx += 1

    if not data_tokens:
        return np.array([], dtype=np.float32), idx

    raw = " ".join(data_tokens)
    raw = raw[raw.find("[") + 1 : raw.rfind("]")]
    if not raw.strip():
        return np.array([], dtype=np.float32), idx

    values = np.fromstring(raw.replace(",", " "), sep=" ", dtype=np.float32)
    if rows > 0 and cols > 0 and values.size == rows * cols:
        return values.reshape(rows, cols), idx
    return values, idx


def _load_decoder_yaml(path: Path) -> Dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    params: Dict[str, object] = {}
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("%") or line == "---":
            continue

        matrix_match = _MATRIX_HEADER_RE.match(line)
        if matrix_match:
            key = matrix_match.group(1)
            arr, idx = _parse_opencv_matrix_block(lines, idx)
            params[key] = arr
            continue

        scalar_match = _SCALAR_RE.match(line)
        if scalar_match:
            key, value = scalar_match.groups()
            if value == "!!opencv-matrix":
                arr, idx = _parse_opencv_matrix_block(lines, idx)
                params[key] = arr
            else:
                params[key] = _parse_scalar(value)
    return params


def _resolve_decoder_params(model_path: str, decoder_params: Optional[str]) -> Path:
    if decoder_params:
        return Path(decoder_params)
    path_lower = model_path.lower()
    repo_root = Path(__file__).resolve().parent
    if "rtmo-m" in path_lower or "rtmo_m" in path_lower:
        return repo_root / "convert" / "decoder" / "rtmo_m_dcc_decoder_params.yml"
    return repo_root / "convert" / "decoder" / "rtmo_s_dcc_decoder_params.yml"


def _resolve_no_nms_onnx(rknn_path: str, rknn_onnx: Optional[str]) -> Path:
    if rknn_onnx:
        return Path(rknn_onnx)
    rknn = Path(rknn_path)
    stem = rknn.stem
    for suffix in (".fp16", ".int8"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return rknn.resolve().parents[1] / "no_nms_onnx" / f"{stem}-no-nms.onnx"


def _detect_rknn_quant(rknn_path: str) -> str:
    lower = rknn_path.lower()
    if ".int8." in lower or lower.endswith(".int8.rknn"):
        return "int8"
    return "fp16"


def _collect_calibration_images(sample_data_dir: Path, limit: int = 4) -> List[Path]:
    if not sample_data_dir.exists():
        raise FileNotFoundError(f"Sample data directory not found: {sample_data_dir}")
    image_files = sorted(
        p
        for p in sample_data_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if len(image_files) < limit:
        raise ValueError(
            f"INT8 simulator build needs at least {limit} calibration images, "
            f"found {len(image_files)} in {sample_data_dir}"
        )
    return image_files[:limit]


class DCCDecoder:
    """CPU GAU decoder for RTMO no-NMS RKNN outputs."""

    def __init__(self, yaml_path: str):
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"DCC decoder params not found: {yaml_path}")

        params = _load_decoder_yaml(path)

        num_keypoints = int(params["num_keypoints"])
        if num_keypoints != NUM_KEYPOINTS:
            raise ValueError(
                f"Expected {NUM_KEYPOINTS} keypoints in decoder params, got {num_keypoints}"
            )

        self.in_channels = int(params["in_channels"])
        self.feat_channels = int(params["feat_channels"])
        self.num_bins_x = int(params["num_bins_x"])
        self.num_bins_y = int(params["num_bins_y"])
        self.gau_s = int(params["gau_s"])
        self.gau_e = int(params["gau_e"])
        self.gau_sqrt_s = float(params["gau_sqrt_s"])
        self.gau_ln_scale = float(params["gau_ln_scale"])
        self.gau_ln_eps = float(params["gau_ln_eps"])
        self.bbox_padding = float(params["bbox_padding"])
        self.gau_pos_enc_mode = str(params["gau_pos_enc_mode"])

        def _flat(key: str) -> np.ndarray:
            return np.asarray(params[key], dtype=np.float32).reshape(-1)

        self.x_bins = _flat("x_bins")
        self.y_bins = _flat("y_bins")
        self.spe_dim_t = _flat("spe_dim_t")
        self.gau_pos_enc = _flat("gau_pos_enc")
        self.pose_to_kpts_weight = _flat("pose_to_kpts_weight")
        self.pose_to_kpts_bias = _flat("pose_to_kpts_bias")
        self.x_fc_weight = _flat("x_fc_weight")
        self.x_fc_bias = _flat("x_fc_bias")
        self.y_fc_weight = _flat("y_fc_weight")
        self.y_fc_bias = _flat("y_fc_bias")
        self.gau_uv_weight = _flat("gau_uv_weight")
        self.gau_uv_bias = _flat("gau_uv_bias")
        self.gau_o_weight = _flat("gau_o_weight")
        self.gau_o_bias = _flat("gau_o_bias")
        self.gau_gamma = _flat("gau_gamma")
        self.gau_beta = _flat("gau_beta")
        self.gau_ln_g = _flat("gau_ln_g")
        self.gau_res_scale = _flat("gau_res_scale")
        self.flatten_priors_640 = _flat("flatten_priors_640")

        if (
            self.in_channels <= 0
            or self.feat_channels <= 0
            or self.num_bins_x <= 0
            or self.num_bins_y <= 0
            or self.gau_s <= 0
            or self.gau_e <= 0
        ):
            raise ValueError(f"Invalid decoder params in {yaml_path}")

    def decode(
        self,
        pose_vec: np.ndarray,
        prior_xy: np.ndarray,
        bbox_xyxy: np.ndarray,
    ) -> np.ndarray:
        """Decode one detection to keypoint xy coordinates in 640-space."""
        k = NUM_KEYPOINTS
        c = self.feat_channels
        d = self.in_channels
        bx = self.num_bins_x
        by = self.num_bins_y
        s = self.gau_s
        e = self.gau_e
        spe = self.spe_dim_t.size * 2

        x1, y1, x2, y2 = bbox_xyxy
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        sx = max(1e-4, (x2 - x1) * self.bbox_padding)
        sy = max(1e-4, (y2 - y1) * self.bbox_padding)
        gx, gy = prior_xy
        center_x = cx - gx
        center_y = cy - gy

        kpt_feats_flat = _linear_forward(
            pose_vec.astype(np.float32),
            self.pose_to_kpts_weight,
            self.pose_to_kpts_bias,
            k * c,
        )
        kpt_feats = np.zeros(k * c, dtype=np.float32)
        uv_tokens = np.zeros(k * (2 * e + s), dtype=np.float32)

        for t in range(k):
            inp = kpt_feats_flat[t * c : (t + 1) * c]
            norm = max(1e-12, float(np.dot(inp, inp)))
            norm = max(np.sqrt(norm) * self.gau_ln_scale, self.gau_ln_eps)
            ln_g = 1.0 if self.gau_ln_g.size == 0 else float(self.gau_ln_g[0])
            x_norm = (inp / norm) * ln_g
            uv = _linear_forward(
                x_norm,
                self.gau_uv_weight,
                self.gau_uv_bias if self.gau_uv_bias.size else None,
                2 * e + s,
            )
            uv = _silu(uv)
            kpt_feats[t * c : (t + 1) * c] = x_norm
            uv_tokens[t * (2 * e + s) : (t + 1) * (2 * e + s)] = uv

        u = np.zeros(k * e, dtype=np.float32)
        v = np.zeros(k * e, dtype=np.float32)
        q = np.zeros(k * s, dtype=np.float32)
        key = np.zeros(k * s, dtype=np.float32)
        for t in range(k):
            uv = uv_tokens[t * (2 * e + s) : (t + 1) * (2 * e + s)]
            for i in range(e):
                u[t * e + i] = uv[i]
                v[t * e + i] = uv[e + i]
            for i in range(s):
                base = uv[2 * e + i]
                q[t * s + i] = base * self.gau_gamma[i] + self.gau_beta[i]
                key[t * s + i] = (
                    base * self.gau_gamma[s + i] + self.gau_beta[s + i]
                )
                if (
                    self.gau_pos_enc_mode == "add"
                    and self.gau_pos_enc.size == k * s
                ):
                    q[t * s + i] += self.gau_pos_enc[t * s + i]
                    key[t * s + i] += self.gau_pos_enc[t * s + i]

        kernel = np.zeros(k * k, dtype=np.float32)
        for i in range(k):
            for j in range(k):
                dot = float(np.dot(q[i * s : (i + 1) * s], key[j * s : (j + 1) * s]))
                z = max(0.0, dot / self.gau_sqrt_s)
                kernel[i * k + j] = z * z

        kv = np.zeros(k * e, dtype=np.float32)
        for i in range(k):
            for e_idx in range(e):
                kv[i * e + e_idx] = float(
                    np.dot(
                        kernel[i * k : (i + 1) * k],
                        v[: k * e].reshape(k, e)[:, e_idx],
                    )
                )

        gau_out = np.zeros(k * c, dtype=np.float32)
        for i in range(k):
            pre_o = u[i * e : (i + 1) * e] * kv[i * e : (i + 1) * e]
            o_vec = _linear_forward(
                pre_o,
                self.gau_o_weight,
                self.gau_o_bias if self.gau_o_bias.size else None,
                c,
            )
            for c_idx in range(c):
                res_scale = (
                    1.0
                    if self.gau_res_scale.size == 0
                    else float(self.gau_res_scale[c_idx])
                )
                gau_out[i * c + c_idx] = (
                    res_scale * kpt_feats[i * c + c_idx] + o_vec[c_idx]
                )

        x_bins = self.x_bins[:bx] * sx + center_x
        y_bins = self.y_bins[:by] * sy + center_y

        x_bins_enc = np.zeros(bx * c, dtype=np.float32)
        y_bins_enc = np.zeros(by * c, dtype=np.float32)
        for b in range(bx):
            spe_vec = np.zeros(spe, dtype=np.float32)
            for t_idx, dim_t in enumerate(self.spe_dim_t):
                f = x_bins[b] / dim_t
                spe_vec[t_idx] = np.cos(f)
                spe_vec[t_idx + self.spe_dim_t.size] = np.sin(f)
            enc = _linear_forward(
                spe_vec,
                self.x_fc_weight,
                self.x_fc_bias,
                c,
            )
            x_bins_enc[b * c : (b + 1) * c] = enc
        for b in range(by):
            spe_vec = np.zeros(spe, dtype=np.float32)
            for t_idx, dim_t in enumerate(self.spe_dim_t):
                f = y_bins[b] / dim_t
                spe_vec[t_idx] = np.cos(f)
                spe_vec[t_idx + self.spe_dim_t.size] = np.sin(f)
            enc = _linear_forward(
                spe_vec,
                self.y_fc_weight,
                self.y_fc_bias,
                c,
            )
            y_bins_enc[b * c : (b + 1) * c] = enc

        out_kpts = np.zeros((k, 2), dtype=np.float32)
        for kpt_idx in range(k):
            feat = gau_out[kpt_idx * c : (kpt_idx + 1) * c]
            hx = np.zeros(bx, dtype=np.float32)
            hy = np.zeros(by, dtype=np.float32)
            for b in range(bx):
                hx[b] = float(np.dot(feat, x_bins_enc[b * c : (b + 1) * c]))
            for b in range(by):
                hy[b] = float(np.dot(feat, y_bins_enc[b * c : (b + 1) * c]))
            hx = _softmax_stable(hx)
            hy = _softmax_stable(hy)
            x = gx + float(np.dot(hx, x_bins))
            y = gy + float(np.dot(hy, y_bins))
            out_kpts[kpt_idx, 0] = x
            out_kpts[kpt_idx, 1] = y
        return out_kpts


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    xx1 = max(a[0], b[0])
    yy1 = max(a[1], b[1])
    xx2 = min(a[2], b[2])
    yy2 = min(a[3], b[3])
    w = max(0.0, xx2 - xx1)
    h = max(0.0, yy2 - yy1)
    inter = w * h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _is_plausible_detection_box(
    box: np.ndarray, input_size: Tuple[int, int] = (640, 640)
) -> bool:
    """Drop quantized false positives with off-canvas boxes (common INT8 artifact)."""
    coords = np.asarray(box, dtype=np.float32).reshape(4)
    x1, y1, x2, y2 = coords
    h_lim, w_lim = input_size
    w = float(x2 - x1)
    h = float(y2 - y1)
    if w < 12.0 or h < 12.0:
        return False
    if x1 < -0.1 * w_lim or y1 < -0.1 * h_lim:
        return False
    if x2 > 1.1 * w_lim or y2 > 1.1 * h_lim:
        return False
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    if cx < 0.0 or cy < 0.0 or cx > w_lim or cy > h_lim:
        return False
    return True


def _scores_for_nms(
    scores: np.ndarray, bboxes: np.ndarray, input_size: Tuple[int, int]
) -> np.ndarray:
    masked = np.asarray(scores, dtype=np.float32).copy()
    for idx in range(masked.shape[0]):
        if not _is_plausible_detection_box(bboxes[idx], input_size):
            masked[idx] = 0.0
    return masked


def _nms_xyxy_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
    max_detections: int,
) -> List[int]:
    order = scores.argsort()[::-1]
    keep: List[int] = []
    for idx in order:
        if len(keep) >= max_detections:
            break
        suppress = False
        for kept_idx in keep:
            if _iou_xyxy(boxes[idx], boxes[kept_idx]) > iou_threshold:
                suppress = True
                break
        if not suppress:
            keep.append(int(idx))
    return keep


def _nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    score_threshold: float,
    iou_threshold: float,
    max_detections: int,
) -> List[int]:
    if boxes.size == 0:
        return []

    keep_mask = scores >= score_threshold
    if not np.any(keep_mask):
        return []

    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    orig_indices = np.nonzero(keep_mask)[0]

    selected = _nms_xyxy_numpy(boxes, scores, iou_threshold, max_detections)
    return [int(orig_indices[i]) for i in selected]


def _reorder_by_sort_indices(
    tensor: np.ndarray, sort_indices: np.ndarray
) -> np.ndarray:
    """Map raw-anchor tensors into the same order as boxes/scores."""
    order = np.asarray(sort_indices, dtype=np.int64).reshape(-1)
    if order.size == 0:
        return tensor
    if order.size > 0 and int(order.max()) >= tensor.shape[0]:
        raise ValueError(
            f"sort_indices max {int(order.max())} out of range for "
            f"{tensor.shape[0]} candidates"
        )
    return tensor[order]


def _outputs_to_dict(
    outputs: List[np.ndarray], output_names: Optional[List[str]]
) -> Dict[str, np.ndarray]:
    if output_names is None:
        default_names = [
            "bboxes",
            "scores",
            "pose_vecs",
            "kpt_vis",
            "priors",
            "sort_indices",
        ]
        output_names = default_names[: len(outputs)]
    if len(output_names) != len(outputs):
        raise ValueError(
            f"Output name count {len(output_names)} does not match "
            f"tensor count {len(outputs)}"
        )
    return {name: out for name, out in zip(output_names, outputs)}


def _cpu_topk_sort_indices(scores: np.ndarray) -> np.ndarray:
    """Match ONNX TopK(topk_inds): descending sort of all anchor scores."""
    scores_flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    return np.argsort(-scores_flat, kind="stable").astype(np.int64)


def _log_array_stats(label: str, arr: np.ndarray) -> None:
    data = np.asarray(arr)
    flat = data.reshape(-1)
    finite = flat[np.isfinite(flat)] if flat.size else flat
    if finite.size == 0:
        print(f"[TRACE] {label}: shape={data.shape} (empty or all non-finite)")
        return
    print(
        f"[TRACE] {label}: shape={data.shape} dtype={data.dtype} "
        f"min={float(finite.min()):.6f} max={float(finite.max()):.6f} "
        f"mean={float(finite.mean()):.6f} "
        f"p50={float(np.percentile(finite, 50)):.6f} "
        f"p99={float(np.percentile(finite, 99)):.6f}"
    )


def _log_top_scores(
    scores: np.ndarray,
    bboxes: np.ndarray,
    *,
    prefix: str,
    top_k: int = 10,
) -> None:
    scores_flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if scores_flat.size == 0:
        print(f"[TRACE] {prefix}: no scores")
        return
    order = np.argsort(-scores_flat)[:top_k]
    print(f"[TRACE] {prefix} top-{min(top_k, order.size)}:")
    for rank, idx in enumerate(order, start=1):
        bb = boxes[idx]
        print(
            f"  #{rank:2d} idx={idx:4d} score={scores_flat[idx]:.4f} "
            f"bbox=[{bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f},{bb[3]:.1f}]"
        )


def _to_float32(arr: np.ndarray) -> np.ndarray:
    """RKNN may return float16 outputs for hybrid INT8 models."""
    return np.asarray(arr, dtype=np.float32)


def _parse_no_nms_output_dict(
    parsed: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(parsed) == 5:
        bboxes_raw = _to_float32(parsed["bboxes"]).reshape(-1, 4)
        scores_raw = _to_float32(parsed["scores"]).reshape(-1)
        pose_vecs = _to_float32(parsed["pose_vecs"]).reshape(
            bboxes_raw.shape[0], -1
        )
        kpt_vis = _to_float32(parsed["kpt_vis"]).reshape(
            bboxes_raw.shape[0], NUM_KEYPOINTS
        )
        priors = _to_float32(parsed["priors"]).reshape(
            bboxes_raw.shape[0], 2
        )
        sort_indices = _cpu_topk_sort_indices(scores_raw)
        bboxes = bboxes_raw[sort_indices]
        scores = scores_raw[sort_indices]
        pose_vecs = _reorder_by_sort_indices(pose_vecs, sort_indices)
        kpt_vis = _reorder_by_sort_indices(kpt_vis, sort_indices)
        priors = _reorder_by_sort_indices(priors, sort_indices)
        return bboxes, scores, pose_vecs, kpt_vis, priors, sort_indices

    if len(parsed) != 6:
        raise ValueError(
            f"Expected 5 or 6 no-NMS outputs, got {len(parsed)}. "
            "Re-export with convert/export_no_nms.py."
        )

    bboxes = _to_float32(parsed["bboxes"]).reshape(-1, 4)
    scores = _to_float32(parsed["scores"]).reshape(-1)
    pose_vecs = _to_float32(parsed["pose_vecs"]).reshape(
        bboxes.shape[0], -1
    )
    kpt_vis = _to_float32(parsed["kpt_vis"]).reshape(
        bboxes.shape[0], NUM_KEYPOINTS
    )
    priors = _to_float32(parsed["priors"]).reshape(
        bboxes.shape[0], 2
    )
    sort_indices = np.asarray(parsed["sort_indices"], dtype=np.int64).reshape(-1)

    pose_vecs = _reorder_by_sort_indices(pose_vecs, sort_indices)
    kpt_vis = _reorder_by_sort_indices(kpt_vis, sort_indices)
    priors = _reorder_by_sort_indices(priors, sort_indices)
    return bboxes, scores, pose_vecs, kpt_vis, priors, sort_indices


def match_detections_by_iou(
    onnx_boxes: np.ndarray,
    rknn_boxes: np.ndarray,
    iou_threshold: float = 0.3,
) -> List[Tuple[int, int, float]]:
    """Greedy IoU matching: ONNX index -> RKNN index."""
    pairs: List[Tuple[int, int, float]] = []
    used_rknn: set = set()
    for onnx_idx, onnx_box in enumerate(np.asarray(onnx_boxes, dtype=np.float32).reshape(-1, 4)):
        best_rknn_idx = -1
        best_iou = 0.0
        for rknn_idx, rknn_box in enumerate(np.asarray(rknn_boxes, dtype=np.float32).reshape(-1, 4)):
            if rknn_idx in used_rknn:
                continue
            iou = _iou_xyxy(onnx_box, rknn_box)
            if iou > best_iou:
                best_iou = iou
                best_rknn_idx = rknn_idx
        if best_rknn_idx >= 0 and best_iou >= iou_threshold:
            pairs.append((onnx_idx, best_rknn_idx, best_iou))
            used_rknn.add(best_rknn_idx)
    return pairs


class RTMO_RKNN:
    def __init__(
        self,
        model_path: str,
        target: str = "rk3588",
        device_id: int = 0,
        use_simulator: bool = True,
        backend: Optional[str] = None,
        score_threshold: float = ONNX_OUTPUT_SCORE_THRESHOLD,
        nms_score_threshold: float = ONNX_NMS_SCORE_THRESHOLD,
        nms_iou: float = 0.65,
        nms_max: int = 200,
        model_input_size: Tuple[int, int] = (640, 640),
        decoder_params: Optional[str] = None,
        rknn_onnx: Optional[str] = None,
        sample_data: Optional[str] = None,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKNN model not found: {model_path}")

        self.model_path = model_path
        self.target = target
        self.device_id = str(device_id)
        if backend is None:
            backend = "simulator" if use_simulator else "device"
        backend = backend.lower()
        if backend not in {"simulator", "onnx", "device"}:
            raise ValueError(
                f"Invalid backend={backend!r}. Use 'simulator', 'onnx', or 'device'."
            )
        self.backend = backend
        self.use_simulator = backend == "simulator"
        self.score_threshold = score_threshold
        self.nms_score_threshold = nms_score_threshold
        self.nms_iou = nms_iou
        self.nms_max = nms_max
        self.model_input_size = model_input_size
        self.rknn_onnx = rknn_onnx
        repo_root = Path(__file__).resolve().parent
        self.sample_data = (
            Path(sample_data) if sample_data else repo_root / "sample-data"
        )
        self.rknn = None
        self.onnx_session = None
        self.onnx_output_names: Optional[List[str]] = None
        self._rknn_input_layout = "nhwc"
        self._rknn_input_dtype = np.uint8
        self._rknn_input_pass_through = False
        self._debug = os.environ.get("RTMO_RKNN_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
        }

        decoder_path = _resolve_decoder_params(model_path, decoder_params)
        self.decoder = DCCDecoder(str(decoder_path))

        self._init_runtime()

    def _init_simulator_from_onnx(self, rknn) -> None:
        onnx_path = _resolve_no_nms_onnx(self.model_path, self.rknn_onnx)
        if not onnx_path.exists():
            raise FileNotFoundError(
                "rknn-toolkit2 cannot run pre-built .rknn files on the PC simulator "
                "(load_rknn + init_runtime). Rebuild from a no-NMS ONNX instead.\n"
                f"Expected ONNX path: {onnx_path}\n"
                "Run: python convert/export_no_nms.py --model rtmo/rtmo-<s|m>.onnx\n"
                "Or connect an RK3588 and use --use_simulator false."
            )

        input_h, input_w = self.model_input_size
        quant = _detect_rknn_quant(self.model_path)
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            target_platform=self.target,
            output_optimize=False,
        )
        if ret != 0:
            raise RuntimeError(f"RKNN.config failed with ret={ret}")

        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["input"],
            input_size_list=[[1, 3, input_h, input_w]],
        )
        if ret != 0:
            raise RuntimeError(f"RKNN.load_onnx failed with ret={ret}")

        if quant == "int8":
            dataset_path = Path(self.model_path).parent / "dataset.txt"
            image_paths = _collect_calibration_images(self.sample_data, limit=4)
            dataset_path.write_text(
                "\n".join(str(p.resolve()) for p in image_paths) + "\n",
                encoding="utf-8",
            )
            ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN.build failed with ret={ret}")

        ret = rknn.init_runtime(target=None)
        if ret != 0:
            raise RuntimeError(f"Failed to init RKNN simulator runtime: ret={ret}")

    def _init_onnx_backend(self) -> None:
        import onnxruntime as ort

        onnx_path = _resolve_no_nms_onnx(self.model_path, self.rknn_onnx)
        if not onnx_path.exists():
            raise FileNotFoundError(
                "ONNX backend requires a no-NMS ONNX export.\n"
                f"Expected ONNX path: {onnx_path}\n"
                "Run: python convert/export_no_nms.py --model rtmo/rtmo-<s|m>.onnx"
            )
        self.onnx_session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        self.onnx_output_names = [o.name for o in self.onnx_session.get_outputs()]

    def _init_runtime(self) -> None:
        if self.backend == "onnx":
            self._init_onnx_backend()
            return
        if self.use_simulator:
            try:
                from rknn.api import RKNN
            except ImportError as exc:
                raise ImportError(
                    "rknn-toolkit2 is required for simulator mode. Install rknn-toolkit2."
                ) from exc

            self.rknn = RKNN(verbose=False)
            self._init_simulator_from_onnx(self.rknn)
        else:
            try:
                from rknnlite.api import RKNNLite
            except ImportError as exc:
                raise ImportError(
                    "rknnlite2 is required for on-device mode. Install rknnlite2 on RK3588."
                ) from exc

            self.rknn = RKNNLite(verbose=False)
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"Failed to load RKNN model: ret={ret}")

            # Plain init_runtime() is most stable across rknnlite 2.3.x board builds.
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"Failed to init RKNN device runtime: ret={ret}")
            self._query_rknn_input_spec()

    def _query_rknn_input_spec(self) -> None:
        """Read expected input layout/dtype from rknnlite (device) or keep defaults."""
        if self.backend == "onnx" or self.rknn is None:
            return

        # RKNN models here are converted with mean=[0,0,0], std=[1,1,1] so uint8
        # pixels pass through as float 0–255 (matching ONNX). rknnlite expects
        # uint8 4D NHWC and applies mean/std from conversion internally.
        self._rknn_input_layout = "nhwc"
        self._rknn_input_dtype = np.uint8
        self._rknn_input_pass_through = False

        env_mode = os.environ.get("RTMO_RKNN_INPUT", "").strip().lower()
        if self.backend == "device" and env_mode:
            print(
                "[WARN] RTMO_RKNN_INPUT is ignored on device (causes segfault). "
                "Using uint8 NHWC; do not set RTMO_RKNN_INPUT=float32."
            )

        try:
            if hasattr(self.rknn, "get_input_detail"):
                details = self.rknn.get_input_detail()
                if details:
                    detail = details[0]
                    fmt = str(detail.get("fmt", detail.get("format", ""))).lower()
                    if "nchw" in fmt:
                        self._rknn_input_layout = "nchw"
                    elif "nhwc" in fmt:
                        self._rknn_input_layout = "nhwc"

                    dtype_name = str(
                        detail.get("dtype", detail.get("type", ""))
                    ).lower()
                    if "uint8" in dtype_name:
                        self._rknn_input_dtype = np.uint8
                        self._rknn_input_pass_through = False
                    elif "float16" in dtype_name:
                        self._rknn_input_dtype = np.float16
                        self._rknn_input_pass_through = False
                    elif "float32" in dtype_name or dtype_name.endswith("float"):
                        self._rknn_input_dtype = np.float32
                        self._rknn_input_pass_through = False
        except Exception as exc:
            print(f"[WARN] Could not query RKNN input attributes: {exc}")

        self._log_rknn_io_details()

        print(
            f"[INFO] RKNN input: layout={self._rknn_input_layout}, "
            f"dtype={self._rknn_dtype_str()}, "
            f"pass_through={self._rknn_input_pass_through}"
        )

    def _log_rknn_io_details(self) -> None:
        if self.rknn is None or self.backend == "onnx":
            return
        for method_name in ("get_input_detail", "get_output_detail", "get_sdk_version"):
            if not hasattr(self.rknn, method_name):
                continue
            try:
                detail = getattr(self.rknn, method_name)()
                print(f"[INFO] RKNN {method_name}: {detail}")
            except Exception as exc:
                print(f"[WARN] RKNN {method_name} failed: {exc}")

    def _rknn_dtype_str(self) -> str:
        if self._rknn_input_dtype == np.float32:
            return "float32"
        if self._rknn_input_dtype == np.float16:
            return "float16"
        return "uint8"

    def _prepare_rknn_input(self, img: np.ndarray) -> np.ndarray:
        """Build 4D rknnlite input (batch, H, W, C) or NCHW when required."""
        if img.ndim == 4:
            if img.shape[0] != 1:
                raise ValueError(
                    f"RKNN inference expects batch size 1, got shape {img.shape}"
                )
            batched = img
        elif img.ndim == 3:
            batched = np.expand_dims(img, axis=0)
        else:
            raise ValueError(f"RKNN inference expects HWC image, got shape {img.shape}")

        layout = self._rknn_input_layout
        if layout == "nchw" and batched.shape[-1] == 3:
            batched = batched.transpose(0, 3, 1, 2)

        if self._rknn_input_dtype == np.float32:
            inp = np.ascontiguousarray(batched, dtype=np.float32)
        elif self._rknn_input_dtype == np.float16:
            inp = np.ascontiguousarray(batched, dtype=np.float16)
        else:
            inp = np.ascontiguousarray(batched, dtype=np.uint8)
        return inp

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3), dtype=np.uint8
            ) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(
            self.model_input_size[0] / img.shape[0],
            self.model_input_size[1] / img.shape[1],
        )
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[: padded_shape[0], : padded_shape[1]] = resized_img

        return padded_img, ratio

    def inference(self, img: np.ndarray) -> List[np.ndarray]:
        if self.backend == "onnx":
            if img.ndim == 3:
                img = np.expand_dims(img, axis=0)
            inp = np.ascontiguousarray(img.transpose(0, 3, 1, 2), dtype=np.float32)
            outputs = self.onnx_session.run(None, {"input": inp})
            return outputs

        inp = self._prepare_rknn_input(img)
        if self._debug:
            print(
                f"[DEBUG] rknn inference input shape={inp.shape}, "
                f"dtype={inp.dtype}, layout={self._rknn_input_layout}"
            )

        if self.backend == "device":
            # rknnlite default: uint8 NHWC 4D, mean/std applied by runtime.
            outputs = self.rknn.inference(inputs=[inp], data_type="uint8")
        else:
            dtype_name = self._rknn_dtype_str()
            pass_through = [1 if self._rknn_input_pass_through else 0]
            call_variants = [
                dict(
                    inputs=[inp],
                    data_type=dtype_name,
                    data_format=self._rknn_input_layout,
                    inputs_pass_through=pass_through,
                ),
                dict(
                    inputs=[inp],
                    data_type=dtype_name,
                    data_format=self._rknn_input_layout,
                ),
                dict(inputs=[inp], data_type=dtype_name),
                dict(inputs=[inp]),
            ]

            outputs = None
            last_exc: Optional[Exception] = None
            for kwargs in call_variants:
                try:
                    outputs = self.rknn.inference(**kwargs)
                    break
                except (TypeError, ValueError) as exc:
                    last_exc = exc
                    continue

            if outputs is None and last_exc is not None:
                raise RuntimeError(f"RKNN inference failed: {last_exc}") from last_exc

        if outputs is None:
            raise RuntimeError(
                "RKNN inference returned None. Check input layout/dtype "
                f"(expected 4D NHWC uint8 for device)."
            )
        if self._debug:
            print(f"[DEBUG] rknn inference outputs={len(outputs)} tensors")
        return outputs

    def _parse_no_nms_outputs(
        self, outputs: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if outputs is None:
            raise ValueError("RKNN outputs are None (inference failed)")
        output_names = (
            self.onnx_output_names if self.backend == "onnx" else None
        )
        parsed = _outputs_to_dict(outputs, output_names)
        return _parse_no_nms_output_dict(parsed)

    def postprocess(
        self, outputs: List[np.ndarray], ratio: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        bboxes, scores, pose_vecs, kpt_vis, _priors, sort_indices = (
            self._parse_no_nms_outputs(outputs)
        )

        if pose_vecs.shape[1] != self.decoder.in_channels:
            raise ValueError(
                f"pose_vecs dim {pose_vecs.shape[1]} does not match decoder "
                f"in_channels {self.decoder.in_channels}"
            )

        # DCC decode needs static grid priors (flatten_priors_640), indexed by the
        # raw anchor id from sort_indices. The network "priors" output (Add_1312)
        # is an offset bbox tensor and must not be used here.
        flatten_priors = self.decoder.flatten_priors_640.reshape(-1, 2)

        nms_scores = _scores_for_nms(scores, bboxes, self.model_input_size)
        nms_kept = _nms_xyxy(
            bboxes,
            nms_scores,
            self.nms_score_threshold,
            self.nms_iou,
            self.nms_max,
        )
        kept = [idx for idx in nms_kept if scores[idx] >= self.score_threshold]
        if not kept:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32),
                np.zeros((0, NUM_KEYPOINTS), dtype=np.float32),
            )

        final_boxes = bboxes[kept] / ratio
        final_scores = scores[kept]
        final_kpt_vis = kpt_vis[kept]

        keypoints = []
        for idx in kept:
            raw_idx = int(sort_indices[idx])
            if raw_idx < 0 or raw_idx >= flatten_priors.shape[0]:
                raise ValueError(
                    f"sort_indices[{idx}]={raw_idx} out of range for "
                    f"{flatten_priors.shape[0]} flatten_priors entries"
                )
            prior = flatten_priors[raw_idx]
            kpts_640 = self.decoder.decode(pose_vecs[idx], prior, bboxes[idx])
            keypoints.append(kpts_640 / ratio)

        keypoints = np.stack(keypoints, axis=0)
        return final_boxes, final_scores, keypoints, final_kpt_vis

    def diagnose(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run inference with detailed stdout trace (for zero-detection debugging)."""
        print("\n=== RTMO_RKNN diagnose ===")
        print(
            f"[TRACE] model={self.model_path} backend={self.backend} "
            f"nms_score>={self.nms_score_threshold} final_score>={self.score_threshold} "
            f"nms_iou={self.nms_iou} decoder_in={self.decoder.in_channels}"
        )
        img = np.asarray(image)
        print(
            f"[TRACE] input image: shape={img.shape} dtype={img.dtype} "
            f"min={int(img.min())} max={int(img.max())} mean={float(img.mean()):.1f}"
        )

        preprocessed, ratio = self.preprocess(img)
        print(
            f"[TRACE] letterbox ratio={ratio:.4f} "
            f"content~{int(img.shape[1] * ratio)}x{int(img.shape[0] * ratio)} "
            f"in 640x640"
        )
        _log_array_stats("preprocessed uint8", preprocessed)

        outputs = self.inference(preprocessed)
        output_names = (
            self.onnx_output_names
            if self.backend == "onnx"
            else ["bboxes", "scores", "pose_vecs", "kpt_vis", "priors", "sort_indices"]
        )
        print(f"[TRACE] rknn outputs: {len(outputs)} tensors")
        for i, out in enumerate(outputs):
            name = output_names[i] if i < len(output_names) else f"out{i}"
            _log_array_stats(f"raw output[{i}] {name}", out)

        parsed = _outputs_to_dict(
            outputs,
            self.onnx_output_names if self.backend == "onnx" else None,
        )
        print(f"[TRACE] parsed output keys ({len(parsed)}): {list(parsed.keys())}")

        if len(parsed) == 5:
            scores_raw = np.asarray(parsed["scores"], dtype=np.float32).reshape(-1)
            bboxes_raw = np.asarray(parsed["bboxes"], dtype=np.float32).reshape(-1, 4)
            _log_array_stats("scores before CPU TopK", scores_raw)
            _log_top_scores(scores_raw, bboxes_raw, prefix="before TopK")
            ge_nms_raw = int(np.sum(scores_raw >= self.nms_score_threshold))
            ge_final_raw = int(np.sum(scores_raw >= self.score_threshold))
            print(
                f"[TRACE] raw anchors >={self.nms_score_threshold}: {ge_nms_raw} | "
                f">={self.score_threshold}: {ge_final_raw}"
            )

        bboxes, scores, pose_vecs, kpt_vis, _priors, sort_indices = (
            _parse_no_nms_output_dict(parsed)
        )
        _log_array_stats("scores after TopK sort", scores)
        _log_top_scores(scores, bboxes, prefix="after TopK")
        ge_nms = int(np.sum(scores >= self.nms_score_threshold))
        ge_final = int(np.sum(scores >= self.score_threshold))
        print(
            f"[TRACE] sorted anchors >={self.nms_score_threshold}: {ge_nms} | "
            f">={self.score_threshold}: {ge_final}"
        )

        nms_scores = _scores_for_nms(scores, bboxes, self.model_input_size)
        plausible = int(np.sum(nms_scores >= self.nms_score_threshold))
        if plausible < ge_nms:
            print(
                f"[TRACE] plausible-box filter: {ge_nms - plausible} high-score "
                f"anchors dropped (off-canvas / invalid INT8 boxes)"
            )
        nms_kept = _nms_xyxy(
            bboxes,
            nms_scores,
            self.nms_score_threshold,
            self.nms_iou,
            self.nms_max,
        )
        kept = [idx for idx in nms_kept if scores[idx] >= self.score_threshold]
        print(
            f"[TRACE] NMS kept {len(nms_kept)} (pre-score-filter) -> "
            f"final {len(kept)} after score>={self.score_threshold}"
        )
        if nms_kept and not kept:
            print(
                "[TRACE] NMS found boxes but all below final score_threshold. "
                "Try --score_threshold 0.15 or inspect top scores above."
            )
            near = sorted(
                ((int(i), float(scores[i])) for i in nms_kept),
                key=lambda x: -x[1],
            )[:5]
            for idx, sc in near:
                bb = bboxes[idx]
                print(
                    f"  nms idx={idx} score={sc:.4f} "
                    f"bbox=[{bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f},{bb[3]:.1f}]"
                )

        if not kept:
            for thr in (0.05, 0.15, 0.25, 0.3):
                n = int(np.sum(scores >= thr))
                print(f"[TRACE] sorted anchors >={thr}: {n}")

        print("=== end diagnose ===\n")
        bboxes, bbox_scores, keypoints, kpt_scores = self.postprocess(outputs, ratio)
        return image.copy(), bboxes, bbox_scores, keypoints, kpt_scores

    def __call__(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        img_out = image.copy()
        if self._debug:
            print("[DEBUG] preprocess")
        preprocessed, ratio = self.preprocess(image)
        if self._debug:
            print(
                f"[DEBUG] preprocessed shape={preprocessed.shape}, "
                f"dtype={preprocessed.dtype}"
            )
        outputs = self.inference(preprocessed)
        if self._debug:
            print("[DEBUG] postprocess")
        bboxes, bbox_scores, keypoints, kpt_scores = self.postprocess(outputs, ratio)
        return img_out, bboxes, bbox_scores, keypoints, kpt_scores

    def release(self) -> None:
        if self.rknn is not None:
            self.rknn.release()
        self.onnx_session = None
        self.onnx_output_names = None
