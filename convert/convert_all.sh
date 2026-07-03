#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_SCRIPT="${ROOT_DIR}/convert/export_no_nms.py"
CONVERT_SCRIPT="${ROOT_DIR}/convert/convert_rknn.py"
NO_NMS_DIR="${ROOT_DIR}/convert/no_nms_onnx"
OUT_DIR="${ROOT_DIR}/convert/models"
SAMPLE_DATA="${ROOT_DIR}/sample-data"
TARGET="${TARGET:-rk3588}"

mkdir -p "${NO_NMS_DIR}" "${OUT_DIR}"

# ── Step 1: export NMS-free ONNX + decoder params for each model ─────────
for model in rtmo-s rtmo-m; do
  echo "[INFO] Exporting no-NMS ONNX for ${model} …"
  python "${EXPORT_SCRIPT}" --model "${ROOT_DIR}/rtmo/${model}.onnx"
done

# ── Step 2: convert each no-NMS ONNX to RKNN (fp16 + int8) ──────────────
for model in rtmo-s rtmo-m; do
  for quant in fp16 int8; do
    input_model="${NO_NMS_DIR}/${model}-no-nms.onnx"
    output_model="${OUT_DIR}/${model}.${quant}.rknn"
    echo "[INFO] Converting ${model} (${quant}) -> ${output_model}"
    extra_args=()
    if [[ "${quant}" == "int8" ]]; then
      extra_args+=(--int8_mode hybrid)
    fi
    python "${CONVERT_SCRIPT}" \
      --model "${input_model}" \
      --output "${output_model}" \
      --quant "${quant}" \
      --target "${TARGET}" \
      --keep_float_io \
      --sample_data "${SAMPLE_DATA}" \
      "${extra_args[@]}"
  done
done

echo "[INFO] Batch conversion finished."
