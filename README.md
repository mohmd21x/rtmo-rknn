# RTMO RKNN Toolkit

Toolkit for converting RTMO ONNX models to RKNN, comparing ONNX vs RKNN outputs on video, and benchmarking keypoint accuracy plus inference speed.

This repository targets:
- `rtmo-s` and `rtmo-m` (640x640 input)
- RK3588 deployment with optional PC simulation

## Repository Layout

```text
rtmo-rknn/
├── rtmo/                      # existing ONNX / TensorRT GPU code
├── sample-data/               # calibration images for INT8 conversion
├── rtmo_rknn.py               # RKNN inference + CPU DCC decoder + NMS
├── convert/
│   ├── export_no_nms.py       # cut ONNX before NMS + extract decoder YAML
│   ├── convert_rknn.py        # no-NMS ONNX -> RKNN (fp16 or int8)
│   ├── convert_all.sh         # export then convert rtmo-s/rtmo-m
│   └── decoder/               # GAU decoder weights (rtmo_s/m_dcc_decoder_params.yml)
├── test/
│   └── compare_video.py       # side-by-side ONNX vs RKNN visualization
├── benchmark/
│   └── benchmark.py           # L2/MAE + inference speed benchmark
├── requirements.txt
└── README.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

On RK3588 devices, install `rknnlite2` (Rockchip-provided wheel) instead of `rknn-toolkit2`.

## Why a 2-Step Conversion?

The original `rtmo-s.onnx` / `rtmo-m.onnx` models embed `NonMaxSuppression` with runtime-computed inputs. RKNN requires all NMS inputs to be constant, so direct conversion fails.

The pipeline cuts the ONNX graph **before** NMS (5 raw outputs) and moves postprocess to Python:

1. **Export** — `export_no_nms.py` writes a no-NMS ONNX plus DCC decoder YAML
2. **Convert** — `convert_rknn.py` builds RKNN from the cut ONNX

At runtime, `rtmo_rknn.py` runs RKNN inference, then on CPU:
- score filter (default 0.15)
- NMS (`cv2.dnn.NMSBoxes`, IoU 0.65, max 200 detections)
- DCC GAU decoder (pose_vecs + priors → 17 keypoint xy)

## Step 1: Export No-NMS ONNX + Decoder Params

```bash
python convert/export_no_nms.py --model rtmo/rtmo-s.onnx
python convert/export_no_nms.py --model rtmo/rtmo-m.onnx
```

Outputs:
- `convert/no_nms_onnx/rtmo-s-no-nms.onnx` (and `rtmo-m-no-nms.onnx`)
- `convert/decoder/rtmo_s_dcc_decoder_params.yml` (and `rtmo_m_…`)

The decoder YAML holds GAU weights extracted from the original ONNX. `RTMO_RKNN` auto-selects the matching file from the RKNN model name (`rtmo-s` vs `rtmo-m`), or you can pass `--decoder_params` explicitly.

## Step 2: Convert No-NMS ONNX to RKNN

Preflight check (optional):

```bash
python convert/check_onnx_for_rknn.py --model convert/no_nms_onnx/rtmo-s-no-nms.onnx
```

Single model conversion:

```bash
python convert/convert_rknn.py \
  --model convert/no_nms_onnx/rtmo-s-no-nms.onnx \
  --output convert/models/rtmo-s.fp16.rknn \
  --quant fp16 \
  --target rk3588 \
  --sample_data sample-data
```

Batch export + conversion (`rtmo-s` + `rtmo-m`, each in `fp16` and `int8`):

```bash
bash convert/convert_all.sh
```

Notes:
- Always convert from the **no-NMS** ONNX produced in step 1, not the original `rtmo/*.onnx`.
- INT8 conversion auto-builds a `dataset.txt` from images in `sample-data`.
- Mean/std normalization is baked into RKNN at conversion time, so runtime input is raw `uint8` NHWC.
- Converter uses fixed static input shape `1x3x640x640`.

## Compare ONNX vs RKNN on Video

```bash
python test/compare_video.py \
  --video rtmo/video/"Oldest video ever - 1888 [tc-L9_4jGc4].mp4" \
  --onnx rtmo/rtmo-s.onnx \
  --rknn convert/models/rtmo-s.fp16.rknn \
  --output test/comparison_output.mp4 \
  --display
```

This script writes a side-by-side output video:
- Left panel: ONNX keypoints/bboxes (embedded NMS + decoder in ONNX)
- Right panel: RKNN keypoints/bboxes (RKNN + CPU decoder in `rtmo_rknn.py`)
- Panel label includes instantaneous FPS

## Benchmark Accuracy + Speed

```bash
python benchmark/benchmark.py \
  --video rtmo/video/"Oldest video ever - 1888 [tc-L9_4jGc4].mp4" \
  --onnx rtmo/rtmo-s.onnx \
  --rknn convert/models/rtmo-s.fp16.rknn \
  --frames 200 \
  --output benchmark/results.csv
```

Metrics are computed on frames where both models detect at least one person:
- Mean L2 distance per keypoint (pixel space)
- Mean absolute error (MAE) per keypoint (pixel space)
- Mean inference time (ms) and FPS for ONNX and RKNN

Output:
- Terminal table
- CSV with columns: `keypoint_name, mean_l2, mae, onnx_ms, rknn_ms`

## Simulator vs Device Runtime

By default, scripts use PC simulator mode:
- `--use_simulator true` -> `rknn-toolkit2` runtime on PC

For RK3588 device runtime:
- `--use_simulator false`
- Optional device args: `--target rk3588 --device_id 0`
- Ensure `rknnlite2` is installed on device

See **[RK3588.md](RK3588.md)** for full board setup, rsync package, convert-on-device, and test commands.

## Quick Start

```bash
# 1) Export no-NMS ONNX + decoder params, then convert to RKNN
bash convert/convert_all.sh

# 2) Visual compare
python test/compare_video.py

# 3) Run benchmark
python benchmark/benchmark.py
```
