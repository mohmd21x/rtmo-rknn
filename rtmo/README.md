---
license: mit
pipeline_tag: object-detection
tags:
- Pose Estimation
---
## RTMO / YOLO-NAS-Pose Inference with CUDAExecutionProvider / TensorrtExecutionProvider DEMO

- `demo.sh`: DEMO main program, which will first install rtmlib, and then use rtmo-s to analyze the .mp4 files in the video folder.
- `demo_batch.sh`: Multi-batch version of demo.sh
- `rtmo_gpu.py`: Defines an RTMO_GPU (& RTMO_GPU_BATCH) class, making fine adjustments to CUDA & TensorRT settings.
- `rtmo_demo.py`: Python main program, which has three arguments:
    - `path`: The folder location that contains the .mp4 files to be analyzed.
    - `model_path`: The local path to the ONNX model or a URL pointing to the RTMO model published on mmpose.
    - `--yolo_nas_pose`: If you run inference with YOLO NAS Pose Model instead of RTMO model.
- `rtmo_demo_batch.py`: Multi-batch version of demo_batch.sh
- `video`: Contains one test video.

# Note

* Original ONNX models come from [MMPOSE/RTMO Project Page](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo) trained on body7. We did only 
* DEMO Inferecne Code is modified from [rtmlib](https://github.com/Tau-J/rtmlib)
* TensorrtExecutionProvider only supports Models with fixed batch size (*_batchN.onnx) while CUDAExecutionProvider can run with dynamic batch size.

We did the following to make them work with TensorRTExecutionProvdier 

1. Shape inference
2. batch size 1,2,4 fixation

PS. FP16 ONNX model is also provided.