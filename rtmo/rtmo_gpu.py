import os
import numpy as np
from typing import List, Tuple, Union
import onnxruntime as ort
import cv2
from queue import Queue

PLUGIN_LIB_PATHS='libmmdeploy_tensorrt_ops.so'
os.environ['ORT_TENSORRT_EXTRA_PLUGIN_LIB_PATHS']=PLUGIN_LIB_PATHS
TRT_BACKEND='POLYGRAPHY'
DEBUG=False

# dictionary from https://github.com/Tau-J/rtmlib/blob/4b29101d54b611048ef165277cebfffff3030074/rtmlib/visualization/skeleton/coco17.py
coco17 = dict(name='coco17',
              keypoint_info={
                  0:
                  dict(name='nose', id=0, color=[51, 153, 255], swap=''),
                  1:
                  dict(name='left_eye',
                       id=1,
                       color=[51, 153, 255],
                       swap='right_eye'),
                  2:
                  dict(name='right_eye',
                       id=2,
                       color=[51, 153, 255],
                       swap='left_eye'),
                  3:
                  dict(name='left_ear',
                       id=3,
                       color=[51, 153, 255],
                       swap='right_ear'),
                  4:
                  dict(name='right_ear',
                       id=4,
                       color=[51, 153, 255],
                       swap='left_ear'),
                  5:
                  dict(name='left_shoulder',
                       id=5,
                       color=[0, 255, 0],
                       swap='right_shoulder'),
                  6:
                  dict(name='right_shoulder',
                       id=6,
                       color=[255, 128, 0],
                       swap='left_shoulder'),
                  7:
                  dict(name='left_elbow',
                       id=7,
                       color=[0, 255, 0],
                       swap='right_elbow'),
                  8:
                  dict(name='right_elbow',
                       id=8,
                       color=[255, 128, 0],
                       swap='left_elbow'),
                  9:
                  dict(name='left_wrist',
                       id=9,
                       color=[0, 255, 0],
                       swap='right_wrist'),
                  10:
                  dict(name='right_wrist',
                       id=10,
                       color=[255, 128, 0],
                       swap='left_wrist'),
                  11:
                  dict(name='left_hip',
                       id=11,
                       color=[0, 255, 0],
                       swap='right_hip'),
                  12:
                  dict(name='right_hip',
                       id=12,
                       color=[255, 128, 0],
                       swap='left_hip'),
                  13:
                  dict(name='left_knee',
                       id=13,
                       color=[0, 255, 0],
                       swap='right_knee'),
                  14:
                  dict(name='right_knee',
                       id=14,
                       color=[255, 128, 0],
                       swap='left_knee'),
                  15:
                  dict(name='left_ankle',
                       id=15,
                       color=[0, 255, 0],
                       swap='right_ankle'),
                  16:
                  dict(name='right_ankle',
                       id=16,
                       color=[255, 128, 0],
                       swap='left_ankle')
              },
              skeleton_info={
                  0:
                  dict(link=('left_ankle', 'left_knee'),
                       id=0,
                       color=[0, 255, 0]),
                  1:
                  dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255,
                                                                    0]),
                  2:
                  dict(link=('right_ankle', 'right_knee'),
                       id=2,
                       color=[255, 128, 0]),
                  3:
                  dict(link=('right_knee', 'right_hip'),
                       id=3,
                       color=[255, 128, 0]),
                  4:
                  dict(link=('left_hip', 'right_hip'),
                       id=4,
                       color=[51, 153, 255]),
                  5:
                  dict(link=('left_shoulder', 'left_hip'),
                       id=5,
                       color=[51, 153, 255]),
                  6:
                  dict(link=('right_shoulder', 'right_hip'),
                       id=6,
                       color=[51, 153, 255]),
                  7:
                  dict(link=('left_shoulder', 'right_shoulder'),
                       id=7,
                       color=[51, 153, 255]),
                  8:
                  dict(link=('left_shoulder', 'left_elbow'),
                       id=8,
                       color=[0, 255, 0]),
                  9:
                  dict(link=('right_shoulder', 'right_elbow'),
                       id=9,
                       color=[255, 128, 0]),
                  10:
                  dict(link=('left_elbow', 'left_wrist'),
                       id=10,
                       color=[0, 255, 0]),
                  11:
                  dict(link=('right_elbow', 'right_wrist'),
                       id=11,
                       color=[255, 128, 0]),
                  12:
                  dict(link=('left_eye', 'right_eye'),
                       id=12,
                       color=[51, 153, 255]),
                  13:
                  dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
                  14:
                  dict(link=('nose', 'right_eye'), id=14, color=[51, 153,
                                                                 255]),
                  15:
                  dict(link=('left_eye', 'left_ear'),
                       id=15,
                       color=[51, 153, 255]),
                  16:
                  dict(link=('right_eye', 'right_ear'),
                       id=16,
                       color=[51, 153, 255]),
                  17:
                  dict(link=('left_ear', 'left_shoulder'),
                       id=17,
                       color=[51, 153, 255]),
                  18:
                  dict(link=('right_ear', 'right_shoulder'),
                       id=18,
                       color=[51, 153, 255])
              })

# functions from https://github.com/Tau-J/rtmlib/blob/4b29101d54b611048ef165277cebfffff3030074/rtmlib/visualization/draw.py#L71
def draw_mmpose(img,
                keypoints,
                scores,
                keypoint_info,
                skeleton_info,
                kpt_thr=0.5,
                radius=2,
                line_width=2):
    assert len(keypoints.shape) == 2

    vis_kpt = [s >= kpt_thr for s in scores]

    link_dict = {}
    for i, kpt_info in keypoint_info.items():
        kpt_color = tuple(kpt_info['color'])
        link_dict[kpt_info['name']] = kpt_info['id']

        kpt = keypoints[i]

        if vis_kpt[i]:
            img = cv2.circle(img, (int(kpt[0]), int(kpt[1])), int(radius),
                             kpt_color, -1)

    for i, ske_info in skeleton_info.items():
        link = ske_info['link']
        pt0, pt1 = link_dict[link[0]], link_dict[link[1]]

        if vis_kpt[pt0] and vis_kpt[pt1]:
            link_color = ske_info['color']
            kpt0 = keypoints[pt0]
            kpt1 = keypoints[pt1]

            img = cv2.line(img, (int(kpt0[0]), int(kpt0[1])),
                           (int(kpt1[0]), int(kpt1[1])),
                           link_color,
                           thickness=line_width)

    return img

def draw_bbox(img, bboxes, bboxes_scores=None, color=None, person_id_list=None, line_width=2):
    green = (0, 255, 0)
    for i, bbox in enumerate(bboxes):
        # Determine the color based on the score if no color is given
        if color is None and bboxes_scores is not None:
            # Scale the score to a color range (green to red)
            score = bboxes_scores[i]
            start_color = np.array([128,128,128],dtype=np.uint8)
            end_color = np.array([128,255,128],dtype=np.uint8)
            box_color = (1 - score) * start_color + score * end_color
        else:
            box_color = color if color is not None else end_color
        
        # Draw the bounding box
        img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                            (int(bbox[2]), int(bbox[3])), box_color, line_width)
        
        green_color = (0,255,0)
        # Display the score at the top-right corner of the bounding box
        if bboxes_scores is not None:
            score_text = f'{bboxes_scores[i]:.2f}'
            text_size, _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = int(bbox[2]) - text_size[0]
            text_y = int(bbox[1]) + text_size[1]
            img = cv2.putText(img, score_text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

        # Display Person ID on the top-right corner edge of the bounding box
        if person_id_list is not None:
            person_id_text = str(person_id_list[i])
            text_size, _ = cv2.getTextSize(person_id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = int(bbox[2]) - text_size[0]
            text_y = int(bbox[1]) - text_size[1]
            img = cv2.putText(img, person_id_text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)
    return img

# with simplification to use onnxruntime only
def draw_skeleton(img,
                  keypoints,
                  scores,
                  kpt_thr=0.5,
                  radius=1,
                  line_width=2):
    num_keypoints = keypoints.shape[1]

    if num_keypoints == 17:
        skeleton = 'coco17'
    else:
        raise NotImplementedError

    skeleton_dict = eval(f'{skeleton}')
    keypoint_info = skeleton_dict['keypoint_info']
    skeleton_info = skeleton_dict['skeleton_info']

    if len(keypoints.shape) == 2:
        keypoints = keypoints[None, :, :]
        scores = scores[None, :, :]

    num_instance = keypoints.shape[0]
    if skeleton in ['coco17']:
        for i in range(num_instance):
            img = draw_mmpose(img, keypoints[i], scores[i], keypoint_info,
                              skeleton_info, kpt_thr, radius, line_width)
    else:
        raise NotImplementedError
    return img

def is_onnx_model(model_path):
    try:
        ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return True
    except Exception as e:
        return False

def is_trt_engine(model_path):
    try:
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes
        engine = EngineFromBytes(BytesFromPath(model_path))
        return engine is not None
    except Exception:
        return False

def get_onnx_input_shapes(model_path):
    from polygraphy.backend.onnx.loader import OnnxFromPath
    from polygraphy.backend.onnx import infer_shapes
    model = OnnxFromPath(model_path)()
    model = infer_shapes(model)
    input_shapes = {inp.name: inp.type.tensor_type.shape for inp in model.graph.input}
    return {name: [dim.dim_value if dim.dim_value > 0 else 'Dynamic' for dim in shape_proto.dim] 
            for name, shape_proto in input_shapes.items()}

def get_trt_input_shapes(model_path):
    input_shapes = {}
    import tensorrt as trt
    with open(model_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        for binding in engine:
            if engine.binding_is_input(binding):
                input_shapes[binding] = engine.get_binding_shape(binding)
    return input_shapes

def get_model_format_and_input_shape(model):
    if is_onnx_model(model):
        model_format = 'onnx'
        input_shape = get_onnx_input_shapes(model)['input']
    elif is_trt_engine(model):
        model_format = 'engine'
        from polygraphy.backend.trt import load_plugins
        load_plugins(plugins=[PLUGIN_LIB_PATHS])
        input_shape = get_trt_input_shapes(model)['input']
    else:
        raise TypeError("Your model is neither ONNX nor Engine !")
    return model_format, input_shape

class RTMO_GPU(object):

    def preprocess(self, img: np.ndarray):
        """Do preprocessing for RTMPose model inference.
        Args:
            img (np.ndarray): Input image in shape.
        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        if len(img.shape) == 3:
            padded_img = np.ones(
                (self.model_input_size[0], self.model_input_size[1], 3),
                dtype=np.uint8) * 114
        else:
            padded_img = np.ones(self.model_input_size, dtype=np.uint8) * 114

        ratio = min(self.model_input_size[0] / img.shape[0],
                    self.model_input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        padded_img[:padded_shape[0], :padded_shape[1]] = resized_img

        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            padded_img = (padded_img - self.mean) / self.std

        return padded_img, ratio

    def postprocess(
        self,
        outputs: List[np.ndarray],
        ratio: float = 1.,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Do postprocessing for RTMO model inference.
        Args:
            outputs (List[np.ndarray]): Outputs of RTMO model.
            ratio (float): Ratio of preprocessing.
        Returns:
            tuple:
            - final_boxes (np.ndarray): Final bounding boxes.
            - final_scores (np.ndarray): Final scores.
        """
        
        if not self.is_yolo_nas_pose:
            # RTMO
            det_outputs, pose_outputs = outputs

            # onnx contains nms module
            pack_dets = (det_outputs[0, :, :4], det_outputs[0, :, 4])
            final_boxes, final_scores = pack_dets
            final_boxes /= ratio
            isscore = final_scores > 0.3
            isbbox = [i for i in isscore]
            final_boxes = final_boxes[isbbox]
            final_boxes_scores = final_scores[isbbox]

            # decode pose outputs
            keypoints, scores = pose_outputs[0, :, :, :2], pose_outputs[0, :, :, 2]
            keypoints = keypoints / ratio

            keypoints = keypoints[isbbox]
            scores = scores[isbbox]
        else:
            # NAS Pose
            flat_predictions = outputs[0]
            if flat_predictions.shape[0] > 0: # at least one person found
                mask = flat_predictions[:, 0] == 0
                final_boxes = flat_predictions[mask, 1:5]
                final_boxes_scores = flat_predictions[mask, 5]
                pred_joints = flat_predictions[mask, 6:].reshape((len(final_boxes), -1, 3))
                keypoints, scores = pred_joints[:,:,:2], pred_joints[:,:,-1]
                keypoints = keypoints / ratio
                final_boxes = final_boxes / ratio
            else: # no detection
                final_boxes, final_boxes_scores, keypoints, scores = np.zeros((0, 4)),np.zeros((0, 1)),np.zeros((0, 17, 2)), np.zeros((0, 17))

        return final_boxes, final_boxes_scores, keypoints, scores

    def inference(self, img: np.ndarray):
            """Inference model.
            Args:
                img (np.ndarray): Input image in shape.
            Returns:
                outputs (np.ndarray): Output of RTMPose model.
            """

            # build input to (1, 3, H, W)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img, dtype=np.float32 if not self.is_yolo_nas_pose else np.uint8)
            input = img[None, :, :, :]

            if self.model_format == 'onnx':

                # Create an IO Binding object
                io_binding = self.session.io_binding()

                if not self.is_yolo_nas_pose:
                    # RTMO
                    io_binding.bind_input(name='input', device_type='cpu', device_id=0, element_type=np.float32, shape=input.shape, buffer_ptr=input.ctypes.data)
                    io_binding.bind_output(name='dets')
                    io_binding.bind_output(name='keypoints')
                else:
                    # NAS Pose, flat format
                    io_binding.bind_input(name='input', device_type='cpu', device_id=0, element_type=np.uint8, shape=input.shape, buffer_ptr=input.ctypes.data)
                    io_binding.bind_output(name='graph2_flat_predictions')

                # Run inference with IO Binding
                self.session.run_with_iobinding(io_binding)

                # Retrieve the outputs from the IO Binding object
                outputs = [output.numpy() for output in io_binding.get_outputs()]

            else: # 'engine'
                if TRT_BACKEND == 'POLYGRAPHY':
                    if not self.session.is_active:
                        self.session.activate()

                    outputs = self.session.infer(feed_dict={'input': input}, check_inputs=False)
                    outputs = [output for output in outputs.values()]
                else: # PYCUDA
                    import pycuda.driver as cuda
                    # Set the input shape dynamically
                    input_shape = input.shape
                    self.context.set_binding_shape(0, input_shape)

                    # Ensure input_data matches the expected shape
                    np.copyto(self.inputs[0]['host'], input.ravel())
                    cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
                    
                    # Run inference
                    self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                    
                    # Transfer predictions back from the GPU
                    for output in self.outputs:
                        cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
                    
                    # Synchronize the stream
                    self.stream.synchronize()
                    
                    # Return only the output values (in their original shapes)
                    outputs = [out['host'].reshape(out['shape']) for out in self.outputs]

            return outputs

    def __exit__(self):
        if self.model_format == 'engine' and TRT_BACKEND == 'POLYGRAPHY':
            if self.session.is_active:
                self.session.deactivate()

    def __call__(self, image: np.ndarray):
            image, ratio = self.preprocess(image)

        
            outputs = self.inference(image)

            bboxes, bboxes_scores, keypoints, scores = self.postprocess(outputs, ratio)

            return bboxes, bboxes_scores, keypoints, scores
    
    def __init__(self,
                 model: str = None,
                 mean: tuple = None,
                 std: tuple = None,
                 device: str = 'cuda',
                 is_yolo_nas_pose = False, 
                 batch_size = 1,
                 plugin_path = PLUGIN_LIB_PATHS):

        self.batch_size = batch_size

        if not os.path.exists(model):
            # If the file does not exist, raise FileNotFoundError
            raise FileNotFoundError(f"The specified ONNX model file was not found: {model}")

        self.model = model
        self.model_format, self.input_shape = get_model_format_and_input_shape(self.model)

        if self.model_format == 'onnx':

            providers = {'cpu': ['CPUExecutionProvider'],
                         'cuda': [
                                 #('TensorrtExecutionProvider', {
                                 # 'trt_fp16_enable':True,
                                 # 'trt_engine_cache_enable':True,
                                 # 'trt_engine_cache_path':'cache'}),
                                 ('CUDAExecutionProvider', {
                                  'cudnn_conv_algo_search': 'DEFAULT',
                                  'cudnn_conv_use_max_workspace': True
                                  }),
                                  'OpenVINOExecutionProvider',
                                  'CPUExecutionProvider']}

            self.session = ort.InferenceSession(path_or_bytes=model,
                                                providers=providers[device])

        else: # 'engine'
            if TRT_BACKEND == 'POLYGRAPHY':
                from polygraphy.backend.common import BytesFromPath
                from polygraphy.backend.trt import EngineFromBytes, TrtRunner
                engine = EngineFromBytes(BytesFromPath(model))
                self.session = TrtRunner(engine)
            else: # PYCUDA
                import tensorrt as trt
                import ctypes
                import pycuda.autoinit
                import pycuda.driver as cuda
                self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
                self.trt_model_path = model
                self.plugin_path = plugin_path

                # Load the custom plugin library
                ctypes.CDLL(self.plugin_path)

                # Load the TensorRT engine
                with open(self.trt_model_path, 'rb') as f:
                    engine_data = f.read()
                
                self.runtime = trt.Runtime(self.TRT_LOGGER)
                self.engine = self.runtime.deserialize_cuda_engine(engine_data)

                if self.engine is None:
                    raise RuntimeError("Failed to load the engine.")

                self.context = self.engine.create_execution_context()

                self.inputs = []
                self.outputs = []
                self.bindings = []
                self.stream = cuda.Stream()

                # Allocate memory for inputs and outputs
                for binding in self.engine:
                    binding_index = self.engine.get_binding_index(binding)
                    shape = self.engine.get_binding_shape(binding_index)
                    if shape[0] == -1:
                        # Handle dynamic batch size by setting max_batch_size
                        shape[0] = self.batch_size
                    size = trt.volume(shape)
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    
                    # Allocate host and device buffers
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    
                    # Append the device buffer to device bindings.
                    self.bindings.append(int(device_mem))
                    
                    # Append to the appropriate list.
                    if self.engine.binding_is_input(binding):
                        self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
                    else:
                        self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

        self.model_input_size = self.input_shape[2:4] # B, C, H, W,
        self.mean = mean
        self.std = std
        self.device = device
        self.is_yolo_nas_pose = is_yolo_nas_pose

        print(f'[I] Detected \'{self.model_format.upper()}\' model', end='')
        print(f', \'{TRT_BACKEND.upper()}\' backend is chosen for inference' if self.model_format == 'engine' else '')
        
class RTMO_GPU_Batch(RTMO_GPU):
    def preprocess_batch(self, imgs: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
        """Process a batch of images for RTMPose model inference.
        Args:
            imgs (List[np.ndarray]): List of input images.
        Returns:
            tuple:
            - batch_img (np.ndarray): Batch of preprocessed images.
            - ratios (List[float]): Ratios used for preprocessing each image.
        """
        batch_img = []
        ratios = []

        for img in imgs:
            preprocessed_img, ratio = super().preprocess(img)
            batch_img.append(preprocessed_img)
            ratios.append(ratio)

        # Stack along the first dimension to create a batch
        batch_img = np.stack(batch_img, axis=0)

        return batch_img, ratios

    def inference(self, batch_img: np.ndarray):
        """Override to handle batch inference.
        Args:
            batch_img (np.ndarray): Batch of preprocessed images.
        Returns:
            outputs (List[np.ndarray]): Outputs of RTMPose model for each image.
        """
        batch_img = batch_img.transpose(0, 3, 1, 2)  # NCHW format
        batch_img = np.ascontiguousarray(batch_img, dtype=np.float32)

        input = batch_img

        if self.model_format == 'onnx':

            # Create an IO Binding object
            io_binding = self.session.io_binding()

            if not self.is_yolo_nas_pose:
                # RTMO
                io_binding.bind_input(name='input', device_type='cpu', device_id=0, element_type=np.float32, shape=input.shape, buffer_ptr=input.ctypes.data)
                io_binding.bind_output(name='dets')
                io_binding.bind_output(name='keypoints')
            else:
                # NAS Pose, flat format
                io_binding.bind_input(name='input', device_type='cpu', device_id=0, element_type=np.uint8, shape=input.shape, buffer_ptr=input.ctypes.data)
                io_binding.bind_output(name='graph2_flat_predictions')

            # Run inference with IO Binding
            self.session.run_with_iobinding(io_binding)

            # Retrieve the outputs from the IO Binding object
            outputs = [output.numpy() for output in io_binding.get_outputs()]

        else: # 'engine'
            if TRT_BACKEND == 'POLYGRAPHY':
                if not self.session.is_active:
                    self.session.activate()

                outputs = self.session.infer(feed_dict={'input': input}, check_inputs=False)
                outputs = [output for output in outputs.values()]
            else: # PYCUDA
                import pycuda.driver as cuda
                # Set the input shape dynamically
                input_shape = input.shape
                self.context.set_binding_shape(0, input_shape)

                # Ensure input_data matches the expected shape
                np.copyto(self.inputs[0]['host'], input.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
                
                # Run inference
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                
                # Transfer predictions back from the GPU
                for output in self.outputs:
                    cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
                
                # Synchronize the stream
                self.stream.synchronize()
                
                # Return only the output values (in their original shapes)
                outputs = [out['host'].reshape(out['shape']) for out in self.outputs]

        return outputs

    def postprocess_batch(
        self,
        outputs: List[np.ndarray],
        ratios: List[float]
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Process outputs for a batch of images.
        Args:
            outputs (List[np.ndarray]): Outputs from the model for each image.
            ratios (List[float]): Ratios used for preprocessing each image.
        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: keypoints and scores for each image.
        """
        batch_keypoints = []
        batch_scores = []
        batch_bboxes = []
        batch_bboxes_scores = []

        b_dets, b_keypoints = outputs
        for i, ratio in enumerate(ratios):
            output = [np.expand_dims(b_dets[i], axis=0), np.expand_dims(b_keypoints[i],axis=0)]
            bboxes, bboxes_scores, keypoints, scores = super().postprocess(output, ratio)
            batch_keypoints.append(keypoints)
            batch_scores.append(scores)
            batch_bboxes.append(bboxes)
            batch_bboxes_scores.append(bboxes_scores)

        return batch_bboxes, batch_bboxes_scores, batch_keypoints, batch_scores

    def __batch_call__(self, images: List[np.ndarray]):
        batch_img, ratios = self.preprocess_batch(images)
        outputs = self.inference(batch_img)
        bboxes, bboxes_scores, keypoints, scores = self.postprocess_batch(outputs, ratios)
        return bboxes, bboxes_scores, keypoints, scores
    
    def free_unused_buffers(self, activate_cameras_ids: List):
        for camera_id in list(self.buffers.keys()):
            if camera_id not in activate_cameras_ids:
                del self.buffers[camera_id]
                del self.in_queues[camera_id]
                del self.out_queues[camera_id]
                if DEBUG:
                    print(f'RTMO buffers to camera "{camera_id}" got freed.', flush=True)

    def __call__(self, image: np.array, camera_id = 0):

        # initialize dedicated buffers & queues for camera with id "camera_id"
        if camera_id not in self.buffers:
            self.buffers[camera_id] = []
            self.in_queues[camera_id] = Queue(maxsize=self.batch_size)
            self.out_queues[camera_id] = Queue(maxsize=self.batch_size)
            if DEBUG:
                print(f'RTMO buffers to camera "{camera_id}" are created.', flush=True)


        in_queue = self.in_queues[camera_id]
        out_queue = self.out_queues[camera_id]
        self.buffers[camera_id].append(image)
        in_queue.put(image)

        if len(self.buffers[camera_id]) == self.batch_size:
            b_bboxes, b_bboxes_scores, b_keypoints, b_scores = self.__batch_call__(self.buffers[camera_id])
            for i, (keypoints, scores) in enumerate(zip(b_keypoints, b_scores)):
                bboxes = b_bboxes[i]
                bboxes_scores = b_bboxes_scores[i]
                out_queue.put((bboxes, bboxes_scores, keypoints, scores))
            self.buffers[camera_id] = []

        frame, bboxes, bboxes_scores, keypoints, scores = None, None, None, None, None
        if not out_queue.empty():
            bboxes, bboxes_scores, keypoints, scores = out_queue.get()
            frame = in_queue.get()
    
        return frame, bboxes, bboxes_scores, keypoints, scores

    
    def __init__(self,
                 model: str = None,
                 mean: tuple = None,
                 std: tuple = None,
                 device: str = 'cuda',
                 is_yolo_nas_pose = False,
                 plugin_path = PLUGIN_LIB_PATHS,
                 batch_size: int = 1):
        super().__init__(model, 
                         mean, 
                         std, 
                         device, 
                         is_yolo_nas_pose, 
                         batch_size,
                         plugin_path)
        
        self.in_queues = dict()
        self.out_queues = dict()
        self.buffers = dict()

def resize_to_fit_screen(image, screen_width, screen_height):
    # Get the dimensions of the image
    h, w = image.shape[:2]

    # Calculate the aspect ratio of the image
    aspect_ratio = w / h

    # Determine the scaling factor
    scale = min(screen_width / w, screen_height / h)

    # Calculate the new dimensions
    new_width = int(w * scale)
    new_height = int(h * scale)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image
