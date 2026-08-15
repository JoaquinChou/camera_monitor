from typing import List, Tuple
import cv2
import numpy as np
import math
import onnxruntime
import time

from utils import (letterbox, sigmoid,
                    read_json)

class Yolo26Detection:
    def __init__(self, config_file: str = None, gpu_id: int = -1) -> None:
        super().__init__()
        self.config = read_json(config_file)
        self.gpu_id = gpu_id
        self.conf_th = self.config["AlgorithmConfig"].get("multi_threshold", [0.1])
        self.class_num = self.config["AlgorithmConfig"].get("class_num", None)
        self.class_names = self.config["AlgorithmConfig"].get("class_names", None)
        self.min_w = self.config["AlgorithmConfig"].get("min_size_w", 15)
        self.min_h = self.config["AlgorithmConfig"].get("min_size_h", 20)
        self.input_w = self.config["ModelConfig"]["inputs"][0].get("width", None)
        self.input_h = self.config["ModelConfig"]["inputs"][0].get("height", None)
        self.mean = self.config["AlgorithmConfig"].get("mean", None)
        self.std = self.config["AlgorithmConfig"].get("std", None)
        self.ratio = None
        self.dxdy = None
        # Initialize onnxruntime
        self.session = None
        self.onnx_init()

    def onnx_init(self) -> None:
        onnxruntime.set_default_logger_severity(3)
        onnx_path = self.config['model_path']
        if self.gpu_id is None or self.gpu_id < 0:
            providers = ['CPUExecutionProvider']
        else:
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': self.gpu_id,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider',
            ]
        self.session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    def infer(self, img: np.ndarray = None, return_time=False) -> List:
        """
        Execute model inference on input img.

        This function contains the complete inference pipeline:
        1. img preprocessing
        2. ONNX Runtime inference
        3. Result postprocessing

        Args:
            img (np.ndarray):
                Input raw img (usually BGR format img read by OpenCV)

        Returns:
            List:
                Postprocessed inference results (detection boxes, classes, confidence scores, etc.)
        """
        if img is None:
            raise ValueError("Input img is None")

        t_total_start = time.perf_counter()

        # 1️⃣ preprocess
        t0 = time.perf_counter()
        ort_inputs, ratio, dxdy = self.pre_process(img)
        self.ratio = ratio
        self.dxdy = dxdy
        t1 = time.perf_counter()

        # 2️⃣ inference
        t2 = time.perf_counter()
        outputs = self.session.run(None, ort_inputs)
        t3 = time.perf_counter()

        # 3️⃣ postprocess
        t4 = time.perf_counter()
        results = self.post_process(outputs, img)
        t5 = time.perf_counter()

        if return_time:
            return results, {
                "pre": t1 - t0,
                "infer": t3 - t2,
                "post": t5 - t4,
                "total": t5 - t_total_start
            }

        return results

    def pre_process(self, img: np.ndarray = None, bgr2rgb: bool = False) -> Tuple[dict, float, Tuple[int, int]]:
        """
        img preprocessing function before model inference.

        Args:
            img (np.ndarray, optional):
                Input img (usually BGR format img read by OpenCV).
            bgr2rgb (bool, optional):
                Whether to convert img from BGR to RGB, default True.

        Returns:
            Tuple:
                ort_inputs (dict):
                    Input dictionary that can be directly fed to ONNX Runtime
                ratio (float):
                    img scaling ratio (used for coordinate restoration in postprocessing)
                dxdy (Tuple[int, int]):
                    Pixel padding values in width and height directions during letterbox
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if bgr2rgb else img
        img, ratio, dxdy = letterbox(img.copy(), new_shape=(self.input_h, self.input_w))        
        img = (img - np.array(self.mean)) * np.array(self.std)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        features = {self.session.get_inputs()[0].name: img[None, :, :, :]}
        return features, ratio, dxdy

    def post_process(self, features: List[np.ndarray] = None, img: np.ndarray = None) -> List:
        """
        Postprocess model inference output to generate final detection results.

        Args:
            features (List[np.ndarray]):
                Raw model output features (ONNX Runtime inference output)
            img (np.ndarray):
                Original input img used to get img dimensions and restore coordinates

        Returns:
            List:
                Detection results list, each detection result contains:
                    - x0 (int): Top-left corner x coordinate of bounding box
                    - y0 (int): Top-left corner y coordinate of bounding box
                    - x1 (int): Bottom-right corner x coordinate of bounding box
                    - y1 (int): Bottom-right corner y coordinate of bounding box
                    - class_name (str): Detection class name
                    - score (float): Confidence score
                    - det_type (str): Detection type ("high_threshold" or "low_threshold")
        """
        img_h, img_w = img.shape[:2]
        features = [np.ascontiguousarray(feature[0].transpose(1, 2, 0)) for feature in features]

        # YOLO26 remove NMS
        det_boxes, det_scores, det_labels = self.__yolo26_decode(features, conf_thres=0.05, num_labels=self.class_num)

        ratio_w, ratio_h = self.ratio
        result = []
        for box, score, label in zip(det_boxes, det_scores, det_labels):
            x0, y0 = box[:2]
            x1 = x0 + box[2]
            y1 = y0 + box[3]
            if score < self.conf_th[self.class_num:][label]:
                continue
            if score > self.conf_th[0:self.class_num][label]:
                det_type = "high_threshold"
            else:
                det_type = "low_threshold"
            if self.dxdy is not None:
                x0 = math.floor(min(max((x0 - self.dxdy[0]) / ratio_w, 1), img_w - 1))
                y0 = math.floor(min(max((y0 - self.dxdy[1]) / ratio_h, 1), img_h - 1))
                x1 = math.ceil(min(max((x1 - self.dxdy[0]) / ratio_w, 1), img_w - 1))
                y1 = math.ceil(min(max((y1 - self.dxdy[1]) / ratio_h, 1), img_h - 1))
            else:
                x0 = math.floor(min(max(x0 / ratio_w, 1), img_w - 1))
                y0 = math.floor(min(max(y0 / ratio_h, 1), img_h - 1))
                x1 = math.ceil(min(max(x1 / ratio_w, 1), img_w - 1))
                y1 = math.ceil(min(max(y1 / ratio_h, 1), img_h - 1))

            if (x1 - x0) < self.min_w or (y1 - y0) < self.min_h:
                continue
            result.append([x0, y0, x1, y1, self.class_names[label], score, det_type])

        return result

    def __yolo26_decode(self, feats: List[np.ndarray], conf_thres: float, num_labels: int = 80, **kwargs) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Decode YOLO26 detection output features into bounding boxes, confidence scores, and class labels.
        YOLO26 removes DFL and NMS, so box predictions are 4 direct distance value instead of 4*reg_max distribution values.
        Args:
            feats (List[np.ndarray]): List of feature maps from different scales. Each feature map
                has shape (H, W, num_labels + 4), where the last dimension contains:
                - num_labels: class confidence scores
                - 4: bounding box coordinates (x, y, w, h) in delta format
            conf_thres (float): Confidence threshold for filtering detections.
            num_labels (int, optional): Number of object classes. Default is 80 (COCO dataset).
            **kwargs: Additional keyword arguments (unused, reserved for future extension).
        Returns:
            Tuple:
                boxes_pro (List[np.ndarray]):
                    Decoded bounding boxes list in format [x0, y0, width, height]
                scores_pro (List[float]):
                    Confidence scores corresponding to each bounding box
                labels_pro (List[int]):
                    Class indices corresponding to each bounding box
        """
        scores_pro = []
        boxes_pro = []
        labels_pro = []
        for i, feat in enumerate(feats):
            stride = 8 << i
            score_feat, box_feat = np.split(feat, [num_labels], -1)
            score_feat = sigmoid(score_feat)
            _argmax = score_feat.argmax(-1)
            _max = score_feat.max(-1)

            indices = np.where(_max > conf_thres)
            hIdx, wIdx = indices
            num_proposal = hIdx.size
            if not num_proposal:
                continue

            scores = _max[hIdx, wIdx]
            # YOLO26: direct 4-value box prediction, no DFL
            boxes = box_feat[hIdx, wIdx]
            labels = _argmax[hIdx, wIdx]
            num_proposal = hIdx.size
            if not num_proposal:
                continue

            for k in range(num_proposal):
                score = scores[k]
                label = labels[k]

                x0, y0, x1, y1 = boxes[k]

                x0 = (wIdx[k] + 0.5 - x0) * stride
                y0 = (hIdx[k] + 0.5 - y0) * stride
                x1 = (wIdx[k] + 0.5 + x1) * stride
                y1 = (hIdx[k] + 0.5 + y1) * stride

                w = x1 - x0
                h = y1 - y0

                scores_pro.append(float(score))
                boxes_pro.append(np.array([x0, y0, w, h], dtype=np.float32))
                labels_pro.append(int(label))
        
        return boxes_pro, scores_pro, labels_pro