# -*- coding: utf-8 -*-
# Author: SeungHyeon Kim

import gc
from typing import Any, Dict
from abc import ABC, abstractmethod

import torch
import numpy as np
from ultralytics import YOLO


class NotTorchModule(TypeError): pass


class TorchModule(ABC):
    """ Interface """

    def __init__(self, model: Any):
        if not isinstance(model, torch.nn.Module):
            raise NotTorchModule
        self._model = model

    @abstractmethod
    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        pass

    def release(self):
        if next(self._model.parameters()).device.type == 'cuda':
            self._model.to('cpu')
            torch.cuda.empty_cache()
        del self._model
        gc.collect()


class Identity(TorchModule):
    def __init__(self):
        pass

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        return {'empty': np.ndarray([])}

    def release(self):
        pass


class Yolov8n(TorchModule):
    categories = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                  4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
                  8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                  11: 'stop sign', 12: 'parking meter', 13: 'bench',
                  14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
                  18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
                  22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
                  26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                  30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
                  34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
                  37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                  40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
                  44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                  48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
                  52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
                  56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                  60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
                  64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
                  68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                  72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
                  76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
                  79: 'toothbrush'}

    def __init__(self):
        super().__init__(YOLO("yolov8n.pt"))

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        # BOX: x_min, y_min, x_max, y_max, box_id, box_conf, box_category_id
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
        return {"boxes": boxes}


class Yolov8nPose(TorchModule):
    keypoints = {0: 'Nose', 1: 'Left eye', 2: 'Right eye', 3: 'Left ear',
                 4: 'Right ear', 5: 'Left shoulder', 6: 'Right shoulder',
                 7: 'Left elbow', 8: 'Right elbow', 9: 'Left wrist',
                 10: 'Right wrist', 11: 'Left hip', 12: 'Right hip',
                 13: 'Left knee', 14: 'Right knee', 15: 'Left ankle',
                 16: 'Right ankle'}

    connections = {0: [1, 2,], 1: [3,], 2: [4,], 3: [], 4: [], 5: [6, 7, 11,],
                   6: [8, 12,], 7: [9,], 8: [10,], 9: [], 10: [], 11: [12, 13,],
                   12: [14,], 13: [15,], 14: [16,], 15: [], 16: []}

    def __init__(self):
        super().__init__(YOLO('yolov8n-pose.pt'))

    def predict(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        # BOX: x_min, y_min, x_max, y_max, box_id, box_conf, box_category_id
        # KPTS: x, y, conf
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
            kptss = results[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}