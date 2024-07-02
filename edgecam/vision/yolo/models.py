# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


import gc
from typing import Dict

import torch
import numpy as np
import ultralytics


Image = np.ndarray
Results = Dict[str, np.ndarray]


class Yolo:

    def __init__(self):
        self._model = None
        self._infer = None
        self.tracking = False

    def load(self, pt: str='yolov8n.pt', tracking: bool=False):
        self._model = ultralytics.YOLO(pt)
        self.tracking = tracking

    @property
    def tracking(self) -> bool:
        return self._tracking

    @tracking.setter
    def tracking(self, turn_on: bool):
        if turn_on:
            fn = lambda x: self._model.track(x, persist=True, verbose=False)
        else:
            fn = lambda x: self._model.predict(x, verbose=False)
        self._tracking = turn_on
        self._infer = fn

    def infer(self, input: Image) -> Results:
        out = self._infer(input)
        if out is None:
            boxes = np.array([])
        else:
            boxes = out[0].boxes.data.cpu().numpy()
        return {'boxes': boxes}

    def release(self):
        if next(self._model.parameters()).device.type == 'cuda':
            self._model.to('cpu')
            torch.cuda.empty_cache()
        del self._model
        gc.collect()


class YoloPose(Yolo):

    def load(self, pt: str='yolov8n-pose.pt', tracking: bool=False):
        self.load(pt, tracking)
    
    def infer(self, input: Image) -> Results:
        out = self._infer(input)
        if out is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = out[0].boxes.data.cpu().numpy()
            kptss = out[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}
