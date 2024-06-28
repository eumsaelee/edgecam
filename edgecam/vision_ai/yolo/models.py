# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import gc
import abc

import torch
import numpy as np
import ultralytics

from edgecam.vision_ai.common import CommonModel
from edgecam.vision_ai.common import InputImage, InferenceResults


class YOLO(CommonModel):

    def __init__(self, pt: str, tracking: bool):
        self._model = ultralytics.YOLO(pt)
        self.tracking = tracking

    @abc.abstractmethod
    def infer(self, img: InputImage) -> InferenceResults:
        pass

    def release(self):
        if next(self._model.parameters()).device.type == 'cuda':
            self._model.to('cpu')
            torch.cuda.empty_cache()
        del self._model
        gc.collect()


class DetYOLO(YOLO):

    def __init__(self, pt: str='yolov8n.pt', tracking: bool=False):
        super().__init__(pt, tracking)

    def infer(self, img: InputImage) -> InferenceResults:
        if self.tracking:
            output = self._model.track(img, persist=True, verbose=False)
        else:
            output = self._model.predict(img, verbose=False)
        if output is None:
            boxes = np.array([])
        else:
            boxes = output[0].boxes.data.cpu().numpy()
        return {'boxes': boxes}


class EstYOLO(YOLO):

    def __init__(self, pt: str='yolov8n-pose.pt', tracking: bool=False):
        super().__init__(pt, tracking)

    def infer(self, img: InputImage) -> InferenceResults:
        if self.tracking:
            output = self._model.track(img, persist=True, verbose=False)
        else:
            output = self._model.predict(img, verbose=False)
        if output is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = output[0].boxes.data.cpu().numpy()
            kptss = output[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}
