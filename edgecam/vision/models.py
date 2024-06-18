# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import gc
import typing
import abc

import torch
import numpy as np
import ultralytics


Preds = typing.Dict[str, np.ndarray]


class PytorchModel(abc.ABC):
    """ 파이토치 모델 인터페이스 """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> Preds:
        pass

    def release(self) -> None:
        if next(self._model.parameters()).device.type == 'cuda':
            self._model.to('cpu')
            torch.cuda.empty_cache()
        del self._model
        gc.collect()


class Yolo(PytorchModel):
    """ YOLO 객체 탐지 모델 """

    def __init__(self, model_name: str="yolov8m.pt") -> None:
        super().__init__(ultralytics.YOLO(model_name))

    def predict(self, frame: np.ndarray) -> Preds:
        """
        NOTE: boxes
        -----
        boxes.ndim: 2
        boxes.shape: (n, 7)  # 'n' is number of objects
        box columns:
            x_min, y_min, x_max ,y_max, box_id, box_conf, category_id
        """
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
        return {"boxes": boxes}


class YoloPose(PytorchModel):
    """ YOLO 자세 추정 모델 """

    def __init__(self, model_name: str='yolov8m-pose.pt'):
        super().__init__(ultralytics.YOLO(model_name))

    def predict(self, frame: np.ndarray) -> Preds:
        """
        NOTE: boxes
        -----
        boxes.ndim: 2
        boxes.shape: (n, 7)  # 'n' is number of persons
        box columns:
            x_min, y_min, x_max ,y_max, box_id, box_conf, category_id

        NOTE: kptss
        -----
        kptss.ndim: 3
        kptss.shape: (n, 17, 3)  # 'n' is number of persons
        kpts rows:
            nose, left eye, right eye, ..., right ankle
        kpts columns:
            x, y, conf
        """
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
            kptss = results[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}