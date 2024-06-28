# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import threading

import numpy as np
from loguru import logger

from edgecam.utils.buffers import PushQueue
from edgecam.utils.skippers import StepSkipper
from edgecam.vision_ai.yolo.models import YOLO, DetYOLO, EstYOLO


Callback = typing.Callable[[typing.Optional[float]], np.ndarray]
Frame = np.ndarray
Preds = typing.Dict[str, np.ndarray]
Result = typing.Tuple[Frame, Preds]


class Inference:

    def __init__(self, model: YOLO, maxsize: int=1, stepsize: int=3):
        self._model = model
        self._buf = PushQueue(maxsize)
        self._th: threading.Thread=None
        self._stop_th = False
        self._skipper = StepSkipper(stepsize)

    def inference(self, fetch_frame: Callback, timeout: float=10.0):
        if not self._th or not self._th.is_alive():
            if self._stop_th:
                self._stop_th = False
            self._buf.flush()
            self._th = threading.Thread(target=self._buffer, args=[fetch_frame, timeout])
            self._th.start()

    def _buffer(self, fetch_frame: Callback, timeout: float):
        try:
            preds = np.array([])
            while not self._stop_th:
                frame = fetch_frame(timeout)
                if not next(self._skipper):
                    preds = self._model.infer(frame)
                self._buf.push((frame, preds))
        except:
            logger.exception('Aborted.')
        finally:
            if not self._stop_th:
                self._stop_task = True

    def release(self):
        if self._th and self._th.is_alive():
            self._stop_th = True
            self._th.join()
        # NOTE Need to check it.
        # self._model.release()

    def fetch_result(self, timeout: float=10.0) -> Result:
        result = self._buf.get(timeout)
        return result

    def alter_maxsize(self, maxsize: int):
        self._buf.maxsize = maxsize


class ObjectDetectorYolov8(Inference):

    def __init__(self,
                 model_pt: str='yolov8m.pt',
                 maxsize: int=1,
                 tracking: bool=False):
        super().__init__(DetYOLO(model_pt, tracking), maxsize)


class PoseEstimatorYolov8(Inference):

    def __init__(self,
                 model_pt: str='yolov8m-pose.pt',
                 maxsize: int=1,
                 tracking: bool=False):
        super().__init__(EstYOLO(model_pt, tracking), maxsize)
