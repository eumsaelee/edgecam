# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import threading

import numpy as np
from loguru import logger

from edgecam.vision.models import Yolov8
from edgecam.utils.buffers import PushQueue
from edgecam.utils.tasks import SingleThreadTask, TaskError


Callback = typing.Callable[[typing.Optional[float]], np.ndarray]
Frame = np.ndarray
Preds = typing.Dict[str, np.ndarray]
Output = typing.Tuple[Frame, Preds]


class ObjectDetectionTask(SingleThreadTask):

    def __init__(self, model: Yolov8, buf: PushQueue) -> None:
        self._model = model
        self._buf = buf
        super().__init__()

    def start(self, callback: Callback) -> None:
        if self.is_running():
            raise RuntimeError('Task is already running.')
        self._task = threading.Thread(target=self._start, args=[callback])
        self._task.start()

    def _start(self, callback: Callback) -> None:
        try:
            while not self._stop_task:
                frame = callback()
                preds = self._model.predict(frame)
                self._buf.push((frame, preds))
        except:
            logger.exception('Task has been aborted.')
        finally:
            if not self._stop_task:
                self._stop_task = True
                self._stop()

    def _stop(self) -> None:
        self._model.release()


class ObjectDetectionHandler:

    def __init__(self, model_name: str, maxsize: int=10) -> None:
        self._model = Yolov8(model_name)
        self._buf = PushQueue(maxsize)
        self._task = ObjectDetectionTask(self._model, self._buf)

    def change_maxsize(self, maxsize: int) -> None:
        self._buf.maxsize = maxsize

    def start_detection(self, callback: Callback) -> None:
        try:
            self._task.start(callback)
        except TaskError:
            logger.warning(
                'Object detection thread is already running.')

    def stop_detection(self) -> None:
        try:
            self._task.stop()
        except TaskError:
            logger.warning(
                'Object detection thread is not running.')

    def read_output(self, timeout: float=30.0) -> Output:
        try:
            output = self._buf.get(timeout)
            return output
        except:
            logger.warning(
                'Failed to read output.')
