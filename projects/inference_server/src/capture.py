# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import threading

import cv2
import numpy as np
from loguru import logger

from edgecam.utils.buffers import PushQueue
from edgecam.utils.tasks import SingleThreadTask, TaskError


Source = typing.Union[int, str]


class VideoCapture:

    def __init__(self,
                 source: Source=None,
                 api_pref: int=cv2.CAP_ANY) -> None:
        self.mutex = threading.Lock()
        self._cap = cv2.VideoCapture()
        self._cap.setExceptionMode(enable=True)
        if source is not None:
            self.open(source, api_pref)

    def open(self,
             source: Source,
             api_pref: int=cv2.CAP_ANY) -> None:
        with self.mutex:
            self._cap.open(source, api_pref)

    def release(self) -> None:
        with self.mutex:
            self._cap.release()

    def read(self) -> np.ndarray:
        with self.mutex:
            _, frame = self._cap.read()
            return frame


class VideoCaptureTask(SingleThreadTask):

    def __init__(self, cap: VideoCapture, buf: PushQueue) -> None:
        self._cap = cap
        self._buf = buf
        super().__init__()

    def _start(self) -> None:
        try:
            while not self._stop_task:
                frame = self._cap.read()
                self._buf.push(frame)
        except:
            logger.exception('Task has been aborted.')
        finally:
            if not self._stop_task:
                self._stop_task = True
                self._stop()

    def _stop(self) -> None:
        self._cap.release()


class VideoCaptureHandler:

    def __init__(self,
                 source: Source,
                 api_pref: int=cv2.CAP_ANY,
                 maxsize: int=10) -> None:
        self._cap = VideoCapture(source, api_pref)
        self._buf = PushQueue(maxsize)
        self._task = VideoCaptureTask(self._cap, self._buf)

    def change_source(self,
                      source: Source,
                      api_pref: int=cv2.CAP_ANY) -> None:
        self._cap.open(source, api_pref)

    def change_maxsize(self, maxsize: int) -> None:
        self._buf.maxsize = maxsize

    def start_capturing(self) -> None:
        try:
            self._task.start()
        except TaskError:
            logger.warning(
                'Video capture thread is already running.')

    def stop_capturing(self) -> None:
        try:
            self._task.stop()
        except TaskError:
            logger.warning(
                'Video capture thread is not running.')

    def read_frame(self, timeout: float=30.0) -> np.ndarray:
        try:
            frame = self._buf.get(timeout)
            return frame
        except:
            logger.warning(
                'Failed to read frame.')
