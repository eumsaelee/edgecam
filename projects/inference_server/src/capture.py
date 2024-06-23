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


class VideoCapture:

    def __init__(self,
                 source: typing.Union[int, str]=None,
                 api_pref: int=cv2.CAP_ANY,
                 maxsize: int=1):
        self.mutex = threading.Lock()
        self._buf = PushQueue(maxsize)
        self._th: threading.Thread=None
        self._stop_th = False
        self._cap = cv2.VideoCapture()
        self._cap.setExceptionMode(enable=True)
        if source is not None:
            self.open(source, api_pref)

    def open(self,
             source: typing.Union[int, str],
             api_pref: int=cv2.CAP_ANY):
        with self.mutex:
            self._cap.open(source, api_pref)
        if not self._th or not self._th.is_alive():
            if self._stop_th:
                self._stop_th = False
            self._buf.flush()
            self._th = threading.Thread(target=self._buffer)
            self._th.start()

    def _buffer(self):
        try:
            while not self._stop_th:
                _, frame = self._cap.read()
                self._buf.push(frame)
        except:
            logger.exception('Aborted.')
        finally:
            if not self._stop_th:
                self._stop_th = True

    def release(self):
        if self._th and self._th.is_alive():
            self._stop_th = True
            self._th.join()
        self._cap.release()

    def fetch_frame(self, timeout: float=10.0) -> np.ndarray:
        with self.mutex:
            frame = self._buf.get(timeout)
            return frame

    def alter_maxsize(self, maxsize: int):
        self._buf.maxsize = maxsize
