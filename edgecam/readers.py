# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


import threading
from typing import Union

import cv2
import numpy as np


class FailedOpen(Exception): pass
class FailedRead(Exception): pass


Source = Union[int, str]
Frame = np.ndarray


class VideoReader:

    def __init__(self, source: Source=None, api_pref: int=cv2.CAP_ANY):
        self.mutex = threading.Lock()
        self._cap = cv2.VideoCapture()
        self._cap.setExceptionMode(enable=True)
        if source is not None:
            self.open(source, api_pref)

    def open(self, source: Source, api_pref: int=cv2.CAP_ANY):
        with self.mutex:
            try:
                self._cap.open(source, api_pref)
            except Exception as e:
                raise FailedOpen from e

    def read(self) -> Frame:
        try:
            with self.mutex:
                _, frame = self._cap.read()
        except Exception as e:
            raise FailedRead from e
        else:
            return frame

    def release(self):
        with self.mutex:
            self._cap.release()
