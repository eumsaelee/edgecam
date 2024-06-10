import threading
from typing import Union

import cv2
import numpy as np


class FailedOpen(Exception):pass
class FailedRead(Exception):pass


class FrameReader:
    def __init__(self,
                 source: Union[int, str]=None,
                 api_pref: int=cv2.CAP_ANY):
        self._cap = cv2.VideoCapture()
        self._cap.setExceptionMode(enable=True)
        self.mutex = threading.Lock()
        if source is not None:
            self.open(source, api_pref)

    def open(self,
             source: Union[int, str],
             api_pref: int=cv2.CAP_ANY):
        try:
            with self.mutex:
                self._cap.open(source, api_pref)
        except Exception as e:
            raise FailedOpen(
                'Failed to open the video source.') from e

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self):
        with self.mutex:
            self._cap.release()

    def read(self) -> np.ndarray:
        try:
            with self.mutex:
                _, frame = self._cap.read()
            return frame
        except Exception as e:
            raise FailedRead(
                'Failed to read the next frame.') from e
