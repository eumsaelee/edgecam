# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import abc
import typing
import threading

import cv2
import numpy as np


class FailedOpen(Exception):
    """ Raises when failing to open the video source. """
    pass


class FailedRead(Exception):
    """ Raises when failing to read next frame. """
    pass


Source = typing.Union[int, str]


class VideoCapture(abc.ABC):

    @abc.abstractmethod
    def open(self) -> None:
        pass

    @abc.abstractmethod
    def release(self) -> None:
        pass

    @abc.abstractmethod
    def is_opened(self) -> bool:
        pass

    @abc.abstractmethod
    def read(self) -> np.ndarray:
        pass


class FrameCapture(VideoCapture):

    def __init__(self, source: Source=None, api_pref: int=cv2.CAP_ANY) -> None:
        self._cap = cv2.VideoCapture()
        self._cap.setExceptionMode(enable=True)
        self.mutex = threading.Lock()
        if source is not None:
            self.open(source, api_pref)

    def open(self, source: Source, api_pref: int=cv2.CAP_ANY) -> None:
        try:
            with self.mutex:
                self._cap.open(source, api_pref)
        except Exception as e:
            raise FailedOpen('Failed to open the video source.') from e

    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def release(self) -> None:
        with self.mutex:
            self._cap.release()

    def read(self) -> np.ndarray:
        try:
            with self.mutex:
                _, frame = self._cap.read()
            return frame
        except Exception as e:
            raise FailedRead('Failed to read the next frame.') from e
