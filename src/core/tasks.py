# -*- coding: utf-8 -*-
# Author: SeungHyeon Kim

import abc
import queue
import typing
import threading

import cv2
import numpy as np
from loguru import logger

from src.utils.captures import VideoCapture
from src.utils.buffers import StreamQueue
from src.utils.ai.vision.models import TorchModel


class BackgroundTask(abc.ABC):

    def __init__(self) -> None:
        self._th = None
        self._stop_th = False

    def start(self) -> None:
        if self.is_running():
            raise RuntimeError('Task thread is already running.')
        self._th = threading.Thread(target=self._start)
        self._th.start()

    @abc.abstractmethod
    def _start(self) -> None:
        """ NOTE: implementation pattern
            ----------------------------
            try:
                while not self._stop_th:
                    TODO
            except:
                logger.exception('Error')
            finally:
                if not self._stop_th:
                    self._stop_th = True
                    self._stop()
        """
        pass

    def stop(self) -> None:
        if not self.is_running():
            raise RuntimeError('Task thread is not running.')
        self._stop_th = True
        self._th.join()
        self._stop()

    @abc.abstractmethod
    def _stop(self) -> None:
        pass

    def is_running(self) -> bool:
        return self._th and self._th.is_alive()


class VideoCaptureTask(BackgroundTask):

    def __init__(self, cap: VideoCapture, buf: StreamQueue) -> None:
        super().__init__()
        self._cap = cap
        self._buf = buf

    def _start(self) -> None:
        try:
            while not self._stop_th:
                frame = self._cap.read()
                self._buf.push(frame)
        except:
            logger.exception('Error')
        finally:
            if not self._stop_th:
                self._stop_th = True
                self._stop()

    def _stop(self) -> None:
        self._cap.release()


# NOTE: temporary unused.
# -----
# class AIPredictionTask(BackgroundTask):
#
#     def __init__(self, model: TorchModel,
#                  frame_buf: StreamQueue,
#                  model_buf: StreamQueue) -> None:
#         super().__init__()
#         self._model = model
#         self._frame_buf = frame_buf
#         self._model_buf = model_buf
#
#     def _start(self) -> None:
#         try:
#             while not self._stop_th:
#                 frame = self._frame_buf.get(timeout=10)
#                 preds = self._model.predict(frame)
#                 self._model_buf.push((frame, preds))
#         except:
#             logger.exception('Error')
#         finally:
#             if not self._stop_th:
#                 self._stop_th = True
#                 self._stop()
#
#     def _stop(self) -> None:
#         self._model.release()
