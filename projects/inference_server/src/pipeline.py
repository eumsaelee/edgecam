# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

""" 애드혹 파이프라인 - 리팩토링 필요 """

from typing import Union, Callable, Any
from dataclasses import dataclass

import numpy as np

from edgecam.tasks import SingleThreadTask
from edgecam.readers import VideoReader
from edgecam.buffers import SyncEvectingQueue
from edgecam.vision.yolo.models import Yolo


@dataclass
class Config:
    video_source: Union[int, str]
    video_api_pref: int
    video_buffersize: int
    yolo_pt_name: str
    yolo_tracking_on: bool
    yolo_buffersize: int
    buffer_timeout: float


class _Broker(SingleThreadTask):
    def __init__(self, src: Any, dst: Any):
        self._src = src
        self._dst = dst
        super().__init__()

    def start(self, hooker: Callable[[Any], Any], timeout: float):
        super().start(lambda: self._dst.put(hooker(self._src.get(timeout))))


class _VideoReader(VideoReader):
    def get(self, timeout: Any=None) -> np.ndarray:
        return self.read()


class Pipeline:
    def __init__(self):
        self.reader = _VideoReader()
        self.reader_buffer = SyncEvectingQueue()
        self.yolo = Yolo()
        self.yolo_buffer = SyncEvectingQueue()
        self._reader_broker = _Broker(self.reader, self.reader_buffer)
        self._yolo_broker = _Broker(self.reader_buffer, self.yolo_buffer)

    def start(self, config: Config):
        self.reader.open(config.video_source, config.video_api_pref)
        self.reader_buffer.maxsize = config.video_buffersize
        self.yolo.load(config.yolo_pt_name, config.yolo_tracking_on)
        self.yolo_buffer.maxsize = config.yolo_buffersize
        self._reader_broker.start(
            lambda x: x, config.buffer_timeout)
        self._yolo_broker.start(
            lambda x: (x, self.yolo.infer(x)), config.buffer_timeout)

    def stop(self):
        self._yolo_broker.stop()
        self.yolo.release()
        self._reader_broker.stop()
        self.reader.close()
