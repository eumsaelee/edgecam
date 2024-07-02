# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


from abc import ABC, abstractmethod
import asyncio
import threading
import websockets
from typing import Union, Any

import cv2
import numpy as np


class FailedOpen(Exception):
    """ 데이터 소스 연결/열기가 실패하였을 때 """
    pass


class FailedRead(Exception):
    """ 데이터 소스로부터 데이터를 가져오지 못하였을 때 """
    pass


class Reader(ABC):
    """ 인터페이스. 데이터 소스 연결/해제 및 데이터 읽기를 제공. """

    @abstractmethod
    def open(self, source: Any, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def close(self, *args: Any, **kwargs: Any):
        pass

    @abstractmethod
    def read(self, *args: Any, **kwargs: Any) -> Any:
        pass


class VideoReader(Reader):
    """ 비디오 소스로부터 프레임 이미지를 읽는 클래스.

    이 클래스는 cv2.VideoCapture의 인스턴스를 래핑(wrapping)하여 간단한 인터페이스를
    제공한다. 장치, 파일, 네트워크 스트림 등 다양한 비디오 소스를 지원한다.

    사용 방법은 cv2.VideoCapture와 유사하나, 필수적인 메소드만 제공한다.

    >>> video_reader = VideoReader()
    >>> video_reader.open('rtsp://localhost:554/stream')
    >>> frame = video_reader.read()  # 반복호출 가능
    >>> video_reader.close()
    """

    def __init__(self):
        self.mutex = threading.Lock()
        self._cap = cv2.VideoCapture()
        self._cap.setExceptionMode(enable=True)

    def open(self, source: Union[int, str], api_pref: int=cv2.CAP_ANY):
        try:
            with self.mutex:
                self._cap.open(source, api_pref)
        except Exception as e:
            raise FailedOpen from e

    def close(self):
        with self.mutex:
            self._cap.release()

    def read(self) -> np.ndarray:
        try:
            with self.mutex:
                _, frame = self._cap.read()
        except Exception as e:
            raise FailedRead from e
        else:
            return frame


class WebsocketReader(Reader):
    """ 웹소켓 소스로부터 데이터를 읽는 비동기 클래스.

    웹소켓 소스가 전송하는 데이터를 비동기적으로 수신(receiving)한다. 송신(sending)은
    지원하지 않으며, 필요할 경우 상속을 통해 추가 기능을 구현해야 한다.

    >>> ws_reader = WebsocketReader()
    >>> await ws_reader.open('ws://localhost:8000/websocket-endpoint')
    >>> data = await ws_reader.read()  # 반복호출 가능
    >>> await ws_reader.close()
    """

    def __init__(self):
        self.mutex = asyncio.Lock()
        self._ws = None

    async def open(self, source: str):
        try:
            async with self.mutex:
                if self._ws is not None and self._ws.open:
                    await self._ws.close()
                self._ws = await websockets.connect(source)
        except Exception as e:
            raise FailedOpen from e

    async def close(self):
        async with self.mutex:
            if self._ws is not None and self._ws.open:
                await self._ws.close()
        self._ws = None

    async def read(self) -> Any:
        try:
            async with self.mutex:
                data = await self._ws.recv()
        except Exception as e:
            raise FailedRead from e
        else:
            return data