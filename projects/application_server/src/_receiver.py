# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import asyncio
import websockets

import numpy as np
from loguru import logger

from edgecam.buffers import AsyncEvectingQueue
from edgecam.serialize import deserialize


Frame = np.ndarray
Preds = typing.Dict[str, np.ndarray]
Result = typing.Tuple[Frame, Preds]


class Receiver:

    def __init__(self, websocket_uri: str, maxsize: int=1):
        self._uri = websocket_uri
        self._buf = AsyncEvectingQueue(maxsize)
        self._task: asyncio.Task=None
        self._stop_task = False

    async def connect(self):
        if not self._task or not self._task.done():
            self._task = asyncio.create_task(self._buffer())

    async def _buffer(self):
        async with websockets.connect(self._uri) as ws:
            try:
                while not self._stop_task:
                    result = await ws.recv()
                    if not isinstance(result, bytes):
                        logger.warning(
                            'result is not a type of bytes.')
                        continue
                    frame, preds = deserialize(result)
                    await self._buf.put((frame, preds))
            except:
                logger.exception('Aborted.')
            finally:
                if not self._stop_task:
                    self._stop_task = True

    async def disconnect(self):
        if self._task and not self._task.done():
            self._stop_task = True
            await self._task

    async def fetch_result(self, timeout: float=10.0) -> Result:
        result = await self._buf.get(timeout)
        return result

    async def alter_maxsize(self, maxsize: int):
        await self._buf.set_maxsize(maxsize)
