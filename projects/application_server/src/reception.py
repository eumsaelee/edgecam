# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

# 정상적으로 작동하지만 코드 검증이 아직 이루어지지 않음.
# 클래스 이름이 하드코딩되어 있음.

import sys
from pathlib import Path
BASEDIR = Path(__file__).parents[3].absolute()
sys.path.append(str(BASEDIR))

import typing
import websockets

import numpy as np
from loguru import logger

from edgecam.utils.buffers import AsyncPushQueue
from edgecam.utils.tasks import SingleAsyncTask, AlreadyRunning, NotRunning
from edgecam.vision.serialize import deserialize


Frame = np.ndarray
Preds = typing.Dict[str, np.ndarray]
Output = typing.Tuple[Frame, Preds]


class ReceptionTask(SingleAsyncTask):

    def __init__(self, uri: str, buf: AsyncPushQueue):
        self._uri = uri
        self._buf = buf
        super().__init__()

    async def _start(self) -> None:
        async with websockets.connect(self._uri) as ws:
            try:
                while not self._stop_task:
                    output = await ws.recv()
                    if not isinstance(output, bytes):
                        logger.warning(
                            '`output` is not a type of bytes.')
                        continue
                    frame, preds = deserialize(output)
                    await self._buf.push((frame, preds))
            except:
                logger.exception('Task has been aborted.')
            finally:
                if not self._stop_task:
                    self._stop_task = True
                    await self._stop()

    async def _stop(self) -> None:
        pass


class ReceptionHandler:

    def __init__(self, uri: str, maxsize: int=10) -> None:
        self._buf = AsyncPushQueue(maxsize)
        self._task = ReceptionTask(uri, self._buf)

    async def change_maxsize(self, maxsize: int) -> None:
        await self._buf.set_maxsize(maxsize)

    async def start_reception(self) -> None:
        try:
            await self._task.start()
        except AlreadyRunning:
            logger.exception(
                'Reception task already running.')

    async def stop_reception(self) -> None:
        try:
            await self._task.stop()
        except NotRunning:
            logger.exception(
                'Reception task is not running.')

    async def read_output(self, timeout: float=30.0) -> Output:
        try:
            output = await self._buf.get(timeout)
            return output
        except:
            logger.warning(
                'Failed to read output.')
