# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import websockets

import numpy as np
from loguru import logger

from edgecam.utils.buffers import AsyncPushQueue
from edgecam.utils.tasks import SingleAsyncTask, TaskError
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
        except TaskError:
            logger.exception(
                'Reception task already running.')

    async def stop_reception(self) -> None:
        try:
            await self._task.stop()
        except TaskError:
            logger.exception(
                'Reception task is not running.')

    async def read_output(self, timeout: float=30.0) -> Output:
        try:
            output = await self._buf.get(timeout)
            return output
        except:
            logger.warning(
                'Failed to read output.')
