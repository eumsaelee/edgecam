# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim
#
# NOTE: DO NOT USE THIS MODULE. IT WILL BE DEPRECATED!


import abc
import asyncio
import threading


class TaskError(Exception):
    pass


class SingleThreadTask(abc.ABC):

    def __init__(self) -> None:
        self._task = None
        self._stop_task = False

    def is_running(self) -> bool:
        return self._task is not None and self._task.is_alive()

    def start(self) -> None:
        if self.is_running():
            raise TaskError(
                'Task is already running.')
        self._task = threading.Thread(target=self._start)
        self._task.start()

    def stop(self) -> None:
        if not self.is_running():
            raise TaskError(
                'Task is not running.')
        self._stop_task = True
        self._task.join()
        self._stop()

    @abc.abstractmethod
    def _start(self) -> None:
        pass

    @abc.abstractmethod
    def _stop(self) -> None:
        pass


class SingleAsyncTask(abc.ABC):

    def __init__(self) -> None:
        self._task = None
        self._stop_task = False

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running():
            raise TaskError(
                'Task is already running.')
        self._task = asyncio.create_task(self._start())

    async def stop(self) -> None:
        if not self.is_running():
            raise TaskError(
                'Task is not running.')
        self._stop_task = True
        await self._task
        await self._stop()

    @abc.abstractmethod
    async def _start(self) -> None:
        pass

    @abc.abstractmethod
    async def _stop(self) -> None:
        pass
