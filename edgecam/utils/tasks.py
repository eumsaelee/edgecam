# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import abc
import asyncio
import threading

from loguru import logger


class AlreadyRunning(Exception):
    """ 태스크가 이미 실행 중 """
    def __init__(self):
        super().__init__('Task is already running.')


class NotRunning(Exception):
    """ 태스크가 실행 중이지 않음 """
    def __init__(self):
        super().__init__('Task is not running.')


class SingleThreadTask(abc.ABC):
    """ 스레드 기반 단일 백그라운드 태스크 인터페이스 """

    def __init__(self) -> None:
        self._task = None
        self._stop_task = False

    def is_running(self) -> bool:
        return isinstance(self._task, threading.Thread) and self._task.is_alive()

    def start(self) -> None:
        if self.is_running():
            raise AlreadyRunning
        self._task = threading.Thread(target=self._start)
        self._task.start()

    def stop(self) -> None:
        
        if not self.is_running():
            raise NotRunning
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
    """ 비동기 기반 단일 백그라운드 태스크 인터페이스 """

    def __init__(self) -> None:
        self._task = None
        self._stop_task = False

    def is_running(self) -> bool:
        return isinstance(self._task, asyncio.Task) and not self._task.done()

    async def start(self) -> None:
        if self.is_running():
            raise AlreadyRunning
        self._task = asyncio.create_task(self._start())

    async def stop(self) -> None:
        if not self.is_running():
            raise NotRunning
        self._stop_task = True
        await self._task
        await self._stop()

    @abc.abstractmethod
    async def _start(self) -> None:
        pass

    @abc.abstractmethod
    async def _stop(self) -> None:
        pass
