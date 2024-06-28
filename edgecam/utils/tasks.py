# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import abc
import typing
import asyncio
import threading


class TaskError(Exception):
    pass


# NOTE: DO NOT USE THIS CLASS
# class SingleThreadTask(abc.ABC):
#
#     def __init__(self) -> None:
#         self._task = None
#         self._stop_task = False
#
#     def is_running(self) -> bool:
#         return self._task is not None and self._task.is_alive()
#
#     def start(self) -> None:
#         if self.is_running():
#             raise TaskError(
#                 'Task is already running.')
#         self._task = threading.Thread(target=self._start)
#         self._task.start()
#
#     def stop(self) -> None:
#         if not self.is_running():
#             raise TaskError(
#                 'Task is not running.')
#         self._stop_task = True
#         self._task.join()
#         self._stop()
#
#     @abc.abstractmethod
#     def _start(self) -> None:
#         pass
#
#     @abc.abstractmethod
#     def _stop(self) -> None:
#         pass


class Alive(Exception): pass
class NotAlive(Exception): pass


class SingleThreadTask:

    def __init__(self):
        self._task: threading.Thread = None
        self._stop_task = False

    def is_alive(self) -> bool:
        return self._task is not None and self._task.is_alive()

    def start(self, target: typing.Callable, args: list=None):
        if self.is_alive():
            raise Alive
        if args is None:
            args = []
        self._task = threading.Thread(target=self._t, args=[target, *args])
        self._task.start()

    def _t(self, target: typing.Callable, *args: typing.Any):
        try:
            while not self._stop_task:
                target(*args)
        except Exception as e:
            raise RuntimeError from e
        finally:
            if self._stop_task:
                self._stop_task = False

    def stop(self):
        if not self.is_alive():
            raise NotAlive
        self._stop_task = True
        self._task.join()
        self._stop_task = False


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
