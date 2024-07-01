# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


from typing import Callable, Any
import asyncio
import threading


class Alive(Exception): pass
class NotAlive(Exception): pass


class SingleThreadTask:

    def __init__(self):
        self._task: threading.Thread = None
        self._stop_task = False

    def is_alive(self) -> bool:
        return self._task is not None and self._task.is_alive()

    def start(self, target: Callable, args: list=None):
        if self.is_alive():
            raise Alive
        if args is None:
            args = []
        self._task = threading.Thread(target=self._t, args=[target, *args])
        self._task.start()

    def _t(self, target: Callable, *args: Any):
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


class SingleAsyncTask:

    def __init__(self):
        self._task: asyncio.Task = None
        self._stop_task = False

    def is_alive(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self, target: Callable, args: list=None):
        if self.is_alive():
            raise Alive
        if args is None:
            args = []
        self._task = asyncio.create_task(self._t(target, *args))

    async def _t(self, target: Callable, *args: Any):
        try:
            while not self._stop_task:
                await target(*args)
        except Exception as e:
            raise RuntimeError from e
        finally:
            if self._stop_task:
                self._stop_task = False

    async def stop(self):
        if not self.is_alive():
            raise NotAlive
        self._stop_task = True
        await self._task
        self._stop_task = False
