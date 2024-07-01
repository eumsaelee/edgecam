# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


import time
import asyncio
import threading
from typing import Any
from collections import deque


class Full(Exception): pass
class Empty(Exception): pass


class SyncEvectingQueue():

    def __init__(self, maxsize: int=1):
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, arg: int):
        new = self._inspect(arg)
        old = self._maxsize
        with self.mutex:
            if new < old:
                for _ in range(old - new):
                    self._get()
            self._maxsize = new

    @staticmethod
    def _inspect(maxsize: int) -> int:
        if isinstance(maxsize, int) and maxsize > 0:
            return maxsize
        raise ValueError(f'The maxsize must be a positive integer.')

    def qsize(self) -> int:
        with self.mutex:
            return len(self._queue)

    def is_full(self) -> bool:
        with self.mutex:
            return len(self._queue) >= self._maxsize

    def is_empty(self) -> bool:
        with self.mutex:
            return not len(self._queue)

    # NOTE: evecting mechanism not included
    # -------------------------------------
    # def put(self, item: time.Any, timeout: float=None):
    #     with self.not_full:
    #         if timeout is None:
    #             while len(self._queue) >= self._maxsize:
    #                 self.not_full.wait()
    #         elif timeout < 0:
    #             raise ValueError(
    #                 '`timeout` must be a non-negative number.')
    #         else:
    #             endtime = time.monotonic() + timeout
    #             while len(self._queue) >= self._maxsize:
    #                 remaining = endtime - time.monotonic()
    #                 if remaining <= 0:
    #                     raise Full
    #                 self.not_full.wait(remaining)
    #         self._put(item)
    #         self.not_empty.notify()

    def put(self, item: Any):
        with self.mutex:
            if len(self._queue) >= self._maxsize:
                self._get()
            self._put(item)
            self.not_empty.notify()

    def _put(self, item: Any):
        self._queue.append(item)

    def get(self, timeout: float=None) -> Any:
        with self.not_empty:
            if timeout is None:
                while not len(self._queue):
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError('The timeout must be a non-negative number.')
            else:
                endtime = time.monotonic() + timeout
                while not len(self._queue):
                    remaining = endtime - time.monotonic()
                    if remaining <= 0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            self.not_full.notify()
            return item

    def _get(self) -> Any:
        return self._queue.popleft()


class AsyncEvectingQueue:

    def __init__(self, maxsize: int=1):
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self.mutex = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.mutex)
        self.not_full = asyncio.Condition(self.mutex)

    def get_maxsize(self) -> int:
        return self._maxsize

    async def set_maxsize(self, arg: int):
        new = self._inspect(arg)
        old = self._maxsize
        async with self.mutex:
            if new < old:
                for _ in range(old - new):
                    await self._get()
            self._maxsize = new

    @staticmethod
    def _inspect(maxsize: int) -> int:
        if isinstance(maxsize, int) and maxsize > 0:
            return maxsize
        raise ValueError(f'The maxsize must be a positive integer.')

    async def qsize(self) -> int:
        async with self.mutex:
            return len(self._queue)

    async def is_full(self) -> bool:
        async with self.mutex:
            return len(self._queue) >= self._maxsize

    async def is_empty(self) -> bool:
        async with self.mutex:
            return not len(self._queue)

    # NOTE: evecting mechanism not included
    # -------------------------------------
    # async def put(self, item: Any, timeout: float=None):
    #     async with self.not_full:
    #         if timeout is None:
    #             while len(self._queue) >= self._maxsize:
    #                 await self.not_full.wait()
    #         elif timeout < 0:
    #             raise ValueError('The timeout must be a non-negative number.')
    #         else:
    #             try:
    #                 await asyncio.wait_for(self.not_full.wait(), timeout)
    #             except asyncio.TimeoutError:
    #                 raise asyncio.QueueFull()
    #         await self._put(item)
    #         self.not_empty.notify()

    async def put(self, item: Any):
        async with self.mutex:
            if len(self._queue) >= self._maxsize:
                await self._get()
            await self._put(item)
            self.not_empty.notify()

    async def _put(self, item: Any):
        self._queue.append(item)

    async def get(self, timeout: float=None) -> Any:
        async with self.not_empty:
            if timeout is None:
                while not len(self._queue):
                    await self.not_empty.wait()
            elif timeout < 0:
                raise ValueError('The timeout must be a non-negative number.')
            else:
                try:
                    await asyncio.wait_for(self.not_empty.wait(), timeout)
                except asyncio.TimeoutError:
                    raise Empty
            item = await self._get()
            self.not_full.notify()
            return item

    async def _get(self) -> Any:
        return self._queue.popleft()
