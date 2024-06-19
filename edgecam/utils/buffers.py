# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import time
import typing
import asyncio
import threading
from collections import deque
from queue import Full, Empty


class PushQueue:

    def __init__(self, maxsize: int=1) -> None:
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
        self._not_full = threading.Condition(self._mutex)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, maxsize: int) -> None:
        new = self._inspect(maxsize)
        old = self._maxsize
        with self._mutex:
            if new < old:
                for _ in range(old - new):
                    self._get()
            self._maxsize = new

    @staticmethod
    def _inspect(maxsize: int) -> int:
        if isinstance(maxsize, int) and maxsize > 0:
            return maxsize
        raise ValueError(
            f'`maxsize` must be a positive integer.')

    def qsize(self) -> int:
        with self._mutex:
            return len(self._queue)

    def is_full(self) -> bool:
        with self._mutex:
            return len(self._queue) >= self._maxsize

    def is_empty(self) -> bool:
        with self._mutex:
            return not len(self._queue)

    def push(self, item: typing.Any) -> None:
        with self._mutex:
            if len(self._queue) >= self._maxsize:
                self._get()
            self._put(item)
            self._not_empty.notify()

    def put(self, item: typing.Any, timeout: float=None) -> None:
        with self._not_full:
            if timeout is None:
                while len(self._queue) >= self._maxsize:
                    self._not_full.wait()
            elif timeout < 0:
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                endtime = time.monotonic() + timeout
                while len(self._queue) >= self._maxsize:
                    remaining = endtime - time.monotonic()
                    if remaining <= 0:
                        raise Full
                    self._not_full.wait(remaining)
            self._put(item)
            self._not_empty.notify()

    def _put(self, item: typing.Any) -> None:
        self._queue.append(item)

    def get(self, timeout: float=None) -> typing.Any:
        with self._not_empty:
            if timeout is None:
                while not len(self._queue):
                    self._not_empty.wait()
            elif timeout < 0:
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                endtime = time.monotonic() + timeout
                while not len(self._queue):
                    remaining = endtime - time.monotonic()
                    if remaining <= 0:
                        raise Empty
                    self._not_empty.wait(remaining)
            item = self._get()
            self._not_full.notify()
            return item

    def _get(self) -> typing.Any:
        return self._queue.popleft()

    def flush(self) -> None:
        with self._mutex:
            for _ in range(len(self._queue)):
                self._get()


class AsyncPushQueue:

    def __init__(self, maxsize: int=1):
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self._mutex = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._mutex)
        self._not_full = asyncio.Condition(self._mutex)

    def get_maxsize(self) -> int:
        return self._maxsize

    async def set_maxsize(self, maxsize: int) -> None:
        new = self._inspect(maxsize)
        old = self._maxsize
        async with self._mutex:
            if new < old:
                for _ in range(old - new):
                    await self._get()
            self._maxsize = new

    @staticmethod
    def _inspect(maxsize: int) -> int:
        if isinstance(maxsize, int) and maxsize > 0:
            return maxsize
        raise ValueError(
            f'`maxsize` must be a positive integer.')

    async def qsize(self) -> int:
        async with self._mutex:
            return len(self._queue)

    async def is_full(self) -> bool:
        async with self._mutex:
            return len(self._queue) >= self._maxsize

    async def is_empty(self) -> bool:
        async with self._mutex:
            return not len(self._queue)

    async def push(self, item: any) -> None:
        async with self._mutex:
            if len(self._queue) >= self._maxsize:
                await self._get()
            await self._put(item)
            self._not_empty.notify()

    async def put(self, item: any, timeout: float=None) -> None:
        async with self._not_full:
            if timeout is None:
                while len(self._queue) >= self._maxsize:
                    await self._not_full.wait()
            elif timeout < 0:
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                try:
                    await asyncio.wait_for(self._not_full.wait(), timeout)
                except asyncio.TimeoutError:
                    raise asyncio.QueueFull()
            await self._put(item)
            self._not_empty.notify()

    async def _put(self, item: any) -> None:
        self._queue.append(item)

    async def get(self, timeout: float=None) -> any:
        async with self._not_empty:
            if timeout is None:
                while not len(self._queue):
                    await self._not_empty.wait()
            elif timeout < 0:
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout)
                except asyncio.TimeoutError:
                    raise asyncio.QueueEmpty()
            item = await self._get()
            self._not_full.notify()
            return item

    async def _get(self) -> any:
        return self._queue.popleft()

    async def flush(self) -> None:
        async with self._mutex:
            while self._queue:
                await self._get()
