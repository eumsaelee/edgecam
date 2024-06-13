# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import time
import typing
import threading
from collections import deque
from queue import Full, Empty


class InvalidSize(Exception):
    """ Raises when an invalid maxsize is entered. """
    pass


class StreamQueue:

    def __init__(self, maxsize=1) -> None:
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
        self._not_full = threading.Condition(self._mutex)

    def __len__(self) -> int:
        with self._mutex:
            return len(self._queue)

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @maxsize.setter
    def maxsize(self, maxsize) -> None:
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
        raise InvalidSize(
            f'`maxsize` must be a positive integer.')

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
            # infinite waiting
            if timeout is None:
                while len(self._queue) >= self._maxsize:
                    self._not_full.wait()
            # no waiting
            elif timeout == 0:
                if len(self._queue) >= self._maxsize:
                    raise Full
            # error
            elif timeout < 0:
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            # limited waiting
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
            # infinite waiting
            if timeout is None:
                while not len(self._queue):
                    self._not_empty.wait()
            # no waiting
            elif timeout == 0:
                if not len(self._queue):
                    raise Empty
            # error
            elif timeout < 0:
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            # limited waiting
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
