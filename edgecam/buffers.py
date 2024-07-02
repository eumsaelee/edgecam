# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


import time
import asyncio
import threading
from typing import Any
from collections import deque


class Full(Exception):
    """ 큐가 가득 찼을 때 """
    pass


class Empty(Exception):
    """ 큐가 비어있을 때 """
    pass


class SyncEvectingQueue():
    """ 고정된 크기를 유지하는 동기식 자동 제거 큐.

    파이썬 표준 큐는 새로운 아이템을 삽입하려고 할 때 빈 슬롯이 없을 경우, 대기하거나
    가득참(Full) 예외를 발생시킨다. 이러한 메커니즘은 실시간 데이터 처리에 적절하지 않다.

    이 큐는 새 아이템을 추가할 때 빈 슬롯이 없다면 가장 오래된 아이템을 자동으로 제거한다.
    예를 들어, 큐 최대 크기가 5이고 [0, 1, 2, 3, 4]가 저장되어 있으며 삽입은 왼쪽에서
    인출은 오른쪽에서 일어난다고 하자. 만약 새로운 아이템 42를 삽입하려고 한다면, 현재
    큐가 가득 차 있으므로 가장 오래된 아이템인 4가 자동으로 제거되고 42가 삽입된다.
    결과적으로 큐의 아이템은 [42, 0, 1, 2, 3]이 된다.
    
    큐의 최대 크기는 동적으로 변경될 수 있으나 무한 크기 큐는 허용되지 않으며 최대 크기가
    이전보다 작을 경우 그 차이만큼 가장 오래된 아이템들이 제거된다는 점에 유의해야 한다.
    예를 들어 최대 크기 5인 큐 [0, 1, 2, 3, 4]의 크기를 3으로 변경하면, 크기 차이는
    5-3=2이고, 그 차이만큼 가장 오래된 아이템 4와 3이 순차적으로 제거되어 [0, 1, 2]만
    남는다.

    사용 방법은 표준 큐(queue.Queue)와 같다.

    >>> buffer = SyncEvectingQueue(maxsize=1)  # 초기화
    >>> buffer.maxsize = 10  # 최대 크기 변경
    >>> buffer.put(item)  # 아이템 삽입
    >>> item = buffer.get(timeout)  # 아이템 인출
    """

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
        """ 큐 최대 크기를 지정/변경한다. """
        new = self._inspect(arg)
        old = self._maxsize
        with self.mutex:
            if new < old:
                for _ in range(old - new):
                    self._get()
            self._maxsize = new

    @staticmethod
    def _inspect(maxsize: int) -> int:
        # 최대 크기는 반드시 유한한 양의 정수(자연수)이어야 한다.
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

    # NOTE: 파이썬 표준 큐의 put() 메소드와 유사. 자동 제거 기능은 없다.
    # -------------------------------------------------------
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
        """ 아이템을 삽입한다. """
        with self.mutex:
            if len(self._queue) >= self._maxsize:
                self._get()  # 자동 제거
            self._put(item)
            self.not_empty.notify()

    def _put(self, item: Any):
        self._queue.append(item)

    def get(self, timeout: float=None) -> Any:
        """ 아이템을 인출한다. """
        with self.not_empty:
            if timeout is None:
                while not len(self._queue):
                    # 큐에 빈 슬롯이 없을 때까지 무한히 대기한다.
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError('The timeout must be a non-negative number.')
            else:
                endtime = time.monotonic() + timeout
                while not len(self._queue):
                    remaining = endtime - time.monotonic()
                    # 큐에 빈 슬롯이 없을 때까지 지정된 시간만큼 대기한다.
                    # 타임 아웃을 0으로 지정하면 표준 큐의 nowait와 동일하다.
                    if remaining <= 0:
                        raise Empty
                    self.not_empty.wait(remaining)
            item = self._get()
            self.not_full.notify()
            return item

    def _get(self) -> Any:
        return self._queue.popleft()


class AsyncEvectingQueue:
    """ 고정된 크기를 유지하는 비동기식 자동 제거 큐.

    SyncEvectingQueue의 비동기 버전.

    파이썬 표준 큐는 새로운 아이템을 삽입하려고 할 때 빈 슬롯이 없을 경우, 대기하거나
    가득참(Full) 예외를 발생시킨다. 이러한 메커니즘은 실시간 데이터 처리에 적절하지 않다.

    이 큐는 새 아이템을 추가할 때 빈 슬롯이 없다면 가장 오래된 아이템을 자동으로 제거한다.
    예를 들어, 큐 최대 크기가 5이고 [0, 1, 2, 3, 4]가 저장되어 있으며 삽입은 왼쪽에서
    인출은 오른쪽에서 일어난다고 하자. 만약 새로운 아이템 42를 삽입하려고 한다면, 현재
    큐가 가득 차 있으므로 가장 오래된 아이템인 4가 자동으로 제거되고 42가 삽입된다.
    결과적으로 큐의 아이템은 [42, 0, 1, 2, 3]이 된다.
    
    큐의 최대 크기는 동적으로 변경될 수 있으나 무한 크기 큐는 허용하지 않으며 최대 크기가
    이전보다 작을 경우 그 차이만큼 가장 오래된 아이템들이 제거된다는 점에 유의해야 한다.
    예를 들어 최대 크기 5인 큐 [0, 1, 2, 3, 4]의 크기를 3으로 변경하면, 크기 차이는
    5-3=2이고, 그 차이만큼 가장 오래된 아이템 4와 3이 순차적으로 제거되어 [0, 1, 2]만
    남는다.

    사용 방법은 표준 비동기 큐(asyncio.Queue)와 같다.

    >>> buffer = AsyncEvectingQueue(maxsize=1)  # 초기화
    >>> await buffer.set_maxsize(10)  # 최대 크기 변경
    >>> await buffer.put(item)  # 아이템 삽입
    >>> item = await buffer.get(timeout)  # 아이템 인출
    """

    def __init__(self, maxsize: int=1):
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self.mutex = asyncio.Lock()
        self.not_empty = asyncio.Condition(self.mutex)
        self.not_full = asyncio.Condition(self.mutex)

    def get_maxsize(self) -> int:
        return self._maxsize

    async def set_maxsize(self, arg: int):
        """ 큐 최대 크기를 지정/변경한다. """
        new = self._inspect(arg)
        old = self._maxsize
        async with self.mutex:
            if new < old:
                for _ in range(old - new):
                    await self._get()
            self._maxsize = new

    @staticmethod
    def _inspect(maxsize: int) -> int:
        # 최대 크기는 반드시 유한한 양의 정수(자연수)이어야 한다.
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

    # NOTE: 파이썬 표준 비동기 큐의 put() 메소드와 유사. 자동 제거 기능은 없다.
    # -------------------------------------------------------------
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
        """ 아이템을 삽입한다. """
        async with self.mutex:
            if len(self._queue) >= self._maxsize:
                await self._get()  # 자동 제거
            await self._put(item)
            self.not_empty.notify()

    async def _put(self, item: Any):
        self._queue.append(item)

    async def get(self, timeout: float=None) -> Any:
        """ 아이템을 인출한다. """
        async with self.not_empty:
            if timeout is None:
                while not len(self._queue):
                    # 큐에 빈 슬롯이 없을 때까지 무한히 대기한다.
                    await self.not_empty.wait()
            elif timeout < 0:
                raise ValueError('The timeout must be a non-negative number.')
            else:
                try:
                    # 큐에 빈 슬롯이 없을 때까지 지정된 시간만큼 대기한다.
                    # 타임 아웃을 0으로 지정하면 표준 큐의 nowait와 동일하다.
                    await asyncio.wait_for(self.not_empty.wait(), timeout)
                except asyncio.TimeoutError:
                    raise Empty
            item = await self._get()
            self.not_full.notify()
            return item

    async def _get(self) -> Any:
        return self._queue.popleft()
