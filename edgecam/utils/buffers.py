# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import time
import typing
import asyncio
import threading
from collections import deque
from queue import Full, Empty


class InvalidBufferSize(Exception):
    """
    자연수가 아닌 값이 큐 최대 사이즈로 입력되었을 때 발생하는 예외.
    """
    def __init__(self):
        super().__init__(f'`maxsize` must be a positive integer.')


class PushQueue:
    """
    밀어내기 큐.

    이 클래스는 파이썬 표준 큐인 queue.Queue 클래스의 설계를 바탕으로 작성되었다.
    사용 방법은 유사하나 완벽하게 호환되지 않으므로 유의해야 한다.

    Attributes
    ----------
    _maxsize : int
        큐의 최대 크기. 0보다 큰 양의 정수.
    _queue : deque
        아이템 입출력이 발생하는 실제 큐.
    _mutex : threading.Lock
        경쟁 방지를 위한 스레딩 락.
    _not_empty : threading.Condition
        아이템을 대기 중인 스레드들에게 큐가 비어있지 않음을 알리기 위한 스레딩 컨디션.
    _not_full : threading.Condition
        아이템 입력을 대기 중인 스레드들에게 큐가 가득 차지 않았음을 알리기 위한 스레딩 컨디션.

    Additional features
    -------------------
    - 큐 최대 크기를 런타임에 변경할 수 있다. 단, 크기가 작아질 경우 아이템 손실이 발생한다.
    - 자동으로 가장 오래된 아이템을 삭제하고 새 아이템을 삽입할 수 있는 '푸시'를 제공한다.
    - 큐에 존재하는 모든 아이템들을 삭제할 수 있는 '플러시'를 제공한다.

    Restricted features
    -------------------
    - 큐 최대 크기를 무한(maxsize=0)으로 지정할 수 없다. 반드시 0보다 큰 정수이어야 한다.
    - 경량화를 위해 unfinished_task와 관련된 모든 속성 및 메소드를 제거하였다.
    """

    def __init__(self, maxsize: int=1) -> None:
        """ 큐 초기화.

        Parameters
        ----------
        maxsize : int, optional
            큐의 최대 크기. 반드시 0보다 큰 양의 정수이어야 한다. 기본 값은 1.
        """
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self._mutex = threading.Lock()
        self._not_empty = threading.Condition(self._mutex)
        self._not_full = threading.Condition(self._mutex)

    @property
    def maxsize(self) -> int:
        """ 프로퍼티. 큐의 최대 크기를 반환한다.

        Returns
        -------
        int
            _description_
        """
        return self._maxsize

    @maxsize.setter
    def maxsize(self, maxsize: int) -> None:
        """ 프로퍼티-세터. 큐의 최대 크기를 변경한다.

        입력된 maxsize가 기존의 그것보다 작을 경우 그 차이만큼 가장 오래된 아이템 순서로
        제거된다.
        
        Parameters
        ----------
        maxsize : int
            
        """
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
        raise InvalidBufferSize

    def qsize(self) -> int:
        """
        현재 큐에 존재하는 아이템의 개수를 반환한다.
        큐 아이템 삽입/삭제가 빈번한 상황에서 이 함수가 반환하는 값은 이미 과거 시점의 결과값이다.
        따라서 신뢰할 수 없다는 점에 유의해야 한다.
        """
        with self._mutex:
            return len(self._queue)

    def is_full(self) -> bool:
        with self._mutex:
            return len(self._queue) >= self._maxsize

    def is_empty(self) -> bool:
        with self._mutex:
            return not len(self._queue)

    def push(self, item: typing.Any) -> None:
        """
        큐에 빈 슬롯이 없을 때 자동으로 가장 오래된 아이템을 제거하고 새 아이템을 삽입한다.
        """
        with self._mutex:
            if len(self._queue) >= self._maxsize:
                self._get()
            self._put(item)
            self._not_empty.notify()

    def put(self, item: typing.Any, timeout: float=None) -> None:
        with self._not_full:
            if timeout is None:
                # case 1: 빈 슬롯이 나타날 때까지 무한 대기
                while len(self._queue) >= self._maxsize:
                    self._not_full.wait()
            elif timeout < 0:
                # case 2: 잘못된 타임아웃 값 (음수)
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                # case 3: 지정된 타임아웃만큼 대기 -> 슬롯이 없으면 예외 발생
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
                # case 1: 아이템이 삽입될 때까지 무한 대기
                while not len(self._queue):
                    self._not_empty.wait()
            elif timeout < 0:
                # case 2: 잘못된 타임아웃 값(음수)
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                # case 3: 지정된 타임아웃만큼 대기 -> 아이템이 없으면 예외 발생
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
        """
        현재 큐에 존재하는 모든 아이템들을 삭제한다.
        """
        with self._mutex:
            for _ in range(len(self._queue)):
                self._get()


class AsyncPushQueue:
    """
    비동기 방식 밀어내기 큐

    이 클래스는 파이썬 표준 큐인 queue.Queue 클래스의 설계를 참고하여 작성하였다.
    밀어내기 큐(PushQueue)를 비동기 버전으로 포팅한 것이다.
    """

    def __init__(self, maxsize: int=1):
        self._maxsize = self._inspect(maxsize)
        self._queue = deque()
        self._mutex = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._mutex)
        self._not_full = asyncio.Condition(self._mutex)

    def get_maxsize(self) -> int:
        return self._maxsize

    async def set_maxsize(self, maxsize: int) -> None:
        # NOTE '@프로퍼티.세터'는 'async' 키워드를 사용할 수 없음
        """
        큐 최대 크기를 변경한다. 이 값은 0보다 큰 정수이어야 한다.
        원래 사이즈보다 작은 사이즈가 입력될 경우 그 차이만큼 아이템이 손실된다.
        """
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
            '`maxsize` must be a positive integer.')

    async def qsize(self) -> int:
        """
        현재 큐에 존재하는 아이템의 개수를 반환한다.
        큐 아이템 삽입/삭제가 빈번한 상황에서 이 함수가 반환하는 값은 이미 과거 시점의 결과값이다.
        따라서 신뢰할 수 없다는 점에 유의해야 한다.
        """
        async with self._mutex:
            return len(self._queue)

    async def is_full(self) -> bool:
        async with self._mutex:
            return len(self._queue) >= self._maxsize

    async def is_empty(self) -> bool:
        async with self._mutex:
            return not len(self._queue)

    async def push(self, item: any) -> None:
        """
        큐에 빈 슬롯이 없을 때 자동으로 가장 오래된 아이템을 제거하고 새 아이템을 삽입한다.
        """
        async with self._mutex:
            if len(self._queue) >= self._maxsize:
                await self._get()
            await self._put(item)
            self._not_empty.notify()

    async def put(self, item: any, timeout: float=None) -> None:
        async with self._not_full:
            if timeout is None:
                # case 1: 빈 슬롯이 나타날 때까지 무한 대기
                while len(self._queue) >= self._maxsize:
                    await self._not_full.wait()
            elif timeout < 0:
                # case 2: 잘못된 타임아웃 값 (음수)
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                # case 3: 지정된 타임아웃만큼 대기 -> 슬롯이 없으면 예외 발생
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
                # case 1: 아이템이 삽입될 때까지 무한 대기
                while not len(self._queue):
                    await self._not_empty.wait()
            elif timeout < 0:
                # case 2: 잘못된 타임아웃 값(음수)
                raise ValueError(
                    '`timeout` must be a non-negative number.')
            else:
                # case 3: 지정된 타임아웃만큼 대기 -> 아이템이 없으면 예외 발생
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
        """
        현재 큐에 존재하는 모든 아이템들을 삭제한다.
        """
        async with self._mutex:
            while self._queue:
                await self._get()
