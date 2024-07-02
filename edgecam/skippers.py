# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


class StepSkipper:
    """ 반복문에서 매 단위 간격마다 작업 스킵(건너뜀) 여부를 판단하는 클래스.

    단위 간격 크기(stepsize)를 기준으로 첫 사이클을 제외한 나머지를 모두 스킵한다.

    >>> skipper = StepSkipper(stepsize=3)
    >>> for i in range(10):
    ...    is_skip = next(skipper)
    ...    print(is_skip, end=' ')
    ...
    False True True False True True False True True False

    위와 같이 단위 간격 크기가 3일 때, 사이클 3회마다 첫 사이클은 스킵하지 않도록
    만들 수 있다. 단위 간격 크기는 반드시 1보다 큰 양의 정수이어야 하고, 런타임에
    동적으로 변경할 수 있다.
    """

    def __init__(self, stepsize: int):
        self._stepsize = self._inspect(stepsize)
        self._pos = -1

    def __iter__(self) -> 'StepSkipper':
        return self

    def __next__(self) -> bool:
        self._pos += 1
        if self._pos >= self._stepsize:
            self._pos = 0
        return bool(self._pos % self._stepsize)

    @property
    def stepsize(self) -> int:
        return self._stepsize

    @stepsize.setter
    def stepsize(self, stepsize: int):
        self._stepsize = self._inspect(stepsize)

    @staticmethod
    def _inspect(stepsize: int) -> int:
        if isinstance(stepsize, int) and stepsize > 1:
            return stepsize
        raise ValueError(
            f'The stepsize must be a positive interger greater than 1.')