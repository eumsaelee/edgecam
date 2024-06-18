# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

class InvalidStepSize(Exception):
    """ 유효하지 않은 스텝 사이즈 """
    def __init__(self):
        super().__init__(
            f'`stepsize` must be a positive interger greater than 1.')


class StepSkipper:
    """ 일정 주기를 기준으로 스킵 여부를 판단하는 스킵퍼 """

    def __init__(self, stepsize: int) -> None:
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
    def stepsize(self, stepsize: int) -> None:
        self._stepsize = self._inspect(stepsize)

    @staticmethod
    def _inspect(stepsize: int) -> int:
        if isinstance(stepsize, int) and stepsize > 1:
            return stepsize
        raise InvalidStepSize
