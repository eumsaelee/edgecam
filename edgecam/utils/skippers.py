# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

class StepSkipper:

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
        raise ValueError(
            f'`stepsize` must be a positive interger greater than 1.')