# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import abc
import typing

import numpy as np


InputImage = np.ndarray
SubtaskAlias = str
SubtaskResult = np.ndarray
InferenceResults = typing.Dict[SubtaskAlias, SubtaskResult]


class CommonModel(abc.ABC):

    @abc.abstractmethod
    def infer(self, img: InputImage) -> InferenceResults:
        pass

    @abc.abstractmethod
    def release(self):
        pass
