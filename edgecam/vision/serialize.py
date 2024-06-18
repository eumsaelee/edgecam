# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import typing

import cv2
import numpy as np

from edgecam.vision.payload import Payload


Frame = np.ndarray
Preds = typing.Dict[str, np.ndarray]


class EncodeError(Exception):
    """ 이미지를 지정된 포맷으로 인코딩 실패하였을 때 """
    def __init__(self, ext: str):
        super().__init__(f'Failed to encode the image to {ext}.')


def serialize(frame: Frame, preds: Preds, ext: str='.jpg') -> bytes:
    """ ? """
    payload = Payload()
    payload.frame = numpy_to_bytes(frame, ext)
    for name, array in preds.items():
        payload.preds[name].shape.extend(array.shape)
        payload.preds[name].data.extend(array.flatten())
    blob = payload.SerializeToString()
    return blob


def deserialize(blob: bytes) -> typing.Tuple[Frame, Preds]:
    """ ? """
    payload = Payload()
    payload.ParseFromString(blob)
    frame = payload.frame
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)
    preds = {}
    for name, array in payload.preds.items():
        data = np.array(array.data, dtype=np.float64)
        data = data.reshape(tuple(array.shape))
        preds[name] = data
    return frame, preds


# NOTE 위치를 변경해야 할 수도 있음. 이 모듈과 성격이 조금 다름.
def numpy_to_bytes(frame: np.ndarray, ext: str='.jpg') -> bytes:
    """ ? """
    retval, buffer = cv2.imencode(ext, frame)
    if not retval:
        raise RuntimeError(
            f'Failed to encode the image to {ext}.')
    buffer = buffer.tobytes()
    return buffer