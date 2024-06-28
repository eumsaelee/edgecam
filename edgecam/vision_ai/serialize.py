# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import typing

import cv2
import numpy as np

from edgecam.vision_ai.payload import Payload


Frame = np.ndarray
Preds = typing.Dict[str, np.ndarray]


class EncodeError(Exception):
    pass


def serialize(frame: Frame, preds: Preds, ext: str='.jpg') -> bytes:
    payload = Payload()
    payload.frame = numpy_to_bytes(frame, ext)
    for name, array in preds.items():
        payload.preds[name].shape.extend(array.shape)
        payload.preds[name].data.extend(array.flatten())
    blob = payload.SerializeToString()
    return blob


def deserialize(blob: bytes) -> typing.Tuple[Frame, Preds]:
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


def numpy_to_bytes(frame: np.ndarray, ext: str='.jpg') -> bytes:
    retval, buffer = cv2.imencode(ext, frame)
    if not retval:
        raise EncodeError(
            f'Failed to encode the image to {ext}.')
    buffer = buffer.tobytes()
    return buffer