from typing import Dict, Tuple

import cv2
import numpy as np

from src.general.payload import Payload


class EncodeError(RuntimeError): pass


def serialize(frame: np.ndarray, preds: Dict[str, np.ndarray]) -> bytes:
    payload = Payload()

    ret, img = cv2.imencode('.jpg', frame)
    if not ret:
        raise EncodeError
    img = img.tobytes()

    payload.frame = img
    for name, array in preds.items():
        payload.preds[name].shape.extend(array.shape)
        payload.preds[name].data.extend(array.flatten())

    serial = payload.SerializeToString()
    return serial


def deserialize(serial: bytes) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    payload = Payload()
    payload.ParseFromString(serial)

    frame = payload.frame
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)

    preds = {}
    for name, array in payload.preds.items():
        data = np.array(array.data, dtype=np.float64)
        data = data.reshape(tuple(array.shape))
        preds[name] = data

    return frame, preds
