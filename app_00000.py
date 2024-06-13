import time
import queue
import typing
import asyncio
import threading

import cv2
import numpy as np
from loguru import logger
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.templating import Jinja2Templates
from starlette.websockets import WebSocketDisconnect
from starlette.requests import Request

from src.utils.captures import FrameCapture
from src.utils.buffers import StreamQueue
from src.utils.ai.vision.models import TorchModel, Yolov8n
from src.utils.ai.vision.visualization import NanumGothic, plot_text
from src.utils.skipper import StepSkipper
from src.core.tasks import VideoCaptureTask


SOURCE = 'rtsp://192.168.1.101:12554/profile2/media.smp'
API_PREF = cv2.CAP_FFMPEG
MAXSIZE = 10

CAPTURE = None
BUFFER = None
CAPTURE_TASK = None

FONT_PATH = NanumGothic.bold.value


def startup() -> None:
    global CAPTURE, BUFFER, CAPTURE_TASK
    CAPTURE = FrameCapture(SOURCE, API_PREF)
    BUFFER = StreamQueue(MAXSIZE)
    CAPTURE_TASK = VideoCaptureTask(CAPTURE, BUFFER)
    CAPTURE_TASK.start()


def shutdown() -> None:
    if CAPTURE_TASK.is_running():
        CAPTURE_TASK.stop()


def to_bytes(frame: np.ndarray) -> bytes:
    # ret, frame = cv2.imencode('.jpg', frame)
    ret, frame = cv2.imencode('.png', frame)  # jpg -> png
    if not ret:
        raise RuntimeError('Failed to encode the frame.')
    frame = frame.tobytes()
    return frame


# NOTE: temporary unused.
# -----
# def to_multipart(frame: bytes) -> bytes:
#     return (b'--frame\r\n'
#             b'Content-Type: image/jpeg\r\n\r\n'
#             + frame + b'\r\n')


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory='templates')


@app.websocket('/ws/stream')
async def stream(websocket: WebSocket):
    await websocket.accept()
    logger.debug('websocket accepted.')
    model = Yolov8n()
    skip = StepSkipper(stepsize=3)
    boxes = None
    try:
        while True:
            frame = BUFFER.get(timeout=10.0)
            # frame = plot_text(frame, '기술연구소', 30)
            if not next(skip):
                preds = model.predict(frame)
                boxes = preds['boxes']
            for box in boxes:
                pt0 = int(box[0]), int(box[1])
                pt1 = int(box[2]), int(box[3])
                cat = Yolov8n.categories_kor[int(box[-1])]
                cv2.rectangle(frame, pt0, pt1, (0, 255, 0), 1)
                frame = plot_text(frame, cat, pt0, (255, 255, 255), (0, 0, 0), FONT_PATH, 12)
            frame = to_bytes(frame)
            await websocket.send_bytes(frame)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.debug('the websocket was disconnected.')
    except Exception:
        logger.exception(
            'an unexpected error has occurred. websocket will be closed.')
    # NOTE: temporary unused.
    # -----
    # finally:
    #     await websocket.close()


@app.get('/')
async def main(request: Request):
    return templates.TemplateResponse('websocket_client.html',
                                      {'request': request})
