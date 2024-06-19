# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import asyncio

import cv2
from loguru import logger
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.websockets import WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

from src.reception import ReceptionHandler
from src.visualize import NanumGothic, plot_text
from edgecam.vision.serialize import numpy_to_bytes
from edgecam.vision.labels import yolo_categories_kor


OBJECT_DETECTION_URI = 'ws://172.17.0.3:8000/stream/object_detection'
RECEPTION_BUFFER_SIZE = 10

RECEPTION_HANDLER = None


async def startup() -> None:
    logger.info('Called startup().')
    await _init_reception_handler()


async def shutdown() -> None:
    logger.info('Called shutdown().')
    await RECEPTION_HANDLER.stop_reception()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


async def _init_reception_handler() -> None:
    global RECEPTION_HANDLER
    RECEPTION_HANDLER = ReceptionHandler(OBJECT_DETECTION_URI, RECEPTION_BUFFER_SIZE)
    await RECEPTION_HANDLER.start_reception()


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory='templates')
font = NanumGothic.bold.value


@app.websocket('/stream/object_detection')
async def stream(websocket: WebSocket):
    await websocket.accept()
    logger.info('WebSocket: object_detection accepted!')
    try:
        while True:
            frame, preds = await RECEPTION_HANDLER.read_output()
            boxes = preds['boxes']
            for box in boxes:
                pt0 = int(box[0]), int(box[1])
                pt1 = int(box[2]), int(box[3])
                cat = yolo_categories_kor[int(box[-1])]
                cv2.rectangle(frame, pt0, pt1, (0, 255, 0), 1)
                frame = plot_text(frame, cat, pt0, (0, 0, 0), (255, 255, 255), font, 12)
            frame = numpy_to_bytes(frame, '.jpg')
            await websocket.send_bytes(frame)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info('WebSocket: object_detection disconnected!')
    except Exception:
        logger.exception('WebSocket: object_detection error')


@app.get('/preview')
async def preview(request: Request):
    return templates.TemplateResponse('preview.html', {'request': request})


@app.get('/update/var')
async def update(request: Request):
    return templates.TemplateResponse('req_update.html', {'request': request})