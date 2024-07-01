# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import json
import asyncio
from dataclasses import dataclass

import cv2
from loguru import logger
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import RedirectResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from src.receiver import Receiver
from src.visualize import NanumGothic, plot_text
from edgecam.serialize import numpy_to_bytes
from edgecam.vision.yolo.labels import yolo_categories_kor


@dataclass
class DetectionReceiverProps:
    websocket_uri: str='ws://172.27.1.11:8000/inference/det'
    maxsize: int=5


DETRCV_PROP = DetectionReceiverProps()
DETRCV = None


async def init_detrcv():
    global DETRCV
    DETRCV = Receiver(DETRCV_PROP.websocket_uri, DETRCV_PROP.maxsize)
    await DETRCV.connect()


async def startup():
    logger.info('Starting the application server ...')
    await init_detrcv()
    logger.info('The application server has started.')
    logger.info('Waiting for requests ...')


async def shutdown():
    logger.info('Stopping the application server ...')
    await DETRCV.disconnect()
    logger.info('The application server has stopped.')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup()
    yield
    await shutdown()


app = FastAPI(lifespan=lifespan)
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')
font = NanumGothic.bold.value


@app.get('/')
async def read_root():
    return RedirectResponse(url='/home')


@app.get('/home')
async def home(request: Request):
    return templates.TemplateResponse('home.html', {'request': request})


@app.get('/item0')
async def item0(request: Request):
    return templates.TemplateResponse('pages/item0.html', {'request': request})


@app.get('/item1')
async def item1(request: Request):
    return templates.TemplateResponse('pages/item1.html', {'request': request})


@app.get('/preview')
async def preview(request: Request):
    return templates.TemplateResponse('preview.html', {'request': request})


# @app.get('/update/var')
# async def update(request: Request):
#     return templates.TemplateResponse('req_update.html', {'request': request})


DRAW = {'boxes': False}


@app.websocket('/inference/det')
async def stream_det(websocket: WebSocket):
    await websocket.accept()
    logger.info('WebSocket: object_detection accepted!')
    try:
        consumer_task = asyncio.create_task(recv_ctrls(websocket))
        producer_task = asyncio.create_task(send_frame(websocket))
        await asyncio.gather(consumer_task, producer_task)
    except WebSocketDisconnect:
        logger.info('WebSocket: object_detection disconnected!')
    except Exception:
        logger.exception('WebSocket: object_detection error')


async def recv_ctrls(websocket: WebSocket):
    async for recv in websocket.iter_text():
        ctrls = json.loads(recv)
        DRAW['boxes'] = ctrls['resize']
        # print(ctrls['resize'])


# NOTE Temp
EQUIPMENT = {
    0: '귀덮개',
    1: '장갑',
    2: '안전모',
    3: '안전화',
    4: '안전대',
    5: '안전마스크'
}


async def send_frame(websocket: WebSocket):
    while True:
        frame, preds = await DETRCV.fetch_result()
        if DRAW['boxes']:
            boxes = preds['boxes']
            for box in boxes:
                pt0 = int(box[0]), int(box[1])
                pt1 = int(box[2]), int(box[3])
                cat = yolo_categories_kor[int(box[-1])]
                # cat = EQUIPMENT[int(box[-1])]
                cv2.rectangle(frame, pt0, pt1, (0, 255, 0), 1)
                frame = plot_text(frame, cat, pt0, (0, 0, 0), (255, 255, 255), font, 12)
        frame = numpy_to_bytes(frame, '.jpg')
        await websocket.send_bytes(frame)
        await asyncio.sleep(0)
