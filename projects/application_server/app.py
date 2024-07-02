# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import json
import asyncio
from typing import Callable, Any
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

from edgecam.readers import WebsocketReader
from edgecam.buffers import AsyncEvectingQueue
from edgecam.tasks import SingleAsyncTask

from src.visualize import NanumGothic, plot_text
from edgecam.serialize import numpy_to_bytes, deserialize
from edgecam.vision.yolo.labels import yolo_categories_kor


class BufferChainer(SingleAsyncTask):

    def __init__(self, src: Any, dst: Any):
        """ Initialization.
        Args:
            src (Any): An object with a get() method that includes a timeout option.
            dst (Any): An object with a put() method that includes an auto-evicting mechanism like an EvectingQueue.
        """
        self._src = src
        self._dst = dst
        super().__init__()

    async def start(self, hooker: Callable[[Any], Any], timeout: float=30.0):
        """ Start chaining.
        Args:
            hooker (Callable): A function to preprocess the frame image from the src.
            timeout (float): Timeout for the get() method of the src.
        """
        async def target():
            data = hooker(await self._src.get(timeout))
            data = hooker(data)
            data = deserialize(data)
            await self._dst.put(data)
        await super().start(target)


class DataReader(WebsocketReader):

    async def get(self, timeout: Any=None) -> Any:
        # Duck typing.
        return await self.read()


@dataclass
class Configs:
    source: str='ws://172.27.1.11:8000/inference/det'
    maxsize: int=5
    timeout: float=10.0


CONFIGS = Configs()

#READER = DataReader(CONFIGS.source)
READER = DataReader()
READER_BUFFER = AsyncEvectingQueue(CONFIGS.maxsize)
READER_CHAINER = BufferChainer(READER, READER_BUFFER)


async def start_server():
    logger.info('Starting the application server ...')
    await READER.open(CONFIGS.source)
    await READER_CHAINER.start(lambda x: x, CONFIGS.timeout)
    logger.info('The application server has started.')
    logger.info('Waiting for requests ...')


async def stop_server():
    logger.info('Stopping the application server ...')
    await READER_CHAINER.stop()
    await READER.close()
    logger.info('The application server has stopped.')


@asynccontextmanager
async def lifespan(app: FastAPI):
    await start_server()
    yield
    await stop_server()


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
# EQUIPMENT = {
#     0: '귀덮개',
#     1: '장갑',
#     2: '안전모',
#     3: '안전화',
#     4: '안전대',
#     5: '안전마스크'
# }


async def send_frame(websocket: WebSocket):
    while True:
        frame, preds = await READER_BUFFER.get(CONFIGS.timeout)
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
