# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2].absolute()))

import asyncio
from typing import Callable, Union, Any
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from edgecam.readers import Frame, VideoReader
from edgecam.buffers import SyncEvectingQueue
from edgecam.tasks import SingleThreadTask
from edgecam.vision.yolo.models import Yolo
from edgecam.serialize import serialize


class BufferChainer(SingleThreadTask):

    def __init__(self, src: Any, dst: Any):
        """ Initialization.
        Args:
            src (Any): An object with a get() method that includes a timeout option.
            dst (Any): An object with a put() method that includes an auto-evicting mechanism like an EvectingQueue.
        """
        self._src = src
        self._dst = dst
        super().__init__()

    def start(self, hooker: Callable[[Any], Any], timeout: float=30.0):
        """ Start chaining.
        Args:
            hooker (Callable): A function to preprocess the frame image from the src.
            timeout (float): Timeout for the get() method of the src.
        """
        super().start(lambda: self._dst.put(hooker(self._src.get(timeout))))


class FrameReader(VideoReader):

    def get(self, timeout: Any=None) -> Frame:
        # Duck typing.
        return self.read()


@dataclass
class Configs:
    source: Union[int, str]='rtsp://192.168.1.101:12554/profile2/media.smp'
    api_pref: int=cv2.CAP_FFMPEG
    maxsize: int=5
    pt: str='yolov8m.pt'
    tracking: bool=True
    timeout: float=10.0


CONFIGS = Configs()

READER = FrameReader(CONFIGS.source, CONFIGS.api_pref)
READER_BUFFER = SyncEvectingQueue(CONFIGS.maxsize)
READER_CHAINER = BufferChainer(READER, READER_BUFFER)

YOLO = Yolo(CONFIGS.pt, CONFIGS.tracking)
YOLO_BUFFER = SyncEvectingQueue(Configs.maxsize)
YOLO_CHAINER = BufferChainer(READER_BUFFER, YOLO_BUFFER)


def start_server():
    logger.info('Starting the inference server ...')
    READER_CHAINER.start(lambda Frame: Frame, CONFIGS.timeout)
    YOLO_CHAINER.start(lambda Frame: (Frame, YOLO.infer(Frame)), CONFIGS.timeout)
    logger.info('The inference server has started.')
    logger.info('Waiting for requests ...')


def stop_server():
    logger.info('Stopping the inference server ...')
    YOLO_CHAINER.stop()
    READER_CHAINER.stop()
    YOLO.release()
    READER.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_server()
    yield
    stop_server()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://127.0.0.1:8888',
                   'http://172.27.1.11:8888'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


async def websocket_endpoint(websocket: WebSocket, buffer: Any):
    logger.info('Received a request for stream via websocket...')
    await websocket.accept()
    host = websocket.client.host
    port = websocket.client.port
    logger.info(f'The request from {host}:{port} has been accepted!')
    try:
        while True:
            frame, preds = buffer.get(timeout=CONFIGS.timeout)
            blob = serialize(frame, preds)
            await websocket.send_bytes(blob)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info('The websocket connection has closed.')
    except asyncio.CancelledError:
        logger.info('The websocket task was cancelled.')
    except Exception as e:
        logger.exception(f'An unexpected error occurred: {str(e)}')


@app.websocket('/inference/det')
async def websocket_endpoint_det(websocket: WebSocket):
    await websocket_endpoint(websocket, YOLO_BUFFER)


# from pydantic import BaseModel
#
# class UpdateRequest(BaseModel):
#     text: str
#
# @app.put('/var/update')
# def update_var(update_request: UpdateRequest):
#     text = update_request.text
#     CAP.change_source(text)
#     logger.info(f'Receive request `UPDATE`: {text}')
#     return {"message": "Data updated successfully",
#             "new_value": text}
#
# @app.websocket('/inference/est')
# async def websocket_endpoint_det(websocket: WebSocket):
#     await websocket_endpoint(websocket, EST)