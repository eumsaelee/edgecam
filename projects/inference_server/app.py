# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import asyncio
from typing import Any

import cv2
from loguru import logger
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from edgecam.serialize import serialize

from src.pipeline import Config, Pipeline


PIPELINE = Pipeline()
CONFIG = Config(
    video_source='rtsp://192.168.1.102:12554/stream2',
    video_api_pref=cv2.CAP_FFMPEG,
    video_buffersize=5,
    yolo_pt_name='yolov8m.pt',
    yolo_tracking_on=True,
    yolo_buffersize=5,
    buffer_timeout=30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # start server
    logger.info('Starting the server ...')
    PIPELINE.start(CONFIG)
    logger.info('The server has started.')
    logger.info('Waiting for requests ...')
    yield
    # stop server
    logger.info('Stopping the server ...')
    PIPELINE.stop()
    logger.info('Done.')


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://127.0.0.1:8888',
        'http://172.27.1.11:8888'
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


async def websocket_endpoint(ws: WebSocket, buffer: Any):
    logger.info('Received a request via websocket...')
    await ws.accept()
    host = ws.client.host
    port = ws.client.port
    logger.info(f'{host}:{port} has been accepted!')
    try:
        while True:
            frame, preds = buffer.get()
            blob = serialize(frame, preds)
            await ws.send_bytes(blob)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info('Connection has closed.')
    except asyncio.CancelledError:
        logger.info('Connection has cancelled.')
    except Exception as e:
        logger.exception(f'Unexpected error: {str(e)}')


@app.websocket('/inference/det')
async def websocket_endpoint_det(ws: WebSocket):
    await websocket_endpoint(ws, PIPELINE.yolo_buffer)


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
