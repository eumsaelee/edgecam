# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import asyncio

import cv2
from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from src.capture import VideoCaptureHandler
from src.detection import ObjectDetectionHandler
from edgecam.vision.serialize import serialize


CAP_PROP_SOURCE = 'rtsp://192.168.1.101:12554/profile2/media.smp'
CAP_PROP_APIPREF = cv2.CAP_FFMPEG
CAP_PROP_BUFFERSIZE = 4
ODT_PROP_MODELNAME = 'yolov8m.pt'
ODT_PROP_BUFFERSIZE = 8
PET_PROP_MODELNAME = 'yolov8m-pose.pt'
PET_PROP_BUFFERSIZE = 8

CAP_HANDLER = None
ODT_HANDLER = None


def startup() -> None:
    logger.info('Called startup().')
    _init_video_capture_handler()
    _init_object_detection_handler()


def shutdown() -> None:
    logger.info('Called shutdown().')
    ODT_HANDLER.stop_detection()
    CAP_HANDLER.stop_capturing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


def _init_video_capture_handler() -> None:
    global CAP_HANDLER
    CAP_HANDLER = VideoCaptureHandler(CAP_PROP_SOURCE,
                                      CAP_PROP_APIPREF,
                                      CAP_PROP_BUFFERSIZE)
    CAP_HANDLER.start_capturing()


def _init_object_detection_handler() -> None:
    global ODT_HANDLER
    ODT_HANDLER = ObjectDetectionHandler(ODT_PROP_MODELNAME,
                                         ODT_PROP_BUFFERSIZE)
    ODT_HANDLER.start_detection(CAP_HANDLER.read_frame)


app = FastAPI(lifespan=lifespan)


@app.put('/var/update')
def update_var(text: str):
    logger.info(f'Receive request `UPDATE`: {text}')


@app.websocket('/stream/object_detection')
async def detect_objects(websocket: WebSocket):
    await websocket.accept()
    logger.info('WebSocket: object_detection accepted!')
    try:
        while True:
            frame, preds = ODT_HANDLER.read_output()
            blob = serialize(frame, preds)
            await websocket.send_bytes(blob)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info('WebSocket: object_detection disconnected!')
    except:
        logger.exception('WebSocket: object_detection error')
