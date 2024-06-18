# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from pathlib import Path
BASEDIR = Path(__file__).parents[2].absolute()
sys.path.append(str(BASEDIR))

import typing
import asyncio

import cv2
from loguru import logger
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager

from src.capture import VideoCaptureHandler
from src.detection import ObjectDetectionHandler
from edgecam.vision.serialize import serialize


VIDEO_CAPTURE_SOURCE = 'rtsp://192.168.1.102:12554/stream2'
VIDEO_CAPTURE_API_PREF = cv2.CAP_FFMPEG
VIDEO_CAPTURE_BUFFER_SIZE = 4
OBJECT_DETECTION_MODEL_NAME = 'yolov8m.pt'
OBJECT_DETECTION_BUFFER_SIZE = 8

VIDEO_CAPTURE_HANDLER = None
OBJECT_DETECTION_HANDLER = None


def startup() -> None:
    logger.info('Called startup().')
    _init_video_capture_handler()
    _init_object_detection_handler()


def shutdown() -> None:
    logger.info('Called shutdown().')
    OBJECT_DETECTION_HANDLER.stop_detection()
    VIDEO_CAPTURE_HANDLER.stop_capturing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield
    shutdown()


def _init_video_capture_handler() -> None:
    global VIDEO_CAPTURE_HANDLER
    VIDEO_CAPTURE_HANDLER = VideoCaptureHandler(VIDEO_CAPTURE_SOURCE, VIDEO_CAPTURE_API_PREF)
    VIDEO_CAPTURE_HANDLER.start_capturing()


def _init_object_detection_handler() -> None:
    global OBJECT_DETECTION_HANDLER
    OBJECT_DETECTION_HANDLER = ObjectDetectionHandler(OBJECT_DETECTION_MODEL_NAME, OBJECT_DETECTION_BUFFER_SIZE)
    OBJECT_DETECTION_HANDLER.start_detection(VIDEO_CAPTURE_HANDLER.read_frame)


app = FastAPI(lifespan=lifespan)


@app.websocket('/stream/object_detection')
async def detect_objects(websocket: WebSocket):
    await websocket.accept()
    logger.info('WebSocket: object_detection accepted!')
    try:
        while True:
            frame, preds = OBJECT_DETECTION_HANDLER.read_output()
            blob = serialize(frame, preds)
            await websocket.send_bytes(blob)
            await asyncio.sleep(0)
    except WebSocketDisconnect:
        logger.info('WebSocket: object_detection disconnected!')
    except:
        logger.exception('WebSocket: object_detection error')
