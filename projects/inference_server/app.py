# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import typing
import asyncio
from dataclasses import dataclass

import cv2
from loguru import logger
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.capture import VideoCapture
from src.inference import Inference, ObjectDetectorYolov8, PoseEstimatorYolov8
from edgecam.vision_ai.serialize import serialize


@dataclass
class CaptureProps:
    source: typing.Union[int, str]='rtsp://192.168.1.101:12554/profile2/media.smp'
    api_pref: int=cv2.CAP_FFMPEG
    maxsize: int=5


@dataclass
class DetectorProps:
    model_pt: str='yolov8m.pt'
    # model_pt: str='safety-equipment-yolov8n.pt'
    maxsize: int=5
    tracking: bool=True


# @dataclass
# class EstimatorProps:
#     model_pt: str='yolov8m-pose.pt'
#     maxsize: int=5
#     tracking: bool=False


CAP_PROP = CaptureProps()
DET_PROP = DetectorProps()
# EST_PROP = EstimatorProps()

CAP = None
DET = None
# EST = None


def init_cap() -> None:
    global CAP
    CAP = VideoCapture(maxsize=CAP_PROP.maxsize)
    CAP.open(CAP_PROP.source, CAP_PROP.api_pref)


def init_det() -> None:
    global DET
    DET = ObjectDetectorYolov8(DET_PROP.model_pt,
                               DET_PROP.maxsize,
                               DET_PROP.tracking)
    DET.inference(CAP.fetch_frame)


# def init_est() -> None:
#     global EST
    # EST = PoseEstimatorYolov8(EST_PROP.model_pt,
    #                           EST_PROP.maxsize,
    #                           EST_PROP.tracking)
#     EST.inference(CAP.fetch_frame)


def start_server() -> None:
    logger.info('Starting the inference server ...')
    init_cap()
    init_det()
    # init_est()
    logger.info('The inference server has started.')
    logger.info('Waiting for requests ...')


def stop_server() -> None:
    logger.info('Stopping the inference server ...')
    DET.release()
    # EST.release()
    CAP.release()
    logger.info('The inference server has stopped.')


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_server()
    yield
    stop_server()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://127.0.0.1:8888', 'http://172.27.1.11:8888'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


# NOTE IT WILL BE ADDED
# from pydantic import BaseModel
# class UpdateRequest(BaseModel):
#     text: str
# @app.put('/var/update')
# def update_var(update_request: UpdateRequest):
#     text = update_request.text
#     CAP.change_source(text)
#     logger.info(f'Receive request `UPDATE`: {text}')
#     return {"message": "Data updated successfully",
#             "new_value": text}


async def websocket_endpoint(websocket: WebSocket, inference: Inference):
    logger.info('Received a request for stream via websocket...')
    await websocket.accept()
    host = websocket.client.host
    port = websocket.client.port
    logger.info(f'The request from {host}:{port} has been accepted!')
    try:
        while True:
            frame, preds = inference.fetch_result()
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
    await websocket_endpoint(websocket, DET)


# @app.websocket('/inference/est')
# async def websocket_endpoint_det(websocket: WebSocket):
#     await websocket_endpoint(websocket, EST)