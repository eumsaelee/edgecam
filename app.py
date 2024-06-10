# -*- coding: utf-8 -*-
# Author: SeungHyeon Kim
import threading
from queue import Queue
from random import randint  # temporary
from abc import ABC, abstractmethod

import cv2
import numpy as np
from loguru import logger
from flask import (Flask,
                   Response,
                   render_template,
                   stream_with_context,
                   jsonify,
                   request)

from src.general.buffers import ResizablePushoutQueue
from src.video.readers import FrameReader
from src.ai.vision.models import TorchModule, Yolov8n, Yolov8nPose


class AlreadyRunning(Exception): pass
class NotRunning(Exception): pass
class NotOpened(Exception): pass

class ThreadError(RuntimeError): pass
class EncodeError(RuntimeError): pass


class Bufferer(ABC):
    """ Interface """

    def __init__(self, buffer: Queue):
        self._buffer = buffer
        self._stop = False
        self._thread: threading.Thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            raise AlreadyRunning(
                'The buffering thread is already running.')
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
    
    def stop(self):
        if not self._thread or not self._thread.is_alive():
            raise NotRunning(
                'The buffering thread is not running.')
        self._stop = True
        self._thread.join()

    @abstractmethod
    def _run(self):
        pass


class FrameBufferer(Bufferer):

    def __init__(self,
                 reader: FrameReader,
                 buffer: ResizablePushoutQueue):
        self._reader = reader
        super().__init__(buffer)

    def start(self):
        if not self._reader.is_opened():
            raise NotOpened(
                'The reader is not opened.')
        super().start()

    def stop(self):
        super().stop()
        self._reader.release()

    def _run(self):
        try:
            while not self._stop:
                frame = self._reader.read()
                self._buffer.put_nowait(frame, push_out=True)
        except Exception:
            logger.exception(
                'An unexpected error has occurred. Please '
                'check the message.')
        finally:
            if not self._stop:
                self._stop = True
                self._reader.release()
                logger.warning(
                    'The thread has stopped on its own '
                    'without a stop signal. This may be an '
                    'unexpected behavior.')


class PredsBufferer(Bufferer):

    def __init__(self,
                 model: TorchModule,
                 frame_buffer: ResizablePushoutQueue,
                 preds_buffer: ResizablePushoutQueue):
        self._model = model
        self._frame_buffer = frame_buffer
        super().__init__(preds_buffer)

    def stop(self):
        super().stop()
        self._model.release()

    def _run(self):
        try:
            while not self._stop:
                frame = self._frame_buffer.get()
                preds = self._model.predict(frame)
                self._buffer.put_nowait((frame, preds), push_out=True)
        except Exception:
            logger.exception(
                'An unexpected error has occurred. Please '
                'check the message.')
        finally:
            if not self._stop:
                self._stop = True
                self._model.release()
                logger.warning(
                    'The thread has stopped on its own '
                    'without a stop signal. This may be an '
                    'unexpected behavior.')


# ---------------------------------------------------------------------------


app = Flask(__name__)

VIDEO_SOURCE = 'rtsp://192.168.1.101:12554/profile2/media.smp'
VIDEO_API_PREFERENCE = cv2.CAP_FFMPEG
VIDEO_BUFFER_MAXSIZE = 10
MIMETYPE = 'multipart/x-mixed-replace; boundary=frame'

_FRAME_READER = FrameReader()
_FRAME_BUFFER = ResizablePushoutQueue()
_FRAME_BUFFERER = FrameBufferer(_FRAME_READER, _FRAME_BUFFER)

_YOLOV8N_MODEL = Yolov8n()
_YOLOV8N_PREDS_BUFFER = ResizablePushoutQueue()
_YOLOV8N_PREDS_BUFFERER = PredsBufferer(
    _YOLOV8N_MODEL, _FRAME_BUFFER, _YOLOV8N_PREDS_BUFFER)

_YOLOV8NPOSE_MODEL = Yolov8nPose()
_YOLOV8NPOSE_PREDS_BUFFER = ResizablePushoutQueue()
_YOLOV8NPOSE_PREDS_BUFFERER = PredsBufferer(
    _YOLOV8NPOSE_MODEL, _FRAME_BUFFER, _YOLOV8NPOSE_PREDS_BUFFER)


def init():
    logger.debug('called init().')
    _FRAME_READER.open(VIDEO_SOURCE, VIDEO_API_PREFERENCE)
    _FRAME_BUFFER.maxsize = VIDEO_BUFFER_MAXSIZE
    _FRAME_BUFFERER.start()
    _YOLOV8N_PREDS_BUFFERER.start()
    _YOLOV8NPOSE_PREDS_BUFFERER.start()


def halt():
    logger.debug('called halt().')
    for bufferer in [_YOLOV8N_PREDS_BUFFERER,
                     _YOLOV8NPOSE_PREDS_BUFFERER,
                     _FRAME_BUFFERER]:
        try:
            bufferer.stop()
        except NotRunning:
            pass


def to_multipart(frame: np.ndarray) -> bytes:
    retval, frame = cv2.imencode('.jpg', frame)
    if not retval:
        raise EncodeError(
            'Failed to encode the frame.')
    frame = frame.tobytes()
    return (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame + b'\r\n')


# ---------------------------------------------------------------------------


@app.route('/')
def index():
    return render_template('summary.html')


# ---------------------------------------------------------------------------


@app.route('/view0')
def view0():
    def view():
        try:
            while True:
                frame = _FRAME_BUFFER.get()
                pt0 = randint(0, 639), randint(0, 359)
                pt1 = randint(0, 639), randint(0, 359)
                cv2.rectangle(frame, pt0, pt1, (0, 255, 0), 1)
                yield to_multipart(frame)
        except TimeoutError:
            pass

    return Response(stream_with_context(view()),
                    mimetype=MIMETYPE)


# ---------------------------------------------------------------------------


@app.route('/view1')
def view1():
    def view():
        try:
            while True:
                frame = _FRAME_BUFFER.get()
                pt0 = randint(0, 639), randint(0, 359)
                pt1 = randint(0, 639), randint(0, 359)
                cv2.rectangle(frame, pt0, pt1, (0, 0, 255), 1)
                yield to_multipart(frame)
        except TimeoutError:
            pass

    return Response(stream_with_context(view()),
                    mimetype=MIMETYPE)


# ---------------------------------------------------------------------------


_DANGER_AREA_DOTS = None
_DANGER_AREA_MASK = None
_DANGER_AREA_UPDATED = False


@app.route('/view/danger_area')
def danger_area():
    def generate_view():
        global _DANGER_AREA_UPDATED, _DANGER_AREA_MASK
        try:
            while True:
                frame, preds = _YOLOV8N_PREDS_BUFFER.get()

                # Update danger area mask.
                if _DANGER_AREA_UPDATED:
                    if _DANGER_AREA_DOTS is None:
                        _DANGER_AREA_MASK = None
                    else:
                        _DANGER_AREA_MASK = np.zeros(shape=frame.shape[:-1], dtype=np.uint8)
                        cv2.fillPoly(_DANGER_AREA_MASK, [_DANGER_AREA_DOTS], 255)

                # Draw the danger dots and lines.
                if _DANGER_AREA_DOTS is not None:
                    cv2.polylines(frame, [_DANGER_AREA_DOTS], True, (0, 0, 255), 1)
                    for dot in _DANGER_AREA_DOTS[0]:
                        cv2.circle(frame, (dot[0], dot[1]), 3, (0, 0, 255), cv2.FILLED)

                # Draw bounding box in the danger area.
                if _DANGER_AREA_MASK is not None:
                    boxes = preds['boxes']
                    boxes_xyxy = boxes[:, :4].astype(np.uint16)
                    boxes_cxcy = np.empty(shape=(len(boxes_xyxy), 2), dtype=np.uint16)
                    boxes_cxcy[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) >> 1
                    boxes_cxcy[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) >> 1
                    boxes_xyxy = boxes_xyxy[_DANGER_AREA_MASK[boxes_cxcy[:, 1], boxes_cxcy[:, 0]] == 255]
                    for box in boxes_xyxy:
                        pt0 = int(box[0]), int(box[1])
                        pt1 = int(box[2]), int(box[3])
                        cv2.rectangle(frame, pt0, pt1, (0, 0, 255), 1)

                _DANGER_AREA_UPDATED = False

                yield to_multipart(frame)
        except TimeoutError:
            pass

    return Response(stream_with_context(generate_view()),
                    mimetype=MIMETYPE)


@app.route('/settings/view/danger_area')
def settings_danger_area():
    return render_template('settings_danger_area.html')


@app.route('/update/view/danger_area', methods=['POST'])
def update_danger_area():
    global _DANGER_AREA_UPDATED, _DANGER_AREA_DOTS, _DANGER_AREA_MASK
    data = request.json
    dots = data['dots']

    _DANGER_AREA_UPDATED = True
    if len(dots) < 3:
        _DANGER_AREA_DOTS = None
    else:
        _DANGER_AREA_DOTS = np.array(dots).reshape(1, -1, 2)
    return jsonify({'status': 'Coordinates updated successfully!',
                    'received': dots})


# ---------------------------------------------------------------------------


@app.route('/view3')
def view3():
    def generate_view():
        try:
            while True:
                frame, preds = _YOLOV8NPOSE_PREDS_BUFFER.get()
                boxes = preds['boxes'][:, :4]
                for box in boxes:
                    pt0 = int(box[0]), int(box[1])
                    pt1 = int(box[2]), int(box[3])
                    cv2.rectangle(frame, pt0, pt1, (0, 0, 255), 1)
                yield to_multipart(frame)
        except TimeoutError:
            pass

    return Response(stream_with_context(generate_view()),
                    mimetype=MIMETYPE)


# ---------------------------------------------------------------------------


if __name__ == '__main__':
    try:
        init()
        app.run(host='0.0.0.0', port=8000)
    except Exception:
        logger.exception('Exception')
    finally:
        halt()
