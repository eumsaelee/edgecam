# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import gc
import typing
import abc

import torch
import numpy as np
import ultralytics


Preds = typing.Dict[str, np.ndarray]


class PytorchModel(abc.ABC):

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model

    @abc.abstractmethod
    def predict(self, frame: np.ndarray) -> Preds:
        pass

    def release(self) -> None:
        if next(self._model.parameters()).device.type == 'cuda':
            self._model.to('cpu')
            torch.cuda.empty_cache()
        del self._model
        gc.collect()


class Yolov8(PytorchModel):

    def __init__(self, model_pt: str="yolov8m.pt") -> None:
        super().__init__(ultralytics.YOLO(model_pt))

    def predict(self, frame: np.ndarray) -> Preds:
        """ 이미지에서 객체를 탐지하고 결과를 딕셔너리 형태로 반환한다.

        반환값은 {'boxes': np.ndarray} 이며,
        키 'boxes'의 np.ndarray는 객체의 바운딩 박스 정보를 나타낸다.

        바운딩 박스
        --------
        바운딩 박스 배열의 shape은 (n, 7)이며, 여기서 'n'은 실제 탐지된
        객체의 수를 나타낸다. 각 열은 인덱스 순서대로 아래와 같이 구성된다.

            x_min: 박스의 좌상단 x 좌표
            y_min: 박스의 좌상단 y 좌표
            x_max: 박스의 우하단 x 좌표
            y_max: 박스의 우하단 y 좌표
            box_id: 객체 식별자
            box_conf: 박스 신뢰도(확률)
            category_id: 객체의 카테고리 식별자

        매개변수
        ------
        frame (np.ndarray): 프레임 이미지

        반환값
        ----
        typing.Dict[str, np.ndarray]: 객체 탐지 결과를 포함하는 딕셔너리.
        """
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
        return {"boxes": boxes}


class Yolov8Pose(PytorchModel):

    def __init__(self, model_pt: str='yolov8m-pose.pt'):
        super().__init__(ultralytics.YOLO(model_pt))

    def predict(self, frame: np.ndarray) -> Preds:
        """ 이미지에서 사람의 자세를 추정하고 결과를 딕셔너리 형태로 반환한다.

        반환값은 {'boxes': np.ndarray, 'kptss': np.ndarray} 이며,
        키 'boxes'의 np.ndarray는 사람 객체의 바운딩 박스를,
        키 'kptss'의 np.ndarray는 사람 객체의 키포인트 정보를 나타낸다.

        바운딩 박스
        ---------
        바운딩 박스 배열의 shape은 (n, 7)이며, 여기서 'n'은 실제 탐지된
        사람의 수를 나타낸다. 각 열은 인덱스 순서대로 아래와 같이 구성된다.

            x_min: 박스의 좌상단 x 좌표
            y_min: 박스의 좌상단 y 좌표
            x_max: 박스의 우하단 x 좌표
            y_max: 박스의 우하단 y 좌표
            box_id: 객체 식별자
            box_conf: 박스 신뢰도(확률)
            category_id: 객체의 카테고리 식별자

        키포인트
        ------
        키포인트 배열의 shape은 (n, 17, 3)이며, 여기서 'n'은 탐지된
        사람의 수를, 17은 키포인트 수를, 그리고 3은 각 키포인트의 x, y,
        conf를 나타낸다.

        키포인트는 사람 객체의 관절을 마킹한 점(point)을 의미하며, 아래와
        같이 17 부위로 구성된다.

            0: '코',
            1: '왼쪽 눈'
            2: '오른쪽 눈'
            3: '왼쪽 귀'
            4: '오른쪽 귀'
            5: '왼쪽 어깨'
            6: '오른쪽 어깨'
            7: '왼쪽 팔꿈치'
            8: '오른쪽 팔꿈치'
            9: '왼쪽 손목'
            10: '오른쪽 손목'
            11: '왼쪽 엉덩이'
            12: '오른쪽 엉덩이'
            13: '왼쪽 무릎'
            14: '오른쪽 무릎'
            15: '왼쪽 발목'
            16: '오른쪽 발목'

        그리고 각 키포인트는 아래와 같은 정보를 갖는다.

            x: 키포인트의 x좌표
            y: 키포인트의 y좌표
            conf: 키포인트의 신뢰도(확률)

        매개변수
        ------
        frame (np.ndarray): 프레임 이미지

        반환값
        -----
        Dict[str, np.ndarray]: 사람의 자세 추정 결과를 포함하는 딕셔너리.
        """
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
            kptss = results[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}