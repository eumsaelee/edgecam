# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import gc
import typing
import abc

import torch
import numpy as np
import ultralytics


class NotTorchModel(Exception):
    """ 모델 타입이 `torch.nn.Module`이 아닐 경우 발생한다. """
    pass


Preds = typing.Dict[str, np.ndarray]


class TorchModel(abc.ABC):
    """ `torch.nn.Module` 기반 모델 인터페이스. """

    def __init__(self, model: typing.Any) -> None:
        if not isinstance(model, torch.nn.Module):
            raise NotTorchModel
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


class Identity:
    """ 어떠한 추론도 수행하지 않는 모델 클래스. """

    def __init__(self) -> None:
        pass

    def predict(self, frame: typing.Any) -> Preds:
        return {'empty': np.ndarray([])}

    def release(self) -> None:
        pass


class Yolov8n(TorchModel):
    """ Ultralytics Yolov8n 모델을 래핑한 객체 탐지 클래스. """

    categories_eng = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
        8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
        26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
        48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
        52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
        79: 'toothbrush'
    }

    categories_kor = {
        0: '사람', 1: '자전거', 2: '자동차', 3: '오토바이',
        4: '비행기', 5: '버스', 6: '기차', 7: '트럭',
        8: '보트', 9: '신호등', 10: '소화전',
        11: '정지 신호', 12: '주차 미터기', 13: '벤치',
        14: '새', 15: '고양이', 16: '개', 17: '말',
        18: '양', 19: '소', 20: '코끼리', 21: '곰',
        22: '얼룩말', 23: '기린', 24: '백팩', 25: '우산',
        26: '핸드백', 27: '넥타이', 28: '여행 가방', 29: '프리스비',
        30: '스키', 31: '스노보드', 32: '스포츠 볼', 33: '연',
        34: '야구 방망이', 35: '야구 글러브', 36: '스케이트보드',
        37: '서핑보드', 38: '테니스 라켓', 39: '병',
        40: '와인 글라스', 41: '컵', 42: '포크', 43: '나이프',
        44: '스푼', 45: '그릇', 46: '바나나', 47: '사과',
        48: '샌드위치', 49: '오렌지', 50: '브로콜리', 51: '당근',
        52: '핫도그', 53: '피자', 54: '도넛', 55: '케이크',
        56: '의자', 57: '소파', 58: '화분', 59: '침대',
        60: '식탁', 61: '화장실', 62: '텔레비전', 63: '노트북',
        64: '마우스', 65: '리모컨', 66: '키보드', 67: '휴대폰',
        68: '전자레인지', 69: '오븐', 70: '토스터', 71: '싱크대',
        72: '냉장고', 73: '책', 74: '시계', 75: '꽃병',
        76: '가위', 77: '테디 베어', 78: '헤어 드라이어',
        79: '칫솔'
    }

    def __init__(self) -> None:
        super().__init__(ultralytics.YOLO("yolov8n.pt"))

    def predict(self, frame: np.ndarray) -> Preds:
        """ 객체를 탐지한다. 객체 추적 기능이 포함되어 있으므로 전후 프레임에서 박스
            식별자가 같다면 동일한 객체를 의미한다.

            boxes.ndim: 2
            boxes.shape: (n, 7)  # 여기서 n은 탐지된 객체의 수
            box columns:
                x_min: 박스 좌상단 점의 x 좌표,
                y_min: 박스 좌상단 점의 y 좌표,
                x_max: 박스 우하단 점의 x 좌표,
                y_max: 박스 우하단 점의 y 좌표,
                box_id: 박스 식별자,
                box_conf: 박스 신뢰도,
                box_category_id: 박스 카테고리 식별자.
        """
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
        return {"boxes": boxes}


class Yolov8nPose(TorchModel):
    """ Ultralytics Yolov8n Pose 모델을 래핑한 객체탐지 클래스. """

    keypoints_eng = {
        0: 'Nose', 1: 'Left eye', 2: 'Right eye', 3: 'Left ear',
        4: 'Right ear', 5: 'Left shoulder', 6: 'Right shoulder',
        7: 'Left elbow', 8: 'Right elbow', 9: 'Left wrist',
        10: 'Right wrist', 11: 'Left hip', 12: 'Right hip',
        13: 'Left knee', 14: 'Right knee', 15: 'Left ankle',
        16: 'Right ankle'
    }

    keypoints_kor = {
        0: '코', 1: '왼쪽 눈', 2: '오른쪽 눈', 3: '왼쪽 귀',
        4: '오른쪽 귀', 5: '왼쪽 어깨', 6: '오른쪽 어깨',
        7: '왼쪽 팔꿈치', 8: '오른쪽 팔꿈치', 9: '왼쪽 손목',
        10: '오른쪽 손목', 11: '왼쪽 엉덩이', 12: '오른쪽 엉덩이',
        13: '왼쪽 무릎', 14: '오른쪽 무릎', 15: '왼쪽 발목',
        16: '오른쪽 발목'
    }


    # 키포인트 간 연결 관계. 예를 들어, 0: [1, 2]는 코(Nose, 0)가 왼쪽 눈(Left
    # eye, 1)과 오른쪽 눈(Right eye, 2)에 선으로 연결된다는 의미이다.
    connections = {
        0: [1, 2,], 1: [3,], 2: [4,], 3: [], 4: [], 5: [6, 7, 11,],
        6: [8, 12,], 7: [9,], 8: [10,], 9: [], 10: [], 11: [12, 13,],
        12: [14,], 13: [15,], 14: [16,], 15: [], 16: []
    }

    def __init__(self):
        super().__init__(ultralytics.YOLO('yolov8n-pose.pt'))

    def predict(self, frame: np.ndarray) -> Preds:
        """ 사람을 탐지하고 자세를 추정한다. 객체 추적 기능이 포함되어 있으므로 전후
            프레임에서 박스 식별자가 같다면 동일한 객체를 의미한다.

            boxes.ndim: 2
            boxes.shape: (n, 7)  # 여기서 n은 탐지된 객체의 수
            box columns:
                x_min: 박스 좌상단 점의 x 좌표,
                y_min: 박스 좌상단 점의 y 좌표,
                x_max: 박스 우하단 점의 x 좌표,
                y_max: 박스 우하단 점의 y 좌표,
                box_id: 박스 식별자,
                box_conf: 박스 신뢰도,
                box_category_id: 박스 카테고리 식별자.
            
            kptss.ndim: 3
            kptss.shape: (n, 17, 3)  # 여기서 n은 탐지된 사람의 수
            kpts columns:
                x: 키포인트의 x 좌표,
                y: 키포인트의 y 좌표,
                conf: 키포인트 신뢰도.
        """
        results = self._model.track(frame, persist=True, verbose=False)
        if results is None:
            boxes = np.array([])
            kptss = np.array([])
        else:
            boxes = results[0].boxes.data.cpu().numpy()
            kptss = results[0].keypoints.data.cpu().numpy()
        return {"boxes": boxes, "kptss": kptss}