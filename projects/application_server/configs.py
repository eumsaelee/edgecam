# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim
""" 프로젝트 'application_server'의 전반적인 구성 및 환경설정. """

import os
import sys
from pathlib import Path

# NOTE
# 패키지 `edgecam`의 부모 디렉토리 경로. 추후 setup.py를 통해 패키지
# 설치를 지원하면 아래 두 라인을 제거할 것이다.
EDGECAM_DIR = str(Path(__file__).parents[2].absolute())
sys.path.append(EDGECAM_DIR)

# from edgecam.vision.yolo.models import DetYOLO, EstYOLO


# --------------
# 디렉토리 경로 관련
# --------------
#
# 프로젝트 `application_server`의 루트 디렉토리 경로. 리소스 접근을
# 위해 필요한 경우가 있다.
BASE_DIR = str(Path(__file__).parent.absolute())
#
# 트루타입 폰트들이 저장된 디렉토리. pillow를 이용해 이미지에 한글을 표시
# 할 때 한글 지원 폰트가 필요하다.
FONTS_DIR = os.path.join(BASE_DIR, 'static/fonts')


# -------------------
# 추론 및 딥러닝 모델 관련
# -------------------
#
# 서비스 가능한 모델의 이름과 참조 타입을 추가
# MODELS = {
#     'yolov8': DetYOLO,
#     'yolov8-pose': EstYOLO,}
#
# 서비스 가능한 모델이 사용할 가중치 파일 목록을 담은 사전 객체. 마찬가지로
# 새로운 모델을 추가하였다면, 이 항목에서 추가해 주어야 한다.
# MODELS_WEIGHTS = {
#     'yolov8': ['yolov8n.pt',
#                'yolov8s.pt',
#                'yolov8m.pt',
#                'yolov8l.pt',
#                'yolov8x.pt'],
#     'yolov8-pose': ['yolov8n-pose.pt',
#                     'yolov8s-pose.pt',
#                     'yolov8m-pose.pt',
#                     'yolov8l-pose.pt',
#                     'yolov8x-pose.pt'],
# }