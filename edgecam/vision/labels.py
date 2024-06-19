# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

yolo_categories_eng = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


yolo_categories_kor = {
    0: '사람',
    1: '자전거',
    2: '자동차',
    3: '오토바이',
    4: '비행기',
    5: '버스',
    6: '기차',
    7: '트럭',
    8: '보트',
    9: '신호등',
    10: '소화전',
    11: '정지 신호',
    12: '주차 미터기',
    13: '벤치',
    14: '새',
    15: '고양이',
    16: '개',
    17: '말',
    18: '양',
    19: '소',
    20: '코끼리',
    21: '곰',
    22: '얼룩말',
    23: '기린',
    24: '배낭',
    25: '우산',
    26: '핸드백',
    27: '넥타이',
    28: '여행 가방',
    29: '프리스비',
    30: '스키',
    31: '스노보드',
    32: '스포츠 공',
    33: '연',
    34: '야구 방망이',
    35: '야구 글러브',
    36: '스케이트보드',
    37: '서핑보드',
    38: '테니스 라켓',
    39: '병',
    40: '와인잔',
    41: '컵',
    42: '포크',
    43: '나이프',
    44: '숟가락',
    45: '그릇',
    46: '바나나',
    47: '사과',
    48: '샌드위치',
    49: '오렌지',
    50: '브로콜리',
    51: '당근',
    52: '핫도그',
    53: '피자',
    54: '도넛',
    55: '케이크',
    56: '의자',
    57: '소파',
    58: '화분',
    59: '침대',
    60: '식탁',
    61: '화장실',
    62: '텔레비전',
    63: '랩탑',
    64: '마우스',
    65: '리모컨',
    66: '키보드',
    67: '휴대폰',
    68: '전자레인지',
    69: '오븐',
    70: '토스터',
    71: '싱크대',
    72: '냉장고',
    73: '책',
    74: '시계',
    75: '꽃병',
    76: '가위',
    77: '테디 베어',
    78: '헤어 드라이어',
    79: '칫솔'
}


yolov8_keypoints_eng = {
    0: 'Nose',
    1: 'Left eye',
    2: 'Right eye',
    3: 'Left ear',
    4: 'Right ear',
    5: 'Left shoulder',
    6: 'Right shoulder',
    7: 'Left elbow',
    8: 'Right elbow',
    9: 'Left wrist',
    10: 'Right wrist',
    11: 'Left hip',
    12: 'Right hip',
    13: 'Left knee',
    14: 'Right knee',
    15: 'Left ankle',
    16: 'Right ankle'
}


yolov8_keypoints_kor = {
    0: '코', 
    1: '왼쪽 눈', 
    2: '오른쪽 눈', 
    3: '왼쪽 귀', 
    4: '오른쪽 귀', 
    5: '왼쪽 어깨', 
    6: '오른쪽 어깨', 
    7: '왼쪽 팔꿈치', 
    8: '오른쪽 팔꿈치', 
    9: '왼쪽 손목', 
    10: '오른쪽 손목', 
    11: '왼쪽 엉덩이', 
    12: '오른쪽 엉덩이', 
    13: '왼쪽 무릎', 
    14: '오른쪽 무릎', 
    15: '왼쪽 발목', 
    16: '오른쪽 발목'
}


yolov8_keypoints_connections = {
    0: [1, 2,],
    1: [3,],
    2: [4,],
    3: [],
    4: [],
    5: [6, 7, 11,],
    6: [8, 12,],
    7: [9,],
    8: [10,],
    9: [],
    10: [],
    11: [12, 13,],
    12: [14,],
    13: [15,],
    14: [16,],
    15: [],
    16: []
}