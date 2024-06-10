from typing import Tuple

import cv2
import numpy as np


def plot_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    color: Tuple[int, int, int],
    font_face: int=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float=0.5,
    thickness: int=1,
    bgcolor: Tuple[int, int, int]=None,
):
    """ 텍스트를 렌더링한다(OpenCV 4.5.0).

    이 함수는 텍스트를 이미지에 렌더링 한다. 지정된 폰트(font_face)를 사용해
    렌더링할 수 없는 기호는 물음표(?)로 대체된다. 배경색(bgcolor)을 함께
    입력하면 이 색상으로 채워진, 경계선 없는 텍스트 박스도 함께 렌더링된다.

    ┌─────────────┐
    │Hello, World!│
    x─────────────┘

    위 그림에서 'Hello, World!'는 텍스트를, 이것을 감싸고 있는 사각형은
    경계선 없는 텍스트 박스를, 그리고 x는 org의 위치를 나타낸다.

    Args:
        img: 이미지
        text: 텍스트
        org: 텍스트의 좌하단 모서리 좌표
        color: 텍스트 색상
        font_face: 폰트
        font_scale: 폰트 배율
        thickness: 폰트 두께
        bgcolor (optional): 텍스트 박스 색상

    Examples:
        1) 이미지와 텍스트를 준비. 여기서는 임의의 이미지를 사용.
        >>> img = np.zeros(shape=(360, 640, 3), dtype=np.uint8)
        >>> text = 'Hello, World!'
        2) 텍스트는 검정색, 그리고 배경은 초록색으로 설정.
        >>> color = (0, 0, 0)
        >>> bgcolor = (0, 255, 0)
        3) 텍스트 위치는 (x=50, y=50) 설정.
        >>> org = (50, 50)
        4) 텍스트 렌더링. 기타 설정은 디폴트 값을 사용.
        >>> plot_text(img, text, org, color, bgcolor=bgcolor)
    """
    if bgcolor:
        w, h = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
        x, y = org
        pt1 = (x, y - h)
        pt2 = (x + w, y)
        cv2.rectangle(img, pt1, pt2, bgcolor, cv2.FILLED)

    cv2.putText(img, text, org, font_face, font_scale, color, thickness)
