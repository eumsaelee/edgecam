# -*- coding: utf-8 -*-
# Author: Seunghyeon Kim

import sys
from configs import EDGECAM_DIR
sys.path.append(EDGECAM_DIR)

import os
import enum
import typing

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from configs import FONTS_DIR


class NanumGothic(enum.Enum):

    default = os.path.join(FONTS_DIR, 'NanumGothic.ttf')
    light = os.path.join(FONTS_DIR, 'NanumGothicLight.ttf')
    bold = os.path.join(FONTS_DIR, 'NanumGothicBold.ttf')
    extra_bold = os.path.join(FONTS_DIR, 'NanumGothicExtraBold.ttf')


# def plot_text(
#     img: np.ndarray,
#     text: str,
#     org: Tuple[int, int],
#     color: Tuple[int, int, int],
#     font_face: int=cv2.FONT_HERSHEY_SIMPLEX,
#     font_scale: float=0.5,
#     thickness: int=1,
#     bgcolor: Tuple[int, int, int]=None,
# ):
#     if bgcolor:
#         w, h = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
#         x, y = org
#         pt1 = (x, y - h)
#         pt2 = (x + w, y)
#         cv2.rectangle(img, pt1, pt2, bgcolor, cv2.FILLED)
#
#     cv2.putText(img, text, org, font_face, font_scale, color, thickness)


def plot_text(img: np.ndarray,
              text: str,
              org: typing.Tuple[int, int],
              color: typing.Tuple[int, int, int],
              bgcolor: typing.Tuple[int, int, int],
              font_path: str,
              font_size: int):
    img_pil = Image.fromarray(img)
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img_pil)
    textbbox = draw.textbbox((org[0], org[1]-1), text, font=font)  # left, top, light, bottom
    draw.rectangle(textbbox, fill=bgcolor)
    draw.text(org, text, color, font)
    img = np.array(img_pil)
    return img
