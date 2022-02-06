"""
This file is under Apache License 2.0, see more details at https://www.apache.org/licenses/LICENSE-2.0
Author: Coder.AN, contact me at an.hongjun@foxmail.com
Github: https://github.com/BestAnHongjun/VOC-Augmentation

Reference:
[1]Code in function __viz_bbox refers to code by Ge Zheng et al. in project YOLO-X,
see more detail at https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/visualize.py,
which is under Apache License 2.0.
"""

import cv2
import numpy as np


def viz_vdict(vdict):
    image = vdict.get("image").copy()
    objects = vdict.get("objects")
    for obj in objects:
        if "confidence" in obj:
            confidence = obj.get("confidence")
        else:
            confidence = -1
        __viz_bbox(image, obj.get("bbox"), obj.get("class_id"), obj.get("class_name"), confidence)
    return image


def __viz_bbox(image, bbox, cls_id, cls_name, confidence=-1):
    x_min, y_min, x_max, y_max = bbox
    color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
    if confidence == -1:
        text = '{}'.format(cls_name)
    else:
        text = '{}:{:.2f}%'.format(cls_name, confidence * 100)
    txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
    cv2.rectangle(
        image,
        (x_min, y_min + 1),
        (x_min + txt_size[0] + 1, y_min + int(1.5 * txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(image, text, (x_min, y_min + txt_size[1]), font, 0.4, txt_color, thickness=1)


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
