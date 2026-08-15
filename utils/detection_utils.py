from scipy.special import expit
from typing import List
import cv2
import numpy as np
import json
import random


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad img while maintaining stride multiple constraints
    
    Parameters:
        img: Input img
        new_shape: Target shape, default is (640, 640)
        color: Padding color, default is (114, 114, 114)
        auto: Whether to use minimum rectangle mode
        scaleFill: Whether to stretch and fill
        scaleup: Whether to allow upscaling
        stride: Stride, default is 32
    
    Returns:
        tuple: (processed img, scale ratio, padding amount)
    """
    shape = img.shape[:2]  # [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Calculate scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width and height padding
    if auto:  # Minimum rectangle mode
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # Use stride modulo
    elif scaleFill:  # Stretch fill mode
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2
    # Direct resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    if left != 0 or top != 0 or bottom != 0 or right != 0:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # Add border
        img = img.astype(np.uint8)

    return img, np.array(ratio), np.array([dw, dh])

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)

    return y

def sigmoid(x: np.ndarray) -> np.ndarray:

    return expit(x)

def plot_one_box(x: List, img: np.ndarray, color=None, label=None, line_thickness=None) -> None:
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1]), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def plot_detect_results(output, img, is_show_label=True, color=[0, 165, 255]) -> None:
    if len(output) > 0:
        for element in output:
            if element[-1] == "low_threshold":
                continue
            xyxy = [element[0], element[1], element[2], element[3]]
            cls = element[4]
            conf = element[5]
            label = None
            if is_show_label:
                label = f'{cls} {conf:.2f}'

            plot_one_box(xyxy, img, color, label)