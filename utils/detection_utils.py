from scipy.special import expit
from typing import List
import cv2
import numpy as np
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


def get_outer_bbox(bboxes: List) -> List:
    if not bboxes:
        return None

    x_min = min(b[0] for b in bboxes)
    y_min = min(b[1] for b in bboxes)
    x_max = max(b[2] for b in bboxes)
    y_max = max(b[3] for b in bboxes)

    return [x_min, y_min, x_max, y_max]


def expand_bbox_crop(image: np.ndarray, bbox, scale: float = 1.2, fill_color=(114, 114, 114)):
    """
    将bbox向外扩scale倍（宽高各乘scale），从原图中裁剪出扩展区域，超出边界用fill_color填充。

    Args:
        image: np.ndarray, shape (H, W, C)，BGR或RGB均可
        bbox: [x1, y1, x2, y2]，支持int或float
        scale: 扩展倍数，默认1.2
        fill_color: 填充颜色，默认为(114,114,114)（灰色）

    Returns:
        np.ndarray: 裁剪后的图像块，尺寸为 (new_h, new_w, C)
    """
    x1, y1, x2, y2 = bbox
    # 计算中心与宽高
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    new_w = w * scale
    new_h = h * scale

    # 扩展后的左上角和右下角（浮点）
    new_x1 = cx - new_w / 2.0
    new_y1 = cy - new_h / 2.0
    new_x2 = cx + new_w / 2.0
    new_y2 = cy + new_h / 2.0

    # 转为整数（四舍五入）
    x1_int = int(round(new_x1))
    y1_int = int(round(new_y1))
    x2_int = int(round(new_x2))
    y2_int = int(round(new_y2))

    out_w = x2_int - x1_int
    out_h = y2_int - y1_int

    # 如果扩展后尺寸非正，返回原图或空？（这里返回原图，但实际不应该发生）
    if out_w <= 0 or out_h <= 0:
        return image

    # 创建填充图像
    out_img = np.full((out_h, out_w, image.shape[2]), fill_color, dtype=image.dtype)

    # 计算原图中有效区域（与图像边界交集）
    src_x1 = max(0, x1_int)
    src_y1 = max(0, y1_int)
    src_x2 = min(image.shape[1], x2_int)
    src_y2 = min(image.shape[0], y2_int)

    # 若存在有效区域，则复制
    if src_x2 > src_x1 and src_y2 > src_y1:
        # 在目标图像中的对应起始位置
        dst_x1 = src_x1 - x1_int
        dst_y1 = src_y1 - y1_int
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        out_img[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]

    return out_img


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