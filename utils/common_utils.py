from typing import Optional, Dict, Any, List, Tuple
import base64
import cv2
import numpy as np
import os
import json
import re


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encode a BGR numpy array (from OpenCV) as JPEG and return base64 with data:image/jpeg prefix.
    """
    success, encoded = cv2.imencode('.jpg', image)
    if not success:
        raise RuntimeError("Failed to encode image as JPEG")
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"
    

def read_img(img_path: str) -> np.ndarray:
    img = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img, flags=cv2.IMREAD_COLOR)

    return img


def save_img(img: np.ndarray, img_path: str) -> None:
    img_suffix = img_path.split('.')[-1]
    cv2.imencode('.{}'.format(img_suffix), img)[1].tofile(img_path)


def video_writer(
    video_path: str,
    output_path: str,
    fourcc: str = 'mp4v',
    fps: Optional[float] = None,
    size: Optional[Tuple[int, int]] = None,
    auto_create_dir: bool = True
) -> Tuple[cv2.VideoWriter, float, int, int]:
    """
    Create a VideoWriter object for the output video, based on input video properties.

    Args:
        video_path: Path to the source video (used to read properties).
        output_path: Full path to the output video file (directory will be created if needed).
        fourcc: FourCC code as string (e.g., 'mp4v', 'XVID'). Default 'mp4v'.
        fps: Optional; if not provided, uses source video FPS.
        size: Optional (width, height); if not provided, uses source video size.
        auto_create_dir: If True, create output directory automatically.

    Returns:
        Tuple of (cv2.VideoWriter, fps, width, height).
        The VideoWriter is already opened and ready to write.

    Raises:
        IOError: If source video cannot be opened or output writer cannot be created.
    """
    if auto_create_dir:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open source video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # fps and size
    out_fps = fps if fps is not None else src_fps
    out_width, out_height = size if size is not None else (src_width, src_height)

    # create VideoWriter
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    out = cv2.VideoWriter(output_path, fourcc_code, out_fps, (out_width, out_height))
    if not out.isOpened():
        raise IOError(f"Cannot create output video writer: {output_path}")

    return out, out_fps, out_width, out_height


def read_txt(txt_path: str) -> List[str]:
    with open(txt_path, mode='r', encoding='utf-8') as f:
        lines = f.readlines()

    return [line.strip() for line in lines]


def read_json(json_path: str) -> dict:
    with open(json_path, mode='r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def write_json(data: dict, json_path: str) -> None:
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON object from model response text.

    Args:
        text: Raw response string from VLM.

    Returns:
        Parsed JSON dict if successful, otherwise None.
    """
    # Try direct parsing first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: extract first JSON block
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None