from typing import Optional, Dict, Any, List
import base64
import cv2
import numpy as np
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