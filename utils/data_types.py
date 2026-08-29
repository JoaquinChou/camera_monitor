from dataclasses import dataclass, field
from typing import List, Dict, Optional
import cv2
import numpy as np
import time


@dataclass
class ObjInfo:
    """Class representing detection object information."""
    frame_index: int = None
    bbox: list = None  # [x1, y1, x2, y2, score]
    class_name: str = None  # person / boat


@dataclass
class FrameInfo:
    """Class representing information about a frame."""

    frame_index: int = None
    frame_data: np.ndarray = None
    width: int = 0
    height: int = 0
    timestamp: float = time.time()
    all_objects: List[ObjInfo] = field(default_factory=list)
    recognition: Optional[Dict] = None

    def clear_image_feature_buffer(self):
        self.frame_data = None

@dataclass
class FrameResult:
    frame_index: int
    all_objects: List[List] = field(default_factory=list),
    recognition: List[Dict] = field(default_factory=list)
    def to_dict(self):
        return {
            "frame_index": self.frame_index,
            "all_objects": [[obj.class_name] + obj.bbox for obj in self.all_objects],
            "recognition": self.recognition
        }


@dataclass
class VideoCache:
    """Class representing video cache."""

    video_path: str = None
    frame_infos: List[FrameInfo] = field(default_factory=list)

    def clear_image_feature_buffer(self):
        for frame_info in self.frame_infos:
            frame_info.clear_image_feature_buffer()

    def reset(self, video_path):
        self.video_path = video_path
        self.frame_infos = []


class Camera():
    VIDEO_RECORD_START = 1
    VIDEO_RECORD_END = 0
    CameraStatus = VIDEO_RECORD_END,
    
    @classmethod
    def get_camera_status(cls):
        return cls.CameraStatus
    
    @classmethod
    def set_camera_status(cls, status=None):
        if status is None:
            status = cls.VIDEO_RECORD_END
    
    @classmethod
    def read_frame_by_video(cls, video_path):
        cls.set_camera_status(cls.VIDEO_RECORD_START)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        # video open fail
        if not cap.isOpened() or fps < 1:
            print("Error: Could not open video.")
            cls.set_camera_status(cls.VIDEO_RECORD_END)
            cap.release()
            frame = FrameInfo(frame_index=-1, frame_data=cls.get_end_frame(), width=1, height=1, timestamp=time.time())
            yield frame
            return
        
        frame_index = 0
        while True:
            ret, frame_data = cap.read()
            frame = FrameInfo(frame_index=frame_index, frame_data=frame_data, timestamp=float(frame_index)/fps)
            if not ret:
                frame.frame_data = cls.get_end_frame()
                frame.width = frame.frame_data.shape[1]
                frame.height = frame.frame_data.shape[0]
                cls.set_camera_status(cls.VIDEO_RECORD_END)
                cap.release()
                yield frame
                return
            frame.width = frame.frame_data.shape[1]
            frame.height = frame.frame_data.shape[0]
            yield frame
            frame_index = frame_index + 1



    @classmethod
    def get_end_frame(cls):
        frame = np.zeros((1, 1, 3), np.uint8)
        frame[0, 0] = 222
        return frame
    

    @classmethod
    def get_video_info(cls, video_path):

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file '{video_path}'")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()
        return {
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'width': width,
            'height': height
        }