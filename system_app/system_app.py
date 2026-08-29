from typing import List, Dict
import time

from models import Yolo26Detection
from utils import (FrameInfo, VideoCache, ObjInfo,
                   logger)
from tasks import MultiDetectionTask


class SystemApp:
    def __init__(self, model_sessions: Dict[str, Yolo26Detection]):
        multi_detect_session = model_sessions.get("multi_detect")
        self.multi_detection_task = MultiDetectionTask(multi_detect_session)
        self.last_frame_index = -1
        self.frames_cache = []

    def clear_cache(self):
        self.last_frame_index = -1
        self.frames_cache = []

    def process(self, frames_cache: List[FrameInfo], system_cache: VideoCache):
        frame_index, frame = frames_cache[-1].frame_index, frames_cache[-1].frame_data
        if frame_index == 0:
            self.clear_cache()
        
        if frame_index == self.last_frame_index:
            return
        self.last_frame_index = frame_index

        logger.info("*" * 20 + f" Processing frame {frame_index} " + "*" * 20)
        frame_info = FrameInfo(frame_index=frame_index, frame_data=frame)

        # 1. [[x1, y1, x2, y2, name, score]]
        start_time = time.time()
        bboxes = self.multi_detection_task.run(frame)
        end_time = time.time()
        logger.warning(f"Multi detection time: {end_time - start_time} seconds")

        for bbox in bboxes:
            if bbox[6] == "low_threshold":
                continue
            # crop_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            obj_info = ObjInfo(bbox=[bbox[0], bbox[1], bbox[2], bbox[3], bbox[5]],
                               class_name=bbox[4])
            
            frame_info.all_objects.append(obj_info)

        system_cache.frame_infos.append(frame_info)