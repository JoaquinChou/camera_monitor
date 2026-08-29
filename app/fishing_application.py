from typing import List
import os
import time

from models import Yolo26Detection, Qwen3VLClient
from tasks import FishingRecognitionTask
from utils import (Camera, FrameInfo, FrameResult, VideoCache,
                   logger, get_outer_bbox, expand_bbox_crop, write_json)
from system_app import SystemApp


class Fishing:
    frames_cache_size = 30
    expand_scale = 1.2
    gap_time = 0.5
    # select_frame_fps = 1

    def __init__(self, config_path: str, gpu_id: int = 0) -> None:
        self.fishing_session = None
        self.frames_cache: List[FrameInfo] = []
        self.model_sessions = self.init_model_sessions(config_path, gpu_id)
        self.system_cache = VideoCache()
        self.system_app = SystemApp(self.model_sessions)
        self.key_frames = []
        self.key_frame_ids = []
        self.last_collect_time = -1.0        
        self.fishing_task = None            
        
    
    def init_model_sessions(self, config_path: str, gpu_id: int = 0) -> dict:
        model_session_dict = {}
        config = read_json(config_path)
        multi_detect_config = config.get("multi_detect")
        if multi_detect_config:
            yolo26Detection = Yolo26Detection(multi_detect_config, gpu_id=gpu_id)
            yolo26Detection.onnx_init()
            logger.info(f"Multi detect model initialized with config: {multi_detect_config}")
            model_session_dict["multi_detect"] = yolo26Detection

        vllm_config = config.get("fishing_vllm_recognition")
        if vllm_config:
            client = Qwen3VLClient(
                base_url=vllm_config["base_url"],
                model=vllm_config.get("model"),
            )
            self.fishing_session = {
                "client": client,
                "default_params": {
                    "max_tokens": vllm_config.get("max_tokens", 512),
                    "temperature": vllm_config.get("temperature", 0.7),
                    "chat_template_kwargs": {
                        "enable_thinking": vllm_config.get("enable_thinking", False)
                    }
                },
                "max_infer_num": vllm_config.get("max_infer_num", 8)
            }
            self.fishing_task = FishingRecognitionTask(self.fishing_session["client"])
            logger.info(f"VLLM client initialized for model: {client.model}")

        return model_session_dict


    def init_system_state(self, video_path) -> None:
        self.frames_cache.clear()
        self.system_cache.reset(video_path=str(video_path))


    @classmethod
    def system_read(cls, input, frames_cache):
        if cls.frames_cache_size > 0 and len(frames_cache) >= cls.frames_cache_size:
            frames_cache.pop(0)
        frames_cache.append(input)


    def system_run_sync(self, video_path):
        logger.warning(f"Processing video: {video_path}")
        self.init_system_state(video_path)
        video_info_dict = Camera.get_video_info(video_path)
        logger.info(video_info_dict)
        for frame_info in Camera.read_frame_by_video(video_path):
            self.system_read(frame_info, self.frames_cache)
            self.system_app.process(self.frames_cache, self.system_cache)
            self.fishing_recog_infer(self.system_cache.frame_infos[-1])
        self._vlm_finalize_infer()
        self.system_finish()


    def fishing_recog_infer(self, current_frame: FrameInfo):
        if (self.fishing_session is None
            or not current_frame.all_objects):
            return
        
        # time gap: 2fps(0.5s)
        if self.last_collect_time >= 0 and (current_frame.timestamp - self.last_collect_time) < self.gap_time:
            return

        bboxes = [obj.bbox for obj in current_frame.all_objects]
        bbox_coords = [b[:4] for b in bboxes if len(b) >= 4]
        if not bbox_coords:
            return
        outer_bbox = get_outer_bbox(bbox_coords)
        if outer_bbox is None:
            return

        # expand and crop
        cropped_img = expand_bbox_crop(current_frame.frame_data, outer_bbox, scale=self.expand_scale)
        if cropped_img is None or cropped_img.size == 0:
            return
        self.key_frames.append(cropped_img)
        self.key_frame_ids.append(current_frame.frame_index)
        self.last_collect_time = current_frame.timestamp

        if len(self.key_frames) >= self.fishing_session["max_infer_num"]:
            self._vlm_infer()


    def _vlm_infer(self):
        start_time = time.time()
        result = self.fishing_task.run(self.key_frames, **self.fishing_session["default_params"])
        end_time = time.time()
        logger.info(f"VLM request time: {end_time - start_time} seconds")
        logger.info("*" * 20 + f"VLM Processing frames id {self.key_frame_ids} " + "*" * 20)
        for idx in self.key_frame_ids:
            for finfo in self.system_cache.frame_infos:
                if finfo.frame_index == idx:
                    finfo.recognition = result
                    break
        self.key_frames.clear()
        self.key_frame_ids.clear()


    def _vlm_finalize_infer(self):
        if self.fishing_session is None or not self.key_frames:
            return

        max_num = self.fishing_session["max_infer_num"]
        current_num = len(self.key_frames)
        if current_num < max_num:
            # copy the last frame(img and frame)
            last_img = self.key_frames[-1]
            last_id = self.key_frame_ids[-1]
            for _ in range(max_num - current_num):
                self.key_frames.append(last_img)
                self.key_frame_ids.append(last_id)
            logger.info(f"Padded to {max_num} frames with last frame for inference.")
        # inference
        self._vlm_infer()


    def system_finish(self):
        self.system_cache.clear_image_feature_buffer()
        self.key_frames.clear()
        self.key_frame_ids.clear()


    def get_frames_result(self):
        frames_result = []
        for finfo in self.system_cache.frame_infos:
            fish_recog = finfo.recognition if finfo.recognition is not None else []
            if not isinstance(fish_recog, list):
                fish_recog = [fish_recog]
            frame_res = FrameResult(
                frame_index=finfo.frame_index,
                all_objects=finfo.all_objects,
                recognition=fish_recog
            )
            frames_result.append(frame_res.to_dict())
        return frames_result


    def save_video_result(self, output_dir: str):
        if not self.system_cache.video_path:
            return
        result = self.get_frames_result()
        os.makedirs(output_dir, exist_ok=True)

        video_name = os.path.basename(self.system_cache.video_path)
        base_name = os.path.splitext(video_name)[0]
        save_path = os.path.join(output_dir, f"{base_name}.json")
        write_json(result, save_path)
        logger.info(f"Saved result to {save_path}")



if __name__ == "__main__":
    from utils import read_json

    data_path = "/home/jcx/code/camera_monitor/configs/test_data.json"
    config_path = "/home/jcx/code/camera_monitor/configs/fishing_app.json"
    result_path = "/home/jcx/code/camera_monitor/output/fishing_res"
    gpu_id = 2
    data_list = read_json(data_path)
    fishing = Fishing(config_path, gpu_id)
    for video_path in data_list:
        fishing.system_run_sync(video_path)
        fishing.save_video_result(result_path)