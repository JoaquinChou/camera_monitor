import numpy as np

from models import Yolo26Detection


class MultiDetectionTask:

    def __init__(self, multi_detect_session: Yolo26Detection):
        self.multi_detect_session = multi_detect_session

    def run(self, image: np.ndarray):
        outputs = self.multi_detect_session.infer(image)
        return [output for output in outputs  if output[4] in ("person", "boat")]