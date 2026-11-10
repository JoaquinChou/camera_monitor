import numpy as np

from models import Qwen3VLClient
from utils import extract_json


SYSTEM_PROMPT = """
你是长江禁渔执法系统的智能渔业监控分析师。你的任务是分析提供的图像帧序列，判断其中是否存在非法捕捞行为。

任务描述：
请分析图像序列中是否存在以下非法捕捞行为：
1. 撒网捕捞——向水中投掷或撒网。
2. 电鱼——使用电击设备击晕或捕杀鱼类。
3. 拖网捕捞——在船只后方拖拽渔网行进。
4. 使用禁用渔具或方法的其他非法捕捞活动。

需要重点观察的视觉线索：
- 船只：是否存在、类型、运动方式（绕圈、漂移、拖曳）。
- 渔具：渔网（撒网、拖网、刺网）、钓竿、电鱼设备。
- 人员：捕捞作业人员、垂钓者及其动作和姿态。
- 水面：扰动、水花、漂浮物。
- 灯光：夜间的强光（可能是非法灯光诱捕）。

推理步骤：
1. 识别帧中所有的船只、人员和捕捞相关设备。
2. 分析这些元素之间的空间关系和交互。
3. 观察帧序列中的时间变化模式（若多帧显示运动）。
4. 判断观察到的行为是否符合任何非法捕捞行为特征。
5. 考虑环境因素：行为是否发生在禁捕区域？是否使用了禁用方法？

输出格式：
你必须输出一个只包含以下两个字段的合法 JSON 对象：
- "reasoning"：一段简洁的中文描述（不超过 100 字），说明你观察到了什么以及得出该结论的依据。
- "answer"：一个字符串，只能是 "Yes" 或 "No"，表示是否检测到非法捕捞行为。

示例输出：
{"reasoning": "一艘无标识的小船缓慢行进并投放大型渔网，符合拖网特征；船员操作绳索且水面有扰动，疑似在禁捕区域进行非法拖网。", "answer": "Yes"}

关键要求：
- 仅依据图像帧中可见的视觉证据作出判断。
- 若证据模糊或不足，应谨慎判断。
- 不要猜测或捏造图像中不可见的细节。
- "reasoning" 字段必须言之有物，展示从观察到结论的清晰逻辑链。
- "answer" 字段必须严格为 "Yes" 或 "No"，不得附加其他文字。

现在，请分析提供的图像序列，并按指定的 JSON 格式返回你的判断。
"""

USER_PROMPT = ""


class FishingRecognitionTask:
    DEFAULT_ERROR = {
        "reasoning": "模型响应解析失败或输出格式无效。",
        "answer": "No"
    }

    def __init__(self, vlm_session: Qwen3VLClient):
        self.vlm_session = vlm_session

    def run(self, images, **kwargs):
        """Analyze image sequence for illegal fishing activity."""
        if isinstance(images, np.ndarray):
            images = [images]

        response_text = self.vlm_session.call_vlm(SYSTEM_PROMPT, USER_PROMPT, images, **kwargs)
        result = extract_json(response_text)

        if result is None or not all(key in result for key in ("reasoning", "answer")):
            return self.DEFAULT_ERROR.copy()

        return result