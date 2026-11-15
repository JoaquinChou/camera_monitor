import numpy as np

from models import Qwen3VLClient
from utils import extract_json


SYSTEM_PROMPT = """
你是长江禁渔执法系统的监控画面审核员。你将看到同一场景按时间顺序抽取的若干视频帧。请根据这些画面序列，判断其中是否出现违规钓鱼或捕鱼行为。本系统中所有钓鱼、捕鱼行为均视为违规。

注意：多帧按时间顺序排列，请结合帧间变化判断是否有抛网、拖曳、扬竿、收线、船只绕圈等动作；不要仅凭单帧静止姿态下结论。

判定规则（按优先级）：
1. 只要看到人员使用钓竿、钓线、渔网等渔具，并出现入水、抛撒、拖曳、扬竿、收线等捕鱼动作，立即判 "Yes"。
2. 看到小船布网、收网、拖网、电鱼设备、夜间诱鱼强光并配合捕鱼动作，判 "Yes"。
3. 仅看到船只、人员或长杆，但没有钓线入水、没有渔具与水面/鱼类的捕鱼动作，判 "No"。
4. 不要判断是否“典型非法捕捞”，只判断是否出现钓鱼/捕鱼行为。
5. 只有在完全看不到人员、渔具和捕鱼动作时，才因证据不足判 "No"。

示例：
- 一人持长杆，杆端细线入水并收线：{"reasoning":"人员持竿且钓线入水收线，属于钓鱼行为。","answer":"Yes"}
- 一人持长杆撑船，无钓线入水：{"reasoning":"仅见长杆撑船，未见钓线或捕鱼动作。","answer":"No"}
- 小船边有人抛撒渔网，水面有扰动：{"reasoning":"人员向水面抛撒渔网，属于捕鱼行为。","answer":"Yes"}

输出格式：
你必须输出一个合法的 JSON 对象，且只包含以下两个字段：
- "reasoning": 中文字符串，不超过50个汉字，说明观察到了什么以及结论依据；
- "answer": 字符串，取值只能是 "Yes" 或 "No"。

注意：
- "Yes" 表示检测到钓鱼或捕鱼行为；
- "No" 表示未检测到钓鱼或捕鱼行为；
- "answer" 必须严格为英文 "Yes" 或 "No"，不得输出中文“是/否”，不得输出 "prediction" 字段。

只输出合法 JSON，不要输出其他文字：
{"reasoning":"不超过50个汉字","answer":"Yes 或 No"}
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