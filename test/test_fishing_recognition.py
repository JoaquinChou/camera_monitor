import argparse

from models import Qwen3VLClient
from tasks import FishingRecognitionTask
from utils import encode_image_to_base64, read_img


def test_img_sequence():
    base_url = "http://localhost:7864"
    model = "Qwen3-VL-8B-Instruct"
    imgs_path = [
        "./data/images/test_frame_at_9m10s.jpg",
        "./data/images/test_frame_at_9m11s.jpg"]


    client = Qwen3VLClient(base_url=base_url, model=model)
    fishingRecognitionTask = FishingRecognitionTask(client)


    print("=== Testing call_vlm (vision-language) ===")
    try:
        img_b64_list = [encode_image_to_base64(read_img(img_path)) for img_path in imgs_path]
        response = fishingRecognitionTask.run(
            img_b64_list,
            temperature=0.1,
            max_tokens=512,
            chat_template_kwargs={"enable_thinking": False}
        )
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_img_sequence()