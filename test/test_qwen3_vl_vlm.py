import argparse

from models import Qwen3VLClient
from utils import encode_image_to_base64, read_img


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL client")
    parser.add_argument("--base-url", default="http://localhost:7864", help="vLLM server base URL")
    parser.add_argument("--image", help="Image URL or local file path to test VLM")
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct", help="Model name (optional)")
    args = parser.parse_args()

    client = Qwen3VLClient(base_url=args.base_url, model=args.model)

    print("=== Testing call_llm (pure text) ===")
    system = "You are a helpful assistant."
    user = "Who are you? Please introduce yourself briefly."
    try:
        response = client.call_llm(system, user, temperature=0.7, chat_template_kwargs={"enable_thinking": False})
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*50 + "\n")

    if args.image:
        print("=== Testing call_vlm (vision-language) ===")
        try:
            img = read_img(args.image)
            img_b64 = encode_image_to_base64(img)
            system_vlm = "You are a vision assistant."
            user_vlm = "Please describe what you see in this image in detail."
            response = client.call_vlm(
                system_prompt=system_vlm,
                user_prompt=user_vlm,
                image_base64_list=[img_b64],
                temperature=0.7,
                max_tokens=512,
                chat_template_kwargs={"enable_thinking": False}
            )
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Skipping VLM test (no --image-url provided).")

if __name__ == "__main__":
    main()