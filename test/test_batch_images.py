"""
Batch evaluation of the fishing recognition model on image datasets.

Runs the VLM (single image per call) over two directories:
  - bulao (捕捞): ground-truth "Yes" (illegal fishing)
  - chuidiao (垂钓): ground-truth "Yes" (recreational angling)

Both directories are POSITIVE classes: the model output "Yes" means fishing
was detected, which is the expected answer in BOTH folders. Because there is
no negative-class data, the reported metrics are per-folder detection rates,
the overall detection rate and the overall accuracy (equal to the overall
detection rate when no negatives exist). Detailed per-image results
(prediction + reasoning) are saved to a JSON file.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from models import Qwen3VLClient
from tasks import FishingRecognitionTask
from utils import encode_image_to_base64, read_img


BULAO_DIR = os.environ.get("BULAO_DIR", "/app/fishing_data/bulao-img/images")
CHUIDIAO_DIR = os.environ.get("CHUIDIAO_DIR", "/app/fishing_data/chuidiao-img/images")
OUTPUT_JSON = os.environ.get("OUTPUT_JSON", "/app/output/img_res/batch_images_results.json")

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(dir_path: str) -> List[str]:
    if not os.path.isdir(dir_path):
        return []
    return sorted(
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if name.lower().endswith(IMAGE_EXTS)
    )


def classify_one(task: FishingRecognitionTask, img_path: str, **kwargs) -> Tuple[Optional[Dict], Optional[str]]:
    """Classify one image; returns (result dict, error)."""
    try:
        img = read_img(img_path)
        if img is None or img.size == 0:
            return None, "read failed"
        img_b64 = encode_image_to_base64(img)
        return task.run([img_b64], **kwargs), None
    except Exception as exc:
        return None, str(exc)


def evaluate(bulao_dir: str, chuidiao_dir: str, output_json: str) -> None:
    base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:7864")
    model = os.environ.get("VLLM_MODEL", "Qwen3-VL-8B-Instruct")

    client = Qwen3VLClient(base_url=base_url, model=model)
    task = FishingRecognitionTask(client)

    infer_kwargs = {
        "temperature": 0.1,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    bulao_images = list_images(bulao_dir)
    chuidiao_images = list_images(chuidiao_dir)
    total = len(bulao_images) + len(chuidiao_images)

    print(f"===== 捕捞/垂钓识别批量测试 =====")
    print(f"模型: {model} @ {base_url}")
    print(f"捕捞(bulao) 图片: {len(bulao_images)} 张")
    print(f"垂钓(chuidiao) 图片: {len(chuidiao_images)} 张")
    print(f"合计: {total} 张\n")

    bulao_hits = 0
    chuidiao_hits = 0
    errors = 0
    processed = 0
    results: List[Dict[str, Any]] = []

    def record(path: str, category: str) -> None:
        nonlocal bulao_hits, chuidiao_hits, errors, processed
        result, err = classify_one(task, path, **infer_kwargs)
        processed += 1
        if err is not None:
            errors += 1
            print(f"[error] {os.path.basename(path)}: {err}")
            results.append({
                "image": os.path.basename(path),
                "category": category,
                "ground_truth": "Yes",
                "prediction": None,
                "reasoning": None,
                "correct": False,
                "error": err,
            })
            return
        answer = result.get("answer") if result else None
        correct = (answer == "Yes")
        if correct:
            if category == "bulao":
                bulao_hits += 1
            else:
                chuidiao_hits += 1
        results.append({
            "image": os.path.basename(path),
            "category": category,
            "ground_truth": "Yes",
            "prediction": answer,
            "reasoning": (result or {}).get("reasoning"),
            "correct": correct,
        })
        if processed % 20 == 0:
            print(f"  进度: {processed}/{total}")

    for path in bulao_images:
        record(path, "bulao")
    for path in chuidiao_images:
        record(path, "chuidiao")

    # metrics (no negative class: accuracy == overall detection rate)
    n_bulao = len(bulao_images)
    n_chuidiao = len(chuidiao_images)
    n_total = n_bulao + n_chuidiao

    bulao_rate = bulao_hits / n_bulao if n_bulao > 0 else 0.0
    chuidiao_rate = chuidiao_hits / n_chuidiao if n_chuidiao > 0 else 0.0
    overall_hits = bulao_hits + chuidiao_hits
    overall_rate = overall_hits / n_total if n_total > 0 else 0.0
    accuracy = overall_rate  # identical when there is no negative class

    summary = {
        "total": n_total,
        "bulao_total": n_bulao,
        "chuidiao_total": n_chuidiao,
        "errors": errors,
        "bulao_hits": bulao_hits,
        "chuidiao_hits": chuidiao_hits,
        "overall_hits": overall_hits,
        "bulao_detection_rate": round(bulao_rate, 4),
        "chuidiao_detection_rate": round(chuidiao_rate, 4),
        "overall_detection_rate": round(overall_rate, 4),
        "accuracy": round(accuracy, 4),
    }

    # Save detailed results to JSON
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\n===== 测试结果 =====")
    print(f"有效样本: {n_total} (错误/失败: {errors})")
    print(f"\n捕捞(bulao) 检出率 = {bulao_hits}/{n_bulao} = {bulao_rate:.4f}")
    print(f"垂钓(chuidiao) 检出率 = {chuidiao_hits}/{n_chuidiao} = {chuidiao_rate:.4f}")
    print(f"\n总体检出率 = {overall_hits}/{n_total} = {overall_rate:.4f}")
    print(f"总体准确率 = {accuracy:.4f} (无负类样本, 与总体检出率相同)")
    print(f"\n详细结果已保存到: {output_json}")


if __name__ == "__main__":
    evaluate(BULAO_DIR, CHUIDIAO_DIR, OUTPUT_JSON)
