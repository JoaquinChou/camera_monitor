"""
Parallel inference runner for multiple videos.

Splits the video list from DATA_CONFIG across NUM_PROCESSES worker processes
(round-robin), each bound to one GPU from GPU_IDS via
``gpu_id = gpu_ids[worker_id % len(gpu_ids)]``.

Each worker owns its own Fishing instance and onnxruntime session, created
inside the child process (CUDA contexts are not fork-safe).
"""
import os
import multiprocessing as mp
from typing import List

from utils import read_json, logger
from fishing_application import Fishing


def _parse_gpu_ids() -> List[int]:
    raw = os.environ.get("GPU_IDS", "2,3,4,5")
    ids = [int(part) for part in raw.split(",") if part.strip() != ""]
    if not ids:
        raise ValueError("GPU_IDS must be a non-empty comma-separated list of ints")
    return ids


def _worker(gpu_id: int, config_path: str, video_paths: List[str],
            result_dir: str) -> None:
    logger.info(
        f"[worker gpu={gpu_id}] start, {len(video_paths)} video(s): {video_paths}"
    )
    fishing = Fishing(config_path, gpu_id=gpu_id)
    for video_path in video_paths:
        fishing.system_run_sync(video_path)
        fishing.save_video_result(result_dir)
    logger.info(f"[worker gpu={gpu_id}] done")


def run_parallel(config_path: str, data_config: str, result_dir: str,
                 num_processes: int) -> None:
    gpu_ids = _parse_gpu_ids()
    video_list = read_json(data_config)
    if not video_list:
        logger.warning("No videos found in DATA_CONFIG; nothing to do")
        return

    num_processes = min(num_processes, len(video_list))
    # Round-robin split: worker i handles video_list[i], video_list[i+N], ...
    chunks = [video_list[i::num_processes] for i in range(num_processes)]
    logger.info(
        f"Parallel inference: {len(video_list)} video(s), {num_processes} "
        f"process(es), GPU pool {gpu_ids}"
    )

    ctx = mp.get_context("spawn")
    processes = []
    for i, chunk in enumerate(chunks):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        p = ctx.Process(target=_worker, args=(gpu_id, config_path, chunk, result_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = [p.exitcode for p in processes if p.exitcode not in (0, None)]
    if failed:
        logger.error(f"{len(failed)} worker(s) failed, exit codes {failed}")
    else:
        logger.info("Parallel inference finished")


if __name__ == "__main__":
    data_config = os.environ.get("DATA_CONFIG", "/app/configs/test_data.json")
    config_path = os.environ.get("APP_CONFIG", "/app/configs/fishing_app.json")
    result_dir = os.environ.get("RESULT_DIR", "/app/output/fishing_res")
    num_processes = int(os.environ.get("NUM_PROCESSES", "8"))
    run_parallel(config_path, data_config, result_dir, num_processes)
