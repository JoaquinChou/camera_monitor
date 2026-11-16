"""Data access layer.

Reads the pipeline outputs (per-video result JSON, evidence montages, batch
image evaluation results) and returns plain dict / list structures that the HTTP
layer serialises to JSON. This module has no knowledge of HTTP or routing; it
only talks to the filesystem, so it can be reused or unit-tested independently.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

CATEGORY_LABELS = {"bulao": "捕捞", "chuidiao": "垂钓"}


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _video_files() -> List[Path]:
    if not config.VIDEOS_DIR.is_dir():
        return []
    return sorted(config.VIDEOS_DIR.glob("*.mp4"))


def _detection_stats(frames: List[Dict]) -> Dict[str, int]:
    """Aggregate person / boat detections across all frames of a video."""
    stats = {
        "person_frames": 0,
        "boat_frames": 0,
        "person_boxes": 0,
        "boat_boxes": 0,
    }
    for frame in frames:
        classes = [obj[0] for obj in frame.get("all_objects", []) if obj]
        if "person" in classes:
            stats["person_frames"] += 1
            stats["person_boxes"] += classes.count("person")
        if "boat" in classes:
            stats["boat_frames"] += 1
            stats["boat_boxes"] += classes.count("boat")
    return stats


def _fishing_events(frames: List[Dict]) -> List[Dict]:
    """Collect the unique VLM recognition events (answer + reasoning)."""
    events: List[Dict] = []
    seen = set()
    for frame in frames:
        recognition = frame.get("recognition") or []
        if isinstance(recognition, dict):
            recognition = [recognition]
        for rec in recognition:
            if not isinstance(rec, dict):
                continue
            key = (rec.get("answer"), tuple(rec.get("frame_ids", [])))
            if key in seen:
                continue
            seen.add(key)
            events.append(
                {
                    "answer": rec.get("answer"),
                    "reasoning": rec.get("reasoning", ""),
                    "frame_ids": rec.get("frame_ids", []),
                    "timestamp": frame.get("timestamp"),
                }
            )
    return events


def load_video_cases() -> List[Dict[str, Any]]:
    """Build one case object per annotated video.

    Each case exposes ``evidence_cases``: the evidence montage paired 1:1 (by
    chronological order) with its VLM verdict (answer / reasoning / timestamp),
    which is what the demo actually showcases.
    """
    cases: List[Dict[str, Any]] = []
    for video in _video_files():
        base = video.stem
        result_path = config.RESULTS_DIR / f"{base}.json"
        evidence_dir = config.EVIDENCE_DIR / base

        evidence = []
        if evidence_dir.is_dir():
            evidence = [
                {
                    "seq": i + 1,
                    "url": f"/media/evidence/{base}/{p.name}",
                }
                for i, p in enumerate(sorted(evidence_dir.glob("evidence_*.jpg")))
            ]

        frames = _read_json(result_path)
        if not isinstance(frames, list):
            frames = []

        yes_events = [e for e in _fishing_events(frames) if e.get("answer") == "Yes"]

        # Pair each evidence montage with its corresponding verdict (same order).
        evidence_cases = []
        for i, ev in enumerate(evidence):
            event = yes_events[i] if i < len(yes_events) else {}
            evidence_cases.append(
                {
                    "seq": ev["seq"],
                    "url": ev["url"],
                    "answer": event.get("answer", "Yes"),
                    "reasoning": event.get("reasoning", ""),
                    "timestamp": event.get("timestamp"),
                }
            )

        cases.append(
            {
                "id": base,
                "video_name": video.name,
                "video_url": f"/media/videos/{video.name}",
                "evidence_count": len(evidence),
                "fishing_detected": len(yes_events) > 0,
                "total_frames": len(frames),
                "detection": _detection_stats(frames),
                "evidence_cases": evidence_cases,
            }
        )
    return cases


def load_image_metrics() -> Dict[str, Any]:
    """Return the batch image evaluation summary (detection rates)."""
    data = _read_json(config.IMG_RESULTS_FILE) or {}
    return data.get("summary", {})


def _image_case(rec: Dict, category: str) -> Dict[str, Any]:
    name = rec.get("image", "")
    return {
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "image": name,
        "image_url": f"/media/images/{category}/{name}",
        "prediction": rec.get("prediction"),
        "reasoning": rec.get("reasoning", ""),
        "correct": bool(rec.get("correct")),
    }


def load_image_cases() -> Dict[str, Any]:
    """Curate a small gallery of correct and error image cases for the demo."""
    data = _read_json(config.IMG_RESULTS_FILE) or {}
    results = data.get("results", [])

    cases: List[Dict[str, Any]] = []
    error_cases: List[Dict[str, Any]] = []
    for category in CATEGORY_LABELS:
        cat_results = [r for r in results if r.get("category") == category]
        correct = [r for r in cat_results if r.get("correct") and r.get("prediction") == "Yes"]
        errors = [r for r in cat_results if not r.get("correct")]

        cases.extend(_image_case(r, category) for r in correct[: config.IMAGE_CASES_PER_CATEGORY])
        error_cases.extend(_image_case(r, category) for r in errors[: config.IMAGE_ERROR_CASES])

    return {"cases": cases, "error_cases": error_cases}


def load_summary() -> Dict[str, Any]:
    """Aggregated overview used by the landing metrics strip."""
    video_cases = load_video_cases()
    return {
        "video_count": len(video_cases),
        "evidence_count": sum(c["evidence_count"] for c in video_cases),
        "fishing_video_count": sum(1 for c in video_cases if c["fishing_detected"]),
        "image_metrics": load_image_metrics(),
    }
