"""
Evidence helpers: stitch VLM key frames into a labeled grid montage and save it.
"""
import os
import cv2
import numpy as np


def build_evidence_montage(frames, frame_ids, cols=4, rows=2):
    """
    Stitch `frames` into a `cols` x `rows` grid, labeling each tile with its frame id.

    Args:
        frames: list of BGR numpy arrays (all the same size).
        frame_ids: frame indices aligned with `frames`.
        cols: number of columns in the grid.
        rows: number of rows in the grid.

    Returns:
        np.ndarray montage, or None if `frames` is empty.
    """
    if not frames:
        return None
    tile_h, tile_w = frames[0].shape[:2]
    canvas = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
    for i, (frame, fid) in enumerate(zip(frames, frame_ids)):
        if i >= cols * rows:
            break
        r, c = divmod(i, cols)
        y0, x0 = r * tile_h, c * tile_w
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = frame
        cv2.putText(canvas, f"frame {fid}", (x0 + 8, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return canvas


def save_evidence_montage(frames, frame_ids, save_path, cols=4, rows=2):
    """
    Build a labeled evidence montage and save it as JPEG to `save_path`.

    Returns:
        bool: True on success, False otherwise.
    """
    montage = build_evidence_montage(frames, frame_ids, cols, rows)
    if montage is None:
        return False
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ok, buf = cv2.imencode('.jpg', montage)
    if not ok:
        return False
    buf.tofile(save_path)
    return True
