import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from utils import Camera, read_json, plot_one_box, video_writer


class VideoVisualizer:
    """
    Video visualization tool:
    - Read original video
    - Load detection & recognition results from JSON
    - Draw per frame: frame number (top-right), detection boxes,
      VLM judgment (Answer + short Reasoning) with color coding
    - Write output video to specified directory
    """

    def __init__(self, video_path: str, json_path: str, output_dir: str):
        self.video_path = video_path
        self.json_path = json_path
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        self.output_path = os.path.join(output_dir, f"{video_basename}.mp4")

    def run(self) -> None:
        data = read_json(self.json_path)
        frame_map: Dict[int, Dict[str, Any]] = {
            item['frame_index']: item for item in data
        }

        out, _, width, height = video_writer(
            self.video_path, self.output_path, fourcc='mp4v'
        )

        frame_count = 0
        for frame_info in Camera.read_frame_by_video(self.video_path):
            if frame_info.frame_index == -1:
                break

            img = frame_info.frame_data
            if img is None or img.size == 0:
                continue

            frame_idx = frame_info.frame_index

            # ---- Draw detection boxes ----
            if frame_idx in frame_map:
                item = frame_map[frame_idx]
                for obj in item.get('all_objects', []):
                    if len(obj) < 6:
                        continue
                    cls_name, x1, y1, x2, y2, score = obj
                    plot_one_box(
                        [x1, y1, x2, y2],
                        img,
                        color=[0, 165, 255],
                        label=f"{cls_name} {score:.2f}"
                    )

                # ---- Draw VLM info at top-right ----
                recog_list = item.get('recognition', [])
                if recog_list:
                    rec = recog_list[0]
                    answer = rec.get('answer', 'N/A')
                    reasoning = rec.get('reasoning', '')

                    # Color coding: Yes -> red, No -> green
                    if answer == "Yes":
                        answer_color = (0, 0, 255)       # BGR red
                    elif answer == "No":
                        answer_color = (0, 255, 0)       # BGR green
                    else:
                        answer_color = (0, 165, 255)     # orange for unknown

                    # Build text lines
                    base_lines = [f"Frame: {frame_idx}", f"VLM Answer: {answer}", "Reasoning:"]
                    reasoning_lines = self._wrap_text(reasoning, 60)
                    lines = base_lines + reasoning_lines

                    # Assign colors: default white, except answer line (index 1)
                    default_color = (255, 255, 255)
                    line_colors = [default_color] * len(lines)
                    if len(lines) >= 2:
                        line_colors[1] = answer_color

                    self._draw_text_block(
                        img,
                        lines=lines,
                        anchor='top-right',
                        font_scale=0.6,
                        color=default_color,
                        bg_color=(0, 0, 0, 0.5),
                        line_colors=line_colors
                    )
                else:
                    self._draw_text_block(
                        img,
                        lines=[f"Frame: {frame_idx}", "No VLM result"],
                        anchor='top-right',
                        font_scale=0.7,
                        color=(0, 0, 255),
                        bg_color=(0, 0, 0, 0.5)
                    )
            else:
                self._draw_text_block(
                    img,
                    lines=[f"Frame: {frame_idx}", "No data for this frame"],
                    anchor='top-right',
                    font_scale=0.7,
                    color=(0, 0, 255),
                    bg_color=(0, 0, 0, 0.5)
                )

            out.write(img)
            frame_count += 1

        out.release()
        print(f"Visualization completed! Processed {frame_count} frames. "
              f"Output saved to: {self.output_path}")

    def _draw_text_block(self, img: np.ndarray, lines: List[str],
                         anchor: str = 'top-right',
                         font_scale: float = 0.6,
                         color=(255, 255, 255),
                         bg_color=(0, 0, 0, 0.5),
                         margin: int = 15,
                         line_gap: int = 5,
                         line_colors: Optional[List] = None) -> None:
        """
        Draw a block of text on the image with optional semi-transparent background.
        Each line can have individual color via `line_colors`.
        """
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 1

        # Calculate text sizes
        line_heights = []
        line_widths = []
        for line in lines:
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            line_widths.append(tw)
            line_heights.append(th)

        max_line_width = max(line_widths) if line_widths else 0
        total_text_height = sum(line_heights) + line_gap * (len(lines) - 1)

        # Determine anchor position (top-right)
        if anchor == 'top-right':
            text_x = w - margin - max_line_width
            text_y = margin + line_heights[0]  # baseline of first line
            bg_x1 = text_x - margin // 2
            bg_y1 = margin - margin // 2
            bg_x2 = w - margin // 2
            bg_y2 = margin + total_text_height + margin // 2
        else:
            # fallback to top-left
            text_x = margin
            text_y = margin + line_heights[0]
            bg_x1 = margin // 2
            bg_y1 = margin // 2
            bg_x2 = margin + max_line_width + margin // 2
            bg_y2 = margin + total_text_height + margin // 2

        # Clamp background to image boundaries
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(w, bg_x2)
        bg_y2 = min(h, bg_y2)

        # Draw semi-transparent background
        if bg_color and len(bg_color) == 4:
            overlay = img.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                          bg_color[:3], -1)
            cv2.addWeighted(overlay, bg_color[3], img, 1 - bg_color[3], 0, img)

        # Draw each line with optional per-line color
        y_offset = text_y
        for i, line in enumerate(lines):
            if i == 0:
                y = y_offset
            else:
                y = y_offset + line_gap + line_heights[i]
            line_color = line_colors[i] if line_colors and i < len(line_colors) else color
            cv2.putText(img, line, (text_x, y),
                        font, font_scale, line_color, thickness, cv2.LINE_AA)
            y_offset = y

    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text into lines of at most `width` characters."""
        words = text.split()
        lines = []
        line = ""
        for w in words:
            if len(line) + len(w) + 1 <= width:
                line += (" " + w) if line else w
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines


if __name__ == "__main__":
    video_path = "/home/jcx/code/camera_monitor/data/videos/test_data_part_2.mp4"
    json_path = "/home/jcx/code/camera_monitor/output/fishing_res/test_data_part_2.json"
    output_dir = "/home/jcx/code/camera_monitor/output/videos"

    visualizer = VideoVisualizer(video_path, json_path, output_dir)
    visualizer.run()