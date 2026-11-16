import os
import subprocess
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from PIL import Image, ImageDraw, ImageFont
from utils import Camera, read_json, plot_one_box, video_writer


# Chinese-capable font candidates (checked in order). cv2.putText only renders
# ASCII, so PIL + a CJK font is required for the Chinese reasoning text.
_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _bgr_to_rgb(color):
    return (color[2], color[1], color[0])


class VideoVisualizer:
    """
    Video visualization tool:
    - Read original video
    - Load detection & recognition results from JSON
    - Draw per frame: frame number (top-right), detection boxes,
      VLM judgment (Answer + short Reasoning) with color coding.
      The VLM result lingers on screen until the next result appears.
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

        # Persistent VLM result: lingers until the next recognition frame updates it.
        current_recog: Optional[Dict] = None

        frame_count = 0
        for frame_info in Camera.read_frame_by_video(self.video_path):
            if frame_info.frame_index == -1:
                break

            img = frame_info.frame_data
            if img is None or img.size == 0:
                continue

            frame_idx = frame_info.frame_index

            # ---- Draw detection boxes (per frame) ----
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
                # Update the persistent recognition when this frame has one.
                recog_list = item.get('recognition', [])
                if recog_list:
                    current_recog = recog_list[0]

            # ---- Draw the persistent VLM result (top-right) ----
            if current_recog is not None:
                answer = current_recog.get('answer', 'N/A')
                reasoning = current_recog.get('reasoning', '')

                # Color coding: Yes -> red, No -> green
                if answer == "Yes":
                    answer_color = (0, 0, 255)       # BGR red
                elif answer == "No":
                    answer_color = (0, 255, 0)       # BGR green
                else:
                    answer_color = (0, 165, 255)     # orange for unknown

                base_lines = [f"Frame: {frame_idx}", f"VLM Answer: {answer}", "Reasoning:"]
                lines = base_lines + self._wrap_text(reasoning, 28)

                default_color = (255, 255, 255)
                line_colors = [default_color] * len(lines)
                if len(lines) >= 2:
                    line_colors[1] = answer_color

                self._draw_text_block(
                    img,
                    lines=lines,
                    anchor='top-right',
                    font_size=18,
                    color=default_color,
                    bg_color=(0, 0, 0, 0.5),
                    line_colors=line_colors
                )
            else:
                self._draw_text_block(
                    img,
                    lines=[f"Frame: {frame_idx}", "No VLM result yet"],
                    anchor='top-right',
                    font_size=18,
                    color=(0, 0, 255),
                    bg_color=(0, 0, 0, 0.5)
                )

            out.write(img)
            frame_count += 1

        out.release()
        self._transcode_to_h264()
        print(f"Visualization completed! Processed {frame_count} frames. "
              f"Output saved to: {self.output_path}")

    def _transcode_to_h264(self) -> None:
        """Re-encode the OpenCV output to browser-friendly H.264 in place.

        OpenCV's bundled FFmpeg only exposes the MPEG-4 Part 2 encoder, which
        HTML5 <video> cannot play. The system ffmpeg (libx264) transcodes the
        intermediate file so the result plays in any modern browser.
        """
        tmp_path = self.output_path + ".h264.tmp.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", self.output_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
            tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0 and os.path.isfile(tmp_path):
            os.replace(tmp_path, self.output_path)
        else:
            # Non-fatal: keep the original MPEG-4 output if the transcode fails.
            print("Warning: H.264 transcode failed, keeping the original output. "
                  f"ffmpeg: {result.stderr.decode('utf-8', 'ignore')[:200]}")

    def _draw_text_block(self, img: np.ndarray, lines: List[str],
                         anchor: str = 'top-right',
                         font_size: int = 18,
                         color=(255, 255, 255),
                         bg_color=(0, 0, 0, 0.5),
                         margin: int = 15,
                         line_gap: int = 5,
                         line_colors: Optional[List] = None) -> None:
        """
        Draw a text block with a semi-transparent background using PIL
        (supports Chinese), then write it back into the BGR image.
        """
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        font = _load_font(font_size)
        draw = ImageDraw.Draw(pil_img)

        # Measure each line (clamp to non-negative: the fallback default font
        # may report negative metrics for unsupported (CJK) characters).
        line_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        line_widths = [max(0, b[2] - b[0]) for b in line_boxes]
        line_heights = [max(0, b[3] - b[1]) for b in line_boxes]
        max_line_width = max(line_widths) if line_widths else 0
        total_text_height = sum(line_heights) + line_gap * (len(lines) - 1)

        h, w = img.shape[:2]

        # Determine anchor position (top-right)
        if anchor == 'top-right':
            text_x = w - margin - max_line_width
            text_y = margin
            bg_x1 = text_x - margin // 2
            bg_y1 = margin - margin // 2
            bg_x2 = w - margin // 2
            bg_y2 = margin + total_text_height + margin // 2
        else:
            # fallback to top-left
            text_x = margin
            text_y = margin
            bg_x1 = margin // 2
            bg_y1 = margin // 2
            bg_x2 = margin + max_line_width + margin // 2
            bg_y2 = margin + total_text_height + margin // 2

        # Clamp background to image boundaries
        bg_x1 = max(0, bg_x1)
        bg_y1 = max(0, bg_y1)
        bg_x2 = min(w, bg_x2)
        bg_y2 = min(h, bg_y2)

        # Draw semi-transparent background (skip if the region is empty/invalid)
        if bg_color and len(bg_color) == 4 and bg_x2 > bg_x1 and bg_y2 > bg_y1:
            overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            bg_rgb = _bgr_to_rgb(bg_color[:3])
            overlay_draw.rectangle(
                [bg_x1, bg_y1, bg_x2, bg_y2],
                fill=(bg_rgb[0], bg_rgb[1], bg_rgb[2], int(bg_color[3] * 255))
            )
            pil_img = Image.alpha_composite(pil_img, overlay)
            draw = ImageDraw.Draw(pil_img)

        # Draw each line with optional per-line color
        y = text_y
        for i, line in enumerate(lines):
            line_color = line_colors[i] if line_colors and i < len(line_colors) else color
            draw.text((text_x, y), line, font=font, fill=_bgr_to_rgb(line_color))
            y += line_heights[i] + line_gap

        img[:] = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _wrap_text(text: str, width: int) -> List[str]:
        """Wrap text into lines of at most `width` characters (Chinese-friendly)."""
        if not text:
            return []
        lines = []
        line = ""
        for ch in text:
            if len(line) >= width:
                lines.append(line)
                line = ch
            else:
                line += ch
        if line:
            lines.append(line)
        return lines


if __name__ == "__main__":
    video_path = os.environ.get("INPUT_VIDEO", "/app/data/videos/20260222172410.mp4")
    json_path = os.environ.get("RESULT_JSON", "/app/output/fishing_res/20260222172410.json")
    output_dir = os.environ.get("OUTPUT_DIR", "/app/output/videos")

    visualizer = VideoVisualizer(video_path, json_path, output_dir)
    visualizer.run()
