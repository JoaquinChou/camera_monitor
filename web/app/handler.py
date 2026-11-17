"""HTTP layer: routing and request handling.

Translates HTTP requests into calls to the data layer (JSON APIs) and the media
layer (static assets / videos / evidence / images). No business logic lives
here; it only maps ``/api/...``, ``/media/...`` and ``/static/...`` to the
appropriate backend module.
"""
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse

from . import ais_data, config, data, media


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CameraMonitorDemo/1.0"

    # -- response helpers -----------------------------------------------------
    def _write(self, payload: bytes) -> None:
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, obj, status: int = 200) -> None:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    def _send_static(self, path: Path) -> None:
        # no-cache: static assets change during development; always revalidate.
        media.serve_file(self, path, cache_control="no-cache")

    def _send_download(self, payload: bytes, content_type: str, filename: str) -> None:
        """Stream a raw file as an attachment download."""
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self._write(payload)

    # -- routing ---------------------------------------------------------------
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            return self._send_static(config.STATIC_DIR / "index.html")

        if path.startswith("/static/"):
            return self._send_static(_safe_join(config.STATIC_DIR, path[len("/static/"):]))

        if path.startswith("/api/"):
            return self._handle_api(path)

        if path.startswith("/media/"):
            return self._handle_media(path)

        self.send_error(404)

    def do_HEAD(self) -> None:
        # Reuse GET routing; body writes are suppressed by self.command.
        self.do_GET()

    # -- API -------------------------------------------------------------------
    def _handle_api(self, path: str) -> None:
        routes = {
            "/api/summary": data.load_summary,
            "/api/video-cases": data.load_video_cases,
            "/api/image-metrics": data.load_image_metrics,
            "/api/image-cases": data.load_image_cases,
        }
        if path in routes:
            return self._send_json(routes[path]())

        if path == "/api/ais":
            return self._send_json(ais_data.load_ais())

        if path == "/api/ais/forecast.csv":
            payload = ais_data.read_forecast_csv()
            if payload is None:
                return self.send_error(404)
            return self._send_download(payload, "text/csv; charset=utf-8", "forecast.csv")

        if path == "/api/ais/metrics.json":
            payload = ais_data.read_metrics_json()
            if payload is None:
                return self.send_error(404)
            return self._send_download(
                payload, "application/json; charset=utf-8", "metrics.json"
            )

        self.send_error(404)

    # -- media ------------------------------------------------------------------
    def _handle_media(self, path: str) -> None:
        parts = path[len("/media/"):].split("/")
        if not parts or not parts[0]:
            return self.send_error(404)
        kind = parts[0]
        resolved = media.resolve_media(kind, tuple(parts[1:]))
        if resolved is None:
            return self.send_error(404)
        media.serve_file(self, resolved)

    # -- logging ---------------------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:
        # Keep the console quiet; uncomment the next line for verbose access logs.
        # super().log_message(fmt, *args)
        pass


def _safe_join(base: Path, rel: str) -> Path:
    """Join a static path without escaping the static root."""
    candidate = (base / rel).resolve()
    base_resolved = base.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise FileNotFoundError(rel)
    return candidate
