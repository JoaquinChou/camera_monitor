"""Web demo server entry point.

Starts a threaded HTTP server that serves the demo frontend, the pipeline
outputs (videos / evidence / results / image cases) and the JSON APIs.

Usage:
    python3 web/server.py
    WEB_PORT=8000 python3 web/server.py
"""
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

# Make the ``app`` package importable whether this file is run directly
# (``python3 web/server.py``) or as a module (``python3 -m web.server``).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config  # noqa: E402
from app.handler import DemoHandler  # noqa: E402


def main() -> None:
    server = ThreadingHTTPServer((config.WEB_HOST, config.WEB_PORT), DemoHandler)
    url = f"http://{config.WEB_HOST}:{config.WEB_PORT}"
    print(f"[web] Camera monitor demo server running at {url}")
    print(f"[web] outputs: {config.OUTPUT_DIR}")
    print(f"[web] fishing image data: {config.FISHING_DATA_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web] shutting down ...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
