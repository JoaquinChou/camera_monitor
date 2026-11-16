"""Media file serving.

Resolves a ``/media/<kind>/...`` request against a fixed set of allowed
directories (videos, results, evidence, images) and streams the file back with
HTTP Range support so HTML5 <video> can seek. Path resolution is constrained to
the allowed roots to prevent path traversal.
"""
import re
from pathlib import Path
from typing import Optional, Tuple

from . import config

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}

# kind -> base directory. "images" is special-cased in resolve_media because it
# depends on the dataset category (bulao-img / chuidiao-img).
_ROOT_KINDS = ("videos", "results", "evidence")

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
_CHUNK_SIZE = 64 * 1024


def content_type_for(path: Path) -> str:
    return _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


def resolve_media(kind: str, parts: Tuple[str, ...]) -> Optional[Path]:
    """Resolve a media request to a file inside its allowed root (or None)."""
    if kind in _ROOT_KINDS:
        base = {
            "videos": config.VIDEOS_DIR,
            "results": config.RESULTS_DIR,
            "evidence": config.EVIDENCE_DIR,
        }[kind]
        rel = parts
    elif kind == "images":
        # /media/images/<category>/<filename>  ->  FISHING_DATA/<category>-img/images/<filename>
        if len(parts) != 2:
            return None
        category, filename = parts
        base = config.FISHING_DATA_DIR / f"{category}-img" / "images"
        rel = (filename,)
    else:
        return None

    base_resolved = base.resolve()
    candidate = base_resolved.joinpath(*rel).resolve()
    # Reject anything that escapes the allowed root (path traversal guard).
    if candidate != base_resolved and base_resolved not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def _parse_range(range_header: Optional[str], size: int) -> Tuple[Optional[int], Optional[int]]:
    """Return (start, end) for a 206 response, (None, None) for a full 200,
    or (-1, -1) for an unsatisfiable 416."""
    if not range_header:
        return None, None
    match = _RANGE_RE.match(range_header.strip())
    if not match:
        return -1, -1

    start_s, end_s = match.groups()
    if not start_s and not end_s:
        return -1, -1

    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else size - 1
    end = min(end, size - 1)

    if start >= size or start > end:
        return -1, -1
    return start, end


def serve_file(handler, path: Path) -> None:
    """Stream ``path`` to the client, honouring a byte Range request."""
    if path is None or not path.is_file():
        handler.send_error(404)
        return

    size = path.stat().st_size
    content_type = content_type_for(path)
    start, end = _parse_range(handler.headers.get("Range"), size)
    is_head = handler.command == "HEAD"

    if start is None:
        handler.send_response(200)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(size))
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        if not is_head:
            _stream(handler, path, 0, size)
    elif start == -1:
        handler.send_response(416)
        handler.send_header("Content-Range", f"bytes */{size}")
        handler.end_headers()
    else:
        length = end - start + 1
        handler.send_response(206)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(length))
        handler.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        handler.send_header("Accept-Ranges", "bytes")
        handler.end_headers()
        if not is_head:
            _stream(handler, path, start, length)


def _stream(handler, path: Path, offset: int, length: int) -> None:
    """Write ``length`` bytes starting at ``offset``, ignoring client disconnects."""
    remaining = length
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            while remaining > 0:
                chunk = fh.read(min(_CHUNK_SIZE, remaining))
                if not chunk:
                    break
                handler.wfile.write(chunk)
                remaining -= len(chunk)
    except (BrokenPipeError, ConnectionResetError):
        pass
