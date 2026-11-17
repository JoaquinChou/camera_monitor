"""AIS spatio-temporal analysis data layer.

Serves the AIS prototype experiment results — the vessel-activity grid history,
the backtest metrics and the next-day forecast — as plain JSON-ready structures.
Mirrors ``data.py``: this module only reads bundled files and has no knowledge
of HTTP or routing, so it stays independently testable.
"""
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

# 4-hour windows indexed by their starting hour (0..20). Matches the reference
# prototype's ``window_hours = 4`` convention.
HOUR_LABELS = {
    0: "00:00–04:00",
    4: "04:00–08:00",
    8: "08:00–12:00",
    12: "12:00–16:00",
    16: "16:00–20:00",
    20: "20:00–24:00",
}


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_history() -> Dict[str, Any]:
    """Return ``{"dates": [...], "rows": [...]}``.

    Each row is ``[day_index, lat_grid, lon_grid, slot0..slot5, day, night]``
    where ``lat_grid`` / ``lon_grid`` are 0.1-degree grid indices (x10) and the
    six slots are the 4-hour observation counts; ``day`` / ``night`` are the
    daylight / night-time deduplicated counts for that cell and date.
    """
    return _read_json(config.AIS_HISTORY_FILE)


def load_metrics() -> Dict[str, Any]:
    """Return the raw backtest metrics (validation + test, three methods)."""
    return _read_json(config.AIS_METRICS_FILE)


def load_forecast() -> List[Dict[str, Any]]:
    """Parse the next-day forecast CSV into a list of grid-slot records."""
    records: List[Dict[str, Any]] = []
    with open(config.AIS_FORECAST_FILE, "r", encoding="utf-8") as fh:
        for rec in csv.DictReader(fh):
            records.append(
                {
                    "date": rec["date"],
                    "hour": int(rec["hour"]),
                    "lat": float(rec["lat_center"]),
                    "lon": float(rec["lon_center"]),
                    "predicted": float(rec["predicted_vessels"]),
                    "yesterday": float(rec["yesterday"]),
                    "mean_7d": float(rec["mean_7d"]),
                    "poisson_gbdt": float(rec["poisson_gbdt"]),
                }
            )
    return records


def load_summary() -> Dict[str, Any]:
    """Aggregate the history rows into the headline numbers the cards show.

    These values are recomputed from the data (rather than hard-coded) so the
    summary always agrees with whatever history file is configured.
    """
    history = load_history()
    rows = history.get("rows", [])

    total = 0
    day_count = 0
    night_count = 0
    cell_totals: Dict[tuple, int] = {}
    for row in rows:
        lat_grid, lon_grid = row[1], row[2]
        counts = row[3:]
        six = sum(counts[:6])
        total += six
        day_count += counts[6]
        night_count += counts[7]
        cell_totals[(lat_grid, lon_grid)] = cell_totals.get((lat_grid, lon_grid), 0) + six

    peak_key = max(cell_totals, key=cell_totals.get)
    lat_grid, lon_grid = peak_key
    return {
        "days": len(history.get("dates", [])),
        "grids": len(cell_totals),
        "grid_timeslot_count": total,
        "day_count": day_count,
        "night_count": night_count,
        "peak_grid_lat": (lat_grid + 0.5) / 10,
        "peak_grid_lon": (lon_grid + 0.5) / 10,
        "peak_grid_count": cell_totals[peak_key],
    }


def load_ais() -> Dict[str, Any]:
    """Full bundle consumed by the AIS frontend view."""
    return {
        "summary": load_summary(),
        "metrics": load_metrics(),
        "history": load_history(),
        "forecast": load_forecast(),
    }


def read_forecast_csv() -> Optional[bytes]:
    """Raw forecast CSV bytes (used by the download endpoint)."""
    path = config.AIS_FORECAST_FILE
    if not path.is_file():
        return None
    return path.read_bytes()


def read_metrics_json() -> Optional[bytes]:
    """Raw metrics JSON bytes (used by the download endpoint)."""
    path = config.AIS_METRICS_FILE
    if not path.is_file():
        return None
    return path.read_bytes()
