from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .models import Candidate


def normalize_ts_code(value: str) -> str:
    raw = value.strip().upper()
    if "." in raw:
        return raw
    if raw.startswith(("6", "5")):
        return f"{raw}.SH"
    if raw.startswith(("8", "4", "9")):
        return f"{raw}.BJ"
    return f"{raw}.SZ"


def _parse_date(value: str):
    cleaned = value.strip()
    if cleaned.isdigit() and len(cleaned) == 8:
        return datetime.strptime(cleaned, "%Y%m%d").date()
    return datetime.strptime(cleaned, "%Y-%m-%d").date()


def load_candidates(path: Path) -> list[Candidate]:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"ts_code", "detect_date"}
    missing = required - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Candidate file is missing required columns: {missing_text}")

    candidates: list[Candidate] = []
    for row in frame.to_dict(orient="records"):
        candidates.append(
            Candidate(
                ts_code=normalize_ts_code(row["ts_code"]),
                detect_date=_parse_date(row["detect_date"]),
                note=row.get("note", "").strip(),
            )
        )
    return candidates


def make_run_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
