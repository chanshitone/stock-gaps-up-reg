from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .models import SummaryMetrics, TradeResult


def build_trade_frame(trades: list[TradeResult]) -> pd.DataFrame:
    rows = []
    for trade in trades:
        payload = asdict(trade)
        notes = payload.pop("entry_notes", {})
        for key, value in notes.items():
            payload[f"entry_{key}"] = value
        rows.append(payload)
    return pd.DataFrame(rows)


def build_summary_frame(metrics: SummaryMetrics) -> pd.DataFrame:
    return pd.DataFrame([asdict(metrics)])


def write_reports(run_dir: Path, trades: list[TradeResult], metrics: SummaryMetrics) -> tuple[Path, Path]:
    trades_path = run_dir / "trades.csv"
    summary_path = run_dir / "summary.csv"
    build_trade_frame(trades).to_csv(trades_path, index=False, encoding="utf-8-sig")
    build_summary_frame(metrics).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return trades_path, summary_path
