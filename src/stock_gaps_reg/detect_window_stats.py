from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import load_config
from .io_utils import load_candidates, make_run_dir
from .models import Candidate
from .tushare_client import TushareClient


@dataclass(frozen=True)
class DetectWindowResult:
    ts_code: str
    detect_date: date
    target_date: date | None
    window_trading_days: int
    status: str
    direction: str | None
    detect_close: float | None
    target_close: float | None
    change_amount: float | None
    change_pct: float | None
    error_message: str = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize how candidates performed from detect date through the Nth trading day close."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="Path to strategy config yaml.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("inputs/candidates.sample.csv"),
        help="CSV with ts_code and detect_date columns.",
    )
    parser.add_argument(
        "--window-trading-days",
        type=int,
        default=10,
        help="Number of trading days to include starting from detect date. Default: 10.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output root.",
    )
    return parser


def _trade_days_from_detect(client: TushareClient, detect_date: date, trading_days: int) -> list[date]:
    if trading_days <= 0:
        raise ValueError("window_trading_days must be greater than zero.")

    lookahead_days = max(30, trading_days * 4)
    for _ in range(6):
        calendar = client.list_trade_days(detect_date - timedelta(days=5), detect_date + timedelta(days=lookahead_days))
        filtered = [trade_day for trade_day in calendar if trade_day >= detect_date]
        if len(filtered) >= trading_days:
            return filtered[:trading_days]
        lookahead_days *= 2
    raise ValueError(f"Need {trading_days} trade days from {detect_date}, found {len(filtered)}.")


def _lookup_row_by_date(frame: pd.DataFrame, target_date: date) -> pd.Series:
    matched = frame.loc[frame["trade_date"] == pd.Timestamp(target_date)]
    if matched.empty:
        raise ValueError(f"Missing daily row for {target_date}.")
    return matched.iloc[0]


def analyze_candidate(candidate: Candidate, client: TushareClient, window_trading_days: int) -> DetectWindowResult:
    try:
        trade_days = _trade_days_from_detect(client, candidate.detect_date, window_trading_days)
        target_date = trade_days[-1]
        daily = client.get_daily(candidate.ts_code, candidate.detect_date, target_date)
        detect_row = _lookup_row_by_date(daily, candidate.detect_date)
        target_row = _lookup_row_by_date(daily, target_date)

        detect_close = float(detect_row["close"])
        target_close = float(target_row["close"])
        change_amount = target_close - detect_close
        change_pct = 0.0 if detect_close == 0 else change_amount / detect_close

        if change_amount > 0:
            direction = "up"
        elif change_amount < 0:
            direction = "down"
        else:
            direction = "flat"

        return DetectWindowResult(
            ts_code=candidate.ts_code,
            detect_date=candidate.detect_date,
            target_date=target_date,
            window_trading_days=window_trading_days,
            status="ok",
            direction=direction,
            detect_close=detect_close,
            target_close=target_close,
            change_amount=change_amount,
            change_pct=change_pct,
        )
    except Exception as exc:
        return DetectWindowResult(
            ts_code=candidate.ts_code,
            detect_date=candidate.detect_date,
            target_date=None,
            window_trading_days=window_trading_days,
            status="error",
            direction=None,
            detect_close=None,
            target_close=None,
            change_amount=None,
            change_pct=None,
            error_message=str(exc),
        )


def run_analysis(candidates: list[Candidate], client: TushareClient, window_trading_days: int) -> list[DetectWindowResult]:
    return [analyze_candidate(candidate, client, window_trading_days) for candidate in candidates]


def build_detail_frame(results: list[DetectWindowResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(result) for result in results])


def build_summary_frame(results: list[DetectWindowResult]) -> pd.DataFrame:
    detail = build_detail_frame(results)
    success = detail[detail["status"] == "ok"].copy()
    errors = detail[detail["status"] == "error"].copy()

    up = success[success["direction"] == "up"]
    down = success[success["direction"] == "down"]
    flat = success[success["direction"] == "flat"]

    summary = {
        "total_candidates": int(len(detail)),
        "successful_candidates": int(len(success)),
        "error_candidates": int(len(errors)),
        "up_count": int(len(up)),
        "down_count": int(len(down)),
        "flat_count": int(len(flat)),
        "up_total_change_amount": float(up["change_amount"].sum()) if not up.empty else 0.0,
        "down_total_change_amount": float(down["change_amount"].sum()) if not down.empty else 0.0,
        "net_total_change_amount": float(success["change_amount"].sum()) if not success.empty else 0.0,
        "up_total_change_pct": float(up["change_pct"].sum()) if not up.empty else 0.0,
        "down_total_change_pct": float(down["change_pct"].sum()) if not down.empty else 0.0,
        "net_total_change_pct": float(success["change_pct"].sum()) if not success.empty else 0.0,
        "up_avg_change_pct": float(up["change_pct"].mean()) if not up.empty else 0.0,
        "down_avg_change_pct": float(down["change_pct"].mean()) if not down.empty else 0.0,
        "overall_avg_change_pct": float(success["change_pct"].mean()) if not success.empty else 0.0,
    }
    return pd.DataFrame([summary])


def write_reports(run_dir: Path, results: list[DetectWindowResult]) -> tuple[Path, Path]:
    details_path = run_dir / "details.csv"
    summary_path = run_dir / "summary.csv"
    build_detail_frame(results).to_csv(details_path, index=False, encoding="utf-8-sig")
    build_summary_frame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return details_path, summary_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    candidates = load_candidates(args.candidates.resolve())
    output_root = args.output_dir.resolve() if args.output_dir else config.data.output_dir / "detect_window_stats"
    run_dir = make_run_dir(output_root)

    client = TushareClient(cache_dir=config.data.cache_dir, exchange=config.market.exchange)
    results = run_analysis(candidates, client, args.window_trading_days)
    details_path, summary_path = write_reports(run_dir, results)
    summary = build_summary_frame(results).iloc[0]

    print(f"Details report: {details_path}")
    print(f"Summary report: {summary_path}")
    print(f"Total candidates: {int(summary['total_candidates'])}")
    print(f"Up count: {int(summary['up_count'])}")
    print(f"Down count: {int(summary['down_count'])}")
    print(f"Flat count: {int(summary['flat_count'])}")
    print(f"Up total pct: {summary['up_total_change_pct']:.4%}")
    print(f"Down total pct: {summary['down_total_change_pct']:.4%}")


if __name__ == "__main__":
    main()