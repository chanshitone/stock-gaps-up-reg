from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import load_config
from .io_utils import make_run_dir
from .tushare_client import TushareClient


@dataclass(frozen=True)
class DiscoverySummary:
    start_date: date
    end_date: date
    trade_days: int
    universe_size: int
    raw_signal_rows: int
    st_filtered_rows: int
    candidate_count: int
    st_filter_mode: str


def _parse_date(value: str) -> date:
    cleaned = value.strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Invalid date: {value!r}. Use YYYY-MM-DD or YYYYMMDD.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover gap-up A-share candidates for backtests using Tushare daily data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="Path to strategy config yaml.",
    )
    parser.add_argument(
        "--start-date",
        type=_parse_date,
        required=True,
        help="Start date for discovery window.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_date,
        required=True,
        help="End date for discovery window.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output root.",
    )
    return parser


def _load_market_daily(client: TushareClient, start_date: date, end_date: date) -> tuple[pd.DataFrame, list[date]]:
    history_start = start_date - timedelta(days=450)
    trade_days = client.list_trade_days(history_start, end_date)
    frames = [client.get_daily_for_trade_date(trade_day) for trade_day in trade_days]
    market_daily = pd.concat(frames, ignore_index=True)
    market_daily = market_daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return market_daily, trade_days


def _apply_signal_rules(daily: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    grouped = daily.groupby("ts_code", sort=False)
    signal_frame = daily.copy()
    signal_frame["prev_high"] = grouped["high"].shift(1)
    signal_frame["ma50"] = grouped["close"].transform(lambda series: series.rolling(50, min_periods=50).mean())
    signal_frame["ma150"] = grouped["close"].transform(lambda series: series.rolling(150, min_periods=150).mean())
    signal_frame["ma200"] = grouped["close"].transform(lambda series: series.rolling(200, min_periods=200).mean())

    in_window = signal_frame["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ma_alignment = (signal_frame["ma50"] > signal_frame["ma150"]) & (signal_frame["ma150"] > signal_frame["ma200"])
    gap_up = signal_frame["low"] > signal_frame["prev_high"]
    filtered = signal_frame.loc[in_window & ma_alignment & gap_up].copy()
    filtered["gap_amount"] = filtered["low"] - filtered["prev_high"]
    filtered["gap_pct_vs_prev_high"] = filtered["gap_amount"] / filtered["prev_high"]
    return filtered


def _exclude_st_rows(client: TushareClient, candidates: pd.DataFrame, start_date: date, end_date: date) -> tuple[pd.DataFrame, int, str]:
    del client, start_date, end_date
    if candidates.empty:
        return candidates, 0, "not_needed"
    return candidates.reset_index(drop=True), 0, "disabled_keep_all"


def discover_candidates(client: TushareClient, start_date: date, end_date: date) -> tuple[pd.DataFrame, DiscoverySummary]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date.")

    universe = client.list_a_share_universe(start_date, end_date)
    market_daily, trade_days = _load_market_daily(client, start_date, end_date)
    filtered_daily = market_daily[market_daily["ts_code"].isin(universe["ts_code"])].copy()
    with_meta = filtered_daily.merge(universe[["ts_code", "name", "market", "exchange"]], on="ts_code", how="inner")
    raw_candidates = _apply_signal_rules(with_meta, start_date, end_date)
    filtered_candidates, st_filtered_rows, st_filter_mode = _exclude_st_rows(client, raw_candidates, start_date, end_date)
    filtered_candidates = filtered_candidates.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    filtered_candidates["detect_date"] = filtered_candidates["trade_date"].dt.strftime("%Y-%m-%d")
    filtered_candidates["note"] = filtered_candidates.apply(
        lambda row: (
            f"market={row['market']};ma50={row['ma50']:.2f};ma150={row['ma150']:.2f};"
            f"ma200={row['ma200']:.2f};gap_pct={row['gap_pct_vs_prev_high']:.4%}"
        ),
        axis=1,
    )

    summary = DiscoverySummary(
        start_date=start_date,
        end_date=end_date,
        trade_days=sum(1 for trade_day in trade_days if start_date <= trade_day <= end_date),
        universe_size=int(universe["ts_code"].nunique()),
        raw_signal_rows=int(len(raw_candidates)),
        st_filtered_rows=st_filtered_rows,
        candidate_count=int(len(filtered_candidates)),
        st_filter_mode=st_filter_mode,
    )
    return filtered_candidates, summary


def build_candidate_csv_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    ordered_columns = [
        "ts_code",
        "detect_date",
        "note",
        "name",
        "market",
        "exchange",
        "open",
        "high",
        "low",
        "close",
        "prev_high",
        "gap_amount",
        "gap_pct_vs_prev_high",
        "ma50",
        "ma150",
        "ma200",
    ]
    return candidates.loc[:, ordered_columns].copy()


def build_summary_frame(summary: DiscoverySummary) -> pd.DataFrame:
    return pd.DataFrame([asdict(summary)])


def write_reports(run_dir: Path, candidates: pd.DataFrame, summary: DiscoverySummary) -> tuple[Path, Path]:
    candidates_path = run_dir / "candidates.csv"
    summary_path = run_dir / "summary.csv"
    build_candidate_csv_frame(candidates).to_csv(candidates_path, index=False, encoding="utf-8-sig")
    build_summary_frame(summary).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return candidates_path, summary_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    output_root = args.output_dir.resolve() if args.output_dir else config.data.output_dir / "discover_candidates"
    run_dir = make_run_dir(output_root)

    print("Initializing Tushare client...", flush=True)
    client = TushareClient(cache_dir=config.data.cache_dir, exchange=config.market.exchange)
    print("Discovering candidates from daily data...", flush=True)
    candidates, summary = discover_candidates(client, args.start_date, args.end_date)
    print("Writing reports...", flush=True)
    candidates_path, summary_path = write_reports(run_dir, candidates, summary)

    print(f"Candidates report: {candidates_path}")
    print(f"Summary report: {summary_path}")
    print(f"Discovery window: {summary.start_date} -> {summary.end_date}")
    print(f"Trade days scanned: {summary.trade_days}")
    print(f"Universe size: {summary.universe_size}")
    print(f"Raw signal rows: {summary.raw_signal_rows}")
    print(f"ST-filtered rows: {summary.st_filtered_rows} ({summary.st_filter_mode})")
    print(f"Final candidates: {summary.candidate_count}")


if __name__ == "__main__":
    main()