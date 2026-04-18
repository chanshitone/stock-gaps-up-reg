from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .io_utils import load_candidates, make_run_dir
from .reporting import write_reports
from .strategy import run_strategy, summarize_results
from .tushare_client import TushareClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gap-up pullback regression tester based on Tushare minute data.")
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
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for run output root.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    candidates = load_candidates(args.candidates.resolve())
    output_root = args.output_dir.resolve() if args.output_dir else config.data.output_dir
    run_dir = make_run_dir(output_root)

    client = TushareClient(cache_dir=config.data.cache_dir, exchange=config.market.exchange)
    results = run_strategy(candidates, config, client)
    summary = summarize_results(results)
    trades_path, summary_path = write_reports(run_dir, results, summary)

    print(f"Trades report: {trades_path}")
    print(f"Summary report: {summary_path}")
    print(f"Total candidates: {summary.total_candidates}")
    print(f"Total trades: {summary.total_trades}")
    print(f"Total R: {summary.total_r:.3f}")
    print(f"Win rate: {summary.win_rate:.2%}")
