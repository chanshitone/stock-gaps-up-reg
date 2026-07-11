from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from src.stock_gaps_reg.config import load_config
from src.stock_gaps_reg.io_utils import load_candidates, make_run_dir
from src.stock_gaps_reg.reporting import write_reports
from src.stock_gaps_reg.strategy import run_strategy, summarize_results
from src.stock_gaps_reg.tushare_client import TushareClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gap-up pullback regression tester v0.1. "
            "This variant removes the detect-day strength gate "
            "based on day1_change_rule and day1_close_strength_rule."
        )
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
    config = replace(
        config,
        entry=replace(
            config.entry,
            day1_min_change_pct=float("-inf"),
            day1_min_close_strength=float("-inf"),
        ),
    )
    candidates = load_candidates(args.candidates.resolve())
    output_root = args.output_dir.resolve() if args.output_dir else config.data.output_dir
    run_dir = make_run_dir(output_root)

    client = TushareClient(
        cache_dir=config.data.cache_dir,
        exchange=config.market.exchange,
        local_minute_data_dir=config.data.minute_data_dir,
    )
    results = run_strategy(candidates, config, client)
    summary = summarize_results(results)
    trades_path, summary_path = write_reports(run_dir, results, summary)

    print("Variant: v0.1 (day1_change_rule and day1_close_strength_rule removed)")
    print(f"Trades report: {trades_path}")
    print(f"Summary report: {summary_path}")
    print(f"Total candidates: {summary.total_candidates}")
    print(f"Total trades: {summary.total_trades}")
    print(f"Total R: {summary.total_r:.3f}")
    print(f"Win rate: {summary.win_rate:.2%}")


if __name__ == "__main__":
    main()