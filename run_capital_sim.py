"""
Fixed-capital P&L simulator.

For every traded candidate in a trades CSV, allocate a fixed capital amount,
compute the maximum whole-share position, and report actual monetary P&L.

Usage:
    python run_capital_sim.py --trades outputs/<run>/trades.csv
    python run_capital_sim.py --trades outputs/<run>/trades.csv --capital 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def run(trades_path: Path, capital: float) -> None:
    df = pd.read_csv(trades_path)

    traded = df[df["status"] == "traded"].copy()
    if traded.empty:
        print("No traded candidates found in the file.")
        return

    traded["shares"] = (capital / traded["buy_price"] / 100).apply(round).mul(100)
    traded["actual_cost"] = traded["shares"] * traded["buy_price"]
    traded["pnl_cny"] = traded["shares"] * (traded["exit_price"] - traded["buy_price"])
    traded["pnl_pct_actual"] = traded["pnl_cny"] / traded["actual_cost"] * 100

    display_cols = [
        "ts_code",
        "buy_date",
        "exit_date",
        "exit_reason",
        "buy_price",
        "exit_price",
        "shares",
        "actual_cost",
        "pnl_cny",
        "pnl_pct_actual",
        "hold_days",
    ]
    print(f"\n{'='*80}")
    print(f"  Fixed-capital simulation  |  Capital per trade: ¥{capital:,.0f}")
    print(f"  Source: {trades_path}")
    print(f"{'='*80}\n")

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    print(traded[display_cols].to_string(index=False))

    total_pnl = traded["pnl_cny"].sum()
    total_deployed = traded["actual_cost"].sum()
    winners = traded[traded["pnl_cny"] > 0]
    losers = traded[traded["pnl_cny"] < 0]
    win_rate = len(winners) / len(traded) * 100

    print(f"\n{'─'*60}")
    print(f"  Trades simulated    : {len(traded)}")
    print(f"  Winners / Losers    : {len(winners)} / {len(losers)}")
    print(f"  Win rate            : {win_rate:.1f}%")
    print(f"  Total deployed      : ¥{total_deployed:,.2f}")
    print(f"  Total P&L           : ¥{total_pnl:,.2f}")
    print(f"  Avg P&L per trade   : ¥{total_pnl / len(traded):,.2f}")
    if len(winners):
        print(f"  Best trade          : ¥{winners['pnl_cny'].max():,.2f}  ({winners.loc[winners['pnl_cny'].idxmax(), 'ts_code']})")
    if len(losers):
        print(f"  Worst trade         : ¥{losers['pnl_cny'].min():,.2f}  ({losers.loc[losers['pnl_cny'].idxmin(), 'ts_code']})")
    print(f"{'─'*60}\n")

    # Exit reason breakdown
    reason_summary = (
        traded.groupby("exit_reason")["pnl_cny"]
        .agg(count="count", total_pnl="sum", avg_pnl="mean")
        .sort_values("total_pnl", ascending=False)
    )
    print("  P&L by exit reason:")
    print(reason_summary.to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate fixed-capital P&L from a trades CSV.")
    parser.add_argument(
        "--trades",
        type=Path,
        required=True,
        help="Path to the trades CSV (e.g. outputs/<run>/trades.csv).",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=15000.0,
        help="Fixed capital to allocate per trade in CNY (default: 15000).",
    )
    args = parser.parse_args()
    run(args.trades.resolve(), args.capital)


if __name__ == "__main__":
    main()
