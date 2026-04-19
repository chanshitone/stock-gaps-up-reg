"""
Peak-capital calculator.

Simulates the timeline of all trades and finds the maximum concurrent capital
deployed at any single point in time.  Capital is freed on exit_date and can
be reused by later trades.

Usage:
    python run_peak_capital.py --trades outputs/<run>/trades.csv
    python run_peak_capital.py --trades outputs/<run>/trades.csv --per-trade 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _lot_shares(capital: float, price: float) -> int:
    """A-share: round to nearest 100-share lot."""
    return int(round(capital / price / 100) * 100)


def run(trades_path: Path, per_trade: float) -> None:
    df = pd.read_csv(trades_path)
    traded = df[df["status"] == "traded"].copy()
    if traded.empty:
        print("No traded rows found.")
        return

    traded["buy_date"] = pd.to_datetime(traded["buy_date"])
    traded["exit_date"] = pd.to_datetime(traded["exit_date"])
    traded["shares"] = traded["buy_price"].apply(lambda p: _lot_shares(per_trade, p))
    traded["actual_cost"] = traded["shares"] * traded["buy_price"]
    traded["exit_proceeds"] = traded["shares"] * traded["exit_price"]

    # Build event list: buy = cash outflow, exit = cash inflow (actual proceeds)
    events: list[tuple] = []  # (datetime, order_tiebreak, cash_delta, position_delta)
    for _, row in traded.iterrows():
        buy_dt = pd.Timestamp(row["buy_time"])
        exit_dt = pd.Timestamp(row["exit_time"])
        # order_tiebreak: 0=exit first when same datetime (edge case)
        events.append((buy_dt, 1, -row["actual_cost"], 1))     # buy: spend cash
        events.append((exit_dt, 0, row["exit_proceeds"], -1))  # exit: receive proceeds

    ev = pd.DataFrame(events, columns=["event_time", "order", "cash_delta", "pos_delta"])
    # Sort by actual event time; ties broken by exits before buys
    ev = ev.sort_values(["event_time", "order"]).reset_index(drop=True)

    # Find minimum principal P so that (P + cumulative cash_delta) >= 0 at all times
    ev["cum_cash"] = ev["cash_delta"].cumsum()
    ev["cum_pos"] = ev["pos_delta"].cumsum()

    # P + cum_cash >= 0  =>  P >= -min(cum_cash)
    min_cum_cash = ev["cum_cash"].min()
    peak_capital = max(0.0, -min_cum_cash)
    peak_idx = ev["cum_cash"].idxmin()
    peak_date = ev.loc[peak_idx, "event_time"]
    peak_positions = ev.loc[peak_idx, "cum_pos"]

    # Simulate with peak_capital as starting cash
    ev["cash_balance"] = peak_capital + ev["cum_cash"]

    # Daily snapshot for display
    ev["date"] = ev["event_time"].dt.date
    daily = (
        ev.groupby("date")
        .agg(cash_balance=("cash_balance", "last"), positions=("cum_pos", "last"))
        .sort_index()
    )

    total_pnl = (traded["exit_proceeds"] - traded["actual_cost"]).sum()

    print(f"\n{'='*70}")
    print(f"  Peak Capital Calculator  (cash-flow model)")
    print(f"  Source        : {trades_path}")
    print(f"  Per-trade     : ¥{per_trade:,.0f}")
    print(f"  Total trades  : {len(traded)}")
    print(f"{'='*70}")

    print(f"\n  Cash balance curve (selected days):\n")
    for dt, row in daily.iterrows():
        bar_len = max(1, int(row["cash_balance"] / (peak_capital + total_pnl + 1) * 30))
        bar = "█" * bar_len
        print(f"    {dt.strftime('%Y-%m-%d')}  {bar}  ¥{row['cash_balance']:>10,.0f}  ({int(row['positions'])} pos)")

    print(f"\n{'─'*70}")
    print(f"  Min principal needed : ¥{peak_capital:,.0f}")
    print(f"  Bottleneck time      : {peak_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Positions at bottom  : {int(peak_positions)}")
    print(f"  Final cash balance   : ¥{peak_capital + ev['cum_cash'].iloc[-1]:,.0f}")
    print(f"  Total P&L            : ¥{total_pnl:,.2f}")
    print(f"  Return on capital    : {total_pnl / peak_capital * 100:.2f}%" if peak_capital > 0 else "  Return on capital    : N/A")
    print(f"{'─'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate minimum principal to cover all trades.")
    parser.add_argument("--trades", type=Path, required=True, help="Path to trades.csv")
    parser.add_argument("--per-trade", type=float, default=15000.0, help="Capital per trade in CNY (default: 15000)")
    args = parser.parse_args()
    run(args.trades.resolve(), args.per_trade)


if __name__ == "__main__":
    main()
