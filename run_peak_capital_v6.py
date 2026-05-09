"""
Peak-capital calculator with index-based initial sizing, fixed add-on logic,
and a cap on concurrent holdings.

Simulates the timeline of all accepted trades and finds the maximum concurrent
capital deployed at any single point in time. Capital is freed on exit_date and
can be reused by later trades.

This v6 variant keeps the v4 sizing logic and holding-cap behavior, but narrows
the add-on rule to a single day-6 check:
    - initial amount starts from 15,000 CNY per stock
    - index-based position sizing scheme is fixed at (0, 0.7, 1.3)
    - A-share quantity is rounded to 100-share lots
    - add-on logic is applied only once for accepted trades
        - only the 6th holding trading day is checked
        - if day 6 high is greater than buy_price + 1R,
            buy one add-on position at the next trading day's opening price
        - if day 6 does not trigger, later holding days are not checked again
        - the add-on amount defaults to the same value as --per-trade, but can be
            overridden with --add-on-per-trade
    - concurrent holdings can be limited with --max-positions
        - the cap applies to accepted initial positions only
        - add-on legs do not consume an extra holding slot
        - holdings are processed in buy_time order, and exits at the same time
            free a slot before new buys are considered

The added position exits at the same exit_time and exit_price as the original
trade recorded in the trades CSV.

Usage:
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv --per-trade 15000 --add-on-per-trade 20000
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv --max-positions 10
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv --max-positions 10 --initial-principal 132470
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv --config config/strategy.yaml
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv --add-on-csv outputs/<run>/add_on_orders.csv
    python run_peak_capital_v6.py --trades outputs/<run>/trades.csv --daily-win-loss-csv outputs/<run>/daily_win_loss.csv
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import pandas as pd

from run_peak_capital_v4 import (
    ADD_ON_TRIGGER_R,
    DEFAULT_INDEX_COLUMN,
    DEFAULT_PER_TRADE,
    POSITION_SCHEME,
    _append_position_leg,
    _apply_initial_position_sizing,
    _build_add_on_execution,
    _build_daily_equity,
    _default_add_on_csv_path,
    _default_daily_win_loss_csv_path,
    _export_daily_win_loss,
    _load_traded_rows,
    _max_pullback_stats,
    _max_raise_stats,
    _print_add_on_orders,
    _print_cash_balance_curve,
    _print_daily_win_loss,
)
from run_plot_daily_win_loss import default_daily_win_loss_chart_path, plot_daily_win_loss
from src.stock_gaps_reg.config import load_config
from src.stock_gaps_reg.tushare_client import TushareClient


ADD_ON_SIGNAL_HOLD_DAY = 6


def _find_add_on_order(
    row: pd.Series,
    client: TushareClient,
    add_on_per_trade: float,
) -> dict[str, object] | None:
    buy_date = row["buy_date"]
    exit_date = row["exit_date"]
    exit_time = row["exit_time"]
    buy_price = row["buy_price"]
    initial_r = row["initial_r"]

    if pd.isna(buy_date) or pd.isna(exit_date) or pd.isna(exit_time):
        return None
    if pd.isna(buy_price) or pd.isna(initial_r) or float(initial_r) <= 0:
        return None

    daily = client.get_daily(
        row["ts_code"],
        buy_date.date(),
        exit_date.date(),
    ).sort_values("trade_date").reset_index(drop=True)
    if daily.empty:
        return None

    trigger_index = ADD_ON_SIGNAL_HOLD_DAY - 1
    if trigger_index >= len(daily):
        return None

    trigger_price = float(buy_price) + ADD_ON_TRIGGER_R * float(initial_r)
    day_six_row = daily.iloc[trigger_index]
    if float(day_six_row["high"]) <= trigger_price:
        return None

    exit_dt = pd.Timestamp(exit_time)
    exit_price = float(row["exit_price"])
    return _build_add_on_execution(daily, trigger_index, exit_dt, add_on_per_trade, exit_price)


def _build_trade_timeline(
    traded: pd.DataFrame,
    client: TushareClient,
    resolved_add_on_per_trade: float,
    max_positions: int | None,
) -> tuple[list[tuple], list[dict[str, object]], list[dict[str, object]], list[int], list[int]]:
    events: list[tuple] = []
    add_on_orders: list[dict[str, object]] = []
    position_legs: list[dict[str, object]] = []
    accepted_indices: list[int] = []
    capped_out_indices: list[int] = []
    active_exit_times: list[pd.Timestamp] = []

    ordered = traded.sort_values(["buy_time", "ts_code", "exit_time"]).reset_index()
    for _, ordered_row in ordered.iterrows():
        row_index = int(ordered_row["index"])
        row = traded.loc[row_index]
        if int(row["shares"]) <= 0:
            continue

        buy_dt = pd.Timestamp(row["buy_time"])
        exit_dt = pd.Timestamp(row["exit_time"])

        while active_exit_times and active_exit_times[0] <= buy_dt:
            heapq.heappop(active_exit_times)

        if max_positions is not None and len(active_exit_times) >= max_positions:
            capped_out_indices.append(row_index)
            continue

        heapq.heappush(active_exit_times, exit_dt)
        accepted_indices.append(row_index)

        events.append((buy_dt, 1, -row["actual_cost"], 1, row["ts_code"], "initial_buy"))
        events.append((exit_dt, 0, row["exit_proceeds"], -1, row["ts_code"], "initial_exit"))
        _append_position_leg(position_legs, row["ts_code"], buy_dt, exit_dt, int(row["shares"]))

        add_on = _find_add_on_order(row, client, resolved_add_on_per_trade)
        if add_on is None:
            continue

        add_on_orders.append(
            {
                "ts_code": row["ts_code"],
                "position_bucket": row["position_bucket"],
                "size_multiplier": float(row["size_multiplier"]),
                "signal_date": add_on["signal_date"],
                "signal_hold_days": add_on["signal_hold_days"],
                "add_date": add_on["add_date"],
                "add_price": add_on["add_price"],
                "add_shares": add_on["add_shares"],
                "add_cost": add_on["add_cost"],
                "exit_date": exit_dt.date(),
                "exit_time": exit_dt,
                "exit_price": row["exit_price"],
                "exit_proceeds": add_on["exit_proceeds"],
                "pnl": float(add_on["exit_proceeds"]) - float(add_on["add_cost"]),
            }
        )
        events.append((add_on["add_time"], 1, -add_on["add_cost"], 0, row["ts_code"], "add_on_buy"))
        events.append((exit_dt, 0, add_on["exit_proceeds"], 0, row["ts_code"], "add_on_exit"))
        _append_position_leg(position_legs, row["ts_code"], pd.Timestamp(add_on["add_time"]), exit_dt, int(add_on["add_shares"]))

    return events, add_on_orders, position_legs, accepted_indices, capped_out_indices


def run(
    trades_path: Path,
    per_trade: float,
    add_on_per_trade: float | None,
    config_path: Path,
    add_on_csv_path: Path | None,
    daily_win_loss_csv_path: Path | None,
    initial_principal: float | None,
    max_positions: int | None,
) -> None:
    traded = _load_traded_rows(trades_path)
    if traded.empty:
        print("No traded rows found.")
        return

    config = load_config(config_path)
    client = TushareClient(
        cache_dir=Path(config.data.cache_dir),
        exchange=config.market.exchange,
    )
    resolved_add_on_per_trade = per_trade if add_on_per_trade is None else float(add_on_per_trade)

    traded = _apply_initial_position_sizing(traded, per_trade)
    events, add_on_orders, position_legs, accepted_indices, capped_out_indices = _build_trade_timeline(
        traded,
        client,
        resolved_add_on_per_trade,
        max_positions,
    )

    if not events:
        print("No initial positions were opened after applying the index sizing rule and position cap.")
        return

    ev = pd.DataFrame(
        events,
        columns=["event_time", "order", "cash_delta", "holding_delta", "ts_code", "event_type"],
    )
    ev = ev.sort_values(["event_time", "order", "ts_code", "event_type"]).reset_index(drop=True)

    ev["cum_cash"] = ev["cash_delta"].cumsum()
    ev["cum_holdings"] = ev["holding_delta"].cumsum()

    min_cum_cash = ev["cum_cash"].min()
    peak_capital = max(0.0, -min_cum_cash)
    peak_idx = ev["cum_cash"].idxmin()
    peak_date = ev.loc[peak_idx, "event_time"]
    peak_positions = ev.loc[peak_idx, "cum_holdings"]

    starting_principal = peak_capital if initial_principal is None else float(initial_principal)
    ev["cash_balance"] = starting_principal + ev["cum_cash"]
    ev["date"] = ev["event_time"].dt.date
    daily = _build_daily_equity(ev, position_legs, client, starting_principal)
    max_raise = _max_raise_stats(daily)
    max_pullback = _max_pullback_stats(daily)

    executed_trades = traded.loc[accepted_indices].copy()
    base_total_pnl = (executed_trades["exit_proceeds"] - executed_trades["actual_cost"]).sum()
    add_on_total_pnl = sum(float(item["exit_proceeds"]) - float(item["add_cost"]) for item in add_on_orders)
    total_pnl = base_total_pnl + add_on_total_pnl
    export_path = add_on_csv_path or _default_add_on_csv_path(trades_path)
    daily_win_loss_export_path = daily_win_loss_csv_path or _default_daily_win_loss_csv_path(trades_path)
    daily_win_loss_chart_path = default_daily_win_loss_chart_path(daily_win_loss_export_path)
    add_on_df = pd.DataFrame(add_on_orders)
    if add_on_df.empty:
        add_on_df = pd.DataFrame(
            columns=[
                "ts_code",
                "position_bucket",
                "size_multiplier",
                "signal_date",
                "signal_hold_days",
                "add_date",
                "add_price",
                "add_shares",
                "add_cost",
                "exit_date",
                "exit_time",
                "exit_price",
                "exit_proceeds",
                "pnl",
            ]
        )
    add_on_df.to_csv(export_path, index=False)
    _export_daily_win_loss(daily, daily_win_loss_export_path)
    plot_daily_win_loss(daily_win_loss_export_path, daily_win_loss_chart_path)

    bucket_counts = executed_trades["position_bucket"].value_counts().to_dict()
    skipped_zero_size = int((traded["shares"] <= 0).sum())
    skipped_by_cap = len(capped_out_indices)
    cap_label = "unlimited" if max_positions is None else str(max_positions)

    print(f"\n{'='*70}")
    print("  Peak Capital Calculator V6  (index-sized initial buys + day-6-only add-on buys + holding cap)")
    print(f"  Source        : {trades_path}")
    print(f"  Config        : {config_path}")
    print(f"  Add-on CSV    : {export_path}")
    print(f"  Daily W/L CSV : {daily_win_loss_export_path}")
    print(f"  Daily W/L HTML: {daily_win_loss_chart_path}")
    print(f"  Position rule : {POSITION_SCHEME} on {DEFAULT_INDEX_COLUMN}")
    print(f"  Max holdings  : {cap_label}")
    print(f"  Add-on rule   : check only hold day {ADD_ON_SIGNAL_HOLD_DAY}, then buy next open if triggered")
    print(f"  Initial trade : ¥{per_trade:,.0f} base")
    print(f"  Add-on trade  : ¥{resolved_add_on_per_trade:,.0f}")
    print(f"  Initial cash  : ¥{starting_principal:,.0f}")
    print(f"  Traded rows   : {len(traded)}")
    print(f"  Opened buys   : {len(executed_trades)}")
    print(f"  Skipped 0x    : {skipped_zero_size}")
    print(f"  Skipped by cap: {skipped_by_cap}")
    print(f"  Add-on buys   : {len(add_on_orders)}")
    print(f"  Buckets       : {bucket_counts}")
    print(f"{'='*70}")

    if starting_principal < peak_capital:
        print(f"  Warning       : starting cash is short by ¥{peak_capital - starting_principal:,.0f} versus the minimum principal needed")

    _print_add_on_orders(add_on_orders)
    _print_cash_balance_curve(daily, starting_principal, total_pnl)
    _print_daily_win_loss(daily)

    print(f"\n{'-'*70}")
    print(f"  Min principal needed : ¥{peak_capital:,.0f}")
    print(f"  Bottleneck time      : {peak_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Holdings at bottom   : {int(peak_positions)}")
    print(f"  Final cash balance   : ¥{starting_principal + ev['cum_cash'].iloc[-1]:,.0f}")
    print(f"  Final equity         : ¥{daily['equity'].iloc[-1]:,.2f}")
    print(
        f"  Max raise            : ¥{max_raise['amount']:,.2f} "
        f"({max_raise['pct']:.2f}%)"
    )
    if max_raise["trough_date"] is not None and max_raise["peak_date"] is not None:
        print(
            "  Raise window         : "
            f"{pd.Timestamp(max_raise['trough_date']).strftime('%Y-%m-%d')} -> "
            f"{pd.Timestamp(max_raise['peak_date']).strftime('%Y-%m-%d')}"
        )
    print(
        f"  Max pullback         : ¥{max_pullback['amount']:,.2f} "
        f"({max_pullback['pct']:.2f}%)"
    )
    if max_pullback["peak_date"] is not None and max_pullback["trough_date"] is not None:
        print(
            "  Pullback window      : "
            f"{pd.Timestamp(max_pullback['peak_date']).strftime('%Y-%m-%d')} -> "
            f"{pd.Timestamp(max_pullback['trough_date']).strftime('%Y-%m-%d')}"
        )
    print(f"  Base trade P&L       : ¥{base_total_pnl:,.2f}")
    print(f"  Add-on P&L           : ¥{add_on_total_pnl:,.2f}")
    print(f"  Total P&L            : ¥{total_pnl:,.2f}")
    print(f"  Return on capital    : {total_pnl / peak_capital * 100:.2f}%" if peak_capital > 0 else "  Return on capital    : N/A")
    print(f"{'-'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate minimum principal with index-sized initial buys, a day-6-only 1R add-on rule, and an optional holding cap."
    )
    parser.add_argument("--trades", type=Path, required=True, help="Path to trades.csv")
    parser.add_argument(
        "--per-trade",
        type=float,
        default=DEFAULT_PER_TRADE,
        help="Base capital for the initial buy leg in CNY before applying the index sizing multiplier (default: 15000)",
    )
    parser.add_argument(
        "--add-on-per-trade",
        type=float,
        default=None,
        help="Capital for each add-on buy leg in CNY (default: same as --per-trade)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum concurrent holdings for accepted initial trades (default: unlimited)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="Path to strategy.yaml for cache/exchange settings.",
    )
    parser.add_argument(
        "--initial-principal",
        type=float,
        default=None,
        help="Starting cash for daily equity and win/loss tracking (default: min principal needed).",
    )
    parser.add_argument(
        "--add-on-csv",
        type=Path,
        default=None,
        help="Path to export add-on orders as CSV (default: beside trades CSV).",
    )
    parser.add_argument(
        "--daily-win-loss-csv",
        type=Path,
        default=None,
        help="Path to export daily win/loss as CSV (default: beside trades CSV).",
    )
    args = parser.parse_args()
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be a positive integer")

    run(
        args.trades.resolve(),
        args.per_trade,
        args.add_on_per_trade,
        args.config.resolve(),
        args.add_on_csv.resolve() if args.add_on_csv else None,
        args.daily_win_loss_csv.resolve() if args.daily_win_loss_csv else None,
        args.initial_principal,
        args.max_positions,
    )


if __name__ == "__main__":
    main()