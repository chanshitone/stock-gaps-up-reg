"""
Peak-capital calculator with index-based initial sizing, fixed add-on logic,
and a same-day priority queue for constrained holdings.

This v5 variant keeps the v4 cash-flow, sizing, and add-on logic, but changes
how initial buy slots are allocated when there are more same-day signals than
available holdings.

Portfolio rules:
    - initial amount starts from 15,000 CNY per stock
    - index-based position sizing scheme is fixed at (0, 0.7, 1.3)
    - add-on logic is unchanged from v4 for accepted trades
    - concurrent holdings can be limited with --max-positions
    - when same-day signals exceed available slots, accepted initial buys are
      prioritized by these fields in order:
        1. price_vs_vwap       descending (higher is better)
        2. vol_ratio_14_30     ascending  (lower is better)
        3. gap_momentum        descending (higher is better)
        4. day1_close_strength descending (higher is better)

Feature mapping for trades.csv:
    - price_vs_vwap: buy_price / entry_vwap_at_1430 - 1
    - vol_ratio_14_30: entry_day2_volume_1430 / entry_detect_day_volume
    - gap_momentum: entry_price_up_ratio when available, otherwise price_up_ratio
    - day1_close_strength: entry_day1_close_strength

The same-day prioritization assumes entries are generated in a common session
window. The current trades dataset buys at 14:30, so this allocation rule is
applied per buy_date.

Usage:
    python run_peak_capital_v5.py --trades outputs/<run>/trades.csv
    python run_peak_capital_v5.py --trades outputs/<run>/trades.csv --max-positions 8
    python run_peak_capital_v5.py --trades outputs/<run>/trades.csv --per-trade 15000 --add-on-per-trade 20000 --max-positions 8
"""
from __future__ import annotations

import argparse
import heapq
from pathlib import Path

import pandas as pd

from run_peak_capital_v4 import (
    ADD_ON_TRIGGER_R,
    ADD_ON_MIN_HOLD_DAYS,
    DEFAULT_INDEX_COLUMN,
    DEFAULT_PER_TRADE,
    POSITION_SCHEME,
    _append_position_leg,
    _apply_initial_position_sizing,
    _build_daily_equity,
    _default_add_on_csv_path,
    _default_daily_win_loss_csv_path,
    _export_daily_win_loss,
    _find_add_on_order,
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


PRIORITY_COLUMNS = [
    "price_vs_vwap",
    "vol_ratio_14_30",
    "gap_momentum",
    "day1_close_strength",
]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator.astype(float) / denominator.astype(float)
    return result.where(denominator.astype(float) > 0)


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def _derive_price_vs_vwap(traded: pd.DataFrame) -> pd.Series:
    if "price_vs_vwap" in traded.columns:
        return _numeric_column(traded, "price_vs_vwap")
    if "entry_vwap_at_1430" not in traded.columns:
        return pd.Series(float("nan"), index=traded.index)
    return _safe_divide(traded["buy_price"], _numeric_column(traded, "entry_vwap_at_1430")) - 1.0


def _derive_vol_ratio_14_30(traded: pd.DataFrame) -> pd.Series:
    if "vol_ratio_14_30" in traded.columns:
        return _numeric_column(traded, "vol_ratio_14_30")
    required = {"entry_day2_volume_1430", "entry_detect_day_volume"}
    if not required.issubset(traded.columns):
        return pd.Series(float("nan"), index=traded.index)
    return _safe_divide(
        _numeric_column(traded, "entry_day2_volume_1430"),
        _numeric_column(traded, "entry_detect_day_volume"),
    )


def _derive_gap_momentum(traded: pd.DataFrame) -> pd.Series:
    if "gap_momentum" in traded.columns:
        return _numeric_column(traded, "gap_momentum")
    if "entry_price_up_ratio" in traded.columns:
        return _numeric_column(traded, "entry_price_up_ratio")
    if "price_up_ratio" in traded.columns:
        return _numeric_column(traded, "price_up_ratio")
    return pd.Series(float("nan"), index=traded.index)


def _derive_day1_close_strength(traded: pd.DataFrame) -> pd.Series:
    if "day1_close_strength" in traded.columns:
        return _numeric_column(traded, "day1_close_strength")
    if "entry_day1_close_strength" in traded.columns:
        return _numeric_column(traded, "entry_day1_close_strength")
    return pd.Series(float("nan"), index=traded.index)


def _add_priority_features(traded: pd.DataFrame) -> pd.DataFrame:
    prioritized = traded.copy()
    prioritized["price_vs_vwap"] = _derive_price_vs_vwap(prioritized)
    prioritized["vol_ratio_14_30"] = _derive_vol_ratio_14_30(prioritized)
    prioritized["gap_momentum"] = _derive_gap_momentum(prioritized)
    prioritized["day1_close_strength"] = _derive_day1_close_strength(prioritized)

    prioritized["buy_session_date"] = pd.to_datetime(prioritized["buy_time"], errors="coerce").dt.date
    return prioritized


def _sort_same_day_signals(signals: pd.DataFrame) -> pd.DataFrame:
    return signals.sort_values(
        by=[
            "price_vs_vwap",
            "vol_ratio_14_30",
            "gap_momentum",
            "day1_close_strength",
            "buy_time",
            "ts_code",
            "exit_time",
        ],
        ascending=[
            False,
            True,
            False,
            False,
            True,
            True,
            True,
        ],
        na_position="last",
    )


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

    prioritized = _add_priority_features(traded)
    daily_groups = prioritized.sort_values(["buy_session_date", "buy_time", "ts_code", "exit_time"]).groupby(
        "buy_session_date",
        sort=True,
        dropna=False,
    )

    for _, day_signals in daily_groups:
        day_signals = day_signals[day_signals["shares"] > 0].copy()
        if day_signals.empty:
            continue

        day_anchor_time = pd.Timestamp(day_signals["buy_time"].min())
        while active_exit_times and active_exit_times[0] <= day_anchor_time:
            heapq.heappop(active_exit_times)

        ranked_signals = _sort_same_day_signals(day_signals)
        if max_positions is None:
            accepted_signals = ranked_signals
            rejected_signals = ranked_signals.iloc[0:0]
        else:
            slots_remaining = max(0, max_positions - len(active_exit_times))
            accepted_signals = ranked_signals.head(slots_remaining)
            rejected_signals = ranked_signals.iloc[slots_remaining:]

        capped_out_indices.extend(rejected_signals.index.tolist())

        for row_index, row in accepted_signals.iterrows():
            buy_dt = pd.Timestamp(row["buy_time"])
            exit_dt = pd.Timestamp(row["exit_time"])

            heapq.heappush(active_exit_times, exit_dt)
            accepted_indices.append(int(row_index))

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
    print("  Peak Capital Calculator V5  (same-day ranked slot allocation + fixed add-on buys)")
    print(f"  Source        : {trades_path}")
    print(f"  Config        : {config_path}")
    print(f"  Add-on CSV    : {export_path}")
    print(f"  Daily W/L CSV : {daily_win_loss_export_path}")
    print(f"  Daily W/L HTML: {daily_win_loss_chart_path}")
    print(f"  Position rule : {POSITION_SCHEME} on {DEFAULT_INDEX_COLUMN}")
    print(f"  Max holdings  : {cap_label}")
    print(f"  Priority rule : {', '.join(PRIORITY_COLUMNS)}")
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
        description="Calculate minimum principal with same-day ranked slot allocation and a >5-day 1R add-on rule."
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