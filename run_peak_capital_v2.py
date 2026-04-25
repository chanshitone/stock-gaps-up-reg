"""
Peak-capital calculator with add-on position logic.

Simulates the timeline of all trades and finds the maximum concurrent capital
deployed at any single point in time. Capital is freed on exit_date and can
be reused by later trades.

This v2 variant keeps the original initial allocation rule and adds one more
position rule:
    - initial amount is still 15,000 CNY per stock
    - A-share quantity is rounded to 100-share lots
        - if a trade has been held for more than 5 trading days and its daily high
      is greater than entry + 1R, buy another 15,000 CNY at the next trading
      day's opening price

The added position exits at the same exit_time and exit_price as the original
trade recorded in the trades CSV.

Usage:
    python run_peak_capital_v2.py --trades outputs/<run>/trades.csv
    python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --per-trade 20000
    python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --initial-principal 132470
    python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --config config/strategy.yaml
    python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --add-on-csv outputs/<run>/add_on_orders.csv
    python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --daily-win-loss-csv outputs/<run>/daily_win_loss.csv
"""
from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path

import pandas as pd

from src.stock_gaps_reg.config import load_config
from src.stock_gaps_reg.tushare_client import TushareClient


DEFAULT_PER_TRADE = 15000.0
ADD_ON_MIN_HOLD_DAYS = 6
ADD_ON_TRIGGER_R = 1.0


def _default_add_on_csv_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_add_on_orders.csv")


def _default_daily_win_loss_csv_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_daily_win_loss.csv")


def _lot_shares(capital: float, price: float) -> int:
    """A-share: round to nearest 100-share lot."""
    return int(round(capital / price / 100) * 100)


def _load_traded_rows(trades_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_path)
    traded = df[df["status"] == "traded"].copy()
    if traded.empty:
        return traded

    for col in ("buy_date", "exit_date", "buy_time", "exit_time"):
        traded[col] = pd.to_datetime(traded[col], errors="coerce")
    for col in ("buy_price", "exit_price", "initial_r"):
        traded[col] = pd.to_numeric(traded[col], errors="coerce")
    return traded


def _append_position_leg(
    position_legs: list[dict[str, object]],
    ts_code: str,
    buy_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    shares: int,
) -> None:
    position_legs.append(
        {
            "ts_code": ts_code,
            "buy_time": buy_time,
            "exit_time": exit_time,
            "shares": shares,
        }
    )


def _build_daily_equity(
    events: pd.DataFrame,
    position_legs: list[dict[str, object]],
    client: TushareClient,
    initial_principal: float,
) -> pd.DataFrame:
    all_dates = set(events["date"].tolist())
    daily_prices_by_code: dict[str, pd.DataFrame] = {}

    for ts_code in sorted({str(leg["ts_code"]) for leg in position_legs}):
        code_legs = [leg for leg in position_legs if str(leg["ts_code"]) == ts_code]
        start_date = min(pd.Timestamp(leg["buy_time"]).date() for leg in code_legs)
        end_date = max(pd.Timestamp(leg["exit_time"]).date() for leg in code_legs)
        daily_prices = client.get_daily(ts_code, start_date, end_date)[["trade_date", "close"]].copy()
        daily_prices["date"] = daily_prices["trade_date"].dt.date
        daily_prices["close_time"] = daily_prices["trade_date"] + pd.Timedelta(hours=15)
        daily_prices_by_code[ts_code] = daily_prices
        all_dates.update(daily_prices["date"].tolist())

    date_index = pd.Index(sorted(all_dates), name="date")
    daily = (
        events.groupby("date")
        .agg(cash_balance=("cash_balance", "last"), positions=("cum_pos", "last"))
        .reindex(date_index)
        .ffill()
    )
    daily["cash_balance"] = daily["cash_balance"].fillna(initial_principal)
    daily["positions"] = daily["positions"].fillna(0)

    market_value = pd.Series(0.0, index=date_index)
    for leg in position_legs:
        ts_code = str(leg["ts_code"])
        daily_prices = daily_prices_by_code[ts_code]
        active_prices = daily_prices.loc[
            (daily_prices["close_time"] >= pd.Timestamp(leg["buy_time"]))
            & (daily_prices["close_time"] < pd.Timestamp(leg["exit_time"])),
            ["date", "close"],
        ]
        if active_prices.empty:
            continue
        leg_value = pd.Series(
            active_prices["close"].to_numpy() * float(leg["shares"]),
            index=pd.Index(active_prices["date"].tolist()),
        )
        market_value = market_value.add(leg_value, fill_value=0.0)

    daily["market_value"] = market_value.reindex(date_index, fill_value=0.0)
    daily["equity"] = daily["cash_balance"] + daily["market_value"]

    previous_equity = daily["equity"].shift(1).fillna(initial_principal)
    daily["daily_pnl"] = daily["equity"].diff().fillna(daily["equity"] - initial_principal)
    daily["daily_return_pct"] = daily["daily_pnl"] / previous_equity * 100
    return daily


def _build_add_on_execution(
    daily: pd.DataFrame,
    trigger_index: int,
    exit_dt: pd.Timestamp,
    per_trade: float,
    exit_price: float,
) -> dict[str, object] | None:
    if trigger_index + 1 >= len(daily):
        return None

    add_row = daily.iloc[trigger_index + 1]
    add_date = pd.Timestamp(add_row["trade_date"]).date()
    add_time = datetime.combine(add_date, time(hour=9, minute=30))
    if pd.Timestamp(add_time) >= exit_dt:
        return None

    add_price = float(add_row["open"])
    add_shares = _lot_shares(per_trade, add_price)
    if add_shares <= 0:
        return None

    return {
        "signal_date": pd.Timestamp(daily.iloc[trigger_index]["trade_date"]).date(),
        "signal_hold_days": trigger_index + 1,
        "add_date": add_date,
        "add_time": add_time,
        "add_price": add_price,
        "add_shares": add_shares,
        "add_cost": add_shares * add_price,
        "exit_proceeds": add_shares * exit_price,
    }


def _print_add_on_orders(add_on_orders: list[dict[str, object]]) -> None:
    if not add_on_orders:
        return

    print("\n  Add-on orders:\n")
    for item in add_on_orders:
        print(
            "    "
            f"{item['ts_code']}  signal={item['signal_date']} "
            f"(hold={int(item['signal_hold_days'])})  "
            f"add={item['add_date']} @ ¥{float(item['add_price']):.2f}  "
            f"shares={int(item['add_shares'])}  cost=¥{float(item['add_cost']):,.0f}  "
            f"exit={item['exit_date']} @ ¥{float(item['exit_price']):.2f}  "
            f"p&l=¥{float(item['pnl']):,.2f}"
        )


def _print_cash_balance_curve(daily: pd.DataFrame, starting_principal: float, total_pnl: float) -> None:
    print("\n  Cash balance curve (selected days):\n")
    bar_scale = max(abs(starting_principal) + abs(total_pnl), 1.0)
    for dt, row in daily.iterrows():
        bar_len = max(1, int(max(row["cash_balance"], 0.0) / bar_scale * 30))
        bar = "█" * bar_len
        print(f"    {dt.strftime('%Y-%m-%d')}  {bar}  ¥{row['cash_balance']:>10,.0f}  ({int(row['positions'])} pos)")


def _daily_result_label(value: float) -> str:
    if value > 0:
        return "WIN"
    if value < 0:
        return "LOSS"
    return "FLAT"


def _print_daily_win_loss(daily: pd.DataFrame) -> None:
    print("\n  Daily win/loss (days with exposure or P&L):\n")
    visible_daily = _visible_daily_win_loss(daily)
    for dt, row in visible_daily.iterrows():
        day_label = _daily_result_label(float(row["daily_pnl"]))
        print(
            f"    {dt.strftime('%Y-%m-%d')}  {day_label}  "
            f"p&l=¥{row['daily_pnl']:>10,.2f}  "
            f"equity=¥{row['equity']:>11,.2f}  "
            f"cash=¥{row['cash_balance']:>10,.0f}  "
            f"mv=¥{row['market_value']:>10,.0f}  "
            f"ret={row['daily_return_pct']:>7.2f}%  "
            f"({int(row['positions'])} pos)"
        )


def _visible_daily_win_loss(daily: pd.DataFrame) -> pd.DataFrame:
    return daily[(daily["positions"] > 0) | (daily["daily_pnl"].abs() > 1e-9)].copy()


def _export_daily_win_loss(daily: pd.DataFrame, export_path: Path) -> None:
    visible_daily = _visible_daily_win_loss(daily)
    if visible_daily.empty:
        visible_daily = pd.DataFrame(
            columns=[
                "date",
                "result",
                "daily_pnl",
                "equity",
                "cash_balance",
                "market_value",
                "daily_return_pct",
                "positions",
            ]
        )
    else:
        visible_daily = visible_daily.reset_index()
        visible_daily["result"] = visible_daily["daily_pnl"].apply(lambda value: _daily_result_label(float(value)))
        visible_daily = visible_daily[
            [
                "date",
                "result",
                "daily_pnl",
                "equity",
                "cash_balance",
                "market_value",
                "daily_return_pct",
                "positions",
            ]
        ]
    visible_daily.to_csv(export_path, index=False)


def _find_add_on_order(
    row: pd.Series,
    client: TushareClient,
    per_trade: float,
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

    trigger_price = float(buy_price) + ADD_ON_TRIGGER_R * float(initial_r)
    exit_dt = pd.Timestamp(exit_time)
    exit_price = float(row["exit_price"])

    for index, daily_row in daily.iterrows():
        hold_days = index + 1
        if hold_days < ADD_ON_MIN_HOLD_DAYS or float(daily_row["high"]) <= trigger_price:
            continue

        return _build_add_on_execution(daily, index, exit_dt, per_trade, exit_price)

    return None


def run(
    trades_path: Path,
    per_trade: float,
    config_path: Path,
    add_on_csv_path: Path | None,
    daily_win_loss_csv_path: Path | None,
    initial_principal: float | None,
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

    traded["shares"] = traded["buy_price"].apply(lambda p: _lot_shares(per_trade, p))
    traded["actual_cost"] = traded["shares"] * traded["buy_price"]
    traded["exit_proceeds"] = traded["shares"] * traded["exit_price"]

    events: list[tuple] = []
    add_on_orders: list[dict[str, object]] = []
    position_legs: list[dict[str, object]] = []

    for _, row in traded.iterrows():
        buy_dt = pd.Timestamp(row["buy_time"])
        exit_dt = pd.Timestamp(row["exit_time"])

        events.append((buy_dt, 1, -row["actual_cost"], 1, row["ts_code"], "initial_buy"))
        events.append((exit_dt, 0, row["exit_proceeds"], -1, row["ts_code"], "initial_exit"))
        _append_position_leg(position_legs, row["ts_code"], buy_dt, exit_dt, int(row["shares"]))

        add_on = _find_add_on_order(row, client, per_trade)
        if add_on is None:
            continue

        add_on_orders.append(
            {
                "ts_code": row["ts_code"],
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
        events.append((add_on["add_time"], 1, -add_on["add_cost"], 1, row["ts_code"], "add_on_buy"))
        events.append((exit_dt, 0, add_on["exit_proceeds"], -1, row["ts_code"], "add_on_exit"))
        _append_position_leg(position_legs, row["ts_code"], pd.Timestamp(add_on["add_time"]), exit_dt, int(add_on["add_shares"]))

    ev = pd.DataFrame(
        events,
        columns=["event_time", "order", "cash_delta", "pos_delta", "ts_code", "event_type"],
    )
    ev = ev.sort_values(["event_time", "order", "ts_code", "event_type"]).reset_index(drop=True)

    ev["cum_cash"] = ev["cash_delta"].cumsum()
    ev["cum_pos"] = ev["pos_delta"].cumsum()

    min_cum_cash = ev["cum_cash"].min()
    peak_capital = max(0.0, -min_cum_cash)
    peak_idx = ev["cum_cash"].idxmin()
    peak_date = ev.loc[peak_idx, "event_time"]
    peak_positions = ev.loc[peak_idx, "cum_pos"]

    starting_principal = peak_capital if initial_principal is None else float(initial_principal)
    ev["cash_balance"] = starting_principal + ev["cum_cash"]
    ev["date"] = ev["event_time"].dt.date
    daily = _build_daily_equity(ev, position_legs, client, starting_principal)

    base_total_pnl = (traded["exit_proceeds"] - traded["actual_cost"]).sum()
    add_on_total_pnl = sum(float(item["exit_proceeds"]) - float(item["add_cost"]) for item in add_on_orders)
    total_pnl = base_total_pnl + add_on_total_pnl
    export_path = add_on_csv_path or _default_add_on_csv_path(trades_path)
    daily_win_loss_export_path = daily_win_loss_csv_path or _default_daily_win_loss_csv_path(trades_path)
    add_on_df = pd.DataFrame(add_on_orders)
    if add_on_df.empty:
        add_on_df = pd.DataFrame(
            columns=[
                "ts_code",
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

    print(f"\n{'='*70}")
    print("  Peak Capital Calculator V2  (cash-flow model with add-on buys)")
    print(f"  Source        : {trades_path}")
    print(f"  Config        : {config_path}")
    print(f"  Add-on CSV    : {export_path}")
    print(f"  Daily W/L CSV : {daily_win_loss_export_path}")
    print(f"  Per-trade     : ¥{per_trade:,.0f}")
    print(f"  Initial cash  : ¥{starting_principal:,.0f}")
    print(f"  Total trades  : {len(traded)}")
    print(f"  Add-on buys   : {len(add_on_orders)}")
    print(f"{'='*70}")

    if starting_principal < peak_capital:
        print(f"  Warning       : starting cash is short by ¥{peak_capital - starting_principal:,.0f} versus the minimum principal needed")

    _print_add_on_orders(add_on_orders)
    _print_cash_balance_curve(daily, starting_principal, total_pnl)
    _print_daily_win_loss(daily)

    print(f"\n{'─'*70}")
    print(f"  Min principal needed : ¥{peak_capital:,.0f}")
    print(f"  Bottleneck time      : {peak_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Positions at bottom  : {int(peak_positions)}")
    print(f"  Final cash balance   : ¥{starting_principal + ev['cum_cash'].iloc[-1]:,.0f}")
    print(f"  Final equity        : ¥{daily['equity'].iloc[-1]:,.2f}")
    print(f"  Base trade P&L       : ¥{base_total_pnl:,.2f}")
    print(f"  Add-on P&L           : ¥{add_on_total_pnl:,.2f}")
    print(f"  Total P&L            : ¥{total_pnl:,.2f}")
    print(f"  Return on capital    : {total_pnl / peak_capital * 100:.2f}%" if peak_capital > 0 else "  Return on capital    : N/A")
    print(f"{'─'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate minimum principal with a >5-day 1R add-on rule.")
    parser.add_argument("--trades", type=Path, required=True, help="Path to trades.csv")
    parser.add_argument(
        "--per-trade",
        type=float,
        default=DEFAULT_PER_TRADE,
        help="Capital per buy leg in CNY (default: 15000)",
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
    run(
        args.trades.resolve(),
        args.per_trade,
        args.config.resolve(),
        args.add_on_csv.resolve() if args.add_on_csv else None,
        args.daily_win_loss_csv.resolve() if args.daily_win_loss_csv else None,
        args.initial_principal,
    )


if __name__ == "__main__":
    main()