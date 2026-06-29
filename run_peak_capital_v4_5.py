"""
Peak-capital calculator with index-based initial sizing, fixed add-on logic,
and a cap on concurrent holdings.

Simulates the timeline of all accepted trades and finds the maximum concurrent
capital deployed at any single point in time. Capital is freed on exit_date and
can be reused by later trades.

This v4.5 variant keeps the v4 cash-flow logic, but simplifies the index rule:
    - initial amount starts from 15,000 CNY per stock
    - index-based position sizing scheme is fixed at (0, 0.7, 1.3)
        - Shenzhen Component minute data is read from D:\\BaiduNetdiskDownload\\指数分时
        - only the 14:30 intraday pct change is used as the buyability/sizing basis
        - if entry_shenzhen_index_pct_chg <= 0, initial position size is 0x
        - if 0 < entry_shenzhen_index_pct_chg <= 0.2, initial position size is 0.7x
        - if entry_shenzhen_index_pct_chg > 0.2, initial position size is 1.3x
    - A-share quantity is rounded to 100-share lots
    - add-on logic is unchanged from v3 for accepted trades
        - after 5 holding days, if the day's high is greater than buy_price + 1R,
            buy one add-on position at the next trading day's opening price
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
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --per-trade 15000 --add-on-per-trade 20000
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --max-positions 10
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --max-positions 10 --initial-principal 132470
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --config config/strategy.yaml
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --add-on-csv outputs/<run>/add_on_orders.csv
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --daily-win-loss-csv outputs/<run>/daily_win_loss.csv
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --daily-positions-csv outputs/<run>/daily_positions.csv
    python run_peak_capital_v4_5.py --trades outputs/<run>/trades.csv --output-txt outputs/<run>/peak_capital_v4_5_report.txt
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import heapq
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import TextIO
import zipfile

import pandas as pd

from run_plot_daily_win_loss import default_daily_win_loss_chart_path, plot_daily_win_loss
from src.stock_gaps_reg.config import load_config
from src.stock_gaps_reg.tushare_client import TushareClient


DEFAULT_PER_TRADE = 15000.0
DEFAULT_INDEX_COLUMN = "entry_shenzhen_index_pct_chg"
INDEX_1430_COLUMN = "entry_shenzhen_index_1430_pct_chg"
DEFAULT_INDEX_MINUTE_DIR = Path(r"D:\BaiduNetdiskDownload\指数分时")
SHENZHEN_INDEX_CODE = "399001"
SHENZHEN_INDEX_TS_CODE = "399001.SZ"
INDEX_DECISION_TIME = time(hour=14, minute=30)
POSITION_SCHEME = (0.0, 0.7, 1.3)
ADD_ON_MIN_HOLD_DAYS = 6
ADD_ON_TRIGGER_R = 1.0


def _default_add_on_csv_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_add_on_orders.csv")


def _default_daily_win_loss_csv_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_daily_win_loss.csv")


def _default_daily_positions_csv_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_daily_positions.csv")


def _default_output_txt_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_peak_capital_v4_5_report.txt")


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _lot_shares(capital: float, price: float) -> int:
    """A-share: round to nearest 100-share lot."""
    return int(round(capital / price / 100) * 100)


def _classify_position_size(index_change: float) -> tuple[str, float]:
    weak, mid, strong = POSITION_SCHEME
    if index_change <= 0:
        return "weak", weak
    if index_change <= 0.2:
        return "mid", mid
    return "strong", strong


def _load_traded_rows(trades_path: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_path)

    traded = df[df["status"] == "traded"].copy()
    if traded.empty:
        return traded

    for col in ("buy_date", "exit_date", "buy_time", "exit_time"):
        traded[col] = pd.to_datetime(traded[col], errors="coerce")
    for col in ("buy_price", "exit_price", "initial_r", DEFAULT_INDEX_COLUMN):
        if col in traded.columns:
            traded[col] = pd.to_numeric(traded[col], errors="coerce")
    return traded


def _read_index_minute_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="utf-8-sig")


def _read_index_minute_zip_entry(zip_path: Path, entry_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(entry_name) as handle:
            return pd.read_csv(handle, encoding="utf-8-sig")


def _load_index_minutes_for_date(index_minute_dir: Path, trade_date: date) -> pd.DataFrame:
    direct_csv = index_minute_dir / f"{SHENZHEN_INDEX_CODE}.csv"
    if direct_csv.exists():
        return _read_index_minute_csv(direct_csv)

    year_zip = index_minute_dir / f"{trade_date.year}_1min.zip"
    year_entry = f"{SHENZHEN_INDEX_CODE}_{trade_date.year}.csv"
    if year_zip.exists():
        return _read_index_minute_zip_entry(year_zip, year_entry)

    day_zip = index_minute_dir / f"{trade_date:%Y%m%d}_1min.zip"
    if day_zip.exists():
        return _read_index_minute_zip_entry(day_zip, f"{SHENZHEN_INDEX_CODE}.csv")

    raise FileNotFoundError(
        f"Missing {SHENZHEN_INDEX_TS_CODE} minute data for {trade_date}: "
        f"expected {direct_csv}, {year_zip}, or {day_zip}"
    )


def _normalize_index_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    required = {"时间", "开盘价", "收盘价"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Index minute CSV is missing required columns: {sorted(missing)}")

    minutes = raw.copy()
    minutes["trade_time"] = pd.to_datetime(minutes["时间"], errors="coerce")
    minutes["open"] = pd.to_numeric(minutes["开盘价"], errors="coerce")
    minutes["close"] = pd.to_numeric(minutes["收盘价"], errors="coerce")
    return minutes.dropna(subset=["trade_time", "open", "close"]).sort_values("trade_time").reset_index(drop=True)


def _index_bar_at_or_before(day_minutes: pd.DataFrame, target_time: time) -> pd.Series | None:
    matched = day_minutes[day_minutes["trade_time"].dt.time <= target_time]
    if matched.empty:
        return None
    return matched.iloc[-1]


def _build_index_condition_map(
    traded: pd.DataFrame,
    index_minute_dir: Path,
) -> dict[date, dict[str, object]]:
    buy_dates = sorted(
        {
            pd.Timestamp(value).date()
            for value in traded["buy_date"]
            if pd.notna(value)
        }
    )
    result: dict[date, dict[str, object]] = {}
    cached_by_year: dict[int, pd.DataFrame] = {}

    for buy_date in buy_dates:
        year_zip = index_minute_dir / f"{buy_date.year}_1min.zip"
        direct_csv = index_minute_dir / f"{SHENZHEN_INDEX_CODE}.csv"
        if direct_csv.exists() or year_zip.exists():
            if buy_date.year not in cached_by_year:
                cached_by_year[buy_date.year] = _normalize_index_minutes(_load_index_minutes_for_date(index_minute_dir, buy_date))
            minutes = cached_by_year[buy_date.year]
        else:
            minutes = _normalize_index_minutes(_load_index_minutes_for_date(index_minute_dir, buy_date))
        day_minutes = minutes[minutes["trade_time"].dt.date == buy_date].copy()
        if day_minutes.empty:
            result[buy_date] = {
                DEFAULT_INDEX_COLUMN: float("nan"),
                INDEX_1430_COLUMN: float("nan"),
            }
            continue

        open_price = float(day_minutes.iloc[0]["open"])
        bar_1430 = _index_bar_at_or_before(day_minutes, INDEX_DECISION_TIME)
        if open_price <= 0 or bar_1430 is None:
            result[buy_date] = {
                DEFAULT_INDEX_COLUMN: float("nan"),
                INDEX_1430_COLUMN: float("nan"),
            }
            continue

        pct_1430 = (float(bar_1430["close"]) / open_price - 1.0) * 100.0
        result[buy_date] = {
            DEFAULT_INDEX_COLUMN: pct_1430,
            INDEX_1430_COLUMN: pct_1430,
        }

    return result


def _apply_index_conditions(
    traded: pd.DataFrame,
    index_minute_dir: Path,
) -> pd.DataFrame:
    condition_map = _build_index_condition_map(traded, index_minute_dir)
    enriched = traded.copy()
    for column in (DEFAULT_INDEX_COLUMN, INDEX_1430_COLUMN):
        enriched[column] = enriched["buy_date"].apply(
            lambda value, col=column: condition_map.get(pd.Timestamp(value).date(), {}).get(col)
            if pd.notna(value)
            else None
        )
    return enriched


def _append_position_leg(
    position_legs: list[dict[str, object]],
    ts_code: str,
    buy_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    shares: int,
    leg_type: str = "position",
    buy_price: float | None = None,
    buy_cost: float | None = None,
) -> None:
    position_legs.append(
        {
            "ts_code": ts_code,
            "buy_time": buy_time,
            "exit_time": exit_time,
            "shares": shares,
            "leg_type": leg_type,
            "buy_price": buy_price,
            "buy_cost": buy_cost,
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
        .agg(cash_balance=("cash_balance", "last"), positions=("cum_holdings", "last"))
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
    daily["equity_trough"] = daily["equity"].cummin()
    daily["raise"] = daily["equity"] - daily["equity_trough"]
    daily["raise_pct"] = daily["raise"] / daily["equity_trough"] * 100
    daily["equity_peak"] = daily["equity"].cummax()
    daily["pullback"] = daily["equity"] - daily["equity_peak"]
    daily["pullback_pct"] = daily["pullback"] / daily["equity_peak"] * 100
    return daily


def _max_pullback_stats(daily: pd.DataFrame) -> dict[str, object]:
    if daily.empty:
        return {
            "amount": 0.0,
            "pct": 0.0,
            "peak_date": None,
            "trough_date": None,
        }

    trough_idx = daily["pullback"].idxmin()
    peak_slice = daily.loc[:trough_idx, "equity"]
    peak_idx = peak_slice.idxmax()
    return {
        "amount": float(-daily.loc[trough_idx, "pullback"]),
        "pct": float(-daily.loc[trough_idx, "pullback_pct"]),
        "peak_date": peak_idx,
        "trough_date": trough_idx,
    }


def _max_raise_stats(daily: pd.DataFrame) -> dict[str, object]:
    if daily.empty:
        return {
            "amount": 0.0,
            "pct": 0.0,
            "trough_date": None,
            "peak_date": None,
        }

    peak_idx = daily["raise"].idxmax()
    trough_slice = daily.loc[:peak_idx, "equity"]
    trough_idx = trough_slice.idxmin()
    return {
        "amount": float(daily.loc[peak_idx, "raise"]),
        "pct": float(daily.loc[peak_idx, "raise_pct"]),
        "trough_date": trough_idx,
        "peak_date": peak_idx,
    }


def _build_add_on_execution(
    daily: pd.DataFrame,
    trigger_index: int,
    exit_dt: pd.Timestamp,
    add_on_per_trade: float,
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
    add_shares = _lot_shares(add_on_per_trade, add_price)
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
        bar = "#" * bar_len
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
                "equity_trough",
                "raise",
                "raise_pct",
                "equity_peak",
                "pullback",
                "pullback_pct",
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
                "equity_trough",
                "raise",
                "raise_pct",
                "equity_peak",
                "pullback",
                "pullback_pct",
                "positions",
            ]
        ]
    visible_daily.to_csv(export_path, index=False)


def _format_position_item(leg: dict[str, object]) -> str:
    details = []
    if leg.get("leg_type"):
        details.append(str(leg["leg_type"]))
    if leg.get("shares") is not None:
        details.append(f"{int(leg['shares'])}sh")
    if leg.get("buy_price") is not None and pd.notna(leg.get("buy_price")):
        details.append(f"@{float(leg['buy_price']):.2f}")
    return f"{leg['ts_code']}({','.join(details)})" if details else str(leg["ts_code"])


def _join_position_items(legs: list[dict[str, object]]) -> str:
    return "; ".join(_format_position_item(leg) for leg in sorted(legs, key=lambda item: (str(item["ts_code"]), str(item.get("leg_type", "")))))


def _join_ts_codes(legs: list[dict[str, object]]) -> str:
    return "; ".join(sorted({str(leg["ts_code"]) for leg in legs}))


def _build_daily_positions(
    daily: pd.DataFrame,
    position_legs: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dt in daily.index:
        close_time = pd.Timestamp(datetime.combine(pd.Timestamp(dt).date(), time(hour=15)))
        day_legs = [
            leg
            for leg in position_legs
            if pd.Timestamp(leg["buy_time"]) <= close_time < pd.Timestamp(leg["exit_time"])
        ]
        opening_legs = [
            leg
            for leg in position_legs
            if pd.Timestamp(leg["buy_time"]).date() == pd.Timestamp(dt).date()
        ]
        initial_opening_legs = [leg for leg in opening_legs if leg.get("leg_type") == "initial"]
        add_on_opening_legs = [leg for leg in opening_legs if leg.get("leg_type") == "add_on"]

        rows.append(
            {
                "date": pd.Timestamp(dt).date(),
                "holding_count": len(day_legs),
                "holding_stock_count": len({str(leg["ts_code"]) for leg in day_legs}),
                "holding_stock_list": _join_ts_codes(day_legs),
                "holding_list": _join_position_items(day_legs),
                "opening_count": len(opening_legs),
                "opening_stock_count": len({str(leg["ts_code"]) for leg in opening_legs}),
                "opening_stock_list": _join_ts_codes(opening_legs),
                "opening_list": _join_position_items(opening_legs),
                "initial_opening_count": len(initial_opening_legs),
                "initial_opening_list": _join_position_items(initial_opening_legs),
                "add_on_opening_count": len(add_on_opening_legs),
                "add_on_opening_list": _join_position_items(add_on_opening_legs),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "holding_count",
            "holding_stock_count",
            "holding_stock_list",
            "holding_list",
            "opening_count",
            "opening_stock_count",
            "opening_stock_list",
            "opening_list",
            "initial_opening_count",
            "initial_opening_list",
            "add_on_opening_count",
            "add_on_opening_list",
        ],
    )


def _export_daily_positions(daily: pd.DataFrame, position_legs: list[dict[str, object]], export_path: Path) -> None:
    _build_daily_positions(daily, position_legs).to_csv(export_path, index=False, encoding="utf-8-sig")


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

    trigger_price = float(buy_price) + ADD_ON_TRIGGER_R * float(initial_r)
    exit_dt = pd.Timestamp(exit_time)
    exit_price = float(row["exit_price"])

    for index, daily_row in daily.iterrows():
        hold_days = index + 1
        if hold_days < ADD_ON_MIN_HOLD_DAYS or float(daily_row["high"]) <= trigger_price:
            continue

        return _build_add_on_execution(daily, index, exit_dt, add_on_per_trade, exit_price)

    return None


def _apply_initial_position_sizing(traded: pd.DataFrame, per_trade: float) -> pd.DataFrame:
    sized = traded.copy()
    sized[["position_bucket", "size_multiplier"]] = sized[DEFAULT_INDEX_COLUMN].apply(
        lambda value: pd.Series(_classify_position_size(float(value))) if pd.notna(value) else pd.Series(("missing", 0.0))
    )
    sized["position_capital"] = per_trade * sized["size_multiplier"]
    sized["shares"] = sized.apply(
        lambda row: _lot_shares(float(row["position_capital"]), float(row["buy_price"]))
        if pd.notna(row["buy_price"]) and float(row["position_capital"]) > 0
        else 0,
        axis=1,
    )
    sized["actual_cost"] = sized["shares"] * sized["buy_price"]
    sized["exit_proceeds"] = sized["shares"] * sized["exit_price"]
    return sized


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
        _append_position_leg(
            position_legs,
            row["ts_code"],
            buy_dt,
            exit_dt,
            int(row["shares"]),
            "initial",
            float(row["buy_price"]),
            float(row["actual_cost"]),
        )

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
        _append_position_leg(
            position_legs,
            row["ts_code"],
            pd.Timestamp(add_on["add_time"]),
            exit_dt,
            int(add_on["add_shares"]),
            "add_on",
            float(add_on["add_price"]),
            float(add_on["add_cost"]),
        )

    return events, add_on_orders, position_legs, accepted_indices, capped_out_indices


def run(
    trades_path: Path,
    per_trade: float,
    add_on_per_trade: float | None,
    config_path: Path,
    add_on_csv_path: Path | None,
    daily_win_loss_csv_path: Path | None,
    daily_positions_csv_path: Path | None,
    initial_principal: float | None,
    max_positions: int | None,
    index_minute_dir: Path,
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

    traded = _apply_index_conditions(traded, index_minute_dir)
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
    max_market_value_idx = daily["market_value"].idxmax()
    max_market_value = float(daily.loc[max_market_value_idx, "market_value"])
    max_market_value_positions = int(daily.loc[max_market_value_idx, "positions"])

    executed_trades = traded.loc[accepted_indices].copy()
    base_total_pnl = (executed_trades["exit_proceeds"] - executed_trades["actual_cost"]).sum()
    add_on_total_pnl = sum(float(item["exit_proceeds"]) - float(item["add_cost"]) for item in add_on_orders)
    total_pnl = base_total_pnl + add_on_total_pnl
    export_path = add_on_csv_path or _default_add_on_csv_path(trades_path)
    daily_win_loss_export_path = daily_win_loss_csv_path or _default_daily_win_loss_csv_path(trades_path)
    daily_positions_export_path = daily_positions_csv_path or _default_daily_positions_csv_path(trades_path)
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
    _export_daily_positions(daily, position_legs, daily_positions_export_path)
    plot_daily_win_loss(daily_win_loss_export_path, daily_win_loss_chart_path)

    bucket_counts = executed_trades["position_bucket"].value_counts().to_dict()
    skipped_zero_size = int((traded["shares"] <= 0).sum())
    skipped_by_cap = len(capped_out_indices)
    cap_label = "unlimited" if max_positions is None else str(max_positions)

    print(f"\n{'='*70}")
    print("  Peak Capital Calculator V4.5  (14:30 index-sized initial buys + fixed add-on buys + holding cap)")
    print(f"  Source        : {trades_path}")
    print(f"  Config        : {config_path}")
    print(f"  Add-on CSV    : {export_path}")
    print(f"  Daily W/L CSV : {daily_win_loss_export_path}")
    print(f"  Daily Pos CSV : {daily_positions_export_path}")
    print(f"  Daily W/L HTML: {daily_win_loss_chart_path}")
    print(f"  Position rule : {POSITION_SCHEME} on {DEFAULT_INDEX_COLUMN}")
    print(f"  Max holdings  : {cap_label}")
    print(f"  Initial trade : ¥{per_trade:,.0f} base")
    print(f"  Add-on trade  : ¥{resolved_add_on_per_trade:,.0f}")
    print(f"  Initial cash  : ¥{starting_principal:,.0f}")
    print(f"  Index minutes : {index_minute_dir}")
    print(
        "  Index rule    : "
        f"{SHENZHEN_INDEX_TS_CODE} sizing only on "
        f"{INDEX_DECISION_TIME.strftime('%H:%M')} intraday pct"
    )
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
    print(f"  Max market value     : ¥{max_market_value:,.2f}")
    print(f"  Max MV date          : {pd.Timestamp(max_market_value_idx).strftime('%Y-%m-%d')}")
    print(f"  Max MV positions     : {max_market_value_positions}")
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
        description="Calculate minimum principal with index-sized initial buys, a >5-day 1R add-on rule, and an optional holding cap."
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
    parser.add_argument(
        "--daily-positions-csv",
        type=Path,
        default=None,
        help="Path to export daily holding and opening lists as CSV (default: beside trades CSV).",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=None,
        help="Path to write the console report as TXT (default: beside trades CSV).",
    )
    parser.add_argument(
        "--index-minute-dir",
        type=Path,
        default=DEFAULT_INDEX_MINUTE_DIR,
        help=f"Directory containing {SHENZHEN_INDEX_TS_CODE} minute data (default: {DEFAULT_INDEX_MINUTE_DIR}).",
    )
    args = parser.parse_args()
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be a positive integer")

    trades_path = args.trades.resolve()
    output_txt_path = args.output_txt.resolve() if args.output_txt else _default_output_txt_path(trades_path)
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with output_txt_path.open("w", encoding="utf-8") as report_file:
        with redirect_stdout(_Tee(sys.stdout, report_file)):
            print(f"TXT report    : {output_txt_path}")
            run(
                trades_path,
                args.per_trade,
                args.add_on_per_trade,
                args.config.resolve(),
                args.add_on_csv.resolve() if args.add_on_csv else None,
                args.daily_win_loss_csv.resolve() if args.daily_win_loss_csv else None,
                args.daily_positions_csv.resolve() if args.daily_positions_csv else None,
                args.initial_principal,
                args.max_positions,
                args.index_minute_dir.resolve(),
            )


if __name__ == "__main__":
    main()
