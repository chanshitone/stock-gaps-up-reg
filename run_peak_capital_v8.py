"""
Peak-capital calculator V8.

This variant keeps the v4.5 cash-flow model and 14:30 Shenzhen index sizing,
then adds a Leadership Radar gate before initial positions are opened:
    - strong leadership can open directly
    - leadership can open only when the trend is up or stable
    - active themes and hot/rebound themes can open only when the trend is up
    - everything else is blocked

Usage:
    python run_peak_capital_v8.py --trades outputs/<run>/trades.csv
    python run_peak_capital_v8.py --trades outputs/<run>/trades.csv --max-positions 10
    python run_peak_capital_v8.py --trades outputs/<run>/trades.csv --leadership-csv outputs/<run>/leadership_filter.csv
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import date
from pathlib import Path
import sys
from typing import TextIO

import pandas as pd

from run_peak_capital_v4_5 import (
    DEFAULT_INDEX_COLUMN,
    DEFAULT_INDEX_MINUTE_DIR,
    DEFAULT_PER_TRADE,
    INDEX_DECISION_TIME,
    POSITION_SCHEME,
    SHENZHEN_INDEX_TS_CODE,
    _Tee,
    _apply_index_conditions,
    _apply_initial_position_sizing,
    _build_daily_equity,
    _build_trade_timeline,
    _default_add_on_csv_path,
    _default_daily_positions_csv_path,
    _default_daily_win_loss_csv_path,
    _default_output_txt_path,
    _export_daily_positions,
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
from src.stock_gaps_reg.leadership_radar import (
    LeadershipConfig,
    calculate_leadership_radar,
    load_leadership_config,
)
from src.stock_gaps_reg.tushare_client import TushareClient


STRONG_LEADERSHIP = "\u5f3a\u4e3b\u7ebf"
LEADERSHIP = "\u4e3b\u7ebf"
ACTIVE_THEME = "\u6d3b\u8dc3\u9898\u6750"
HOT_REBOUND = "\u70ed\u70b9 / \u53cd\u5f39"
TREND_UP = "\u8d8b\u52bf\u5411\u4e0a"
TREND_STABLE = "\u8d8b\u52bf\u5e73\u7a33"


def _default_leadership_csv_path(trades_path: Path) -> Path:
    return trades_path.with_name(f"{trades_path.stem}_leadership_filter.csv")


def _leadership_allows_entry(status: object, trend: object) -> bool:
    status_text = "" if pd.isna(status) else str(status).strip()
    trend_text = "" if pd.isna(trend) else str(trend).strip()
    if status_text == STRONG_LEADERSHIP:
        return True
    if status_text == LEADERSHIP:
        return trend_text in {TREND_UP, TREND_STABLE}
    if status_text in {ACTIVE_THEME, HOT_REBOUND}:
        return trend_text == TREND_UP
    return False


def _coerce_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _disabled_leadership_result(traded: pd.DataFrame) -> pd.DataFrame:
    result = traded.copy()
    result["leadership_as_of"] = pd.NaT
    result["leadership_status"] = pd.NA
    result["leadership_trend"] = pd.NA
    result["leadership_industry_score"] = pd.NA
    result["leadership_industry_rank"] = pd.NA
    result["leadership_sw_l3_code"] = pd.NA
    result["leadership_sw_l3_name"] = pd.NA
    result["leadership_buyable"] = True
    result["leadership_block_reason"] = "leadership radar disabled"
    return result


def _apply_leadership_filter(
    traded: pd.DataFrame,
    client: TushareClient,
    leadership_config: LeadershipConfig,
    enabled: bool,
    as_of_mode: str,
) -> pd.DataFrame:
    if not enabled:
        return _disabled_leadership_result(traded)

    enriched = traded.copy()
    for column in (
        "leadership_as_of",
        "leadership_status",
        "leadership_trend",
        "leadership_industry_score",
        "leadership_industry_rank",
        "leadership_sw_l3_code",
        "leadership_sw_l3_name",
        "leadership_buyable",
        "leadership_block_reason",
    ):
        enriched[column] = pd.NA

    valid_buy_dates = sorted(
        {pd.Timestamp(value).date() for value in enriched["buy_date"] if pd.notna(value)}
    )
    if as_of_mode == "latest":
        evaluation_dates: list[tuple[date, pd.Series]] = [
            (valid_buy_dates[-1], pd.Series(True, index=enriched.index))
        ] if valid_buy_dates else []
    else:
        evaluation_dates = [
            (buy_date, enriched["buy_date"].dt.date == buy_date)
            for buy_date in valid_buy_dates
        ]

    for buy_date, mask in evaluation_dates:
        stock_codes = sorted(enriched.loc[mask, "ts_code"].astype(str).dropna().unique().tolist())
        if not stock_codes:
            continue
        try:
            _sectors, stocks, unmatched, _trends = calculate_leadership_radar(
                stock_codes, client, buy_date, leadership_config
            )
        except Exception as exc:
            enriched.loc[mask, "leadership_as_of"] = buy_date
            enriched.loc[mask, "leadership_buyable"] = False
            enriched.loc[mask, "leadership_block_reason"] = f"leadership radar failed: {exc}"
            continue

        stock_map = stocks.set_index("ts_code") if not stocks.empty else pd.DataFrame()
        unmatched_reasons = (
            unmatched.set_index("ts_code")["reason"].to_dict() if not unmatched.empty else {}
        )
        for row_index in enriched.loc[mask].index:
            ts_code = str(enriched.at[row_index, "ts_code"])
            enriched.at[row_index, "leadership_as_of"] = buy_date
            if not stock_map.empty and ts_code in stock_map.index:
                radar_row = stock_map.loc[ts_code]
                status = radar_row["leadership_status"]
                trend = radar_row["leadership_trend"]
                buyable = _leadership_allows_entry(status, trend)
                enriched.at[row_index, "leadership_status"] = status
                enriched.at[row_index, "leadership_trend"] = trend
                enriched.at[row_index, "leadership_industry_score"] = radar_row["industry_score"]
                enriched.at[row_index, "leadership_industry_rank"] = radar_row["industry_rank"]
                enriched.at[row_index, "leadership_sw_l3_code"] = radar_row["sw_l3_code"]
                enriched.at[row_index, "leadership_sw_l3_name"] = radar_row["sw_l3_name"]
                enriched.at[row_index, "leadership_buyable"] = buyable
                if buyable:
                    enriched.at[row_index, "leadership_block_reason"] = ""
                else:
                    enriched.at[row_index, "leadership_block_reason"] = (
                        f"blocked by leadership status/trend: {status} / {trend}"
                    )
            else:
                enriched.at[row_index, "leadership_buyable"] = False
                enriched.at[row_index, "leadership_block_reason"] = unmatched_reasons.get(
                    ts_code, "not mapped by leadership radar"
                )

    enriched["leadership_buyable"] = enriched["leadership_buyable"].fillna(False).astype(bool)
    enriched["leadership_block_reason"] = enriched["leadership_block_reason"].fillna("not evaluated")
    return enriched


def _load_leadership_filter_from_csv(traded: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    leadership = pd.read_csv(csv_path, dtype={"ts_code": str})
    required = {"ts_code", "buy_date", "leadership_buyable"}
    missing = required.difference(leadership.columns)
    if missing:
        raise ValueError(f"Leadership input CSV is missing required columns: {sorted(missing)}")

    merge_columns = [
        "ts_code",
        "buy_date",
        "leadership_as_of",
        "leadership_buyable",
        "leadership_status",
        "leadership_trend",
        "leadership_industry_score",
        "leadership_industry_rank",
        "leadership_sw_l3_code",
        "leadership_sw_l3_name",
        "leadership_block_reason",
    ]
    available = [column for column in merge_columns if column in leadership.columns]
    leadership = leadership[available].copy()
    leadership["buy_date_key"] = pd.to_datetime(leadership["buy_date"], errors="coerce").dt.date
    leadership = leadership.drop(columns=["buy_date"]).drop_duplicates(["ts_code", "buy_date_key"], keep="last")

    base = traded.copy()
    base["buy_date_key"] = base["buy_date"].dt.date
    merged = base.merge(leadership, on=["ts_code", "buy_date_key"], how="left").drop(columns=["buy_date_key"])
    merged["leadership_buyable"] = merged["leadership_buyable"].map(_coerce_bool)
    merged["leadership_block_reason"] = merged["leadership_block_reason"].fillna("not found in leadership input CSV")
    return merged


def _apply_leadership_gate_to_sizing(traded: pd.DataFrame) -> pd.DataFrame:
    gated = traded.copy()
    blocked = ~gated["leadership_buyable"].astype(bool)
    gated.loc[blocked, "shares"] = 0
    gated.loc[blocked, "actual_cost"] = 0.0
    gated.loc[blocked, "exit_proceeds"] = 0.0
    gated.loc[blocked, "position_bucket"] = gated.loc[blocked, "position_bucket"].astype(str) + "+leadership_blocked"
    return gated


def _export_leadership_filter(traded: pd.DataFrame, export_path: Path) -> None:
    columns = [
        "ts_code",
        "buy_date",
        "leadership_as_of",
        "leadership_buyable",
        "leadership_status",
        "leadership_trend",
        "leadership_industry_score",
        "leadership_industry_rank",
        "leadership_sw_l3_code",
        "leadership_sw_l3_name",
        "position_bucket",
        "size_multiplier",
        "shares",
        "leadership_block_reason",
    ]
    available = [column for column in columns if column in traded.columns]
    traded[available].to_csv(export_path, index=False, encoding="utf-8-sig")


def run(
    trades_path: Path,
    per_trade: float,
    add_on_per_trade: float | None,
    config_path: Path,
    leadership_config_path: Path,
    add_on_csv_path: Path | None,
    daily_win_loss_csv_path: Path | None,
    daily_positions_csv_path: Path | None,
    leadership_csv_path: Path | None,
    leadership_input_csv_path: Path | None,
    initial_principal: float | None,
    max_positions: int | None,
    index_minute_dir: Path,
    disable_leadership_filter: bool,
    leadership_as_of_mode: str,
) -> None:
    traded = _load_traded_rows(trades_path)
    if traded.empty:
        print("No traded rows found.")
        return

    config = load_config(config_path)
    leadership_config = load_leadership_config(leadership_config_path)
    client = TushareClient(
        cache_dir=Path(config.data.cache_dir),
        exchange=config.market.exchange,
    )
    resolved_add_on_per_trade = per_trade if add_on_per_trade is None else float(add_on_per_trade)

    traded = _apply_index_conditions(traded, index_minute_dir)
    traded = _apply_initial_position_sizing(traded, per_trade)
    if leadership_input_csv_path is not None:
        traded = _load_leadership_filter_from_csv(traded, leadership_input_csv_path)
    else:
        traded = _apply_leadership_filter(
            traded,
            client,
            leadership_config,
            enabled=not disable_leadership_filter,
            as_of_mode=leadership_as_of_mode,
        )
    traded = _apply_leadership_gate_to_sizing(traded)
    leadership_export_path = leadership_csv_path or _default_leadership_csv_path(trades_path)
    _export_leadership_filter(traded, leadership_export_path)

    events, add_on_orders, position_legs, accepted_indices, capped_out_indices = _build_trade_timeline(
        traded,
        client,
        resolved_add_on_per_trade,
        max_positions,
    )

    if not events:
        print("No initial positions were opened after applying index sizing, leadership filter, and position cap.")
        print(f"Leadership CSV: {leadership_export_path}")
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
    leadership_counts = traded["leadership_buyable"].value_counts().to_dict()
    skipped_zero_size = int((traded["shares"] <= 0).sum())
    skipped_by_leadership = int((~traded["leadership_buyable"]).sum())
    skipped_by_cap = len(capped_out_indices)
    cap_label = "unlimited" if max_positions is None else str(max_positions)

    print(f"\n{'='*70}")
    print("  Peak Capital Calculator V8  (14:30 index sizing + Leadership Radar gate + fixed add-ons)")
    print(f"  Source        : {trades_path}")
    print(f"  Config        : {config_path}")
    print(f"  Leadership cfg: {leadership_config_path}")
    print(f"  Leadership CSV: {leadership_export_path}")
    print(f"  Add-on CSV    : {export_path}")
    print(f"  Daily W/L CSV : {daily_win_loss_export_path}")
    print(f"  Daily Pos CSV : {daily_positions_export_path}")
    print(f"  Daily W/L HTML: {daily_win_loss_chart_path}")
    print(f"  Position rule : {POSITION_SCHEME} on {DEFAULT_INDEX_COLUMN}")
    print(f"  Leadership    : {'disabled' if disable_leadership_filter else 'enabled'}")
    print(f"  Radar as-of   : {leadership_as_of_mode}")
    print(f"  Max holdings  : {cap_label}")
    print(f"  Initial trade : {per_trade:,.0f} base")
    print(f"  Add-on trade  : {resolved_add_on_per_trade:,.0f}")
    print(f"  Initial cash  : {starting_principal:,.0f}")
    print(f"  Index minutes : {index_minute_dir}")
    print(
        "  Index rule    : "
        f"{SHENZHEN_INDEX_TS_CODE} sizing only on "
        f"{INDEX_DECISION_TIME.strftime('%H:%M')} intraday pct"
    )
    print(f"  Traded rows   : {len(traded)}")
    print(f"  Opened buys   : {len(executed_trades)}")
    print(f"  Skipped 0x    : {skipped_zero_size}")
    print(f"  Blocked radar : {skipped_by_leadership}")
    print(f"  Skipped by cap: {skipped_by_cap}")
    print(f"  Add-on buys   : {len(add_on_orders)}")
    print(f"  Buckets       : {bucket_counts}")
    print(f"  Radar buyable : {leadership_counts}")
    print(f"{'='*70}")

    if starting_principal < peak_capital:
        print(
            f"  Warning       : starting cash is short by {peak_capital - starting_principal:,.0f} "
            "versus the minimum principal needed"
        )

    _print_add_on_orders(add_on_orders)
    _print_cash_balance_curve(daily, starting_principal, total_pnl)
    _print_daily_win_loss(daily)

    print(f"\n{'-'*70}")
    print(f"  Min principal needed : {peak_capital:,.0f}")
    print(f"  Bottleneck time      : {peak_date.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Holdings at bottom   : {int(peak_positions)}")
    print(f"  Max market value     : {max_market_value:,.2f}")
    print(f"  Max MV date          : {pd.Timestamp(max_market_value_idx).strftime('%Y-%m-%d')}")
    print(f"  Max MV positions     : {max_market_value_positions}")
    print(f"  Final cash balance   : {starting_principal + ev['cum_cash'].iloc[-1]:,.0f}")
    print(f"  Final equity         : {daily['equity'].iloc[-1]:,.2f}")
    print(f"  Max raise            : {max_raise['amount']:,.2f} ({max_raise['pct']:.2f}%)")
    if max_raise["trough_date"] is not None and max_raise["peak_date"] is not None:
        print(
            "  Raise window         : "
            f"{pd.Timestamp(max_raise['trough_date']).strftime('%Y-%m-%d')} -> "
            f"{pd.Timestamp(max_raise['peak_date']).strftime('%Y-%m-%d')}"
        )
    print(f"  Max pullback         : {max_pullback['amount']:,.2f} ({max_pullback['pct']:.2f}%)")
    if max_pullback["peak_date"] is not None and max_pullback["trough_date"] is not None:
        print(
            "  Pullback window      : "
            f"{pd.Timestamp(max_pullback['peak_date']).strftime('%Y-%m-%d')} -> "
            f"{pd.Timestamp(max_pullback['trough_date']).strftime('%Y-%m-%d')}"
        )
    print(f"  Base trade P&L       : {base_total_pnl:,.2f}")
    print(f"  Add-on P&L           : {add_on_total_pnl:,.2f}")
    print(f"  Total P&L            : {total_pnl:,.2f}")
    print(f"  Return on capital    : {total_pnl / peak_capital * 100:.2f}%" if peak_capital > 0 else "  Return on capital    : N/A")
    print(f"{'-'*70}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate minimum principal with index-sized buys, Leadership Radar gating, add-ons, and a holding cap."
    )
    parser.add_argument("--trades", type=Path, required=True, help="Path to trades.csv")
    parser.add_argument("--per-trade", type=float, default=DEFAULT_PER_TRADE)
    parser.add_argument("--add-on-per-trade", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("config/strategy.yaml"))
    parser.add_argument("--leadership-config", type=Path, default=Path("config/leadership_radar.yaml"))
    parser.add_argument("--initial-principal", type=float, default=None)
    parser.add_argument("--add-on-csv", type=Path, default=None)
    parser.add_argument("--daily-win-loss-csv", type=Path, default=None)
    parser.add_argument("--daily-positions-csv", type=Path, default=None)
    parser.add_argument("--leadership-csv", type=Path, default=None)
    parser.add_argument(
        "--leadership-input-csv",
        type=Path,
        default=None,
        help="Reuse a previously exported v8 leadership filter CSV instead of recalculating radar.",
    )
    parser.add_argument("--output-txt", type=Path, default=None)
    parser.add_argument("--index-minute-dir", type=Path, default=DEFAULT_INDEX_MINUTE_DIR)
    parser.add_argument(
        "--leadership-as-of-mode",
        choices=["buy-date", "latest"],
        default="buy-date",
        help="Use each trade's buy_date for precise gating, or one latest buy_date snapshot for faster regression.",
    )
    parser.add_argument(
        "--disable-leadership-filter",
        action="store_true",
        help="Skip Leadership Radar gating and allow rows through after index sizing.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_positions is not None and args.max_positions <= 0:
        parser.error("--max-positions must be a positive integer")

    trades_path = args.trades.resolve()
    output_txt_path = args.output_txt.resolve() if args.output_txt else _default_output_txt_path(trades_path)
    output_txt_path = output_txt_path.with_name(output_txt_path.name.replace("v4_5", "v8"))
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with output_txt_path.open("w", encoding="utf-8") as report_file:
        with redirect_stdout(_Tee(sys.stdout, report_file)):
            print(f"TXT report    : {output_txt_path}")
            run(
                trades_path,
                args.per_trade,
                args.add_on_per_trade,
                args.config.resolve(),
                args.leadership_config.resolve(),
                args.add_on_csv.resolve() if args.add_on_csv else None,
                args.daily_win_loss_csv.resolve() if args.daily_win_loss_csv else None,
                args.daily_positions_csv.resolve() if args.daily_positions_csv else None,
                args.leadership_csv.resolve() if args.leadership_csv else None,
                args.leadership_input_csv.resolve() if args.leadership_input_csv else None,
                args.initial_principal,
                args.max_positions,
                args.index_minute_dir.resolve(),
                args.disable_leadership_filter,
                args.leadership_as_of_mode,
            )


if __name__ == "__main__":
    main()
