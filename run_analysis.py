"""
Winner vs Loser feature analysis for gap-up pullback trades.

Categorises each traded position into four groups based on pnl_r:
    Strong Winner : pnl_r >= 2
    Winner        : 0 < pnl_r < 2
    Loser         : -1 < pnl_r <= 0
    Strong Loser  : pnl_r <= -1

For every analysed feature it prints:
    * Median comparison across groups
    * Percentile distribution (P25 / P50 / P75)
    * For binary flags: % True per group
    * A suggested rule threshold

Usage:
    python run_analysis.py --trades outputs/<run>/trades.csv
    python run_analysis.py --trades outputs/<run>/trades.csv --enrich
    python run_analysis.py --trades outputs/<run>/trades.csv --enrich --config config/strategy.yaml
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Group labelling
# ---------------------------------------------------------------------------

def assign_group(pnl_r: float) -> str:
    if pnl_r >= 2.0:
        return "Strong Winner"
    elif pnl_r > 0.0:
        return "Winner"
    elif pnl_r > -1.0:
        return "Loser"
    else:
        return "Strong Loser"


GROUP_ORDER = ["Strong Winner", "Winner", "Loser", "Strong Loser"]
WINNER_GROUPS = {"Strong Winner", "Winner"}
LOSER_GROUPS = {"Loser", "Strong Loser"}


# ---------------------------------------------------------------------------
# Derived features from existing CSV columns (no API required)
# ---------------------------------------------------------------------------

def _safe_div(a, b, default: float = float("nan")) -> float:
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return default
        return float(a) / float(b)
    except Exception:
        return default


def compute_free_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Detect-day (Day1) strength ──────────────────────────────────────────
    df["gap_pct"] = df.apply(
        lambda r: _safe_div(r.get("entry_gap_size"), r.get("entry_detect_prev_high")),
        axis=1,
    )
    df["day1_change_pct"] = df.apply(
        lambda r: _safe_div(
            r.get("entry_detect_day_close", float("nan")) - r.get("entry_detect_day_open", float("nan")),
            r.get("entry_detect_day_open"),
        ),
        axis=1,
    )
    df["day1_close_strength"] = df.apply(
        lambda r: _safe_div(
            r.get("entry_detect_day_close", float("nan")) - r.get("entry_detect_day_low", float("nan")),
            r.get("entry_detect_day_high", float("nan")) - r.get("entry_detect_day_low", float("nan")),
        ),
        axis=1,
    )
    df["day1_range_pct"] = df.apply(
        lambda r: _safe_div(
            r.get("entry_detect_day_high", float("nan")) - r.get("entry_detect_day_low", float("nan")),
            r.get("entry_detect_day_open"),
        ),
        axis=1,
    )

    # ── Day2 pullback & volume ──────────────────────────────────────────────
    df["pullback_ratio"] = pd.to_numeric(df.get("entry_pullback_ratio"), errors="coerce")
    df["vol_ratio_14_30"] = df.apply(
        lambda r: _safe_div(r.get("entry_day2_volume_1430"), r.get("entry_detect_day_volume")),
        axis=1,
    )
    # Positive = buy price extended above detect close by this fraction of the original gap.
    if "entry_price_up_ratio" in df.columns:
        df["price_up_ratio"] = pd.to_numeric(df.get("entry_price_up_ratio"), errors="coerce")
    else:
        df["price_up_ratio"] = -pd.to_numeric(df.get("entry_gap_fill_ratio"), errors="coerce")

    # ── Binary behavioral features already captured in entry_notes ──────────
    for col in ("entry_has_long_lower_shadow", "entry_stabilized_after_1400", "entry_gap_unfilled"):
        if col in df.columns:
            df[col.replace("entry_", "")] = df[col].map(
                lambda v: bool(v) if not pd.isna(v) else False
            )

    # ── Trade outcome supplementary ─────────────────────────────────────────
    df["pnl_r"] = pd.to_numeric(df.get("pnl_r"), errors="coerce")
    df["mfe_r"] = pd.to_numeric(df.get("max_favorable_excursion_r"), errors="coerce")
    df["mae_r"] = pd.to_numeric(df.get("max_adverse_excursion_r"), errors="coerce")
    df["hold_days"] = pd.to_numeric(df.get("hold_days"), errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Optional minute-bar enrichment (VWAP, afternoon behaviour)
# ---------------------------------------------------------------------------

def _minute_slice(frame: pd.DataFrame, start_hhmm: str | None, end_hhmm: str | None) -> pd.DataFrame:
    out = frame.copy()
    if start_hhmm:
        t = datetime.strptime(start_hhmm, "%H:%M").time()
        out = out[out["trade_time"].dt.time >= t]
    if end_hhmm:
        t = datetime.strptime(end_hhmm, "%H:%M").time()
        out = out[out["trade_time"].dt.time <= t]
    return out.reset_index(drop=True)


def _compute_vwap(minutes: pd.DataFrame) -> float:
    total_vol = minutes["vol"].sum()
    if total_vol <= 0:
        return float("nan")
    return float((minutes["close"] * minutes["vol"]).sum() / total_vol)


def enrich_with_minute_features(df: pd.DataFrame, client, freq: str = "1min") -> pd.DataFrame:
    """
    Adds minute-bar based features for each row in df.
    Requires a TushareClient instance and local parquet minute data.
    Silently skips rows where minute data is unavailable.
    """
    extra_cols = [
        "price_vs_vwap",
        "close_vs_vwap",
        "no_new_low_after_1400",
        "close_vs_1400_open",
        "close_vs_low_after_1400",
        "entry_vs_day2_low",
        "afternoon_volume_trend",
    ]
    for col in extra_cols:
        df[col] = float("nan")

    for idx, row in df.iterrows():
        ts_code = row["ts_code"]
        buy_date_raw = row.get("buy_date")
        buy_price = row.get("buy_price")

        if pd.isna(buy_date_raw) or pd.isna(buy_price):
            continue

        buy_date = (
            buy_date_raw.date()
            if isinstance(buy_date_raw, (datetime, pd.Timestamp))
            else datetime.strptime(str(buy_date_raw)[:10], "%Y-%m-%d").date()
        )

        try:
            minutes = client.get_minutes_for_day(ts_code, buy_date, freq)
        except Exception:
            continue

        if minutes.empty:
            continue

        minutes["trade_time"] = pd.to_datetime(minutes["trade_time"])

        morning = _minute_slice(minutes, "09:30", "14:30")
        after_1400 = _minute_slice(minutes, "14:00", "14:30")

        if morning.empty:
            continue

        vwap = _compute_vwap(morning)
        day2_low = float(morning["low"].min())
        day2_close = float(morning.iloc[-1]["close"])

        df.at[idx, "price_vs_vwap"] = _safe_div(float(buy_price) - vwap, vwap)
        df.at[idx, "close_vs_vwap"] = _safe_div(day2_close - vwap, vwap)
        df.at[idx, "entry_vs_day2_low"] = _safe_div(float(buy_price) - day2_low, float(buy_price))

        if not after_1400.empty:
            open_1400 = float(after_1400.iloc[0]["open"])
            low_after_1400 = float(after_1400["low"].min())
            morning_low = float(_minute_slice(morning, "09:30", "13:59")["low"].min()) if len(morning) > 1 else day2_low

            no_new_low = low_after_1400 >= morning_low
            df.at[idx, "no_new_low_after_1400"] = float(no_new_low)
            df.at[idx, "close_vs_1400_open"] = _safe_div(day2_close - open_1400, open_1400)
            df.at[idx, "close_vs_low_after_1400"] = _safe_div(day2_close - low_after_1400, low_after_1400)

            vol_morning = float(_minute_slice(morning, "09:30", "13:59")["vol"].sum())
            vol_afternoon = float(after_1400["vol"].sum())
            afternoon_bars = max(len(after_1400), 1)
            morning_bars = max(len(_minute_slice(morning, "09:30", "13:59")), 1)
            df.at[idx, "afternoon_volume_trend"] = _safe_div(
                vol_afternoon / afternoon_bars,
                vol_morning / morning_bars,
            )

    return df


# ---------------------------------------------------------------------------
# Analysis core
# ---------------------------------------------------------------------------

CONTINUOUS_FEATURES = [
    ("gap_pct",                 "Gap pct (detect day)",           ">= thresh"),
    ("day1_change_pct",         "Day1 change pct",                ">= thresh"),
    ("day1_close_strength",     "Day1 close strength (0–1)",      ">= thresh"),
    ("day1_range_pct",          "Day1 range pct",                 "any"),
    ("pullback_ratio",          "Day2 pullback ratio",            "<= thresh"),
    ("price_up_ratio",          "Price-up ratio vs detect close", ">= thresh"),
    ("vol_ratio_14_30",         "Vol ratio 14:30 / day1 vol",     "<= thresh"),
    ("hold_days",               "Hold days",                      "any"),
    ("mfe_r",                   "Max favorable excursion (R)",    ">= thresh"),
    ("mae_r",                   "Max adverse excursion (R)",      ">= thresh (less negative)"),
    # minute-bar enriched (may be all NaN without --enrich)
    ("price_vs_vwap",           "Price vs VWAP at entry",         ">= 0"),
    ("close_vs_vwap",           "Day2 close vs VWAP",             ">= 0"),
    ("entry_vs_day2_low",       "Entry above day2 low (pct)",     ">= thresh"),
    ("close_vs_1400_open",      "Close vs 14:00 open",            ">= 0"),
    ("close_vs_low_after_1400", "Close vs low after 14:00",       ">= 0"),
    ("afternoon_volume_trend",  "Afternoon vol trend (pm/am bar)", "any"),
]

BINARY_FEATURES = [
    ("has_long_lower_shadow",   "Has long lower shadow"),
    ("stabilized_after_1400",   "Stabilized after 14:00"),
    ("gap_unfilled",            "Gap unfilled at entry"),
    ("no_new_low_after_1400",   "No new low after 14:00"),
]


def _pct_true(series: pd.Series) -> str:
    valid = series.dropna()
    if valid.empty:
        return "n/a"
    # treat numeric 0/1 or boolean
    return f"{float(valid.astype(float).mean()) * 100:.0f}%"


def _percentiles(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (
        float(valid.median()),
        float(valid.quantile(0.25)),
        float(valid.quantile(0.75)),
        len(valid),
    )


def _suggest_rule(feature: str, direction: str, sw_med: float, sl_med: float) -> str:
    if direction == "any" or any(np.isnan(v) for v in [sw_med, sl_med]):
        return "—"
    mid = (sw_med + sl_med) / 2
    if direction.startswith(">="):
        return f"{feature} >= {mid:.3f}"
    elif direction.startswith("<="):
        return f"{feature} <= {mid:.3f}"
    return "—"


def analyse(df: pd.DataFrame) -> None:
    traded = df[df["status"] == "traded"].copy()
    traded = traded[traded["pnl_r"].notna()]

    if traded.empty:
        print("No traded rows with pnl_r found.")
        return

    traded["group"] = traded["pnl_r"].apply(assign_group)

    counts = traded["group"].value_counts()
    print(f"\n{'='*90}")
    print("  TRADE GROUP BREAKDOWN")
    print(f"{'='*90}")
    for g in GROUP_ORDER:
        n = counts.get(g, 0)
        print(f"  {g:<18}: {n:>3} trades")
    print(f"  {'Total':<18}: {len(traded):>3} trades")

    groups: dict[str, pd.DataFrame] = {g: traded[traded["group"] == g] for g in GROUP_ORDER}

    # ── Continuous features ─────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  CONTINUOUS FEATURE COMPARISON")
    print(f"{'='*90}")

    header = f"  {'Feature':<32} {'SW med':>8} {'W med':>8} {'L med':>8} {'SL med':>8} {'n(SW/W/L/SL)':>14}  Suggested rule"
    print(header)
    print(f"  {'-'*88}")

    rows_for_csv = []

    for feat, label, direction in CONTINUOUS_FEATURES:
        if feat not in traded.columns:
            continue
        # skip if entirely NaN (feature not computed)
        if traded[feat].isna().all():
            continue

        sw_med, sw_p25, sw_p75, sw_n = _percentiles(groups["Strong Winner"][feat])
        w_med,  w_p25,  w_p75,  w_n  = _percentiles(groups["Winner"][feat])
        l_med,  l_p25,  l_p75,  l_n  = _percentiles(groups["Loser"][feat])
        sl_med, sl_p25, sl_p75, sl_n = _percentiles(groups["Strong Loser"][feat])

        rule = _suggest_rule(feat, direction, sw_med, sl_med)

        def _fmt(v):
            return f"{v:>8.3f}" if not np.isnan(v) else f"{'n/a':>8}"

        counts_str = f"{int(sw_n)}/{int(w_n)}/{int(l_n)}/{int(sl_n)}"
        print(f"  {label:<32} {_fmt(sw_med)} {_fmt(w_med)} {_fmt(l_med)} {_fmt(sl_med)} {counts_str:>14}  {rule}")

        rows_for_csv.append({
            "feature": feat,
            "label": label,
            "SW_median": sw_med, "SW_p25": sw_p25, "SW_p75": sw_p75, "SW_n": sw_n,
            "W_median":  w_med,  "W_p25":  w_p25,  "W_p75":  w_p75,  "W_n":  w_n,
            "L_median":  l_med,  "L_p25":  l_p25,  "L_p75":  l_p75,  "L_n":  l_n,
            "SL_median": sl_med, "SL_p25": sl_p25, "SL_p75": sl_p75, "SL_n": sl_n,
            "suggested_rule": rule,
        })

    # Percentile detail for top features
    print(f"\n{'='*90}")
    print("  PERCENTILE DETAIL (P25 / P50 / P75) for Strong Winner vs Strong Loser")
    print(f"{'='*90}")
    pct_header = f"  {'Feature':<32} {'SW P25':>8} {'SW P50':>8} {'SW P75':>8}  |  {'SL P25':>8} {'SL P50':>8} {'SL P75':>8}"
    print(pct_header)
    print(f"  {'-'*88}")
    for feat, label, direction in CONTINUOUS_FEATURES:
        if feat not in traded.columns or traded[feat].isna().all():
            continue
        sw_med, sw_p25, sw_p75, sw_n = _percentiles(groups["Strong Winner"][feat])
        sl_med, sl_p25, sl_p75, sl_n = _percentiles(groups["Strong Loser"][feat])
        if sw_n == 0 and sl_n == 0:
            continue

        def _f(v):
            return f"{v:>8.3f}" if not np.isnan(v) else f"{'n/a':>8}"

        print(f"  {label:<32} {_f(sw_p25)} {_f(sw_med)} {_f(sw_p75)}  |  {_f(sl_p25)} {_f(sl_med)} {_f(sl_p75)}")

    # ── Binary features ─────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  BINARY FEATURE COMPARISON  (% True per group)")
    print(f"{'='*90}")
    bin_header = f"  {'Feature':<32} {'SW':>8} {'W':>8} {'L':>8} {'SL':>8}  Suggested rule"
    print(bin_header)
    print(f"  {'-'*80}")
    for feat, label in BINARY_FEATURES:
        if feat not in traded.columns or traded[feat].isna().all():
            continue
        sw_pct = _pct_true(groups["Strong Winner"][feat])
        w_pct  = _pct_true(groups["Winner"][feat])
        l_pct  = _pct_true(groups["Loser"][feat])
        sl_pct = _pct_true(groups["Strong Loser"][feat])

        # Suggest rule if SW >> SL
        try:
            sw_val = float(groups["Strong Winner"][feat].astype(float).mean()) if not groups["Strong Winner"][feat].empty else float("nan")
            sl_val = float(groups["Strong Loser"][feat].astype(float).mean()) if not groups["Strong Loser"][feat].empty else float("nan")
            rule = f"require {feat} == True" if (not np.isnan(sw_val) and not np.isnan(sl_val) and sw_val - sl_val >= 0.20) else "—"
        except Exception:
            rule = "—"

        print(f"  {label:<32} {sw_pct:>8} {w_pct:>8} {l_pct:>8} {sl_pct:>8}  {rule}")

    # ── Exit reason breakdown ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  EXIT REASON BREAKDOWN  (count per group)")
    print(f"{'='*90}")
    pivot = (
        traded.groupby(["exit_reason", "group"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[g for g in GROUP_ORDER if g in traded["group"].unique()], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    print(pivot.to_string())

    # ── pnl_r summary per exit reason ───────────────────────────────────────
    print(f"\n{'='*90}")
    print("  MEDIAN pnl_r PER EXIT REASON")
    print(f"{'='*90}")
    reason_stats = (
        traded.groupby("exit_reason")["pnl_r"]
        .agg(count="count", median="median", mean="mean", min="min", max="max")
        .sort_values("median", ascending=False)
    )
    print(reason_stats.to_string(float_format="{:.3f}".format))

    return rows_for_csv


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse winner vs loser features from a trades CSV."
    )
    parser.add_argument(
        "--trades",
        type=Path,
        required=True,
        help="Path to trades CSV (e.g. outputs/<run>/trades.csv).",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        default=False,
        help="Fetch minute-bar data to compute VWAP / afternoon features (requires TUSHARE_TOKEN and local parquet).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/strategy.yaml"),
        help="Path to strategy.yaml (used when --enrich is set).",
    )
    args = parser.parse_args()

    if not args.trades.exists():
        print(f"ERROR: trades file not found: {args.trades}")
        raise SystemExit(1)

    print(f"\nLoading {args.trades} …")
    df = pd.read_csv(args.trades)

    df = compute_free_features(df)

    if args.enrich:
        print("Enriching with minute-bar features …")
        try:
            import yaml
            from src.stock_gaps_reg.config import load_config
            from src.stock_gaps_reg.tushare_client import TushareClient

            config = load_config(args.config)
            client = TushareClient(
                cache_dir=Path(config.data.cache_dir),
                exchange=config.market.exchange,
            )
            traded_mask = df["status"] == "traded"
            df.loc[traded_mask] = enrich_with_minute_features(
                df[traded_mask].copy(), client, config.data.minute_freq
            )
            print(f"  Enriched {traded_mask.sum()} traded rows.")
        except Exception as exc:
            print(f"  WARNING: enrichment failed ({exc}). Continuing without minute-bar features.")

    analyse(df)


if __name__ == "__main__":
    main()
