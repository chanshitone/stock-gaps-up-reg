"""Analyse fixed_stop trades to identify common failure patterns and test filter rules.

Usage:
    python run_fixed_stop_analysis.py --trades outputs/<run>/trades.csv
    python run_fixed_stop_analysis.py --trades outputs/<run>/trades.csv --enrich --config config/strategy.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------

def _bool_col(series: pd.Series) -> pd.Series:
    """Normalise True/False stored as strings or booleans to int 0/1."""
    return series.map(lambda x: 1 if str(x).strip().lower() == "true" else 0)


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df[df["status"] == "traded"].copy()

    # price relative to VWAP at 14:30  (positive = above VWAP)
    t["price_vs_vwap"] = (
        t["entry_buy_price"].astype(float) / t["entry_vwap_at_1430"].astype(float) - 1
    ).where(t["entry_vwap_at_1430"].astype(float) > 0)

    # VWAP slope proxy: 1 = rising 14:00→14:30, 0 = flat/falling
    t["vwap_slope_pm"] = _bool_col(t["entry_vwap_rising_after_1400"])

    # positive = buy price extended above detect close by this fraction of the original gap
    if "entry_price_up_ratio" in t.columns:
        t["price_up_ratio"] = t["entry_price_up_ratio"].astype(float)
    else:
        t["price_up_ratio"] = -t["entry_gap_fill_ratio"].astype(float)

    # pullback fraction from detect close
    t["pullback_pct"] = t["entry_pullback_ratio"].astype(float)

    # day2 volume up to 14:30 / detect-day total volume
    t["vol_ratio_14_30"] = (
        t["entry_day2_volume_1430"].astype(float) / t["entry_detect_day_volume"].astype(float)
    ).where(t["entry_detect_day_volume"].astype(float) > 0)

    # afternoon held higher than morning low  (1 = yes, no new low after 14:00)
    if "entry_day2_low_before_1400" in t.columns and "entry_day2_low_after_1400" in t.columns:
        t["no_new_low_after_1400"] = (
            t["entry_day2_low_after_1400"].astype(float)
            > t["entry_day2_low_before_1400"].astype(float)
        ).fillna(False).astype(int)
    else:
        t["no_new_low_after_1400"] = pd.NA  # old CSV without these columns

    # entry position in day2 range – requires day2_high from minute data (set by enrich step)
    if "day2_high" not in t.columns:
        t["day2_high"] = float("nan")
    if "entry_day2_low_before_1400" in t.columns:
        t["day2_low"] = t[["entry_day2_low_before_1400", "entry_day2_low_after_1400"]].astype(float).min(axis=1)
    else:
        t["day2_low"] = float("nan")
    day2_range = t["day2_high"] - t["day2_low"]
    t["entry_pos_in_day2"] = (
        (t["entry_buy_price"].astype(float) - t["day2_low"]) / day2_range
    ).where(day2_range > 0)

    # gap pct relative to previous-day high
    t["gap_pct"] = (
        t["entry_gap_size"].astype(float) / t["entry_detect_prev_high"].astype(float)
    ).where(t["entry_detect_prev_high"].astype(float) > 0)

    t["day1_change_pct"] = t["entry_day1_change_pct"].astype(float)
    t["day1_close_strength"] = t["entry_day1_close_strength"].astype(float)

    return t


def enrich_day2_high(
    t: pd.DataFrame,
    config_path: Path,
) -> pd.DataFrame:
    """Fetch day2 session high from minute data (requires Tushare token)."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from src.stock_gaps_reg.config import load_config
    from src.stock_gaps_reg.tushare_client import TushareClient

    cfg = load_config(config_path)
    client = TushareClient(token=os.environ["TUSHARE_TOKEN"], cache_dir=Path(cfg.data.cache_dir))

    highs: list[float] = []
    for _, row in t.iterrows():
        try:
            mins = client.get_minutes_for_day(row["ts_code"], row["buy_date"], cfg.data.minute_freq)
            highs.append(float(mins["high"].max()) if not mins.empty else float("nan"))
        except Exception:
            highs.append(float("nan"))
    t = t.copy()
    t["day2_high"] = highs
    return t


# ---------------------------------------------------------------------------
# Pattern tagging
# ---------------------------------------------------------------------------

def tag_pattern(row: pd.Series) -> str:
    tags = []
    pv = row.get("price_vs_vwap")
    vs = row.get("vwap_slope_pm")
    nln = row.get("no_new_low_after_1400")
    price_up_ratio = row.get("price_up_ratio")
    vr = row.get("vol_ratio_14_30")

    if pd.notna(pv) and pv < 0:
        tags.append("below_vwap")
    if pd.notna(vs) and vs <= 0:
        tags.append("vwap_flat_down")
    if pd.notna(nln) and nln == 0:
        tags.append("new_low_pm")
    if pd.notna(price_up_ratio) and price_up_ratio < 0.8:
        tags.append("weak_price_up")
    if pd.notna(vr) and vr > 0.7:
        tags.append("high_vol_pullback")
    return "|".join(tags) if tags else "clean"


# ---------------------------------------------------------------------------
# Rule impact
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "price_vs_vwap",
    "vwap_slope_pm",
    "price_up_ratio",
    "pullback_pct",
    "vol_ratio_14_30",
    "no_new_low_after_1400",
    "entry_pos_in_day2",
    "gap_pct",
    "day1_change_pct",
    "day1_close_strength",
]


def _safe(s: pd.Series, fill) -> pd.Series:
    return s.fillna(fill)


def build_rules(t: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    rules: list[tuple[str, pd.Series]] = [
        # Single rules
        ("no_new_low_after_1400 == 1",
         t["no_new_low_after_1400"] == 1),
        ("price_vs_vwap >= 0.002",
         _safe(t["price_vs_vwap"], -999) >= 0.002),
        ("vwap_slope_pm > 0",
         t["vwap_slope_pm"] > 0),
        ("price_up_ratio >= 0.9",
         _safe(t["price_up_ratio"], 0) >= 0.9),
        ("vol_ratio_14_30 <= 0.65",
         _safe(t["vol_ratio_14_30"], 1) <= 0.65),
    ]
    if t["entry_pos_in_day2"].notna().any():
        rules.append((
            "entry_pos_in_day2 >= 0.6",
            _safe(t["entry_pos_in_day2"], 0) >= 0.6,
        ))
    # Combined rules
    rules += [
        ("no_new_low + price_vs_vwap >= 0.002",
         (t["no_new_low_after_1400"] == 1)
         & (_safe(t["price_vs_vwap"], -999) >= 0.002)),
        ("no_new_low + vwap_slope_pm > 0",
         (t["no_new_low_after_1400"] == 1)
         & (t["vwap_slope_pm"] > 0)),
    ]
    return rules


def impact_row(t: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    kept = t[mask]
    removed = t[~mask]
    return {
        "rule": name,
        "kept": len(kept),
        "kept_R": round(kept["pnl_r"].sum(), 2),
        "kept_wr": f"{(kept['pnl_r'] > 0).mean():.1%}" if len(kept) > 0 else "-",
        "removed_fs": int((removed["exit_reason"] == "fixed_stop").sum()),
        "removed_wins": int((removed["pnl_r"] > 0).sum()),
        "removed_R": round(removed["pnl_r"].sum(), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse fixed_stop failure patterns.")
    parser.add_argument("--trades", required=True, type=Path, help="Path to trades.csv")
    parser.add_argument("--enrich", action="store_true", help="Fetch day2 session high from minute data")
    parser.add_argument("--config", default="config/strategy.yaml", type=Path)
    args = parser.parse_args()

    raw = pd.read_csv(args.trades)
    traded_raw = raw[raw["status"] == "traded"]
    skipped = raw[raw["status"] == "skipped"]
    print(f"\nTotal rows : {len(raw)}")
    print(f"Traded     : {len(traded_raw)}")
    print(f"Skipped    : {len(skipped)}")

    t = derive_features(raw)

    if args.enrich:
        print("\n[enrich] Fetching day2 session high from minute data …")
        t = enrich_day2_high(t, args.config)
        day2_range = t["day2_high"] - t["day2_low"]
        t["entry_pos_in_day2"] = (
            (t["entry_buy_price"].astype(float) - t["day2_low"]) / day2_range
        ).where(day2_range > 0)

    # ---- Exit reason overview ------------------------------------------------
    print("\n" + "=" * 60)
    print("Exit Reason Breakdown")
    print("=" * 60)
    breakdown = (
        t.groupby("exit_reason")["pnl_r"]
        .agg(count="count", total_R="sum", mean_R="mean", win_rate=lambda x: (x > 0).mean())
        .round(3)
    )
    print(breakdown.to_string())
    print(f"\nOverall  trades={len(t)}  total_R={t['pnl_r'].sum():.2f}"
          f"  win_rate={( t['pnl_r'] > 0).mean():.1%}")

    # ---- Fixed-stop details --------------------------------------------------
    fs = t[t["exit_reason"] == "fixed_stop"].copy()
    print(f"\n{'=' * 60}")
    print(f"Fixed-Stop Trades ({len(fs)} of {len(t)})")
    print("=" * 60)

    if fs.empty:
        print("No fixed_stop trades – nothing to analyse.")
        return

    detail_cols = ["ts_code", "detect_date", "buy_date", "pnl_r"] + [
        c for c in FEATURE_COLS if c in fs.columns
    ]
    print(fs[detail_cols].sort_values("pnl_r").to_string(index=False))

    # ---- Feature comparison --------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Feature Comparison: fixed_stop vs others (mean)")
    print("=" * 60)
    other = t[t["exit_reason"] != "fixed_stop"]
    avail = [c for c in FEATURE_COLS if c in t.columns]
    comp = pd.DataFrame({
        "fixed_stop": fs[avail].mean(),
        "others": other[avail].mean(),
        "delta": fs[avail].mean() - other[avail].mean(),
    }).round(4)
    print(comp.to_string())

    # ---- Pattern tags --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Pattern Tags on Fixed-Stop Trades")
    print("=" * 60)
    fs["pattern"] = fs.apply(tag_pattern, axis=1)
    print(fs["pattern"].value_counts().to_string())

    print("\n  Tag component breakdown:")
    for tag in ["below_vwap", "vwap_flat_down", "new_low_pm", "weak_price_up", "high_vol_pullback"]:
        cnt = fs["pattern"].str.contains(tag, regex=False).sum()
        print(f"    {tag:30s}: {cnt}/{len(fs)} ({cnt/len(fs)*100:.0f}%)")

    # ---- Rule impact tests ---------------------------------------------------
    print(f"\n{'=' * 60}")
    print("Rule Impact Tests (applied to all traded)")
    print("  kept=trades kept  kept_R=sum pnl_r  kept_wr=win rate")
    print("  removed_fs=fixed_stop trades removed  removed_wins=winning trades removed")
    print("=" * 60)
    rules = build_rules(t)
    rows = [impact_row(t, mask, name) for name, mask in rules]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
