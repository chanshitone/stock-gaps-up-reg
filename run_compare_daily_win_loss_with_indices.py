from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_plot_daily_vs_indices import default_output_path_for_daily_csv, plot_daily_vs_indices
from src.stock_gaps_reg.daily_vs_indices_common import INDEX_SPECS
from src.stock_gaps_reg.tushare_client import TushareClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a daily win/loss CSV against same-day Shanghai, Shenzhen, and ChiNext index percentage changes."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to *_daily_win_loss.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional merged CSV output path (default: beside input CSV)",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=None,
        help="Optional HTML chart output path (default: beside input daily CSV)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory for Tushare responses (default: data/cache)",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_vs_indices.csv")


def _looks_like_percent_strings(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip()
    if values.empty:
        return False
    sample = values.head(min(len(values), 20))
    return bool(sample.str.endswith("%").all())


def _parse_percent_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip().str.rstrip("%")
    return pd.to_numeric(cleaned, errors="coerce")


def resolve_strategy_return_column(frame: pd.DataFrame) -> tuple[str, pd.Series]:
    last_column = frame.columns[-1]
    if _looks_like_percent_strings(frame[last_column]):
        return last_column, _parse_percent_series(frame[last_column])

    if "daily_return_pct" in frame.columns:
        return "daily_return_pct", pd.to_numeric(frame["daily_return_pct"], errors="coerce")

    raise ValueError(
        "Could not find a strategy daily return column. Expected the last column to contain percent strings or a daily_return_pct column."
    )


def load_daily_view(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")

    frame.columns = [str(column).strip() for column in frame.columns]
    if "date" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: date")

    return_column, strategy_returns = resolve_strategy_return_column(frame)
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily["strategy_return_pct"] = strategy_returns
    if daily["strategy_return_pct"].isna().all():
        raise ValueError(f"Could not parse any daily return percentages from column: {return_column}")

    daily["strategy_result"] = daily.get("result", pd.Series(index=daily.index, dtype=object)).fillna("")
    columns = ["date", "strategy_result", "strategy_return_pct"]
    for optional in ["daily_pnl", "equity", "cash_balance", "market_value", "positions"]:
        if optional in daily.columns:
            daily[optional] = pd.to_numeric(daily[optional], errors="coerce")
            columns.append(optional)
    if return_column not in columns:
        columns.append(return_column)
    return daily[columns].sort_values("date").reset_index(drop=True)


def fetch_index_returns(client: TushareClient, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for slug, ts_code, _label in INDEX_SPECS:
        frame = client.get_index_daily(ts_code, start_date.date(), end_date.date())
        subset = frame[["trade_date", "pct_chg"]].rename(
            columns={
                "trade_date": "date",
                "pct_chg": f"{slug}_pct_chg",
            }
        )
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on="date", how="outer")
    assert merged is not None
    return merged.sort_values("date").reset_index(drop=True)


def same_direction_rate(left: pd.Series, right: pd.Series) -> float:
    mask = left.notna() & right.notna()
    if not mask.any():
        return float("nan")
    return float((np.sign(left[mask]) == np.sign(right[mask])).mean() * 100.0)


def summarize(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    strategy = merged["strategy_return_pct"]
    for slug, _ts_code, label in INDEX_SPECS:
        index_col = f"{slug}_pct_chg"
        spread_col = f"strategy_minus_{slug}_pct"
        positive_mask = merged[index_col] > 0
        negative_mask = merged[index_col] < 0
        rows.append(
            {
                "index": label,
                "correlation": float(strategy.corr(merged[index_col])),
                "same_direction_rate_pct": same_direction_rate(strategy, merged[index_col]),
                "avg_strategy_return_when_index_up_pct": float(strategy[positive_mask].mean()),
                "avg_strategy_return_when_index_down_pct": float(strategy[negative_mask].mean()),
                "strategy_win_rate_when_index_up_pct": float((strategy[positive_mask] > 0).mean() * 100.0)
                if positive_mask.any()
                else float("nan"),
                "strategy_win_rate_when_index_down_pct": float((strategy[negative_mask] > 0).mean() * 100.0)
                if negative_mask.any()
                else float("nan"),
                "avg_strategy_minus_index_pct": float(merged[spread_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def print_summary(summary: pd.DataFrame, merged: pd.DataFrame, output_path: Path) -> None:
    print(f"Rows compared: {len(merged)}")
    print(f"Date range   : {merged['date'].min():%Y-%m-%d} to {merged['date'].max():%Y-%m-%d}")
    print(f"Merged CSV   : {output_path}")
    print()

    for row in summary.to_dict(orient="records"):
        print(row["index"])
        print(f"  correlation: {row['correlation']:.4f}")
        print(f"  same-direction rate: {row['same_direction_rate_pct']:.1f}%")
        print(f"  avg strategy return when index up: {row['avg_strategy_return_when_index_up_pct']:.3f}%")
        print(f"  avg strategy return when index down: {row['avg_strategy_return_when_index_down_pct']:.3f}%")
        print(f"  strategy win rate when index up: {row['strategy_win_rate_when_index_up_pct']:.1f}%")
        print(f"  strategy win rate when index down: {row['strategy_win_rate_when_index_down_pct']:.1f}%")
        print(f"  avg strategy minus index: {row['avg_strategy_minus_index_pct']:.3f}%")
        print()


def main() -> None:
    args = parse_args()

    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(csv_path)
    chart_output_path = args.chart_output.resolve() if args.chart_output else default_output_path_for_daily_csv(csv_path)
    daily = load_daily_view(csv_path)
    client = TushareClient(cache_dir=args.cache_dir.resolve())
    index_returns = fetch_index_returns(client, daily["date"].min(), daily["date"].max())
    merged = daily.merge(index_returns, on="date", how="left")

    missing_index_rows = merged[[f"{slug}_pct_chg" for slug, _ts_code, _label in INDEX_SPECS]].isna().any(axis=1)
    if bool(missing_index_rows.any()):
        missing_dates = merged.loc[missing_index_rows, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Missing index data for dates: {', '.join(missing_dates[:10])}")

    for slug, _ts_code, _label in INDEX_SPECS:
        merged[f"strategy_minus_{slug}_pct"] = merged["strategy_return_pct"] - merged[f"{slug}_pct_chg"]

    merged = merged.sort_values("date").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    plot_daily_vs_indices(output_path, chart_output_path)

    summary = summarize(merged)
    print_summary(summary, merged, output_path)
    print(f"Chart HTML   : {chart_output_path}")


if __name__ == "__main__":
    main()