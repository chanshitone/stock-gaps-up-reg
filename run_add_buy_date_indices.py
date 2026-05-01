from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.stock_gaps_reg.daily_vs_indices_common import INDEX_SPECS
from src.stock_gaps_reg.tushare_client import TushareClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append same-day Shanghai, Shenzhen, and ChiNext index pct_chg columns to a trade CSV using buy_date."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to the input trade CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: beside input CSV)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory for Tushare responses (default: data/cache)",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_with_buy_date_indices.csv")


def load_trades(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if "buy_date" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: buy_date")

    trades = frame.copy()
    trades["_buy_date"] = pd.to_datetime(trades["buy_date"], errors="raise").dt.normalize()
    return trades


def fetch_index_returns(client: TushareClient, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for _slug, ts_code, label in INDEX_SPECS:
        frame = client.get_index_daily(ts_code, start_date.date(), end_date.date())
        subset = frame[["trade_date", "pct_chg"]].rename(
            columns={
                "trade_date": "_buy_date",
                "pct_chg": f"{label}_pct_chg",
            }
        )
        if merged is None:
            merged = subset
        else:
            merged = merged.merge(subset, on="_buy_date", how="outer")
    assert merged is not None
    return merged.sort_values("_buy_date").reset_index(drop=True)


def enrich_trades(trades: pd.DataFrame, index_returns: pd.DataFrame) -> pd.DataFrame:
    merged = trades.merge(index_returns, on="_buy_date", how="left")
    index_columns = [f"{label}_pct_chg" for _slug, _ts_code, label in INDEX_SPECS]
    missing_rows = merged["_buy_date"].notna() & merged[index_columns].isna().any(axis=1)
    if bool(missing_rows.any()):
        missing_dates = (
            merged.loc[missing_rows, "_buy_date"].dt.strftime("%Y-%m-%d").dropna().drop_duplicates().tolist()
        )
        raise ValueError(f"Missing index data for buy_date values: {', '.join(missing_dates[:10])}")
    return merged.drop(columns=["_buy_date"])


def main() -> None:
    args = parse_args()

    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(csv_path)

    trades = load_trades(csv_path)
    client = TushareClient(cache_dir=args.cache_dir.resolve())
    index_returns = fetch_index_returns(client, trades["_buy_date"].min(), trades["_buy_date"].max())
    enriched = enrich_trades(trades, index_returns)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Rows written: {len(enriched)}")
    print(f"Output CSV : {output_path}")


if __name__ == "__main__":
    main()