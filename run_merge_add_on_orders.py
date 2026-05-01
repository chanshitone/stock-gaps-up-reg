from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["ts_code", "exit_date", "exit_time"]
MERGE_COLUMNS = ["initial_stop_price", "max_favorable_excursion_r"]
BUY_CSV_CANDIDATES = ["buy_trades.csv", "trades.csv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge add-on order CSVs with trade-level stop and excursion metrics for a given output folder."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path to an output folder containing the trade CSV and add-on order CSV",
    )
    parser.add_argument(
        "--buy-csv",
        help="Optional explicit path to the base trade CSV (defaults to buy_trades.csv or trades.csv in the output folder)",
    )
    parser.add_argument(
        "--add-on-csv",
        help="Optional explicit path to the add-on order CSV (defaults to the only *add_on_orders*.csv in the output folder)",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path (defaults to <add_on_csv_stem>_with_trade_metrics.csv beside the add-on CSV)",
    )
    return parser.parse_args()


def normalize_ts_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def normalize_exit_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), format="mixed", errors="coerce")
    normalized = parsed.dt.strftime("%Y-%m-%d")
    fallback = series.fillna("").astype(str).str.strip()
    return normalized.fillna(fallback)


def normalize_exit_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), format="mixed", errors="coerce")
    normalized = parsed.dt.strftime("%Y-%m-%d %H:%M:%S")
    fallback = series.fillna("").astype(str).str.strip()
    return normalized.fillna(fallback)


def resolve_buy_csv(output_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Base trade CSV not found: {path}")
        return path

    for filename in BUY_CSV_CANDIDATES:
        candidate = output_dir / filename
        if candidate.exists():
            return candidate

    expected = ", ".join(BUY_CSV_CANDIDATES)
    raise FileNotFoundError(
        f"Could not find a base trade CSV in {output_dir}. Expected one of: {expected}"
    )


def resolve_add_on_csv(output_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Add-on order CSV not found: {path}")
        return path

    matches = sorted(
        path for path in output_dir.glob("*add_on_orders*.csv") if path.stem.endswith("add_on_orders")
    )
    if not matches:
        raise FileNotFoundError(f"Could not find any *add_on_orders*.csv files in {output_dir}")
    if len(matches) > 1:
        match_list = ", ".join(str(path.name) for path in matches)
        raise ValueError(
            "Found multiple add-on order CSVs. Pass --add-on-csv explicitly. Matches: "
            f"{match_list}"
        )
    return matches[0]


def validate_required_columns(frame: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def prepare_join_keys(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["ts_code"] = normalize_ts_code(prepared["ts_code"])
    prepared["exit_date"] = normalize_exit_date(prepared["exit_date"])
    prepared["exit_time"] = normalize_exit_time(prepared["exit_time"])
    return prepared


def validate_unique_keys(frame: pd.DataFrame, path: Path) -> None:
    duplicate_mask = frame.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        sample = frame.loc[duplicate_mask, KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(f"{path} has duplicate join keys. Sample: {sample}")


def default_output_path(add_on_csv_path: Path) -> Path:
    return add_on_csv_path.with_name(f"{add_on_csv_path.stem}_with_trade_metrics.csv")


def build_output(add_on_frame: pd.DataFrame, merged_metrics: pd.DataFrame) -> pd.DataFrame:
    merged = add_on_frame.merge(
        merged_metrics[KEY_COLUMNS + MERGE_COLUMNS],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    missing_mask = merged[MERGE_COLUMNS].isna().any(axis=1)
    if missing_mask.any():
        sample = merged.loc[missing_mask, KEY_COLUMNS].head(10).to_dict(orient="records")
        raise ValueError(f"Missing merged trade metrics for some add-on rows. Sample: {sample}")

    columns = list(add_on_frame.columns)
    insert_at = columns.index("exit_date") if "exit_date" in columns else len(columns)
    final_columns = columns[:insert_at] + MERGE_COLUMNS + columns[insert_at:]
    return merged[final_columns]


def main() -> int:
    args = parse_args()

    output_dir = Path(args.output_dir)
    buy_csv_path = resolve_buy_csv(output_dir, args.buy_csv)
    add_on_csv_path = resolve_add_on_csv(output_dir, args.add_on_csv)
    output_path = Path(args.output) if args.output else default_output_path(add_on_csv_path)

    buy_frame = pd.read_csv(buy_csv_path)
    add_on_frame = pd.read_csv(add_on_csv_path)

    validate_required_columns(buy_frame, KEY_COLUMNS + MERGE_COLUMNS, buy_csv_path)
    validate_required_columns(add_on_frame, KEY_COLUMNS, add_on_csv_path)

    prepared_buy = prepare_join_keys(buy_frame)
    prepared_add_on = prepare_join_keys(add_on_frame)
    validate_unique_keys(prepared_buy, buy_csv_path)
    validate_unique_keys(prepared_add_on, add_on_csv_path)

    merged_output = build_output(prepared_add_on, prepared_buy)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_output.to_csv(output_path, index=False)

    print(f"Base trade CSV : {buy_csv_path}")
    print(f"Add-on CSV     : {add_on_csv_path}")
    print(f"Output CSV     : {output_path}")
    print(f"Merged rows    : {len(merged_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())