from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = ["ts_code", "detect_date"]
COMPARE_COLUMNS = [
    "exit_date",
    "exit_time",
    "exit_price",
    "exit_reason",
    "hold_days",
    "pnl_r",
    "max_favorable_excursion_r",
]
NUMERIC_COLUMNS = {"exit_price", "hold_days", "pnl_r", "max_favorable_excursion_r"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two trades.csv files by ts_code + detect_date and report field-level differences."
        )
    )
    parser.add_argument("--left", required=True, help="Path to the first trades.csv file")
    parser.add_argument("--right", required=True, help="Path to the second trades.csv file")
    parser.add_argument(
        "--output",
        help="Optional path to write a CSV of mismatched rows and keys missing from either side",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-9,
        help="Absolute tolerance for numeric comparisons (default: 1e-9)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of differing or missing keys to print per section (default: 20)",
    )
    parser.add_argument(
        "--fail-on-diff",
        action="store_true",
        help="Exit with code 1 when any mismatch or missing key is found",
    )
    return parser.parse_args()


def normalize_detect_date(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str).str.strip(), format="mixed", errors="coerce")
    normalized = parsed.dt.strftime("%Y-%m-%d")
    fallback = series.astype(str).str.strip()
    return normalized.fillna(fallback)


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


def load_trade_view(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    frame.columns = [column.strip() for column in frame.columns]

    required = set(KEY_COLUMNS + COMPARE_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{path} is missing required columns: {missing_text}")

    subset = frame[KEY_COLUMNS + COMPARE_COLUMNS].copy()
    subset["ts_code"] = subset["ts_code"].astype(str).str.strip().str.upper()
    subset["detect_date"] = normalize_detect_date(subset["detect_date"])
    subset["exit_date"] = normalize_exit_date(subset["exit_date"])
    subset["exit_time"] = normalize_exit_time(subset["exit_time"])

    duplicate_mask = subset.duplicated(subset=KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicate_rows = subset.loc[duplicate_mask, KEY_COLUMNS].sort_values(KEY_COLUMNS)
        sample = duplicate_rows.head(10).to_dict(orient="records")
        raise ValueError(f"{path} has duplicate keys. Sample: {sample}")

    return subset.set_index(KEY_COLUMNS).sort_index()


def compare_string_series(left: pd.Series, right: pd.Series) -> pd.Series:
    left_values = left.fillna("").astype(str).str.strip()
    right_values = right.fillna("").astype(str).str.strip()
    return left_values != right_values


def compare_numeric_series(left: pd.Series, right: pd.Series, tolerance: float) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce")
    right_values = pd.to_numeric(right, errors="coerce")
    both_missing = left_values.isna() & right_values.isna()
    both_present_equal = (~left_values.isna()) & (~right_values.isna()) & (
        (left_values - right_values).abs() <= tolerance
    )
    return ~(both_missing | both_present_equal)


def build_mismatch_report(left: pd.DataFrame, right: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    common_index = left.index.intersection(right.index)
    if common_index.empty:
        return pd.DataFrame(columns=KEY_COLUMNS + ["diff_fields"])

    left_common = left.loc[common_index, COMPARE_COLUMNS]
    right_common = right.loc[common_index, COMPARE_COLUMNS]

    mismatch_flags: dict[str, pd.Series] = {}
    overall_mask = pd.Series(False, index=common_index)
    for column in COMPARE_COLUMNS:
        if column in NUMERIC_COLUMNS:
            column_mask = compare_numeric_series(left_common[column], right_common[column], tolerance)
        else:
            column_mask = compare_string_series(left_common[column], right_common[column])
        mismatch_flags[column] = column_mask
        overall_mask = overall_mask | column_mask

    mismatch_index = common_index[overall_mask]
    if len(mismatch_index) == 0:
        return pd.DataFrame(columns=KEY_COLUMNS + ["diff_fields"])

    rows: list[dict[str, object]] = []
    for key in mismatch_index:
        row: dict[str, object] = {
            "ts_code": key[0],
            "detect_date": key[1],
        }
        diff_fields: list[str] = []
        for column in COMPARE_COLUMNS:
            left_value = left_common.at[key, column]
            right_value = right_common.at[key, column]
            row[f"left_{column}"] = left_value
            row[f"right_{column}"] = right_value
            if bool(mismatch_flags[column].at[key]):
                diff_fields.append(column)
        row["diff_fields"] = ",".join(diff_fields)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(KEY_COLUMNS).reset_index(drop=True)


def build_missing_report(index: pd.Index, side: str) -> pd.DataFrame:
    if len(index) == 0:
        return pd.DataFrame(columns=KEY_COLUMNS + ["missing_from"])
    rows = [{"ts_code": key[0], "detect_date": key[1], "missing_from": side} for key in index]
    return pd.DataFrame(rows).sort_values(KEY_COLUMNS).reset_index(drop=True)


def print_key_section(title: str, frame: pd.DataFrame, limit: int) -> None:
    print(title)
    if frame.empty:
        print("  none")
        return

    shown = frame.head(limit)
    for row in shown.to_dict(orient="records"):
        print(f"  {row['ts_code']} {row['detect_date']}")
    if len(frame) > limit:
        print(f"  ... {len(frame) - limit} more")


def print_mismatch_section(frame: pd.DataFrame, limit: int) -> None:
    print("Mismatched keys:")
    if frame.empty:
        print("  none")
        return

    shown = frame.head(limit)
    for row in shown.to_dict(orient="records"):
        print(f"  {row['ts_code']} {row['detect_date']} -> {row['diff_fields']}")
    if len(frame) > limit:
        print(f"  ... {len(frame) - limit} more")


def main() -> int:
    args = parse_args()

    left_path = Path(args.left)
    right_path = Path(args.right)
    left = load_trade_view(left_path)
    right = load_trade_view(right_path)

    left_only = left.index.difference(right.index)
    right_only = right.index.difference(left.index)
    mismatches = build_mismatch_report(left, right, args.tolerance)

    print(f"Left file : {left_path}")
    print(f"Right file: {right_path}")
    print(f"Common keys: {len(left.index.intersection(right.index))}")
    print(f"Left-only keys: {len(left_only)}")
    print(f"Right-only keys: {len(right_only)}")
    print(f"Mismatched keys: {len(mismatches)}")
    print()

    print_key_section("Keys only in left:", build_missing_report(left_only, "right"), args.limit)
    print()
    print_key_section("Keys only in right:", build_missing_report(right_only, "left"), args.limit)
    print()
    print_mismatch_section(mismatches, args.limit)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report = pd.concat(
            [
                build_missing_report(left_only, "right"),
                build_missing_report(right_only, "left"),
                mismatches,
            ],
            ignore_index=True,
            sort=False,
        )
        try:
            report.to_csv(output_path, index=False)
        except PermissionError as exc:
            raise SystemExit(
                f"Cannot write report to {output_path}. Close the file if it is open, or use a different --output path."
            ) from exc
        print()
        print(f"Detailed report written to {output_path}")

    has_diff = bool(len(left_only) or len(right_only) or len(mismatches))
    return 1 if args.fail_on_diff and has_diff else 0


if __name__ == "__main__":
    raise SystemExit(main())