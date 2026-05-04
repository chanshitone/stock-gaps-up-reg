from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_INDEX_COLUMN = "entry_shenzhen_index_pct_chg"
DEFAULT_SIZE_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run position-sizing regression on traded rows using index-change buckets and report total R, max drawdown, and score."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to a trades CSV")
    parser.add_argument(
        "--index-column",
        default=DEFAULT_INDEX_COLUMN,
        help=f"Index change column to use as index_change (default: {DEFAULT_INDEX_COLUMN})",
    )
    parser.add_argument(
        "--size-values",
        default=",".join(str(value) for value in DEFAULT_SIZE_VALUES),
        help="Comma-separated candidate position sizes used to build monotonic (weak, mid, strong) schemes.",
    )
    parser.add_argument(
        "--schemes",
        default=None,
        help="Optional semicolon-separated explicit schemes like '0.5,1.0,1.5;1.0,1.0,1.0'.",
    )
    parser.add_argument(
        "--schemes-file",
        type=Path,
        default=None,
        help="Optional YAML file containing explicit schemes under a top-level 'schemes' list.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top schemes to print (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: beside input CSV)",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path, index_column: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_{index_column}_sizing_regression.csv")


def parse_size_values(raw: str) -> list[float]:
    size_values: list[float] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        size_values.append(float(value))
    if not size_values:
        raise ValueError("No size values provided")
    return size_values


def parse_explicit_schemes(raw: str) -> list[tuple[float, float, float]]:
    schemes: list[tuple[float, float, float]] = []
    for scheme_chunk in raw.split(";"):
        chunk = scheme_chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Invalid scheme '{scheme_chunk}'. Expected three comma-separated values.")
        weak, mid, strong = (float(part) for part in parts)
        schemes.append((weak, mid, strong))
    if not schemes:
        raise ValueError("No schemes provided")
    return schemes


def parse_schemes_payload(payload: object, source: str) -> list[tuple[float, float, float]]:
    if isinstance(payload, dict):
        payload = payload.get("schemes")
    if not isinstance(payload, list):
        raise ValueError(f"{source} must contain a top-level 'schemes' list.")

    schemes: list[tuple[float, float, float]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Invalid scheme at position {index} in {source}. Expected three numeric values.")
        weak, mid, strong = (float(value) for value in item)
        schemes.append((weak, mid, strong))

    if not schemes:
        raise ValueError(f"No schemes provided in {source}")
    return schemes


def parse_schemes_file(path: Path) -> list[tuple[float, float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return parse_schemes_payload(payload, str(path))


def build_schemes(size_values: list[float]) -> list[tuple[float, float, float]]:
    ordered = sorted(set(size_values))
    return [scheme for scheme in itertools.product(ordered, repeat=3) if scheme[0] <= scheme[1] <= scheme[2]]


def load_trades(csv_path: Path, index_column: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if "status" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: status")
    if "pnl_r" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: pnl_r")
    if index_column not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: {index_column}")

    trades = frame.copy()
    trades["pnl_r"] = pd.to_numeric(trades["pnl_r"], errors="coerce")
    trades[index_column] = pd.to_numeric(trades[index_column], errors="coerce")
    if "buy_date" in trades.columns:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"], errors="coerce")

    traded = trades[
        trades["status"].eq("traded") & trades["pnl_r"].notna() & trades[index_column].notna()
    ].copy()
    if traded.empty:
        raise ValueError(f"No traded rows with both pnl_r and {index_column} in {csv_path}")
    if "buy_date" in traded.columns:
        traded = traded.sort_values(["buy_date", "ts_code"], kind="stable")
    traded = traded.rename(columns={index_column: "index_change"})
    return traded.reset_index(drop=True)


def simulate(df: pd.DataFrame, scheme: tuple[float, float, float]) -> dict[str, float]:
    weak, mid, strong = scheme

    def get_size(x: float) -> float:
        if x <= 0:
            return weak
        if x <= 0.2:
            return mid
        return strong

    sized = df.copy()
    sized["size"] = sized["index_change"].apply(get_size)
    sized["adj_pnl"] = sized["pnl_r"] * sized["size"]

    equity = sized["adj_pnl"].cumsum()
    max_dd = float((equity.cummax() - equity).max()) if not equity.empty else 0.0
    total_r = float(sized["adj_pnl"].sum())

    return {
        "total_r": total_r,
        "max_dd": max_dd,
        "score": total_r / (max_dd + 1e-6),
        "avg_size": float(sized["size"].mean()),
    }


def run_regression(trades: pd.DataFrame, schemes: list[tuple[float, float, float]]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for weak, mid, strong in schemes:
        metrics = simulate(trades, (weak, mid, strong))
        rows.append(
            {
                "weak": weak,
                "mid": mid,
                "strong": strong,
                "total_r": metrics["total_r"],
                "max_dd": metrics["max_dd"],
                "score": metrics["score"],
                "avg_size": metrics["avg_size"],
                "trades_n": int(len(trades)),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise ValueError("No regression results generated")
    return results.sort_values(["score", "total_r", "max_dd"], ascending=[False, False, True], kind="stable")


def print_summary(
    results: pd.DataFrame,
    csv_path: Path,
    output_path: Path,
    index_column: str,
    trades_n: int,
    top_n: int,
) -> None:
    print(f"Input CSV    : {csv_path}")
    print(f"Index column : {index_column}")
    print(f"Traded rows  : {trades_n}")
    print(f"Output CSV   : {output_path}")
    print()
    print("Top schemes:")
    print(results.head(top_n).round(4).to_string(index=False))
    print()
    baseline = results[
        results["weak"].apply(lambda value: math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9))
        & results["mid"].apply(lambda value: math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9))
        & results["strong"].apply(lambda value: math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9))
    ]
    if not baseline.empty:
        print("Baseline 1,1,1:")
        print(baseline.round(4).to_string(index=False))


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(csv_path, args.index_column)
    trades = load_trades(csv_path, args.index_column)

    if args.schemes_file:
        schemes = parse_schemes_file(args.schemes_file.resolve())
    elif args.schemes:
        schemes = parse_explicit_schemes(args.schemes)
    else:
        schemes = build_schemes(parse_size_values(args.size_values))

    results = run_regression(trades, schemes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print_summary(results, csv_path, output_path, args.index_column, len(trades), args.top)


if __name__ == "__main__":
    main()