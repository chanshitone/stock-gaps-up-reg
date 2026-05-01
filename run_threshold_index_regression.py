from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_THRESHOLDS = [-0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
DEFAULT_INDEX_COLUMN = "entry_shenzhen_index_pct_chg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep index-change thresholds against traded pnl_r results and report total R, max drawdown, trade count, and score."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to a trades CSV")
    parser.add_argument(
        "--index-column",
        default=DEFAULT_INDEX_COLUMN,
        help=f"Index change column to threshold (default: {DEFAULT_INDEX_COLUMN})",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in DEFAULT_THRESHOLDS),
        help="Comma-separated thresholds in percent units, e.g. -0.5,-0.2,0,0.2,0.5,1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path (default: beside input CSV)",
    )
    parser.add_argument(
        "--chart-output",
        type=Path,
        default=None,
        help="Optional HTML chart path (default: beside input CSV)",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path, index_column: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_{index_column}_threshold_regression.csv")


def default_chart_output_path(csv_path: Path, index_column: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_{index_column}_threshold_regression.html")


def parse_thresholds(raw: str) -> list[float]:
    thresholds: list[float] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        thresholds.append(float(value))
    if not thresholds:
        raise ValueError("No thresholds provided")
    return thresholds


def load_trades(csv_path: Path, index_column: str) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if "pnl_r" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: pnl_r")
    if index_column not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: {index_column}")

    trades = frame.copy()
    trades["pnl_r"] = pd.to_numeric(trades["pnl_r"], errors="coerce")
    trades[index_column] = pd.to_numeric(trades[index_column], errors="coerce")
    if "buy_date" in trades.columns:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"], errors="coerce")

    traded = trades[trades["pnl_r"].notna() & trades[index_column].notna()].copy()
    if traded.empty:
        raise ValueError(f"No traded rows with both pnl_r and {index_column} in {csv_path}")
    if "buy_date" in traded.columns:
        traded = traded.sort_values(["buy_date", "ts_code"], kind="stable")
    return traded.reset_index(drop=True)


def compute_max_drawdown(pnl_r: pd.Series) -> float:
    if pnl_r.empty:
        return 0.0
    equity = pnl_r.cumsum()
    drawdown = equity.cummax() - equity
    return float(drawdown.max())


def run_threshold_sweep(trades: pd.DataFrame, index_column: str, thresholds: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        filtered = trades[trades[index_column] > threshold].copy()
        total_r = float(filtered["pnl_r"].sum()) if not filtered.empty else 0.0
        trades_n = int(len(filtered))
        max_dd = compute_max_drawdown(filtered["pnl_r"]) if trades_n else 0.0
        score = total_r / (max_dd + 1e-6)
        avg_r = float(filtered["pnl_r"].mean()) if trades_n else float("nan")
        win_rate_pct = float((filtered["pnl_r"] > 0).mean() * 100.0) if trades_n else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "total_r": total_r,
                "max_dd": max_dd,
                "trades_n": trades_n,
                "score": score,
                "avg_r": avg_r,
                "win_rate_pct": win_rate_pct,
            }
        )
    return pd.DataFrame(rows)


def build_chart(results: pd.DataFrame, index_column: str) -> go.Figure:
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Total R", "Max Drawdown", "Score"),
    )

    figure.add_trace(
        go.Scatter(x=results["threshold"], y=results["total_r"], mode="lines+markers", name="total_r"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=results["threshold"], y=results["max_dd"], mode="lines+markers", name="max_dd"),
        row=2,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=results["threshold"], y=results["score"], mode="lines+markers", name="score"),
        row=3,
        col=1,
    )
    figure.update_layout(
        title=f"Threshold Regression for {index_column}",
        template="plotly_white",
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figure.update_xaxes(title_text="threshold (%)", row=3, col=1)
    figure.update_yaxes(title_text="R", row=1, col=1)
    figure.update_yaxes(title_text="DD", row=2, col=1)
    figure.update_yaxes(title_text="score", row=3, col=1)
    return figure


def print_summary(results: pd.DataFrame, csv_path: Path, output_path: Path, chart_output_path: Path, index_column: str) -> None:
    print(f"Input CSV    : {csv_path}")
    print(f"Index column : {index_column}")
    print(f"Output CSV   : {output_path}")
    print(f"Chart HTML   : {chart_output_path}")
    print()
    print(results.round(4).to_string(index=False))


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    thresholds = parse_thresholds(args.thresholds)
    output_path = args.output.resolve() if args.output else default_output_path(csv_path, args.index_column)
    chart_output_path = (
        args.chart_output.resolve() if args.chart_output else default_chart_output_path(csv_path, args.index_column)
    )

    trades = load_trades(csv_path, args.index_column)
    results = run_threshold_sweep(trades, args.index_column, thresholds)
    chart = build_chart(results, args.index_column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    chart.write_html(chart_output_path, include_plotlyjs="cdn", full_html=True)
    print_summary(results, csv_path, output_path, chart_output_path, args.index_column)


if __name__ == "__main__":
    main()