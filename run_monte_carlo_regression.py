from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LEVERAGES = [1.0, 1.5, 2.0]
DEFAULT_QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
DEFAULT_N_SIMS = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Monte Carlo regression on historical traded pnl_r by resampling trade outcomes and reporting drawdown and total-R risk."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to a trades CSV")
    parser.add_argument(
        "--leverages",
        default=",".join(str(value) for value in DEFAULT_LEVERAGES),
        help="Comma-separated leverage multipliers, e.g. 1.0,1.5,2.0",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=DEFAULT_N_SIMS,
        help=f"Number of Monte Carlo simulations per leverage (default: {DEFAULT_N_SIMS})",
    )
    parser.add_argument(
        "--n-trades",
        type=int,
        default=None,
        help="Number of trades to sample in each simulation (default: number of traded rows)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible simulations (default: 42)",
    )
    parser.add_argument(
        "--risk-amount",
        type=float,
        default=None,
        help="Optional cash amount represented by 1R. If omitted, cash metrics are skipped unless --capital and --risk-per-trade-pct are provided.",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Optional account capital used to derive 1R cash when combined with --risk-per-trade-pct.",
    )
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        default=None,
        help="Optional per-trade risk percentage expressed as a decimal, e.g. 0.05 for 5%%.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path for all simulation rows (default: beside input CSV)",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional output CSV path for leverage-level summary statistics (default: beside input CSV)",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_monte_carlo.csv")


def default_summary_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_monte_carlo_summary.csv")


def parse_float_list(raw: str, field_name: str) -> list[float]:
    values: list[float] = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        values.append(float(value))
    if not values:
        raise ValueError(f"No values provided for {field_name}")
    return values


def load_trades(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if "pnl_r" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: pnl_r")

    trades = frame.copy()
    trades["pnl_r"] = pd.to_numeric(trades["pnl_r"], errors="coerce")
    if "status" in trades.columns:
        trades = trades[trades["status"].eq("traded")].copy()
    if "buy_date" in trades.columns:
        trades["buy_date"] = pd.to_datetime(trades["buy_date"], errors="coerce")

    traded = trades[trades["pnl_r"].notna()].copy()
    if traded.empty:
        raise ValueError(f"No traded rows with pnl_r in {csv_path}")
    if "buy_date" in traded.columns and "ts_code" in traded.columns:
        traded = traded.sort_values(["buy_date", "ts_code"], kind="stable")
    return traded.reset_index(drop=True)


def resolve_risk_amount(
    risk_amount: float | None,
    capital: float | None,
    risk_per_trade_pct: float | None,
) -> float | None:
    if risk_amount is not None:
        return float(risk_amount)
    if capital is None or risk_per_trade_pct is None:
        return None
    return float(capital) * float(risk_per_trade_pct)


def compute_max_drawdown(equity: np.ndarray) -> np.ndarray:
    peaks = np.maximum.accumulate(equity, axis=1)
    drawdowns = peaks - equity
    return drawdowns.max(axis=1)


def run_monte_carlo(
    pnl_r_values: np.ndarray,
    leverage: float,
    n_sims: int,
    n_trades: int,
    rng: np.random.Generator,
    risk_amount: float | None,
) -> pd.DataFrame:
    sample = rng.choice(pnl_r_values, size=(n_sims, n_trades), replace=True) * leverage
    equity = np.cumsum(sample, axis=1)

    results = pd.DataFrame(
        {
            "leverage": leverage,
            "simulation_id": np.arange(1, n_sims + 1, dtype=int),
            "total_r": equity[:, -1],
            "max_dd_r": compute_max_drawdown(equity),
            "min_equity_r": equity.min(axis=1),
            "win_rate": (sample > 0).mean(axis=1),
        }
    )
    if risk_amount is not None:
        results["risk_amount"] = risk_amount
        results["total_cash"] = results["total_r"] * risk_amount
        results["max_dd_cash"] = results["max_dd_r"] * risk_amount
        results["min_equity_cash"] = results["min_equity_r"] * risk_amount
    return results


def _metric_quantiles(series: pd.Series, prefix: str, quantiles: list[float]) -> dict[str, float]:
    values = series.quantile(quantiles)
    return {
        f"{prefix}_q{int(round(quantile * 100)):02d}": float(values.loc[quantile])
        for quantile in quantiles
    }


def _drawdown_assessment(max_dd_r_q95: float) -> str:
    if max_dd_r_q95 <= 6.0:
        return "acceptable"
    if max_dd_r_q95 <= 8.0:
        return "elevated"
    return "danger"


def _total_r_assessment(total_r_q05: float) -> str:
    return "acceptable" if total_r_q05 > 0 else "danger"


def _loss_probability_assessment(loss_probability: float) -> str:
    if loss_probability < 0.05:
        return "acceptable"
    if loss_probability <= 0.10:
        return "elevated"
    return "danger"


def summarize_results(results: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for leverage, group in results.groupby("leverage", sort=True):
        row: dict[str, float | int | str] = {
            "leverage": float(leverage),
            "n_sims": int(len(group)),
            "n_trades": int(group.attrs.get("n_trades", 0)),
            "loss_probability": float((group["total_r"] < 0).mean()),
            "avg_win_rate": float(group["win_rate"].mean()),
        }
        row.update(_metric_quantiles(group["total_r"], "total_r", quantiles))
        row.update(_metric_quantiles(group["max_dd_r"], "max_dd_r", quantiles))
        row.update(_metric_quantiles(group["min_equity_r"], "min_equity_r", quantiles))

        if "total_cash" in group.columns:
            row.update(_metric_quantiles(group["total_cash"], "total_cash", quantiles))
            row.update(_metric_quantiles(group["max_dd_cash"], "max_dd_cash", quantiles))
            row.update(_metric_quantiles(group["min_equity_cash"], "min_equity_cash", quantiles))
            row["risk_amount"] = float(group["risk_amount"].iloc[0])

        row["max_dd_assessment"] = _drawdown_assessment(float(row["max_dd_r_q95"]))
        row["total_r_assessment"] = _total_r_assessment(float(row["total_r_q05"]))
        row["loss_probability_assessment"] = _loss_probability_assessment(float(row["loss_probability"]))
        rows.append(row)

    return pd.DataFrame(rows).sort_values("leverage", kind="stable").reset_index(drop=True)


def print_summary(
    summary: pd.DataFrame,
    csv_path: Path,
    output_path: Path,
    summary_output_path: Path,
    n_trades: int,
    seed: int,
    risk_amount: float | None,
) -> None:
    print(f"Input CSV      : {csv_path}")
    print(f"Simulation CSV : {output_path}")
    print(f"Summary CSV    : {summary_output_path}")
    print(f"Sample trades  : {n_trades}")
    print(f"Random seed    : {seed}")
    if risk_amount is not None:
        print(f"Risk amount    : {risk_amount:.4f}")
    print()
    print(summary.round(4).to_string(index=False))


def main() -> None:
    args = parse_args()
    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(csv_path)
    summary_output_path = args.summary_output.resolve() if args.summary_output else default_summary_output_path(csv_path)
    leverages = parse_float_list(args.leverages, "leverages")
    quantiles = list(DEFAULT_QUANTILES)

    traded = load_trades(csv_path)
    pnl_r_values = traded["pnl_r"].to_numpy(dtype=float)
    n_trades = len(pnl_r_values) if args.n_trades is None else int(args.n_trades)
    if n_trades <= 0:
        raise ValueError("n_trades must be positive")

    risk_amount = resolve_risk_amount(args.risk_amount, args.capital, args.risk_per_trade_pct)
    rng = np.random.default_rng(args.seed)

    simulations: list[pd.DataFrame] = []
    for leverage in leverages:
        simulation = run_monte_carlo(pnl_r_values, leverage, args.n_sims, n_trades, rng, risk_amount)
        simulation.attrs["n_trades"] = n_trades
        simulations.append(simulation)

    results = pd.concat(simulations, ignore_index=True)
    results.attrs["n_trades"] = n_trades
    summary = summarize_results(results, quantiles)
    summary["n_trades"] = n_trades

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_output_path, index=False, encoding="utf-8-sig")
    print_summary(summary, csv_path, output_path, summary_output_path, n_trades, args.seed, risk_amount)


if __name__ == "__main__":
    main()