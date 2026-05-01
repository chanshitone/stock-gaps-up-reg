from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.stock_gaps_reg.daily_vs_indices_common import INDEX_SPECS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether final pnl_r is related to same-day Shanghai, Shenzhen, and ChiNext index changes on buy_date."
        )
    )
    parser.add_argument("--csv", type=Path, required=True, help="Path to an enriched trade CSV with buy-date index columns")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output CSV path for the stats table (default: beside input CSV)",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=5000,
        help="Permutation iterations for two-sided p-values (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for permutation tests (default: 42)",
    )
    return parser.parse_args()


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_pnl_r_index_stats.csv")


def load_trades(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"No rows found in {csv_path}")
    if "pnl_r" not in frame.columns:
        raise ValueError(f"{csv_path} is missing required column: pnl_r")

    required_index_columns = {f"{label}_pct_chg" for _slug, _ts_code, label in INDEX_SPECS}
    missing = sorted(required_index_columns - set(frame.columns))
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{csv_path} is missing required index columns: {missing_text}")

    trades = frame.copy()
    trades["pnl_r"] = pd.to_numeric(trades["pnl_r"], errors="coerce")
    if trades["pnl_r"].isna().all():
        raise ValueError(f"Could not parse any pnl_r values from {csv_path}")
    return trades


def _two_sided_permutation_pvalue(observed: float, simulated: np.ndarray) -> float:
    if np.isnan(observed):
        return float("nan")
    return float((np.abs(simulated) >= abs(observed)).mean())


def analyze_index(trades: pd.DataFrame, slug: str, label: str, permutations: int, rng: np.random.Generator) -> dict[str, float | int | str]:
    index_column = f"{label}_pct_chg"
    index_returns = pd.to_numeric(trades[index_column], errors="coerce")
    mask = trades["pnl_r"].notna() & index_returns.notna()
    pnl_r = trades.loc[mask, "pnl_r"].reset_index(drop=True)
    index_returns = index_returns.loc[mask].reset_index(drop=True)

    up_mask = index_returns > 0
    down_mask = index_returns < 0

    pearson_corr = float(pnl_r.corr(index_returns))
    spearman_corr = float(pnl_r.rank(method="average").corr(index_returns.rank(method="average")))
    same_direction_rate_pct = float((np.sign(index_returns) == np.sign(pnl_r)).mean() * 100.0)

    avg_pnl_r_when_up = float(pnl_r[up_mask].mean()) if bool(up_mask.any()) else float("nan")
    avg_pnl_r_when_down = float(pnl_r[down_mask].mean()) if bool(down_mask.any()) else float("nan")
    median_pnl_r_when_up = float(pnl_r[up_mask].median()) if bool(up_mask.any()) else float("nan")
    median_pnl_r_when_down = float(pnl_r[down_mask].median()) if bool(down_mask.any()) else float("nan")
    win_rate_when_up_pct = float((pnl_r[up_mask] > 0).mean() * 100.0) if bool(up_mask.any()) else float("nan")
    win_rate_when_down_pct = float((pnl_r[down_mask] > 0).mean() * 100.0) if bool(down_mask.any()) else float("nan")
    avg_pnl_r_up_minus_down = float(avg_pnl_r_when_up - avg_pnl_r_when_down)

    q25 = float(index_returns.quantile(0.25))
    q75 = float(index_returns.quantile(0.75))
    low_quartile = pnl_r[index_returns <= q25]
    high_quartile = pnl_r[index_returns >= q75]

    permuted_corrs = np.empty(permutations, dtype=float)
    permuted_diffs = np.empty(permutations, dtype=float)
    pnl_values = pnl_r.to_numpy()
    index_values = index_returns.to_numpy()
    up_values = up_mask.to_numpy()
    down_values = down_mask.to_numpy()
    for idx in range(permutations):
        shuffled = rng.permutation(pnl_values)
        permuted_corrs[idx] = pd.Series(shuffled).corr(pd.Series(index_values))
        permuted_diffs[idx] = shuffled[up_values].mean() - shuffled[down_values].mean()

    return {
        "index_slug": slug,
        "index_label": label,
        "n": int(mask.sum()),
        "up_days": int(up_mask.sum()),
        "down_days": int(down_mask.sum()),
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "corr_perm_p": _two_sided_permutation_pvalue(pearson_corr, permuted_corrs),
        "same_direction_rate_pct": same_direction_rate_pct,
        "avg_pnl_r_when_up": avg_pnl_r_when_up,
        "avg_pnl_r_when_down": avg_pnl_r_when_down,
        "avg_pnl_r_up_minus_down": avg_pnl_r_up_minus_down,
        "diff_perm_p": _two_sided_permutation_pvalue(avg_pnl_r_up_minus_down, permuted_diffs),
        "median_pnl_r_when_up": median_pnl_r_when_up,
        "median_pnl_r_when_down": median_pnl_r_when_down,
        "win_rate_when_up_pct": win_rate_when_up_pct,
        "win_rate_when_down_pct": win_rate_when_down_pct,
        "q25_index_pct_chg": q25,
        "q75_index_pct_chg": q75,
        "avg_pnl_r_low_quartile": float(low_quartile.mean()),
        "avg_pnl_r_high_quartile": float(high_quartile.mean()),
        "win_rate_low_quartile_pct": float((low_quartile > 0).mean() * 100.0),
        "win_rate_high_quartile_pct": float((high_quartile > 0).mean() * 100.0),
        "n_low_quartile": int(low_quartile.shape[0]),
        "n_high_quartile": int(high_quartile.shape[0]),
    }


def build_stats_table(trades: pd.DataFrame, permutations: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [analyze_index(trades, slug, label, permutations, rng) for slug, _ts_code, label in INDEX_SPECS]
    return pd.DataFrame(rows)


def print_summary_table(stats: pd.DataFrame, csv_path: Path, output_path: Path) -> None:
    printable_columns = [
        "index_slug",
        "n",
        "pearson_corr",
        "spearman_corr",
        "corr_perm_p",
        "avg_pnl_r_when_up",
        "avg_pnl_r_when_down",
        "avg_pnl_r_up_minus_down",
        "diff_perm_p",
        "win_rate_when_up_pct",
        "win_rate_when_down_pct",
    ]
    print(f"Input CSV : {csv_path}")
    print(f"Stats CSV : {output_path}")
    print()
    print(stats[printable_columns].round(4).to_string(index=False))


def main() -> None:
    args = parse_args()

    if args.permutations <= 0:
        raise ValueError("--permutations must be greater than 0")

    csv_path = args.csv.resolve()
    output_path = args.output.resolve() if args.output else default_output_path(csv_path)

    trades = load_trades(csv_path)
    stats = build_stats_table(trades, permutations=args.permutations, seed=args.seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_path, index=False, encoding="utf-8-sig")
    print_summary_table(stats, csv_path, output_path)


if __name__ == "__main__":
    main()