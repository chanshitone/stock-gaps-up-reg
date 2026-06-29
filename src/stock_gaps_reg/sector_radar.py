from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import re
from typing import Protocol

import numpy as np
import pandas as pd
import yaml

from .io_utils import make_run_dir, normalize_ts_code
from .tushare_client import TushareClient


PHASE_LABELS = {
    "A": "强趋势赛道",
    "B": "分化赛道",
    "C": "失败赛道",
    "D": "无结构赛道",
}


@dataclass(frozen=True)
class RadarConfig:
    history_calendar_days: int = 180
    short_return_days: int = 5
    medium_return_days: int = 20
    fast_ma_days: int = 20
    slow_ma_days: int = 60
    volatility_days: int = 20
    phase_a_min_short_return_pct: float = 0.0
    phase_a_min_medium_return_pct: float = 0.0
    phase_c_max_medium_return_pct: float = 0.0


class SectorDataClient(Protocol):
    def get_sw_index_classify(self, level: str = "L1", src: str = "SW2021") -> pd.DataFrame: ...
    def get_sw_memberships(self, ts_code: str) -> pd.DataFrame: ...
    def get_sw_daily(self, index_code: str, start_date: date, end_date: date) -> pd.DataFrame: ...


def load_radar_config(path: Path) -> RadarConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a YAML object.")
    unknown = set(payload) - set(RadarConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown sector radar config keys: {', '.join(sorted(unknown))}")
    config = RadarConfig(**payload)
    required_history = max(
        config.short_return_days,
        config.medium_return_days,
        config.fast_ma_days,
        config.slow_ma_days,
        config.volatility_days,
    )
    windows = [
        config.history_calendar_days,
        config.short_return_days,
        config.medium_return_days,
        config.fast_ma_days,
        config.slow_ma_days,
        config.volatility_days,
    ]
    if min(windows) <= 0 or config.history_calendar_days < required_history:
        raise ValueError("Radar windows must be positive and history_calendar_days must cover every window.")
    return config


def load_stock_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "ts_code" not in frame.columns:
            raise ValueError(f"Stock CSV {path} must contain a ts_code column.")
        values = frame["ts_code"].tolist()
    else:
        values = re.split(r"[\s,;]+", path.read_text(encoding="utf-8-sig"))

    codes: list[str] = []
    seen: set[str] = set()
    for value in values:
        raw = str(value).strip()
        if not raw:
            continue
        code = normalize_ts_code(raw)
        if code not in seen:
            codes.append(code)
            seen.add(code)
    if not codes:
        raise ValueError(f"No stock codes found in {path}.")
    return codes


def select_membership(memberships: pd.DataFrame, as_of: date) -> pd.Series | None:
    if memberships.empty:
        return None
    frame = memberships.copy().fillna("")
    in_dates = pd.to_datetime(frame.get("in_date", ""), format="%Y%m%d", errors="coerce")
    out_dates = pd.to_datetime(frame.get("out_date", ""), format="%Y%m%d", errors="coerce")
    stamp = pd.Timestamp(as_of)
    active = frame.loc[(in_dates.isna() | (in_dates <= stamp)) & (out_dates.isna() | (out_dates >= stamp))].copy()
    if active.empty:
        return None
    active["_in_date"] = pd.to_datetime(active.get("in_date", ""), format="%Y%m%d", errors="coerce")
    return active.sort_values(["_in_date", "l1_code"], na_position="first").iloc[-1]


def calculate_sector_metrics(history: pd.DataFrame, config: RadarConfig) -> dict[str, object]:
    frame = history.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)
    minimum_rows = max(
        config.short_return_days + 1,
        config.medium_return_days + 1,
        config.slow_ma_days,
        config.volatility_days + 1,
    )
    if len(frame) < minimum_rows:
        raise ValueError(f"Need at least {minimum_rows} daily rows, received {len(frame)}.")

    close = pd.to_numeric(frame["close"], errors="coerce")
    latest = float(close.iloc[-1])
    fast_ma = float(close.tail(config.fast_ma_days).mean())
    slow_ma = float(close.tail(config.slow_ma_days).mean())
    short_return = (latest / float(close.iloc[-config.short_return_days - 1]) - 1.0) * 100.0
    medium_return = (latest / float(close.iloc[-config.medium_return_days - 1]) - 1.0) * 100.0
    daily_returns = close.pct_change().tail(config.volatility_days)
    volatility = float(daily_returns.std(ddof=1) * np.sqrt(252) * 100.0)
    above_fast = (latest / fast_ma - 1.0) * 100.0
    fast_vs_slow = (fast_ma / slow_ma - 1.0) * 100.0

    if (
        latest > fast_ma > slow_ma
        and short_return > config.phase_a_min_short_return_pct
        and medium_return > config.phase_a_min_medium_return_pct
    ):
        phase = "A"
    elif latest > slow_ma and (latest <= fast_ma or short_return <= 0.0):
        phase = "B"
    elif fast_ma < slow_ma and medium_return < config.phase_c_max_medium_return_pct:
        phase = "C"
    else:
        phase = "D"

    score = (
        0.40 * medium_return
        + 0.25 * short_return
        + 0.20 * above_fast
        + 0.15 * fast_vs_slow
        - 0.05 * volatility
    )
    return {
        "latest_trade_date": frame.iloc[-1]["trade_date"].date(),
        "latest_close": latest,
        "return_5d_pct": short_return,
        "return_20d_pct": medium_return,
        "ma20": fast_ma,
        "ma60": slow_ma,
        "close_vs_ma20_pct": above_fast,
        "ma20_vs_ma60_pct": fast_vs_slow,
        "volatility_20d_ann_pct": volatility,
        "phase": phase,
        "phase_name": PHASE_LABELS[phase],
        "score": score,
    }


def map_stocks_to_sectors(
    stock_codes: list[str], client: SectorDataClient, catalogue: pd.DataFrame, as_of: date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalogue_names = (
        catalogue.drop_duplicates("index_code").set_index("index_code")["industry_name"].astype(str).to_dict()
    )
    mapped: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    for code in stock_codes:
        membership = select_membership(client.get_sw_memberships(code), as_of)
        if membership is None:
            unmatched.append({"ts_code": code, "reason": f"No active SW2021 membership on {as_of}"})
            continue
        sector_code = str(membership.get("l1_code", "")).strip()
        if not sector_code:
            unmatched.append({"ts_code": code, "reason": "Membership has no level-1 sector code"})
            continue
        mapped.append(
            {
                "ts_code": code,
                "stock_name": str(membership.get("name", "")).strip(),
                "sector_code": sector_code,
                "sector_name": catalogue_names.get(sector_code, str(membership.get("l1_name", "")).strip()),
            }
        )
    return pd.DataFrame(mapped), pd.DataFrame(unmatched, columns=["ts_code", "reason"])


def build_sector_radar(
    stock_codes: list[str], client: SectorDataClient, as_of: date, config: RadarConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    catalogue = client.get_sw_index_classify(level="L1", src="SW2021")
    stocks, unmatched = map_stocks_to_sectors(stock_codes, client, catalogue, as_of)
    if stocks.empty:
        raise ValueError("None of the supplied stocks had an active SW2021 level-1 membership.")

    sector_rows: list[dict[str, object]] = []
    start_date = as_of - timedelta(days=config.history_calendar_days)
    for sector_code, group in stocks.groupby("sector_code", sort=True):
        history = client.get_sw_daily(sector_code, start_date, as_of)
        metrics = calculate_sector_metrics(history, config)
        stock_labels = [
            f"{row.ts_code} {row.stock_name}".strip()
            for row in group.sort_values("ts_code").itertuples(index=False)
        ]
        sector_rows.append(
            {
                "sector_code": sector_code,
                "sector_name": group.iloc[0]["sector_name"],
                "stock_count": len(group),
                "stocks": "; ".join(stock_labels),
                **metrics,
            }
        )

    sectors = pd.DataFrame(sector_rows)
    phase_order = {"A": 0, "B": 1, "C": 2, "D": 3}
    sectors["_phase_order"] = sectors["phase"].map(phase_order)
    sectors = sectors.sort_values(["_phase_order", "score", "sector_code"], ascending=[True, False, True])
    sectors = sectors.drop(columns="_phase_order").reset_index(drop=True)
    sectors.insert(0, "rank", range(1, len(sectors) + 1))
    ranks = sectors.set_index("sector_code")["rank"].to_dict()
    phases = sectors.set_index("sector_code")["phase"].to_dict()
    stocks["sector_rank"] = stocks["sector_code"].map(ranks)
    stocks["phase"] = stocks["sector_code"].map(phases)
    stocks = stocks.sort_values(["sector_rank", "ts_code"]).reset_index(drop=True)
    return sectors, stocks, unmatched


def _format_markdown(sectors: pd.DataFrame, unmatched: pd.DataFrame, as_of: date) -> str:
    lines = [f"# Sector Radar — {as_of}", "", "| Rank | Phase | Sector | Score | 5D | 20D | Stocks |", "|---:|:---:|---|---:|---:|---:|---|"]
    for row in sectors.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.phase} {row.phase_name} | {row.sector_name} ({row.sector_code}) "
            f"| {row.score:.2f} | {row.return_5d_pct:.2f}% | {row.return_20d_pct:.2f}% | {row.stocks} |"
        )
    if not unmatched.empty:
        lines.extend(["", "## Unmatched stocks", ""])
        lines.extend(f"- {row.ts_code}: {row.reason}" for row in unmatched.itertuples(index=False))
    lines.extend(
        [
            "",
            "## Phase rules",
            "",
            "- A: close > MA20 > MA60 and both 5-day and 20-day returns are positive.",
            "- B: close remains above MA60, but price is at/below MA20 or the 5-day return is non-positive.",
            "- C: MA20 < MA60 and the 20-day return is negative.",
            "- D: none of the above; no clear structure.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    run_dir: Path, sectors: pd.DataFrame, stocks: pd.DataFrame, unmatched: pd.DataFrame, as_of: date
) -> tuple[Path, Path, Path, Path]:
    sectors_path = run_dir / "sectors.csv"
    stocks_path = run_dir / "stocks.csv"
    unmatched_path = run_dir / "unmatched_stocks.csv"
    markdown_path = run_dir / "sector_radar.md"
    sectors.to_csv(sectors_path, index=False, encoding="utf-8-sig")
    stocks.to_csv(stocks_path, index=False, encoding="utf-8-sig")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(_format_markdown(sectors, unmatched, as_of), encoding="utf-8")
    return sectors_path, stocks_path, unmatched_path, markdown_path


def _parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD or YYYYMMDD.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank supplied stocks by their Shenwan level-1 sector phase.")
    parser.add_argument("--stocks", type=Path, required=True, help="CSV with ts_code, or a text file of stock codes.")
    parser.add_argument("--as-of", type=_parse_date, default=date.today(), help="Radar date; defaults to today.")
    parser.add_argument("--config", type=Path, default=Path("config/sector_radar.yaml"))
    parser.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sector_radar"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    radar_config = load_radar_config(args.config.resolve())
    with args.strategy_config.resolve().open("r", encoding="utf-8") as handle:
        strategy_payload = yaml.safe_load(handle) or {}
    project_root = args.strategy_config.resolve().parent.parent
    cache_dir = (project_root / strategy_payload.get("data", {}).get("cache_dir", "data/cache")).resolve()
    client = TushareClient(cache_dir=cache_dir)
    stock_codes = load_stock_list(args.stocks.resolve())
    try:
        sectors, stocks, unmatched = build_sector_radar(stock_codes, client, args.as_of, radar_config)
    except PermissionError as exc:
        parser.exit(
            2,
            "Sector radar could not retrieve required Tushare data. "
            f"Grant this account access to index_classify, index_member_all, and sw_daily.\nDetails: {exc}\n",
        )
    run_dir = make_run_dir(args.output_dir.resolve())
    paths = write_reports(run_dir, sectors, stocks, unmatched, args.as_of)
    print(f"Sector report: {paths[0]}")
    print(f"Stock mapping: {paths[1]}")
    print(f"Unmatched stocks: {paths[2]}")
    print(f"Markdown report: {paths[3]}")
    print(f"Sectors ranked: {len(sectors)}; stocks mapped: {len(stocks)}; unmatched: {len(unmatched)}")


if __name__ == "__main__":
    main()
