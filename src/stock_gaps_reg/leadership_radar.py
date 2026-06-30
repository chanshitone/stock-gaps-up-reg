from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pandas as pd
import yaml

from .io_utils import make_run_dir
from .sector_radar import load_stock_list, select_membership
from .tushare_client import TushareClient


@dataclass(frozen=True)
class LeadershipConfig:
    history_calendar_days: int = 240
    trend_days: int = 60
    trend_classification_days: int = 10
    trend_score_threshold: float = 3.0
    trend_score_ma_short_days: int = 5
    trend_score_ma_long_days: int = 20
    trend_min_improving_days: int = 6
    benchmark_index_code: str = "000300.SH"
    rs_short_days: int = 20
    rs_long_days: int = 60
    new_high_days: int = 60
    above_ma_days: int = 20
    volume_ratio_days: int = 20
    leader_return_days: int = 60
    leader_near_high_days: int = 20
    leader_near_high_pct: float = 10.0
    leader_top_n: int = 3
    anchor_top_n: int = 3
    anchor_ma_short_days: int = 20
    anchor_ma_long_days: int = 50
    anchor_amount_short_days: int = 5
    anchor_amount_long_days: int = 20
    concentration_top_n: int = 3
    strong_leadership_score: float = 85.0
    leadership_score: float = 75.0
    candidate_score: float = 65.0
    theme_score: float = 55.0
    hot_score: float = 45.0


class LeadershipDataClient(Protocol):
    def get_sw_memberships(self, ts_code: str) -> pd.DataFrame: ...
    def get_sw_l3_members(self, l3_code: str) -> pd.DataFrame: ...
    def get_sw_daily(self, index_code: str, start_date: date, end_date: date) -> pd.DataFrame: ...
    def get_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame: ...
    def get_index_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame: ...


def load_leadership_config(path: Path) -> LeadershipConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a YAML object.")
    unknown = set(payload) - set(LeadershipConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown leadership radar config keys: {', '.join(sorted(unknown))}")
    config = LeadershipConfig(**payload)
    windows = [
        config.history_calendar_days,
        config.trend_days,
        config.trend_classification_days,
        config.trend_score_ma_short_days,
        config.trend_score_ma_long_days,
        config.trend_min_improving_days,
        config.rs_short_days,
        config.rs_long_days,
        config.new_high_days,
        config.above_ma_days,
        config.volume_ratio_days,
        config.leader_return_days,
        config.leader_near_high_days,
        config.leader_top_n,
        config.anchor_top_n,
        config.anchor_ma_short_days,
        config.anchor_ma_long_days,
        config.anchor_amount_short_days,
        config.anchor_amount_long_days,
        config.concentration_top_n,
    ]
    required_history = max(windows[1:])
    if min(windows) <= 0 or config.history_calendar_days < required_history:
        raise ValueError("Leadership radar windows must be positive and history_calendar_days must cover every window.")
    return config


def map_stocks_to_sw_l3(stock_codes: list[str], client: LeadershipDataClient, as_of: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped: list[dict[str, str]] = []
    unmatched: list[dict[str, str]] = []
    for code in stock_codes:
        membership = select_membership(client.get_sw_memberships(code), as_of)
        if membership is None:
            unmatched.append({"ts_code": code, "reason": f"No active SW2021 membership on {as_of}"})
            continue
        l3_code = str(membership.get("l3_code", "")).strip()
        if not l3_code:
            unmatched.append({"ts_code": code, "reason": "Membership has no level-3 sector code"})
            continue
        mapped.append(
            {
                "ts_code": code,
                "stock_name": str(membership.get("name", "")).strip(),
                "sw_l1_code": str(membership.get("l1_code", "")).strip(),
                "sw_l1_name": str(membership.get("l1_name", "")).strip(),
                "sw_l2_code": str(membership.get("l2_code", "")).strip(),
                "sw_l2_name": str(membership.get("l2_name", "")).strip(),
                "sw_l3_code": l3_code,
                "sw_l3_name": str(membership.get("l3_name", "")).strip(),
            }
        )
    return pd.DataFrame(mapped), pd.DataFrame(unmatched, columns=["ts_code", "reason"])


def active_sw_members(members: pd.DataFrame, as_of: date) -> pd.DataFrame:
    if members.empty:
        return members.copy()
    frame = members.copy().fillna("")
    in_dates = pd.to_datetime(frame.get("in_date", ""), format="%Y%m%d", errors="coerce")
    out_dates = pd.to_datetime(frame.get("out_date", ""), format="%Y%m%d", errors="coerce")
    stamp = pd.Timestamp(as_of)
    active = frame.loc[(in_dates.isna() | (in_dates <= stamp)) & (out_dates.isna() | (out_dates >= stamp))].copy()
    if active.empty:
        return active
    active["_in_date"] = pd.to_datetime(active.get("in_date", ""), format="%Y%m%d", errors="coerce")
    return (
        active.sort_values(["ts_code", "_in_date"], na_position="first")
        .drop_duplicates("ts_code", keep="last")
        .drop(columns="_in_date")
        .reset_index(drop=True)
    )


def _return_pct(history: pd.DataFrame, days: int) -> float:
    close = pd.to_numeric(history["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < days + 1:
        raise ValueError(f"Need at least {days + 1} daily rows, received {len(close)}.")
    return (float(close.iloc[-1]) / float(close.iloc[-days - 1]) - 1.0) * 100.0


def _rank_score(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    if len(values) == 1:
        return {next(iter(values)): 100.0}
    series = pd.Series(values, dtype=float)
    ranks = series.rank(method="average", ascending=True, pct=True)
    return (ranks * 100.0).to_dict()


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _stock_strength_metrics(history: pd.DataFrame, config: LeadershipConfig) -> dict[str, float | bool]:
    frame = history.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)
    close = pd.to_numeric(frame["close"], errors="coerce")
    minimum_rows = max(
        config.new_high_days,
        config.above_ma_days,
        config.leader_return_days + 1,
        config.leader_near_high_days,
    )
    if len(close.dropna()) < minimum_rows:
        raise ValueError(f"Need at least {minimum_rows} stock daily rows, received {len(close.dropna())}.")
    latest = float(close.iloc[-1])
    high_60 = float(close.tail(config.new_high_days).max())
    ma20 = float(close.tail(config.above_ma_days).mean())
    high_20 = float(close.tail(config.leader_near_high_days).max())
    return {
        "return_60d_pct": _return_pct(frame, config.leader_return_days),
        "is_60d_new_high": latest >= high_60,
        "above_ma20": latest > ma20,
        "near_20d_high": latest >= high_20 * (1.0 - config.leader_near_high_pct / 100.0),
    }


def _volume_ratio(history: pd.DataFrame, days: int) -> float:
    amount = pd.to_numeric(history.get("amount", pd.Series(dtype=float)), errors="coerce").dropna().reset_index(drop=True)
    if len(amount) < days:
        return 1.0
    average = float(amount.tail(days).mean())
    if average <= 0:
        return 1.0
    return float(amount.iloc[-1]) / average


def _history_until(history: pd.DataFrame, trade_date: pd.Timestamp) -> pd.DataFrame:
    frame = history.dropna(subset=["trade_date", "close"]).sort_values("trade_date").copy()
    return frame.loc[frame["trade_date"] <= trade_date].reset_index(drop=True)


def _return_pct_until(history: pd.DataFrame, trade_date: pd.Timestamp, days: int) -> float | None:
    frame = _history_until(history, trade_date)
    close = pd.to_numeric(frame["close"], errors="coerce").dropna().reset_index(drop=True)
    if len(close) < days + 1:
        return None
    return (float(close.iloc[-1]) / float(close.iloc[-days - 1]) - 1.0) * 100.0


def _stock_snapshot(
    ts_code: str, history: pd.DataFrame, trade_date: pd.Timestamp, benchmark_ret20: float, config: LeadershipConfig
) -> dict[str, float | bool | str] | None:
    frame = _history_until(history, trade_date)
    close = pd.to_numeric(frame.get("close", pd.Series(dtype=float)), errors="coerce")
    amount = pd.to_numeric(frame.get("amount", pd.Series(dtype=float)), errors="coerce")
    required = max(
        config.new_high_days,
        config.above_ma_days,
        config.leader_return_days + 1,
        config.anchor_ma_long_days,
        config.anchor_amount_long_days,
    )
    if len(close.dropna()) < required:
        return None
    latest = float(close.iloc[-1])
    high60 = float(close.tail(config.new_high_days).max())
    high20 = float(close.tail(config.leader_near_high_days).max())
    ma20 = float(close.tail(config.anchor_ma_short_days).mean())
    ma50 = float(close.tail(config.anchor_ma_long_days).mean())
    amount_today = float(amount.iloc[-1]) if len(amount.dropna()) else 0.0
    amount_ma5 = float(amount.tail(config.anchor_amount_short_days).mean()) if len(amount.dropna()) else 0.0
    amount_ma20 = float(amount.tail(config.anchor_amount_long_days).mean()) if len(amount.dropna()) else 0.0
    ret20 = _return_pct_until(frame, trade_date, config.rs_short_days)
    ret60 = _return_pct_until(frame, trade_date, config.leader_return_days)
    if ret20 is None or ret60 is None:
        return None

    drawdown20 = latest / high20 - 1.0 if high20 > 0 else -1.0
    anchor_score = 0.0
    if latest > ma20:
        anchor_score += 2.0
    if latest > ma50:
        anchor_score += 2.0
    if ret20 > benchmark_ret20:
        anchor_score += 2.0
    if drawdown20 >= -0.05:
        anchor_score += 2.0
    elif drawdown20 >= -0.10:
        anchor_score += 1.0
    if amount_today > amount_ma20:
        anchor_score += 1.0
    if amount_ma5 > amount_ma20:
        anchor_score += 1.0

    if latest >= high60:
        leader_score = 15.0
    elif drawdown20 >= -0.03:
        leader_score = 13.0
    elif drawdown20 >= -0.06:
        leader_score = 10.0
    elif latest > ma20 and drawdown20 >= -0.10:
        leader_score = 7.0
    elif latest > ma50:
        leader_score = 3.0
    else:
        leader_score = 0.0

    return {
        "ts_code": ts_code,
        "close": latest,
        "return_20d_pct": ret20,
        "return_60d_pct": ret60,
        "is_60d_new_high": latest >= high60,
        "above_ma20": latest > ma20,
        "amount_today": amount_today,
        "amount_ma20": amount_ma20,
        "anchor_score": _clamp(anchor_score, 0.0, 10.0),
        "leader_score": _clamp(leader_score, 0.0, 15.0),
    }


def _volume_score(volume_ratio: float, sector_ret20: float, benchmark_ret20: float) -> float:
    if volume_ratio > 2.5:
        raw = 6.0
    elif volume_ratio >= 1.5:
        raw = 10.0
    elif volume_ratio >= 1.2:
        raw = 8.0
    elif volume_ratio >= 1.0:
        raw = 6.0
    elif volume_ratio >= 0.8:
        raw = 4.0
    else:
        raw = 2.0
    if volume_ratio > 1.5 and sector_ret20 <= benchmark_ret20:
        raw = min(raw, 5.0)
    return raw


def _leadership_status_from_points(score: float, config: LeadershipConfig) -> str:
    if score >= config.strong_leadership_score:
        return "强主线"
    if score >= config.leadership_score:
        return "主线"
    if score >= config.candidate_score:
        return "主线候选 / 强题材"
    if score >= config.theme_score:
        return "活跃题材"
    if score >= config.hot_score:
        return "热点 / 反弹"
    return "普通 / 休眠"


def _classify_leadership_trend(industry_trend: pd.DataFrame, config: LeadershipConfig) -> str:
    if industry_trend.empty or "industry_score" not in industry_trend.columns:
        return "趋势平稳"
    scores = industry_trend.sort_values("trade_date")["industry_score"].dropna().astype(float).reset_index(drop=True)
    minimum_rows = max(config.trend_score_ma_long_days, config.trend_classification_days + config.trend_score_ma_short_days)
    if len(scores) < minimum_rows:
        return "趋势平稳"

    score_ma_short = scores.rolling(config.trend_score_ma_short_days).mean()
    score_ma_long = scores.rolling(config.trend_score_ma_long_days).mean()
    recent_short = score_ma_short.dropna().tail(config.trend_classification_days).reset_index(drop=True)
    if len(recent_short) < config.trend_classification_days:
        return "趋势平稳"

    trend_delta = float(recent_short.iloc[-1] - recent_short.iloc[0])
    improving_days = int((recent_short.diff().dropna() > 0).sum())
    declining_days = int((recent_short.diff().dropna() < 0).sum())
    latest_short = float(score_ma_short.iloc[-1])
    latest_long = float(score_ma_long.iloc[-1])

    if (
        trend_delta >= config.trend_score_threshold
        and improving_days >= config.trend_min_improving_days
        and latest_short > latest_long
    ):
        return "趋势向上"
    if (
        trend_delta <= -config.trend_score_threshold
        and declining_days >= config.trend_min_improving_days
        and latest_short < latest_long
    ):
        return "趋势向下"
    return "趋势平稳"


def _build_equal_weight_sector_history(stock_histories: dict[str, pd.DataFrame]) -> pd.DataFrame:
    close_frames: list[pd.DataFrame] = []
    amount_frames: list[pd.DataFrame] = []
    for ts_code, history in stock_histories.items():
        frame = history.dropna(subset=["trade_date", "close"]).sort_values("trade_date").copy()
        if frame.empty:
            continue
        first_close = float(pd.to_numeric(frame["close"], errors="coerce").dropna().iloc[0])
        if first_close <= 0:
            continue
        close_frames.append(
            pd.DataFrame(
                {
                    "trade_date": frame["trade_date"],
                    ts_code: pd.to_numeric(frame["close"], errors="coerce") / first_close * 100.0,
                }
            )
        )
        if "amount" in frame.columns:
            amount_frames.append(
                pd.DataFrame(
                    {
                        "trade_date": frame["trade_date"],
                        ts_code: pd.to_numeric(frame["amount"], errors="coerce"),
                    }
                )
            )
    if not close_frames:
        raise ValueError("Cannot build equal-weight sector history because no stock histories are usable.")

    close_matrix = close_frames[0]
    for frame in close_frames[1:]:
        close_matrix = close_matrix.merge(frame, on="trade_date", how="outer")
    result = pd.DataFrame(
        {
            "trade_date": close_matrix["trade_date"],
            "close": close_matrix.drop(columns="trade_date").mean(axis=1, skipna=True),
        }
    )

    if amount_frames:
        amount_matrix = amount_frames[0]
        for frame in amount_frames[1:]:
            amount_matrix = amount_matrix.merge(frame, on="trade_date", how="outer")
        amount = pd.DataFrame(
            {
                "trade_date": amount_matrix["trade_date"],
                "amount": amount_matrix.drop(columns="trade_date").sum(axis=1, skipna=True),
            }
        )
        result = result.merge(amount, on="trade_date", how="left")
    return result.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)


def _classify_score(score: float, config: LeadershipConfig) -> str:
    if score >= config.strong_leadership_score:
        return "强主线"
    if score >= config.leadership_score:
        return "主线"
    if score >= config.candidate_score:
        return "主线候选 / 强题材"
    if score >= config.theme_score:
        return "题材"
    return "普通方向"


def calculate_leadership_radar(
    stock_codes: list[str], client: LeadershipDataClient, as_of: date, config: LeadershipConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stocks, unmatched = map_stocks_to_sw_l3(stock_codes, client, as_of)
    if stocks.empty:
        raise ValueError("None of the supplied stocks had an active SW2021 level-3 membership.")

    start_date = as_of - timedelta(days=config.history_calendar_days)
    benchmark = client.get_index_daily(config.benchmark_index_code, start_date, as_of)

    raw_sector_rows: list[dict[str, object]] = []
    trend_rows: list[dict[str, object]] = []
    for l3_code, group in stocks.groupby("sw_l3_code", sort=True):
        all_members = active_sw_members(client.get_sw_l3_members(l3_code), as_of)
        if all_members.empty:
            all_members = group.rename(
                columns={"sw_l3_code": "l3_code", "sw_l3_name": "l3_name"}
            )[["ts_code", "stock_name", "l3_code", "l3_name"]].rename(columns={"stock_name": "name"})

        stock_metrics: list[dict[str, float | bool]] = []
        stock_histories: dict[str, pd.DataFrame] = {}
        failed_stocks: list[str] = []
        for row in all_members.itertuples(index=False):
            ts_code = str(row.ts_code).strip()
            if not ts_code:
                continue
            try:
                stock_history = client.get_daily(ts_code, start_date, as_of)
                stock_histories[ts_code] = stock_history
                stock_metrics.append(_stock_strength_metrics(stock_history, config))
            except ValueError as exc:
                failed_stocks.append(f"{ts_code}: {exc}")

        sector_source = "sw_daily"
        sector_warnings: list[str] = []
        try:
            sector_history = client.get_sw_daily(l3_code, start_date, as_of)
        except ValueError as exc:
            sector_source = "equal_weight_input_stocks"
            sector_warnings.append(f"sw_daily fallback: {exc}")
            sector_history = _build_equal_weight_sector_history(stock_histories)

        trade_dates = (
            sector_history["trade_date"]
            .dropna()
            .sort_values()
            .drop_duplicates()
            .tail(config.trend_days)
            .tolist()
        )
        for trade_date in trade_dates:
            benchmark_rs20 = _return_pct_until(benchmark, trade_date, config.rs_short_days)
            benchmark_rs60 = _return_pct_until(benchmark, trade_date, config.rs_long_days)
            sector_ret20 = _return_pct_until(sector_history, trade_date, config.rs_short_days)
            sector_ret60 = _return_pct_until(sector_history, trade_date, config.rs_long_days)
            if benchmark_rs20 is None or benchmark_rs60 is None or sector_ret20 is None or sector_ret60 is None:
                continue

            snapshots = [
                snapshot
                for ts_code, stock_history in stock_histories.items()
                if (snapshot := _stock_snapshot(ts_code, stock_history, trade_date, benchmark_rs20, config)) is not None
            ]
            metric_count = len(snapshots)
            if metric_count:
                nh60 = sum(bool(item["is_60d_new_high"]) for item in snapshots) / metric_count * 100.0
                above_ma20 = sum(bool(item["above_ma20"]) for item in snapshots) / metric_count * 100.0
                leaders = sorted(snapshots, key=lambda item: float(item["return_60d_pct"]), reverse=True)[
                    : config.leader_top_n
                ]
                leader_status_score = sum(float(item["leader_score"]) for item in leaders) / len(leaders)
                anchors = sorted(snapshots, key=lambda item: float(item["amount_ma20"]), reverse=True)[
                    : config.anchor_top_n
                ]
                anchor_weight = sum(float(item["amount_ma20"]) for item in anchors)
                if anchor_weight > 0:
                    anchor_status_score = (
                        sum(float(item["anchor_score"]) * float(item["amount_ma20"]) for item in anchors)
                        / anchor_weight
                    )
                else:
                    anchor_status_score = sum(float(item["anchor_score"]) for item in anchors) / len(anchors)
                total_amount = sum(float(item["amount_today"]) for item in snapshots)
                top_amount = sum(
                    float(item["amount_today"])
                    for item in sorted(snapshots, key=lambda item: float(item["amount_today"]), reverse=True)[
                        : config.concentration_top_n
                    ]
                )
                concentration_ratio = top_amount / total_amount if total_amount > 0 else 0.0
            else:
                nh60 = 0.0
                above_ma20 = 0.0
                leader_status_score = 0.0
                anchor_status_score = 0.0
                concentration_ratio = 0.0

            sector_until = _history_until(sector_history, trade_date)
            volume_ratio = _volume_ratio(sector_until, config.volume_ratio_days)
            trend_rows.append(
                {
                    "trade_date": pd.Timestamp(trade_date).date(),
                    "sw_l3_code": l3_code,
                    "sw_l3_name": group.iloc[0]["sw_l3_name"],
                    "input_stock_count": len(group),
                    "industry_member_count": len(all_members),
                    "metric_stock_count": metric_count,
                    "rs_20_pct": sector_ret20 - benchmark_rs20,
                    "rs_60_pct": sector_ret60 - benchmark_rs60,
                    "nh_60_pct": nh60,
                    "above_ma20_pct": above_ma20,
                    "volume_ratio": volume_ratio,
                    "volume_score": _volume_score(volume_ratio, sector_ret20, benchmark_rs20),
                    "leader_status_score": leader_status_score,
                    "anchor_status_score": anchor_status_score,
                    "concentration_ratio": concentration_ratio,
                    "sector_source": sector_source,
                }
            )

        sector_trend = pd.DataFrame([row for row in trend_rows if row["sw_l3_code"] == l3_code])
        if sector_trend.empty:
            sector_warnings.append("No trend rows could be calculated for this industry.")
            latest = {
                "rs_20_pct": 0.0,
                "rs_60_pct": 0.0,
                "nh_60_pct": 0.0,
                "above_ma20_pct": 0.0,
                "volume_ratio": 1.0,
                "volume_score": 0.0,
                "leader_status_score": 0.0,
                "anchor_status_score": 0.0,
                "concentration_ratio": 0.0,
                "metric_stock_count": len(stock_metrics),
            }
        else:
            latest = sector_trend.sort_values("trade_date").iloc[-1].to_dict()
        raw_sector_rows.append(
            {
                "sw_l3_code": l3_code,
                "sw_l3_name": group.iloc[0]["sw_l3_name"],
                "input_stock_count": len(group),
                "industry_member_count": len(all_members),
                "metric_stock_count": int(latest["metric_stock_count"]),
                "rs_20_pct": float(latest["rs_20_pct"]),
                "rs_60_pct": float(latest["rs_60_pct"]),
                "nh_60_pct": float(latest["nh_60_pct"]),
                "above_ma20_pct": float(latest["above_ma20_pct"]),
                "volume_ratio": float(latest["volume_ratio"]),
                "volume_score": float(latest["volume_score"]),
                "leader_status_score": float(latest["leader_status_score"]),
                "anchor_status_score": float(latest["anchor_status_score"]),
                "concentration_ratio": float(latest["concentration_ratio"]),
                "sector_source": sector_source,
                "metric_warnings": "; ".join([*sector_warnings, *failed_stocks]),
            }
        )

    sectors = pd.DataFrame(raw_sector_rows)
    trends = pd.DataFrame(trend_rows)
    rs20_scores = _rank_score(sectors.set_index("sw_l3_code")["rs_20_pct"].to_dict())
    rs60_scores = _rank_score(sectors.set_index("sw_l3_code")["rs_60_pct"].to_dict())
    sectors["rs_20_rank_score"] = sectors["sw_l3_code"].map(rs20_scores)
    sectors["rs_60_rank_score"] = sectors["sw_l3_code"].map(rs60_scores)
    sectors["nh_60_score"] = (sectors["nh_60_pct"] / 25.0).clip(lower=0.0, upper=1.0) * 15.0
    sectors["above_ma20_score"] = (sectors["above_ma20_pct"] / 80.0).clip(lower=0.0, upper=1.0) * 10.0
    sectors["concentration_score"] = 5.0
    sectors.loc[sectors["concentration_ratio"].between(0.15, 0.35), "concentration_score"] += 2.0
    sectors.loc[(sectors["concentration_ratio"] < 0.15) & (sectors["above_ma20_pct"] > 60.0), "concentration_score"] += 1.0
    sectors.loc[(sectors["concentration_ratio"] > 0.40) & (sectors["nh_60_pct"] < 5.0), "concentration_score"] -= 2.0
    sectors.loc[(sectors["concentration_ratio"] < 0.10) & (sectors["leader_status_score"] < 8.0), "concentration_score"] -= 2.0
    sectors.loc[sectors["anchor_status_score"] < 5.0, "concentration_score"] -= 1.0
    sectors["concentration_score"] = sectors["concentration_score"].clip(lower=0.0, upper=10.0)
    sectors["industry_score"] = (
        18.0 * sectors["rs_20_rank_score"] / 100.0
        + 12.0 * sectors["rs_60_rank_score"] / 100.0
        + sectors["nh_60_score"]
        + sectors["above_ma20_score"]
        + sectors["volume_score"]
        + sectors["leader_status_score"]
        + sectors["anchor_status_score"]
        + sectors["concentration_score"]
    )
    sectors["leadership_status"] = sectors["industry_score"].map(
        lambda value: _leadership_status_from_points(float(value), config)
    )
    sectors = sectors.sort_values(["industry_score", "sw_l3_code"], ascending=[False, True]).reset_index(drop=True)
    sectors.insert(0, "industry_rank", range(1, len(sectors) + 1))

    if not trends.empty:
        trends = trends.merge(
            sectors[["sw_l3_code", "rs_20_rank_score", "rs_60_rank_score"]],
            on="sw_l3_code",
            how="left",
        )
        trends["nh_60_score"] = (trends["nh_60_pct"] / 25.0).clip(lower=0.0, upper=1.0) * 15.0
        trends["above_ma20_score"] = (trends["above_ma20_pct"] / 80.0).clip(lower=0.0, upper=1.0) * 10.0
        trends["concentration_score"] = 5.0
        trends.loc[trends["concentration_ratio"].between(0.15, 0.35), "concentration_score"] += 2.0
        trends.loc[(trends["concentration_ratio"] < 0.15) & (trends["above_ma20_pct"] > 60.0), "concentration_score"] += 1.0
        trends.loc[(trends["concentration_ratio"] > 0.40) & (trends["nh_60_pct"] < 5.0), "concentration_score"] -= 2.0
        trends.loc[(trends["concentration_ratio"] < 0.10) & (trends["leader_status_score"] < 8.0), "concentration_score"] -= 2.0
        trends.loc[trends["anchor_status_score"] < 5.0, "concentration_score"] -= 1.0
        trends["concentration_score"] = trends["concentration_score"].clip(lower=0.0, upper=10.0)
        trends["industry_score"] = (
            18.0 * trends["rs_20_rank_score"] / 100.0
            + 12.0 * trends["rs_60_rank_score"] / 100.0
            + trends["nh_60_score"]
            + trends["above_ma20_score"]
            + trends["volume_score"]
            + trends["leader_status_score"]
            + trends["anchor_status_score"]
            + trends["concentration_score"]
        )
        trends["leadership_status"] = trends["industry_score"].map(
            lambda value: _leadership_status_from_points(float(value), config)
        )

    if trends.empty:
        sectors["leadership_trend"] = "趋势平稳"
    else:
        trend_labels = {
            l3_code: _classify_leadership_trend(group, config)
            for l3_code, group in trends.groupby("sw_l3_code", sort=False)
        }
        sectors["leadership_trend"] = sectors["sw_l3_code"].map(trend_labels).fillna("趋势平稳")

    stock_output = stocks.merge(
        sectors[
            [
                "industry_rank",
                "sw_l3_code",
                "industry_score",
                "industry_member_count",
                "leadership_status",
                "leadership_trend",
                "rs_20_pct",
                "rs_60_pct",
                "nh_60_pct",
                "above_ma20_pct",
                "volume_ratio",
                "leader_status_score",
                "anchor_status_score",
                "concentration_ratio",
            ]
        ],
        on="sw_l3_code",
        how="left",
    )
    stock_output = stock_output.sort_values(["industry_rank", "ts_code"]).reset_index(drop=True)
    return sectors, stock_output, unmatched, trends


def _format_markdown(stocks: pd.DataFrame, sectors: pd.DataFrame, unmatched: pd.DataFrame, as_of: date) -> str:
    lines = [
        f"# Leadership Radar MVP — {as_of}",
        "",
        "## Stocks",
        "",
        "| Stock | SW L3 | Industry Score | Status | Trend |",
        "|---|---|---:|---|---|",
    ]
    for row in stocks.itertuples(index=False):
        lines.append(
            f"| {row.ts_code} {row.stock_name} | {row.sw_l3_name} ({row.sw_l3_code}) "
            f"| {row.industry_score:.2f} | {row.leadership_status} |"
        )
    lines.extend(["", "## Industries", ""])
    lines.append("| Rank | SW L3 | Score | Members | Metric Stocks | RS20 | RS60 | NH60 | Above MA20 | Volume Ratio | Leader |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sectors.itertuples(index=False):
        lines.append(
            f"| {row.industry_rank} | {row.sw_l3_name} ({row.sw_l3_code}) | {row.industry_score:.2f} "
            f"| {row.industry_member_count} | {row.metric_stock_count} "
            f"| {row.rs_20_pct:.2f}% | {row.rs_60_pct:.2f}% | {row.nh_60_pct:.2f}% "
            f"| {row.above_ma20_pct:.2f}% | {row.volume_ratio:.2f} | {row.leader_status_score:.0f} |"
        )
    if not unmatched.empty:
        lines.extend(["", "## Unmatched stocks", ""])
        lines.extend(f"- {row.ts_code}: {row.reason}" for row in unmatched.itertuples(index=False))
    lines.extend(
        [
            "",
            "## MVP scoring formula",
            "",
            "```text",
            "Industry_Score =",
            "25% RS_20 rank score",
            "+ 20% RS_60 rank score",
            "+ 20% NH_60",
            "+ 15% Above_MA20",
            "+ 10% Volume_Ratio score",
            "+ 10% Leader_Status",
            "```",
            "",
            "Note: NH_60, Above_MA20, and Leader_Status are calculated from all active stocks in each SW L3 industry.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    run_dir: Path, sectors: pd.DataFrame, stocks: pd.DataFrame, unmatched: pd.DataFrame, as_of: date
) -> tuple[Path, Path, Path, Path]:
    sectors_path = run_dir / "industries.csv"
    stocks_path = run_dir / "stocks.csv"
    unmatched_path = run_dir / "unmatched_stocks.csv"
    markdown_path = run_dir / "leadership_radar.md"
    sectors.to_csv(sectors_path, index=False, encoding="utf-8-sig")
    stocks.to_csv(stocks_path, index=False, encoding="utf-8-sig")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(_format_markdown(stocks, sectors, unmatched, as_of), encoding="utf-8")
    return sectors_path, stocks_path, unmatched_path, markdown_path


def _format_markdown_v2(stocks: pd.DataFrame, sectors: pd.DataFrame, unmatched: pd.DataFrame, as_of: date) -> str:
    lines = [
        f"# Leadership Radar — {as_of}",
        "",
        "## Stocks",
        "",
        "| Stock | SW L3 | Industry Score | Status | Trend |",
        "|---|---|---:|---|---|",
    ]
    for row in stocks.itertuples(index=False):
        lines.append(
            f"| {row.ts_code} {row.stock_name} | {row.sw_l3_name} ({row.sw_l3_code}) "
            f"| {row.industry_score:.2f} | {row.leadership_status} | {row.leadership_trend} |"
        )
    lines.extend(["", "## Industries", ""])
    lines.append(
        "| Rank | SW L3 | Score | Trend | Members | Metric Stocks | RS20 | RS60 | NH60 | Above MA20 | Vol | Leader | Anchor | Conc |"
    )
    lines.append("|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sectors.itertuples(index=False):
        lines.append(
            f"| {row.industry_rank} | {row.sw_l3_name} ({row.sw_l3_code}) | {row.industry_score:.2f} "
            f"| {row.leadership_trend} "
            f"| {row.industry_member_count} | {row.metric_stock_count} "
            f"| {row.rs_20_pct:.2f}% | {row.rs_60_pct:.2f}% | {row.nh_60_pct:.2f}% "
            f"| {row.above_ma20_pct:.2f}% | {row.volume_ratio:.2f} | {row.leader_status_score:.2f} "
            f"| {row.anchor_status_score:.2f} | {row.concentration_ratio:.2%} |"
        )
    if not unmatched.empty:
        lines.extend(["", "## Unmatched stocks", ""])
        lines.extend(f"- {row.ts_code}: {row.reason}" for row in unmatched.itertuples(index=False))
    lines.extend(
        [
            "",
            "## 8-metric scoring formula",
            "",
            "```text",
            "Industry_Score =",
            "18 * RS_20_rank_norm",
            "+ 12 * RS_60_rank_norm",
            "+ 15 * NH_60_norm",
            "+ 10 * Above_MA20_norm",
            "+ 10 * Volume_Ratio_score_norm",
            "+ 15 * Leader_Status_norm",
            "+ 10 * Anchor_Status_norm",
            "+ 10 * Concentration_Score_norm",
            "```",
            "",
            "Charts are written under `charts/`; daily metric rows are written to `industry_trends.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_trend_charts(run_dir: Path, trends: pd.DataFrame) -> list[Path]:
    if trends.empty:
        return []
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        return []

    chart_dir = run_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_paths: list[Path] = []
    for l3_code, group in trends.groupby("sw_l3_code", sort=True):
        frame = group.sort_values("trade_date")
        title = f"{frame.iloc[-1]['sw_l3_name']} ({l3_code}) leadership metrics"
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Industry score", "Strength / Breadth", "Capital / Core stocks"),
        )
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["industry_score"], name="Score"), row=1, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["rs_20_pct"], name="RS20"), row=2, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["rs_60_pct"], name="RS60"), row=2, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["nh_60_pct"], name="NH60"), row=2, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["above_ma20_pct"], name="Above MA20"), row=2, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["volume_ratio"], name="Volume Ratio"), row=3, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["leader_status_score"], name="Leader"), row=3, col=1)
        fig.add_trace(go.Scatter(x=frame["trade_date"], y=frame["anchor_status_score"], name="Anchor"), row=3, col=1)
        fig.add_trace(
            go.Scatter(x=frame["trade_date"], y=frame["concentration_ratio"] * 100.0, name="Concentration %"),
            row=3,
            col=1,
        )
        fig.update_layout(title=title, height=900, hovermode="x unified")
        chart_path = chart_dir / f"{l3_code.replace('.', '_')}_trend.html"
        fig.write_html(chart_path, include_plotlyjs="cdn")
        chart_paths.append(chart_path)
    return chart_paths


def write_reports_v2(
    run_dir: Path,
    sectors: pd.DataFrame,
    stocks: pd.DataFrame,
    unmatched: pd.DataFrame,
    trends: pd.DataFrame,
    as_of: date,
) -> tuple[Path, Path, Path, Path, Path, list[Path]]:
    sectors_path = run_dir / "industries.csv"
    stocks_path = run_dir / "stocks.csv"
    trends_path = run_dir / "industry_trends.csv"
    unmatched_path = run_dir / "unmatched_stocks.csv"
    markdown_path = run_dir / "leadership_radar.md"
    sectors.to_csv(sectors_path, index=False, encoding="utf-8-sig")
    stocks.to_csv(stocks_path, index=False, encoding="utf-8-sig")
    trends.to_csv(trends_path, index=False, encoding="utf-8-sig")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
    chart_paths = write_trend_charts(run_dir, trends)
    markdown_path.write_text(_format_markdown_v2(stocks, sectors, unmatched, as_of), encoding="utf-8")
    return sectors_path, stocks_path, trends_path, unmatched_path, markdown_path, chart_paths


def _parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD or YYYYMMDD.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leadership radar: map stocks to SW level-3 industries, score 8 metrics, and chart trends."
    )
    parser.add_argument("--stocks", type=Path, required=True, help="CSV with ts_code, or a text file of stock codes.")
    parser.add_argument("--as-of", type=_parse_date, default=date.today(), help="Radar date; defaults to today.")
    parser.add_argument("--config", type=Path, default=Path("config/leadership_radar.yaml"))
    parser.add_argument("--strategy-config", type=Path, default=Path("config/strategy.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/leadership_radar"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    leadership_config = load_leadership_config(args.config.resolve())
    with args.strategy_config.resolve().open("r", encoding="utf-8") as handle:
        strategy_payload = yaml.safe_load(handle) or {}
    project_root = args.strategy_config.resolve().parent.parent
    cache_dir = (project_root / strategy_payload.get("data", {}).get("cache_dir", "data/cache")).resolve()
    client = TushareClient(cache_dir=cache_dir)
    stock_codes = load_stock_list(args.stocks.resolve())
    try:
        sectors, stocks, unmatched, trends = calculate_leadership_radar(
            stock_codes, client, args.as_of, leadership_config
        )
    except PermissionError as exc:
        parser.exit(
            2,
            "Leadership radar could not retrieve required Tushare data. "
            "Grant this account access to index_member_all, sw_daily, daily, and index_daily.\n"
            f"Details: {exc}\n",
        )
    run_dir = make_run_dir(args.output_dir.resolve())
    paths = write_reports_v2(run_dir, sectors, stocks, unmatched, trends, args.as_of)
    print(f"Industry report: {paths[0]}")
    print(f"Stock report: {paths[1]}")
    print(f"Industry trends: {paths[2]}")
    print(f"Unmatched stocks: {paths[3]}")
    print(f"Markdown report: {paths[4]}")
    print(f"Trend charts: {len(paths[5])}")
    print(f"Industries ranked: {len(sectors)}; stocks mapped: {len(stocks)}; unmatched: {len(unmatched)}")


if __name__ == "__main__":
    main()
