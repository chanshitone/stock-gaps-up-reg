from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MarketConfig:
    exchange: str
    buy_time: str
    intrabar_order: str


@dataclass(frozen=True)
class EntryConfig:
    buy_on_nth_trading_day_after_detect: int
    pullback_fraction: float
    pullback_reference: str
    volume_fraction: float
    day1_min_change_pct: float
    day1_min_close_strength: float
    max_gap_fill_ratio: float


@dataclass(frozen=True)
class RiskConfig:
    initial_stop_loss_pct: float


@dataclass(frozen=True)
class ExitConfig:
    ma_window: int
    consecutive_close_below_ma_days: int
    timeout_hold_days: int
    timeout_target_r: float
    simulation_max_days_after_entry: int
    vol_spike_ma_window: int
    vol_spike_ratio: float


@dataclass(frozen=True)
class DataConfig:
    minute_freq: str
    cache_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class StrategyConfig:
    market: MarketConfig
    entry: EntryConfig
    risk: RiskConfig
    exit: ExitConfig
    data: DataConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file {path} must contain a YAML object.")
    return payload


def load_config(path: Path) -> StrategyConfig:
    payload = _read_yaml(path)
    base_dir = path.parent.parent
    market = payload["market"]
    entry = payload["entry"]
    risk = payload["risk"]
    exit_config = payload["exit"]
    data = payload["data"]

    return StrategyConfig(
        market=MarketConfig(**market),
        entry=EntryConfig(**entry),
        risk=RiskConfig(**risk),
        exit=ExitConfig(**exit_config),
        data=DataConfig(
            minute_freq=data["minute_freq"],
            cache_dir=(base_dir / data["cache_dir"]).resolve(),
            output_dir=(base_dir / data["output_dir"]).resolve(),
        ),
    )
