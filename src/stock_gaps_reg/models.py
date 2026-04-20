from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass(frozen=True)
class Candidate:
    ts_code: str
    detect_date: date
    note: str = ""


@dataclass
class EntryDecision:
    eligible: bool
    reason: str
    buy_date: date | None = None
    buy_time: datetime | None = None
    buy_price: float | None = None
    detect_prev_high: float | None = None
    detect_day_open: float | None = None
    detect_day_close: float | None = None
    detect_day_low: float | None = None
    detect_day_high: float | None = None
    detect_day_volume: float | None = None
    day2_volume_1430: float | None = None
    pullback_ratio: float | None = None
    gap_size: float | None = None
    price_up_ratio: float | None = None
    vwap_at_1430: float | None = None
    price_above_vwap: bool = False
    vwap_rising_after_1400: bool = False
    day2_low_before_1400: float | None = None
    day2_low_after_1400: float | None = None
    day1_change_pct: float | None = None
    day1_close_strength: float | None = None
    initial_stop_price: float | None = None
    initial_r: float | None = None


@dataclass
class TradeResult:
    ts_code: str
    detect_date: date
    note: str
    status: str
    status_reason: str
    buy_date: date | None = None
    buy_time: datetime | None = None
    buy_price: float | None = None
    initial_stop_price: float | None = None
    initial_r: float | None = None
    exit_date: date | None = None
    exit_time: datetime | None = None
    exit_price: float | None = None
    exit_reason: str | None = None
    hold_days: int | None = None
    pnl_amount: float | None = None
    pnl_pct: float | None = None
    pnl_r: float | None = None
    max_favorable_excursion_r: float | None = None
    max_adverse_excursion_r: float | None = None
    peak_stop_price: float | None = None
    entry_notes: dict[str, object] = field(default_factory=dict)


@dataclass
class SummaryMetrics:
    total_candidates: int
    skipped_candidates: int
    total_trades: int
    wins: int
    losses: int
    breakeven: int
    win_rate: float
    avg_r: float
    total_r: float
    avg_pct: float
    max_win_r: float
    max_loss_r: float
    median_r: float
    gross_win_r: float
    gross_loss_r: float
    profit_factor: float | None
    expectancy_r: float
