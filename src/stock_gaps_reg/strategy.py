from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from math import isnan

import pandas as pd

from .config import StrategyConfig
from .models import Candidate, EntryDecision, SummaryMetrics, TradeResult
from .tushare_client import TushareClient


def _combine_day_and_hhmm(trade_date: date, hhmm: str) -> datetime:
    target = datetime.strptime(hhmm, "%H:%M").time()
    return datetime.combine(trade_date, target)


def _lookup_row_by_date(frame: pd.DataFrame, target_date: date) -> pd.Series:
    matched = frame.loc[frame["trade_date"] == pd.Timestamp(target_date)]
    if matched.empty:
        raise ValueError(f"Missing daily row for {target_date}.")
    return matched.iloc[0]


def _minute_slice(
    frame: pd.DataFrame,
    trade_date: date,
    start_time: str | None = None,
    end_time: str | None = None,
) -> pd.DataFrame:
    day_start = datetime.combine(trade_date, time.min)
    day_end = day_start + timedelta(days=1)
    matched = frame[(frame["trade_time"] >= day_start) & (frame["trade_time"] < day_end)].copy()
    if start_time:
        matched = matched[matched["trade_time"].dt.time >= datetime.strptime(start_time, "%H:%M").time()]
    if end_time:
        matched = matched[matched["trade_time"].dt.time <= datetime.strptime(end_time, "%H:%M").time()]
    return matched.reset_index(drop=True)


def _price_reference(row: pd.Series, mode: str) -> float:
    body = float(row["close"] - row["open"])
    daily_range = float(row["high"] - row["low"])
    if mode == "body" and body > 0:
        return body
    return max(daily_range, 1e-6)


def _compute_pullback_ratio(detect_row: pd.Series, buy_price: float, reference_mode: str) -> float:
    reference = _price_reference(detect_row, reference_mode)
    pullback_amount = max(float(detect_row["close"]) - buy_price, 0.0)
    return pullback_amount / reference


def _has_long_lower_shadow(
    minute_frame: pd.DataFrame,
    trade_date: date,
    start_time: str,
    min_ratio: float,
    close_in_upper_half: bool,
) -> bool:
    window = _minute_slice(minute_frame, trade_date, start_time=start_time, end_time="14:30")
    if window.empty:
        return False
    candle_open = float(window.iloc[0]["open"])
    candle_close = float(window.iloc[-1]["close"])
    candle_high = float(window["high"].max())
    candle_low = float(window["low"].min())
    body = abs(candle_close - candle_open)
    lower_shadow = min(candle_open, candle_close) - candle_low
    full_range = candle_high - candle_low
    if full_range <= 0:
        return False
    if body == 0:
        body = full_range * 0.1
    if close_in_upper_half and (candle_close - candle_low) / full_range < 0.5:
        return False
    return lower_shadow / body >= min_ratio


def _stabilized_after_1400(minute_frame: pd.DataFrame, trade_date: date, start_time: str) -> bool:
    window = _minute_slice(minute_frame, trade_date, start_time=start_time, end_time="14:30")
    if window.empty:
        return False
    start_price = float(window.iloc[0]["open"])
    last_price = float(window.iloc[-1]["close"])
    low_after_start = float(window["low"].min())
    low_index = int(window["low"].idxmin())
    lows_after_low = window.loc[low_index + 1 :, "low"]
    no_new_low = lows_after_low.empty or bool((lows_after_low >= low_after_start).all())
    return last_price >= start_price and no_new_low


def _gap_unfilled(prev_high: float, minute_frame: pd.DataFrame, trade_date: date, end_time: str) -> bool:
    window = _minute_slice(minute_frame, trade_date, end_time=end_time)
    if window.empty:
        return False
    return float(window["low"].min()) > prev_high


def _cumulative_volume_to(frame: pd.DataFrame, trade_date: date, end_time: str) -> float:
    window = _minute_slice(frame, trade_date, end_time=end_time)
    if window.empty:
        return 0.0
    return float(window["vol"].sum())


def _buy_bar(frame: pd.DataFrame, trade_date: date, buy_time: str) -> pd.Series:
    target_dt = _combine_day_and_hhmm(trade_date, buy_time)
    matched = frame[frame["trade_time"] == target_dt]
    if matched.empty:
        raise ValueError(f"Missing minute bar at {target_dt}.")
    return matched.iloc[0]


def _trade_days_after(client: TushareClient, start_date: date, days_needed: int) -> list[date]:
    lookahead_days = max(30, days_needed * 3 + 20)
    for _ in range(6):
        calendar = client.list_trade_days(start_date - timedelta(days=5), start_date + timedelta(days=lookahead_days))
        filtered = [day for day in calendar if day >= start_date]
        if len(filtered) >= days_needed:
            return filtered[:days_needed]
        lookahead_days *= 2
    raise ValueError(f"Need {days_needed} trade days from {start_date}, found {len(filtered)}.")


def _trade_day_offsets(client: TushareClient, anchor_date: date, previous_needed: int, next_needed: int) -> list[date]:
    lookback_days = max(30, previous_needed * 15)
    lookahead_days = max(30, next_needed * 15)
    last_calendar: list[date] = []

    for _ in range(6):
        calendar = client.list_trade_days(
            anchor_date - timedelta(days=lookback_days),
            anchor_date + timedelta(days=lookahead_days),
        )
        last_calendar = calendar
        if anchor_date not in calendar:
            lookback_days *= 2
            lookahead_days *= 2
            continue

        anchor_index = calendar.index(anchor_date)
        has_previous = anchor_index >= previous_needed
        has_next = len(calendar) - anchor_index - 1 >= next_needed
        if has_previous and has_next:
            start_index = anchor_index - previous_needed
            end_index = anchor_index + next_needed + 1
            return calendar[start_index:end_index]

        lookback_days *= 2
        lookahead_days *= 2

    if anchor_date not in last_calendar:
        raise ValueError(f"Detect date {anchor_date} is not an open trade day.")
    raise ValueError(
        f"Not enough trading days around {anchor_date}: need previous={previous_needed}, next={next_needed}."
    )


def evaluate_entry(candidate: Candidate, config: StrategyConfig, client: TushareClient) -> EntryDecision:
    trade_days = _trade_day_offsets(
        client,
        candidate.detect_date,
        previous_needed=1,
        next_needed=config.entry.buy_on_nth_trading_day_after_detect,
    )
    prev_trade_date = trade_days[0]
    detect_index = 1
    buy_date = trade_days[detect_index + config.entry.buy_on_nth_trading_day_after_detect]

    daily = client.get_daily(candidate.ts_code, prev_trade_date, buy_date)
    detect_row = _lookup_row_by_date(daily, candidate.detect_date)
    prev_row = _lookup_row_by_date(daily, prev_trade_date)

    if float(detect_row["low"]) <= float(prev_row["high"]):
        return EntryDecision(eligible=False, reason="detect_day_gap_not_confirmed")

    detect_minutes = client.get_minutes_for_day(candidate.ts_code, candidate.detect_date, config.data.minute_freq)
    buy_minutes = client.get_minutes_for_day(candidate.ts_code, buy_date, config.data.minute_freq)
    buy_bar = _buy_bar(buy_minutes, buy_date, config.market.buy_time)
    buy_price = float(buy_bar["close"])

    pullback_ratio = _compute_pullback_ratio(detect_row, buy_price, config.entry.pullback_reference)
    detect_day_volume = float(detect_minutes["vol"].sum())
    day2_volume_1430 = _cumulative_volume_to(buy_minutes, buy_date, config.market.buy_time)
    has_long_lower_shadow = _has_long_lower_shadow(
        buy_minutes,
        buy_date,
        config.market.stabilization_start_time,
        config.entry.lower_shadow_ratio,
        config.entry.close_in_upper_half,
    )
    stabilized_after_1400 = _stabilized_after_1400(
        buy_minutes,
        buy_date,
        config.market.stabilization_start_time,
    )
    gap_unfilled = _gap_unfilled(float(prev_row["high"]), buy_minutes, buy_date, config.market.buy_time)

    volume_ok = day2_volume_1430 < detect_day_volume * config.entry.volume_fraction
    price_ok = pullback_ratio <= config.entry.pullback_fraction
    behavior_ok = has_long_lower_shadow or stabilized_after_1400
    eligible = price_ok and volume_ok and behavior_ok and gap_unfilled

    notes = {
        "buy_date": buy_date,
        "buy_time": _combine_day_and_hhmm(buy_date, config.market.buy_time),
        "buy_price": buy_price,
        "detect_prev_high": float(prev_row["high"]),
        "detect_day_open": float(detect_row["open"]),
        "detect_day_close": float(detect_row["close"]),
        "detect_day_low": float(detect_row["low"]),
        "detect_day_high": float(detect_row["high"]),
        "detect_day_volume": detect_day_volume,
        "day2_volume_1430": day2_volume_1430,
        "pullback_ratio": pullback_ratio,
        "has_long_lower_shadow": has_long_lower_shadow,
        "stabilized_after_1400": stabilized_after_1400,
        "gap_unfilled": gap_unfilled,
        "gap_size": float(detect_row["low"] - prev_row["high"]),
    }

    if not eligible:
        failed = []
        if not price_ok:
            failed.append("pullback_rule")
        if not volume_ok:
            failed.append("volume_rule")
        if not behavior_ok:
            failed.append("reversal_rule")
        if not gap_unfilled:
            failed.append("gap_fill_rule")
        return EntryDecision(eligible=False, reason=";".join(failed), **notes)

    stop_by_pct = buy_price * (1 - config.risk.initial_stop_loss_pct)
    stop_by_gap = float(prev_row["high"]) * (1 - config.risk.gap_fill_stop_buffer_pct)
    initial_stop = max(stop_by_pct, stop_by_gap)
    initial_r = buy_price - initial_stop
    if initial_r <= 0:
        return EntryDecision(eligible=False, reason="invalid_initial_r", **notes)

    return EntryDecision(
        eligible=True,
        reason="eligible",
        initial_stop_price=initial_stop,
        initial_r=initial_r,
        **notes,
    )


def _apply_intrabar_path(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    current_stop: float,
    one_r_target: float,
    two_r_target: float,
    buy_price: float,
    initial_r: float,
    intrabar_order: str,
    raise_at_1r: bool,
    raise_at_2r: bool,
) -> tuple[float, float | None]:
    if intrabar_order == "olhc":
        checkpoints = [open_price, low_price, high_price, close_price]
    else:
        checkpoints = [open_price, high_price, low_price, close_price]

    stop = current_stop
    for price in checkpoints:
        if raise_at_2r and price >= two_r_target and stop < buy_price + initial_r:
            stop = buy_price + initial_r
        elif raise_at_1r and price >= one_r_target and stop < buy_price:
            stop = buy_price
        if price <= stop:
            return stop, stop
    return stop, None


def simulate_trade(candidate: Candidate, entry: EntryDecision, config: StrategyConfig, client: TushareClient) -> TradeResult:
    if not entry.eligible:
        return TradeResult(
            ts_code=candidate.ts_code,
            detect_date=candidate.detect_date,
            note=candidate.note,
            status="skipped",
            status_reason=entry.reason,
            buy_date=entry.buy_date,
            buy_time=entry.buy_time,
            buy_price=entry.buy_price,
            entry_notes=asdict(entry),
        )

    buy_date = entry.buy_date
    assert buy_date is not None
    buy_price = float(entry.buy_price)
    initial_stop = float(entry.initial_stop_price)
    initial_r = float(entry.initial_r)
    future_days = _trade_days_after(client, buy_date, config.exit.simulation_max_days_after_entry + 2)
    end_date = future_days[-1]
    daily = client.get_daily_with_ma(candidate.ts_code, buy_date, end_date, config.exit.ma_window)

    stop_price = initial_stop
    one_r_target = buy_price + initial_r
    two_r_target = buy_price + 2 * initial_r
    half_r_target = buy_price + config.exit.timeout_target_r * initial_r
    peak_stop_price = stop_price
    max_high = buy_price
    min_low = buy_price
    reached_half_r = False
    consecutive_below_ma = 0
    scheduled_open_exit: tuple[date, str] | None = None

    for index, trade_day in enumerate(future_days):
        if index > config.exit.simulation_max_days_after_entry:
            break
        minutes = client.get_minutes_for_day(candidate.ts_code, trade_day, config.data.minute_freq)
        if trade_day == buy_date:
            minutes = minutes[minutes["trade_time"] > entry.buy_time].reset_index(drop=True)
        if minutes.empty:
            continue

        if scheduled_open_exit and scheduled_open_exit[0] == trade_day:
            day_open = float(minutes.iloc[0]["open"])
            return _finalize_trade(
                candidate,
                entry,
                exit_date=trade_day,
                exit_time=datetime.combine(trade_day, time(hour=9, minute=30)),
                exit_price=day_open,
                exit_reason=scheduled_open_exit[1],
                hold_sessions=index + 1,
                max_high=max_high,
                min_low=min_low,
                peak_stop_price=peak_stop_price,
            )

        day_open = float(minutes.iloc[0]["open"])
        if day_open <= stop_price:
            return _finalize_trade(
                candidate,
                entry,
                exit_date=trade_day,
                exit_time=minutes.iloc[0]["trade_time"].to_pydatetime(),
                exit_price=day_open,
                exit_reason="gap_down_stop",
                hold_sessions=index + 1,
                max_high=max_high,
                min_low=min_low,
                peak_stop_price=peak_stop_price,
            )

        for row in minutes.itertuples(index=False):
            open_price = float(row.open)
            high_price = float(row.high)
            low_price = float(row.low)
            close_price = float(row.close)

            max_high = max(max_high, high_price)
            min_low = min(min_low, low_price)
            if high_price >= half_r_target:
                reached_half_r = True

            stop_price, triggered_exit = _apply_intrabar_path(
                open_price,
                high_price,
                low_price,
                close_price,
                stop_price,
                one_r_target,
                two_r_target,
                buy_price,
                initial_r,
                config.market.intrabar_order,
                config.risk.raise_stop_at_1r_to_entry,
                config.risk.raise_stop_at_2r_to_entry_plus_1r,
            )
            peak_stop_price = max(peak_stop_price, stop_price)
            if triggered_exit is not None:
                return _finalize_trade(
                    candidate,
                    entry,
                    exit_date=trade_day,
                    exit_time=row.trade_time.to_pydatetime(),
                    exit_price=float(triggered_exit),
                    exit_reason="trailing_stop",
                    hold_sessions=index + 1,
                    max_high=max_high,
                    min_low=min_low,
                    peak_stop_price=peak_stop_price,
                )

        daily_row = _lookup_row_by_date(daily, trade_day)
        ma_value = float(daily_row["ma"]) if pd.notna(daily_row["ma"]) else float("nan")
        if not isnan(ma_value) and float(daily_row["close"]) < ma_value:
            consecutive_below_ma += 1
        else:
            consecutive_below_ma = 0

        if consecutive_below_ma >= config.exit.consecutive_close_below_ma_days:
            scheduled_open_exit = (future_days[index + 1], "two_close_below_ma5")

        hold_days = index + 1
        if hold_days >= config.exit.timeout_hold_days and not reached_half_r and scheduled_open_exit is None:
            scheduled_open_exit = (future_days[index + 1], "timeout_no_0.5r")

    last_day = future_days[min(config.exit.simulation_max_days_after_entry, len(future_days) - 1)]
    last_minutes = client.get_minutes_for_day(candidate.ts_code, last_day, config.data.minute_freq)
    last_row = last_minutes.iloc[-1]
    return _finalize_trade(
        candidate,
        entry,
        exit_date=last_day,
        exit_time=last_row["trade_time"].to_pydatetime(),
        exit_price=float(last_row["close"]),
        exit_reason="forced_end_of_simulation",
        hold_sessions=min(config.exit.simulation_max_days_after_entry + 1, len(future_days)),
        max_high=max_high,
        min_low=min_low,
        peak_stop_price=peak_stop_price,
    )


def _finalize_trade(
    candidate: Candidate,
    entry: EntryDecision,
    exit_date: date,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
    hold_sessions: int,
    max_high: float,
    min_low: float,
    peak_stop_price: float,
) -> TradeResult:
    buy_price = float(entry.buy_price)
    initial_r = float(entry.initial_r)
    pnl_amount = exit_price - buy_price
    pnl_pct = pnl_amount / buy_price if buy_price else None
    pnl_r = pnl_amount / initial_r if initial_r else None
    mfe_r = (max_high - buy_price) / initial_r if initial_r else None
    mae_r = (min_low - buy_price) / initial_r if initial_r else None
    return TradeResult(
        ts_code=candidate.ts_code,
        detect_date=candidate.detect_date,
        note=candidate.note,
        status="traded",
        status_reason="completed",
        buy_date=entry.buy_date,
        buy_time=entry.buy_time,
        buy_price=buy_price,
        initial_stop_price=entry.initial_stop_price,
        initial_r=initial_r,
        exit_date=exit_date,
        exit_time=exit_time,
        exit_price=exit_price,
        exit_reason=exit_reason,
        hold_days=hold_sessions,
        pnl_amount=pnl_amount,
        pnl_pct=pnl_pct,
        pnl_r=pnl_r,
        max_favorable_excursion_r=mfe_r,
        max_adverse_excursion_r=mae_r,
        peak_stop_price=peak_stop_price,
        entry_notes=asdict(entry),
    )


def summarize_results(results: list[TradeResult]) -> SummaryMetrics:
    trades = [item for item in results if item.status == "traded" and item.pnl_r is not None]
    trade_count = len(trades)
    skipped_count = len(results) - trade_count
    wins = sum(1 for item in trades if (item.pnl_r or 0) > 0)
    losses = sum(1 for item in trades if (item.pnl_r or 0) < 0)
    breakeven = trade_count - wins - losses
    pnl_rs = [float(item.pnl_r) for item in trades]
    total_r = sum(pnl_rs)
    avg_r = total_r / trade_count if trade_count else 0.0
    avg_pct = sum(float(item.pnl_pct) for item in trades) / trade_count if trade_count else 0.0
    max_win_r = max(pnl_rs, default=0.0)
    max_loss_r = min(pnl_rs, default=0.0)
    median_r = float(pd.Series(pnl_rs).median()) if pnl_rs else 0.0
    gross_win_r = sum(value for value in pnl_rs if value > 0)
    gross_loss_r = abs(sum(value for value in pnl_rs if value < 0))
    profit_factor = (gross_win_r / gross_loss_r) if gross_loss_r else None
    return SummaryMetrics(
        total_candidates=len(results),
        skipped_candidates=skipped_count,
        total_trades=trade_count,
        wins=wins,
        losses=losses,
        breakeven=breakeven,
        win_rate=(wins / trade_count) if trade_count else 0.0,
        avg_r=avg_r,
        total_r=total_r,
        avg_pct=avg_pct,
        max_win_r=max_win_r,
        max_loss_r=max_loss_r,
        median_r=median_r,
        gross_win_r=gross_win_r,
        gross_loss_r=gross_loss_r,
        profit_factor=profit_factor,
        expectancy_r=avg_r,
    )


def run_strategy(candidates: list[Candidate], config: StrategyConfig, client: TushareClient) -> list[TradeResult]:
    results: list[TradeResult] = []
    for candidate in candidates:
        try:
            entry = evaluate_entry(candidate, config, client)
            results.append(simulate_trade(candidate, entry, config, client))
        except Exception as exc:
            results.append(
                TradeResult(
                    ts_code=candidate.ts_code,
                    detect_date=candidate.detect_date,
                    note=candidate.note,
                    status="error",
                    status_reason=str(exc),
                    entry_notes={"exception": str(exc)},
                )
            )
    return results
