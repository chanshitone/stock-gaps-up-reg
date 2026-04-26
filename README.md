# Stock Gap Pullback Regression Workspace

This workspace runs a regression test for A-share gap-up candidates using Tushare daily and minute data, then exports trade-level details and summary statistics to CSV.

## What It Does

Input:

- Candidate CSV with `ts_code` and `detect_date`

### Entry Conditions

Execution timing:

- Buy at the `market.buy_time` 1-minute bar close on trading day `N` after `detect_date`, where `N = entry.buy_on_nth_trading_day_after_detect`.
- With the current config, that means the `14:30` bar close on the next trading day after `detect_date`.

Active entry filters:

| # | Condition | Config Key | Default |
|---|-----------|-----------|---------|
| 1 | **Gap confirmed**: detect-day low > previous-day high | — | structural |
| 2 | **Volume shrink**: day2 cumulative volume up to 14:30 < detect-day full volume × `volume_fraction` | `entry.volume_fraction` | 0.95 |
| 3 | **Day1 strength** (OR): detect-day change% ≥ `day1_min_change_pct` **or** close strength ≥ `day1_min_close_strength` | `entry.day1_min_change_pct` / `entry.day1_min_close_strength` | 0.01 / 0.5 |
| 4 | **Price-up strength**: `(buy_price - detect_close) / (detect_low - prev_high) ≥ min_price_up_ratio` | `entry.min_price_up_ratio` | 0.8 |
| 5 | **VWAP filter**: buy_price ≥ session VWAP × (1 + `vwap_min_buffer`) **and** VWAP rising from 14:00 to 14:30 | `entry.vwap_min_buffer` | 0.002 |
| 6 | **No new low after 14:00**: day2 low in 14:00–14:30 > day2 low in 09:30–13:59 | — | structural |

Diagnostic metric exported but not currently used as a gate:

- `pullback_ratio`, using `entry.pullback_reference` (current value: `body`)

### Exit Conditions

Exits are checked in priority order each trading day:

| # | Condition | Config Key | Default |
|---|-----------|-----------|---------|
| 1 | **Gap-down stop**: day open ≤ stop price → exit at open | — | — |
| 2 | **Fixed stop**: intrabar price hits stop (5% below entry) → exit at stop | `risk.initial_stop_loss_pct` | 0.05 |
| 3 | **Profit lock**: if any intraday high reaches entry + `profit_lock_target_r` × R, raise the stop to entry + (`profit_lock_target_r` - 1) × R after that day's close for subsequent sessions | `exit.profit_lock_target_r` | 3.0 |
| 4 | **Secondary profit lock**: if any intraday high reaches entry + `profit_lock_secondary_target_r` × R, raise the stop to entry + `profit_lock_secondary_stop_r` × R after that day's close for subsequent sessions | `exit.profit_lock_secondary_target_r` / `exit.profit_lock_secondary_stop_r` | 10.0 / 9.0 |
| 5 | **MA exit**: 2 consecutive closes below MA13 → exit at next open | `exit.ma_window` / `exit.consecutive_close_below_ma_days` | 13 / 2 |
| 6 | **Volume spike**: daily volume > 1.5× vol MA5 with no new closing high → exit at next open | `exit.vol_spike_ratio` / `exit.vol_spike_ma_window` | 1.5 / 5 |
| 7 | **Timeout**: if price hasn't reached 0.5R within 10 holding days → exit at next open | `exit.timeout_hold_days` / `exit.timeout_target_r` | 10 / 0.5 |
| 8 | **Forced end**: simulation cap reached → exit at last bar close | `exit.simulation_max_days_after_entry` | 50 |

Outputs:

- `trades.csv`: all candidates, including skipped names and all traded-stock details
- `summary.csv`: total R, win rate, max win, max loss, average R, median R, profit factor, average return, and counts

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Tushare token:

```bash
copy .env.example .env
```

Then edit `.env` and fill `TUSHARE_TOKEN`.

## Candidate File

Use a CSV like this:

```csv
ts_code,detect_date,note
600000.SH,2024-01-15,example
000001.SZ,20240122,example
```

## Run

### Backtest

Run the main strategy over a candidate list:

```bash
python run_backtest.py --candidates inputs/candidates_b.csv
```

Optional flags:

- `--config config/strategy.yaml`
- `--output-dir outputs`

Reports are written to `outputs/<timestamp>/` by default.

### Script Usage Reference

`run_backtest.py`

- Purpose: run the full gap-up pullback backtest and export `trades.csv` and `summary.csv`
- Input: candidate CSV with `ts_code` and `detect_date`

```bash
python run_backtest.py --candidates inputs/candidates_b.csv
python run_backtest.py --candidates inputs/candidates.csv --config config/strategy.yaml
python run_backtest.py --candidates inputs/candidates.csv --output-dir outputs/rerun
```

`run_detect_window_stats.py`

- Purpose: measure whether each candidate is up, down, or flat from `detect_date` through the close of the Nth trading day
- Output: `details.csv` and `summary.csv` under `outputs/detect_window_stats/<timestamp>/` by default

```bash
python run_detect_window_stats.py --candidates inputs/candidates.csv --window-trading-days 10
python run_detect_window_stats.py --candidates inputs/candidates.sample.csv --config config/strategy.yaml
python run_detect_window_stats.py --candidates inputs/candidates.csv --window-trading-days 5 --output-dir outputs/detect_window_stats/rerun
```

`run_analysis.py`

- Purpose: analyze traded rows in a `trades.csv` file to compare winners vs losers across entry features
- Input: `trades.csv` produced by `run_backtest.py`
- `--enrich` note: fetches minute-bar features, so it requires `TUSHARE_TOKEN` and minute data access

```bash
python run_analysis.py --trades outputs/<run>/trades.csv
python run_analysis.py --trades outputs/<run>/trades.csv --enrich
python run_analysis.py --trades outputs/<run>/trades.csv --enrich --config config/strategy.yaml
```

`run_compare_trades.py`

- Purpose: compare two `trades.csv` files by `ts_code` + `detect_date` and report differences in `exit_date`, `exit_time`, `exit_price`, `exit_reason`, `hold_days`, `pnl_r`, and `max_favorable_excursion_r`
- Input: any two `trades.csv` files produced by `run_backtest.py`

```bash
python run_compare_trades.py --left outputs/<run_a>/trades.csv --right outputs/<run_b>/trades.csv
python run_compare_trades.py --left outputs/<run_a>/trades.csv --right outputs/<run_b>/trades.csv --output outputs/compare_report.csv
python run_compare_trades.py --left outputs/<run_a>/trades.csv --right outputs/<run_b>/trades.csv --fail-on-diff
```

`run_fixed_stop_analysis.py`

- Purpose: analyze only `fixed_stop` exits and test simple filter rules against those failure cases
- Input: `trades.csv` produced by `run_backtest.py`
- `--enrich` note: fetches day-2 session highs from minute data, so it requires `TUSHARE_TOKEN`

```bash
python run_fixed_stop_analysis.py --trades outputs/<run>/trades.csv
python run_fixed_stop_analysis.py --trades outputs/<run>/trades.csv --enrich
python run_fixed_stop_analysis.py --trades outputs/<run>/trades.csv --enrich --config config/strategy.yaml
```

`run_capital_sim.py`

- Purpose: simulate fixed-capital position sizing for each traded row and report actual CNY P&L
- Input: `trades.csv` produced by `run_backtest.py`
- Capital rule: rounds to A-share 100-share lots

```bash
python run_capital_sim.py --trades outputs/<run>/trades.csv
python run_capital_sim.py --trades outputs/<run>/trades.csv --capital 20000
```

`run_peak_capital.py`

- Purpose: calculate the minimum principal required to support all overlapping trades under a fixed per-trade allocation
- Input: `trades.csv` produced by `run_backtest.py`

```bash
python run_peak_capital.py --trades outputs/<run>/trades.csv
python run_peak_capital.py --trades outputs/<run>/trades.csv --per-trade 20000
```

`run_peak_capital_v2.py`

- Purpose: calculate minimum principal under the same fixed per-trade allocation, plus one add-on buy per trade after 6 holding days if daily high exceeds entry + 1R
- Input: `trades.csv` produced by `run_backtest.py`
- Sizing: `--per-trade` controls the initial buy-leg capital; `--add-on-per-trade` controls the add-on buy-leg capital and defaults to the same amount as `--per-trade`
- Add-on rule: after 5 holding days, if the day's high is greater than `buy_price + 1R`, buy one add-on position at the next trading day open, rounded to A-share 100-share lots; that add-on leg exits at the same recorded exit as the original trade
- Extra outputs: add-on order CSV and daily win/loss equity CSV, written beside `trades.csv` by default

```bash
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --per-trade 20000
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --per-trade 15000 --add-on-per-trade 20000
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --initial-principal 132470
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --config config/strategy.yaml
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --add-on-csv outputs/<run>/add_on_orders.csv
python run_peak_capital_v2.py --trades outputs/<run>/trades.csv --daily-win-loss-csv outputs/<run>/daily_win_loss.csv
```

## Implementation Notes

- **Entry timing**: `buy_date` is the `entry.buy_on_nth_trading_day_after_detect`-th trading day after `detect_date` (default `1`). The previous trading day is also loaded so the gap can be confirmed with `detect_day_low > previous_day_high`.
- **Buy price source**: the strategy buys at the exact `market.buy_time` 1-minute bar close (default `14:30`) from local minute data under `inputs/a_share_1_min/` / cache-backed loaders. If that bar is missing, the run raises an error for that candidate.
- **Pullback metric**: `pullback_ratio` is still computed and exported in `entry_notes`, but it is not currently used as an entry filter. With `pullback_reference: body`, the denominator is the detect-day bullish body `close - open`; if that body is non-positive, it falls back to the full detect-day range `high - low`.
- **Active strength metric**: the current price-follow-through filter is `price_up_ratio = (buy_price - detect_close) / (detect_day_low - previous_day_high)`, and it must be at least `entry.min_price_up_ratio`.
- **VWAP rule**: VWAP is session-cumulative from `09:30` through `14:30`. The rule requires both `buy_price >= vwap_at_1430 * (1 + entry.vwap_min_buffer)` and a positive VWAP slope from the first bar at or after `14:00` to the `14:30` bar.
- **Afternoon low rule**: `no_new_low_after_1400` is a strict comparison. The minimum low in `14:00–14:30` must be greater than, not equal to, the minimum low in `09:30–13:59`.
- **Intrabar stop path**: the default intrabar evaluation order is `O → H → L → C` because `market.intrabar_order` is `ohlc` in the current config. If set to `olhc`, the stop check order becomes `O → L → H → C`. Profit-lock stop raises are not applied intraday; the strategy only lifts the stop after a day whose intraday high reaches either `entry + exit.profit_lock_target_r × R` or `entry + exit.profit_lock_secondary_target_r × R`, and the new stop becomes `entry + (exit.profit_lock_target_r - 1) × R` or `entry + exit.profit_lock_secondary_stop_r × R` from the next session onward.
- **Lot sizing**: the capital simulation scripts still round A-share orders to 100-share lots.

## Tushare Interfaces Used

- `trade_cal` for trading days
- `daily` for daily OHLCV
- `stk_mins` for historical minute bars

Official references:

- Tushare `trade_cal`: https://tushare.pro/document/2?doc_id=26
- Tushare `daily`: https://tushare.pro/document/2?doc_id=27
- Tushare `stk_mins`: https://tushare.pro/document/2?doc_id=370
