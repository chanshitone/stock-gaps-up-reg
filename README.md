# Stock Gap Pullback Regression Workspace

This workspace runs a regression test for A-share gap-up candidates using Tushare daily and minute data, then exports trade-level details and summary statistics to CSV.

## What It Does

Input:

- Candidate CSV with `ts_code` and `detect_date`

### Entry Conditions

Buy at the **14:30** 1-minute bar close on the **next trading day** after `detect_date` (day2). All of the following must be satisfied:

| # | Condition | Config Key | Default |
|---|-----------|-----------|---------|
| 1 | **Gap confirmed**: detect-day low > previous-day high | — | structural |
| 2 | **Volume shrink**: day2 cumulative volume up to 14:30 < detect-day full volume × `volume_fraction` | `entry.volume_fraction` | 0.75 |
| 3 | **Day1 strength** (OR): detect-day change% ≥ `day1_min_change_pct` **or** close strength ≥ `day1_min_close_strength` | `entry.day1_min_change_pct` / `entry.day1_min_close_strength` | 0.01 / 0.5 |
| 4 | **Price-up strength**: `(buy_price - detect_close) / (detect_low - prev_high) ≥ min_price_up_ratio` | `entry.min_price_up_ratio` | 0.8 |
| 5 | **VWAP filter**: buy_price ≥ session VWAP × (1 + `vwap_min_buffer`) **and** VWAP rising from 14:00 to 14:30 | `entry.vwap_min_buffer` | 0.002 |
| 6 | **No new low after 14:00**: day2 low in 14:00–14:30 > day2 low in 09:30–13:59 | — | structural |

### Exit Conditions

Exits are checked in priority order each trading day:

| # | Condition | Config Key | Default |
|---|-----------|-----------|---------|
| 1 | **Gap-down stop**: day open ≤ stop price → exit at open | — | — |
| 2 | **Fixed stop**: intrabar price hits stop (5% below entry) → exit at stop | `risk.initial_stop_loss_pct` | 0.05 |
| 3 | **MA exit**: 2 consecutive closes below MA13 → exit at next open | `exit.ma_window` / `exit.consecutive_close_below_ma_days` | 13 / 2 |
| 4 | **Volume spike**: daily volume > 1.5× vol MA5 with no new closing high → exit at next open | `exit.vol_spike_ratio` / `exit.vol_spike_ma_window` | 1.5 / 5 |
| 5 | **Timeout**: if price hasn't reached 0.5R within 10 holding days → exit at next open | `exit.timeout_hold_days` / `exit.timeout_target_r` | 10 / 0.5 |
| 6 | **Forced end**: simulation cap reached → exit at last bar close | `exit.simulation_max_days_after_entry` | 50 |

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

```bash
python run_backtest.py --candidates inputs/candidates_b.csv
```

Reports are written to `outputs/<timestamp>/`.

To count how many candidates are up or down from `detect_date` through the close of the 10th trading day, run:

```bash
python run_detect_window_stats.py --candidates inputs/candidates.csv --window-trading-days 10
```

This script writes `details.csv` and `summary.csv` under `outputs/detect_window_stats/<timestamp>/`.

## Important Assumptions

- **Day 2** is the next trading day after `detect_date`.
- **14:30 market price** is the `14:30` 1-minute bar close from local parquet data (`inputs/a_share_1_min/`).
- **Pullback depth** defaults to `(detect_close - buy_price) / detect_day_body`. If detect-day body ≤ 0, falls back to high–low range.
- **Gap retention vs. detect close** = `(detect_close - buy_price) / (detect_day_low - previous_day_high)`. More negative means the day2 buy price is further above the detect-day close; closer to 0 means more of that strength has faded.
- **VWAP** is session-cumulative from 09:30: `sum(close_i × vol_i) / sum(vol_i)`. "Rising" means VWAP at the last 14:30 bar > VWAP at the first 14:00 bar.
- **No new low after 14:00**: the minimum low of 14:00–14:30 bars must be strictly higher than the minimum low of 09:30–13:59 bars.
- **Intrabar order** defaults to `O → H → L → C`. Change via `market.intrabar_order` in `config/strategy.yaml`.
- **A-share lot size**: capital simulation scripts round shares to the nearest 100-share lot.

## Tushare Interfaces Used

- `trade_cal` for trading days
- `daily` for daily OHLCV
- `stk_mins` for historical minute bars

Official references:

- Tushare `trade_cal`: https://tushare.pro/document/2?doc_id=26
- Tushare `daily`: https://tushare.pro/document/2?doc_id=27
- Tushare `stk_mins`: https://tushare.pro/document/2?doc_id=370
