# Stock Gap Pullback Regression Workspace

This workspace runs a regression test for A-share gap-up candidates using Tushare daily and minute data, then exports trade-level details and summary statistics to CSV.

## What It Does

Input:

- Candidate CSV with `ts_code` and `detect_date`

Entry logic on the second trading day after the detect date:

- 14:30 market-entry proxy using the `14:30` 1-minute close
- Pullback depth must stay within the configured fraction of the detect-day move
- day2 14:30 cumulative volume must be below the configured fraction of detect-day full-session volume
- Must show either a long lower shadow or a `14:00-14:30` stabilization pattern: `14:30 close >= 14:00 open` and no lower low after the post-`14:00` low is made
- Gap must remain unfilled by `14:30`

Exit logic:

- Initial stop is the tighter of `buy_price - 5%` and `gap_fill_price - 1%`
- At `1R`, stop moves to entry
- At `2R`, stop moves to `entry + 1R`
- Two consecutive closes below MA5 trigger exit at next open
- If price does not reach `0.5R` within 10 holding days, exit at day-11 open

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
python run_backtest.py --candidates inputs/candidates.sample.csv
```

Reports are written to `outputs/<timestamp>/`.

To count how many candidates are up or down from `detect_date` through the close of the 10th trading day, run:

```bash
python run_detect_window_stats.py --candidates inputs/candidates.csv --window-trading-days 10
```

This script writes `details.csv` and `summary.csv` under `outputs/detect_window_stats/<timestamp>/`.

## Important Assumptions

- `day 2` is interpreted as the second trading day after `detect_date`.
- `window_trading_days=10` is interpreted as counting `detect_date` itself as day 1, then measuring through the close of the 10th trading day.
- `14:30 market price` is approximated with the `14:30` 1-minute bar close from `stk_mins`.
- Pullback depth defaults to `(detect_close - buy_price) / detect_day_body`. If detect-day body is not positive, the script falls back to detect-day high-low range.
- `gap not filled` is interpreted as `day2 intraday low > previous trading day high`.
- `long lower shadow` is approximated from the `14:00-14:30` intraday window.
- `stabilized after 14:00` means the `14:30` close is at or above the `14:00` open, and once the lowest low after `14:00` is set, later bars do not print a lower low before `14:30`.
- Intrabar order defaults to `O -> H -> L -> C`. You can change it in `config/strategy.yaml`.

## Tushare Interfaces Used

- `trade_cal` for trading days
- `daily` for daily OHLCV
- `stk_mins` for historical minute bars

Official references:

- Tushare `trade_cal`: https://tushare.pro/document/2?doc_id=26
- Tushare `daily`: https://tushare.pro/document/2?doc_id=27
- Tushare `stk_mins`: https://tushare.pro/document/2?doc_id=370
