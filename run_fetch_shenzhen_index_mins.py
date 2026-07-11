from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_TS_CODE = "399001.SZ"
DEFAULT_FREQ = "1min"
DEFAULT_START = "20260601 09:00:00"
DEFAULT_END = "20260630 19:00:00"
DEFAULT_OUTPUT = Path("outputs/index_mins/399001_SZ_1min_202606.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Shenzhen Component Index 1-minute history for June 2026 via MiniShare."
    )
    parser.add_argument(
        "--token",
        default=None,
        help="MiniShare auth token. Defaults to MINISHARE_TOKEN from .env or the shell environment.",
    )
    parser.add_argument(
        "--ts-code",
        default=DEFAULT_TS_CODE,
        help=f"Index code to fetch (default: {DEFAULT_TS_CODE})",
    )
    parser.add_argument(
        "--freq",
        default=DEFAULT_FREQ,
        help=f"Minute frequency such as 1min/5min/15min (default: {DEFAULT_FREQ})",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START,
        help=f"Inclusive start timestamp in YYYYMMDD HH:MM:SS format (default: {DEFAULT_START})",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END,
        help=f"Inclusive end timestamp in YYYYMMDD HH:MM:SS format (default: {DEFAULT_END})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV output path (default: {DEFAULT_OUTPUT.as_posix()})",
    )
    return parser.parse_args()


def resolve_token(cli_token: str | None) -> str:
    load_dotenv()
    token = cli_token or os.getenv("MINISHARE_TOKEN")
    if not token:
        raise ValueError("Missing MiniShare token. Pass --token or set MINISHARE_TOKEN in .env.")
    return token.strip()


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)

    try:
        import minishare as ms
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "minishare is not installed in the active Python environment. Run `pip install -r requirements.txt`."
        ) from exc

    df = ms.pro_api(token).idx_mins(
        ts_code=args.ts_code,
        freq=args.freq,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    if df.empty:
        raise ValueError(
            f"No data returned for {args.ts_code} between {args.start_date} and {args.end_date}."
        )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sort_columns = [column for column in ("trade_time", "ts_code") if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns).reset_index(drop=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(df)
    print(f"Rows written: {len(df)}")
    print(f"Output CSV : {output_path}")


if __name__ == "__main__":
    main()