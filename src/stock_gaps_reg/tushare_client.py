from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
import pandas as pd
import pyarrow.dataset as ds
import tushare as ts


@dataclass
class TushareClient:
    cache_dir: Path
    exchange: str = "SSE"
    pause_seconds: float = 0.2
    rate_limit_retry_seconds: float = 1.0
    max_retries: int = 1

    def __post_init__(self) -> None:
        load_dotenv()
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise ValueError("TUSHARE_TOKEN is not set. Put it in your environment or .env file.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.daily_cache_dir = self.cache_dir / "daily"
        self.minute_cache_dir = self.cache_dir / "minute"
        self.calendar_cache_dir = self.cache_dir / "calendar"
        self.daily_cache_dir.mkdir(parents=True, exist_ok=True)
        self.minute_cache_dir.mkdir(parents=True, exist_ok=True)
        self.calendar_cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_minute_data_dir = Path(__file__).resolve().parents[2] / "inputs" / "a_share_1_min"
        self.pro = ts.pro_api(token)

    def get_trade_calendar(self, start_date: date, end_date: date) -> pd.DataFrame:
        cache_path = self.calendar_cache_dir / f"{self.exchange}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, dtype={"cal_date": str, "pretrade_date": str})

        frame = self._call_api(
            "trade_cal",
            self.pro.trade_cal,
            exchange=self.exchange,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            is_open="1",
        )
        if frame.empty:
            raise ValueError(f"No trade calendar returned for {start_date} to {end_date}.")
        frame = frame.sort_values("cal_date").reset_index(drop=True)
        frame.to_csv(cache_path, index=False)
        return frame

    def list_trade_days(self, start_date: date, end_date: date) -> list[date]:
        frame = self.get_trade_calendar(start_date, end_date)
        return [datetime.strptime(value, "%Y%m%d").date() for value in frame["cal_date"].tolist()]

    def get_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        cache_path = self.daily_cache_dir / f"{ts_code}_{start_date:%Y%m%d}_{end_date:%Y%m%d}.csv"
        if cache_path.exists():
            return self._load_daily_csv(cache_path)

        frame = self._call_api(
            "daily",
            self.pro.daily,
            ts_code=ts_code,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
        if frame.empty:
            raise ValueError(f"No daily data returned for {ts_code} between {start_date} and {end_date}.")
        frame = frame.sort_values("trade_date").reset_index(drop=True)
        frame.to_csv(cache_path, index=False)
        return self._normalize_daily(frame)

    def get_daily_with_ma(self, ts_code: str, start_date: date, end_date: date, ma_window: int, vol_ma_window: int = 5) -> pd.DataFrame:
        history_start = start_date - timedelta(days=max(ma_window, vol_ma_window) * 3)
        frame = self.get_daily(ts_code, history_start, end_date).copy()
        frame["ma"] = frame["close"].rolling(ma_window).mean()
        frame["vol_ma"] = frame["vol"].rolling(vol_ma_window).mean()
        return frame[frame["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].reset_index(drop=True)

    def get_minutes_for_day(self, ts_code: str, trade_date: date, freq: str = "1min") -> pd.DataFrame:
        cache_path = self.minute_cache_dir / f"{ts_code}_{trade_date:%Y%m%d}_{freq}.csv"
        if cache_path.exists():
            return self._load_minute_csv(cache_path)

        frame = self._load_minutes_from_parquet(ts_code, trade_date, freq)
        if frame.empty:
            raise ValueError(f"No minute data returned for {ts_code} on {trade_date}.")
        frame = frame.sort_values("trade_time").reset_index(drop=True)
        frame.to_csv(cache_path, index=False)
        return self._normalize_minutes(frame)

    def _load_minutes_from_parquet(self, ts_code: str, trade_date: date, freq: str) -> pd.DataFrame:
        if freq != "1min":
            raise ValueError(f"Local parquet minute data only supports freq='1min', got {freq!r}.")
        if not self.local_minute_data_dir.exists():
            raise FileNotFoundError(f"Local minute parquet directory not found: {self.local_minute_data_dir}")

        parquet_files = sorted(self.local_minute_data_dir.glob(f"*_{trade_date:%Y-%m}_batch*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet shards found for {trade_date:%Y-%m} in {self.local_minute_data_dir}"
            )

        parquet_code = self._to_parquet_code(ts_code)
        start_dt = datetime.combine(trade_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        dataset = ds.dataset([str(path) for path in parquet_files], format="parquet")
        table = dataset.to_table(
            columns=["time", "code", "open", "close", "high", "low", "volume", "money"],
            filter=(ds.field("code") == parquet_code)
            & (ds.field("time") >= start_dt)
            & (ds.field("time") < end_dt),
        )
        frame = table.to_pandas()
        if frame.empty:
            return pd.DataFrame(columns=["trade_time", "ts_code", "open", "close", "high", "low", "vol", "amount"])

        frame = frame.rename(columns={"time": "trade_time", "code": "ts_code", "volume": "vol", "money": "amount"})
        return frame[["trade_time", "ts_code", "open", "close", "high", "low", "vol", "amount"]]

    @staticmethod
    def _to_parquet_code(ts_code: str) -> str:
        normalized = ts_code.strip().upper()
        if normalized.endswith(".SH"):
            return f"{normalized[:-3]}.XSHG"
        if normalized.endswith(".SZ"):
            return f"{normalized[:-3]}.XSHE"
        if normalized.endswith((".XSHG", ".XSHE")):
            return normalized
        raise ValueError(f"Unsupported ts_code format for local parquet minute lookup: {ts_code}")

    def _load_daily_csv(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, dtype={"trade_date": str})
        return self._normalize_daily(frame)

    def _load_minute_csv(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        return self._normalize_minutes(frame)

    def _call_api(self, endpoint: str, func: Callable[..., Any], **kwargs: Any) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                frame = func(**kwargs)
                time.sleep(self.pause_seconds)
                return frame
            except Exception as exc:
                last_error = exc
                self._raise_if_permission_error(endpoint, exc)
                if self._should_retry_rate_limit(exc, attempt):
                    time.sleep(self.rate_limit_retry_seconds)
                    continue
                raise
        assert last_error is not None
        raise RuntimeError(f"{endpoint} failed after {self.max_retries} attempts: {last_error}")

    def _raise_if_permission_error(self, endpoint: str, exc: Exception) -> None:
        message = str(exc)
        if "每天最多访问该接口" in message:
            raise PermissionError(f"{endpoint} daily quota exceeded or permission too low: {message}") from exc
        if "没有权限" in message or "积分" in message:
            raise PermissionError(f"{endpoint} permission denied: {message}") from exc

    def _should_retry_rate_limit(self, exc: Exception, attempt: int) -> bool:
        message = str(exc)
        if "每分钟最多访问该接口" not in message and "rate limit" not in message.lower():
            return False
        return attempt < self.max_retries

    @staticmethod
    def _normalize_daily(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], format="%Y%m%d")
        numeric_columns = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
        for column in numeric_columns:
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized.sort_values("trade_date").reset_index(drop=True)

    @staticmethod
    def _normalize_minutes(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized["trade_time"] = pd.to_datetime(normalized["trade_time"])
        numeric_columns = ["open", "high", "low", "close", "vol", "amount"]
        for column in numeric_columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        return normalized.sort_values("trade_time").reset_index(drop=True)
