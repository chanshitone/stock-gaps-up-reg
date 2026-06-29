from datetime import date
import unittest

import numpy as np
import pandas as pd

from src.stock_gaps_reg.sector_radar import RadarConfig, calculate_sector_metrics, select_membership


def history(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=len(values), freq="B"),
            "close": values,
        }
    )


class SectorMetricsTests(unittest.TestCase):
    def test_steady_uptrend_is_phase_a(self) -> None:
        metrics = calculate_sector_metrics(history(np.linspace(100.0, 180.0, 80).tolist()), RadarConfig())
        self.assertEqual(metrics["phase"], "A")
        self.assertGreater(metrics["score"], 0)

    def test_steady_downtrend_is_phase_c(self) -> None:
        metrics = calculate_sector_metrics(history(np.linspace(180.0, 100.0, 80).tolist()), RadarConfig())
        self.assertEqual(metrics["phase"], "C")

    def test_insufficient_history_fails_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 60"):
            calculate_sector_metrics(history([100.0] * 30), RadarConfig())


class MembershipTests(unittest.TestCase):
    def test_selects_membership_active_on_date(self) -> None:
        memberships = pd.DataFrame(
            [
                {"l1_code": "old", "in_date": "20100101", "out_date": "20201231"},
                {"l1_code": "new", "in_date": "20210101", "out_date": ""},
            ]
        )
        selected = select_membership(memberships, date(2025, 1, 1))
        self.assertIsNotNone(selected)
        self.assertEqual(selected["l1_code"], "new")

    def test_returns_none_outside_membership_periods(self) -> None:
        memberships = pd.DataFrame([{"l1_code": "old", "in_date": "20100101", "out_date": "20201231"}])
        self.assertIsNone(select_membership(memberships, date(2025, 1, 1)))


if __name__ == "__main__":
    unittest.main()
