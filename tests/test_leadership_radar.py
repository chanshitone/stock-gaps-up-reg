from datetime import date
import unittest

import numpy as np
import pandas as pd

from src.stock_gaps_reg.leadership_radar import (
    LeadershipConfig,
    _classify_leadership_trend,
    calculate_leadership_radar,
    map_stocks_to_sw_l3,
)


def history(values: list[float], amounts: list[float] | None = None) -> pd.DataFrame:
    if amounts is None:
        amounts = [1000.0] * len(values)
        amounts[-1] = 1500.0
    return pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=len(values), freq="B"),
            "close": values,
            "amount": amounts,
        }
    )


class FakeLeadershipClient:
    def __init__(self) -> None:
        self.memberships = {
            "000001.SZ": pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "name": "One",
                        "l1_code": "801000.SI",
                        "l1_name": "L1",
                        "l2_code": "801010.SI",
                        "l2_name": "L2",
                        "l3_code": "850001.SI",
                        "l3_name": "Strong L3",
                        "in_date": "20200101",
                        "out_date": "",
                    }
                ]
            ),
            "000002.SZ": pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "name": "Two",
                        "l1_code": "801000.SI",
                        "l1_name": "L1",
                        "l2_code": "801020.SI",
                        "l2_name": "L2B",
                        "l3_code": "850002.SI",
                        "l3_name": "Weak L3",
                        "in_date": "20200101",
                        "out_date": "",
                    }
                ]
            ),
            "000003.SZ": pd.DataFrame(
                [
                    {
                        "ts_code": "000003.SZ",
                        "name": "Three",
                        "l1_code": "801000.SI",
                        "l1_name": "L1",
                        "l2_code": "801010.SI",
                        "l2_name": "L2",
                        "l3_code": "850001.SI",
                        "l3_name": "Strong L3",
                        "in_date": "20200101",
                        "out_date": "",
                    }
                ]
            ),
        }

    def get_sw_memberships(self, ts_code: str) -> pd.DataFrame:
        return self.memberships.get(ts_code, pd.DataFrame())

    def get_sw_l3_members(self, l3_code: str) -> pd.DataFrame:
        frames = [
            frame
            for frame in self.memberships.values()
            if not frame.empty and str(frame.iloc[0]["l3_code"]) == l3_code
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def get_sw_daily(self, index_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        if index_code == "850001.SI":
            return history(np.linspace(100.0, 180.0, 90).tolist())
        return history(np.linspace(100.0, 95.0, 90).tolist())

    def get_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        if ts_code in {"000001.SZ", "000003.SZ"}:
            return history(np.linspace(100.0, 180.0, 90).tolist())
        return history(np.linspace(100.0, 95.0, 90).tolist())

    def get_index_daily(self, ts_code: str, start_date: date, end_date: date) -> pd.DataFrame:
        return history(np.linspace(100.0, 105.0, 90).tolist())


class LeadershipRadarTests(unittest.TestCase):
    def test_maps_stocks_to_sw_l3(self) -> None:
        stocks, unmatched = map_stocks_to_sw_l3(["000001.SZ"], FakeLeadershipClient(), date(2025, 5, 1))
        self.assertTrue(unmatched.empty)
        self.assertEqual(stocks.iloc[0]["sw_l3_code"], "850001.SI")

    def test_stronger_l3_scores_above_weaker_l3(self) -> None:
        sectors, stocks, unmatched, trends = calculate_leadership_radar(
            ["000001.SZ", "000002.SZ"], FakeLeadershipClient(), date(2025, 5, 1), LeadershipConfig()
        )
        self.assertTrue(unmatched.empty)
        self.assertEqual(sectors.iloc[0]["sw_l3_code"], "850001.SI")
        strong_score = float(stocks.loc[stocks["ts_code"] == "000001.SZ", "industry_score"].iloc[0])
        weak_score = float(stocks.loc[stocks["ts_code"] == "000002.SZ", "industry_score"].iloc[0])
        self.assertGreater(strong_score, weak_score)
        self.assertEqual(int(sectors.loc[sectors["sw_l3_code"] == "850001.SI", "industry_member_count"].iloc[0]), 2)
        self.assertEqual(int(sectors.loc[sectors["sw_l3_code"] == "850001.SI", "metric_stock_count"].iloc[0]), 2)
        self.assertIn("anchor_status_score", sectors.columns)
        self.assertIn("concentration_ratio", sectors.columns)
        self.assertIn("leadership_trend", sectors.columns)
        self.assertIn("leadership_trend", stocks.columns)
        self.assertIn(
            sectors.loc[sectors["sw_l3_code"] == "850001.SI", "leadership_trend"].iloc[0],
            {"趋势向上", "趋势向下", "趋势平稳"},
        )
        self.assertIn("industry_score", trends.columns)
        self.assertFalse(trends.empty)

    def test_trend_classifier_uses_smoothed_score_path(self) -> None:
        config = LeadershipConfig()
        up = pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=30, freq="B"),
                "industry_score": list(range(30)),
            }
        )
        down = pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=30, freq="B"),
                "industry_score": list(range(30, 0, -1)),
            }
        )
        flat = pd.DataFrame(
            {
                "trade_date": pd.date_range("2025-01-01", periods=30, freq="B"),
                "industry_score": [60.0] * 30,
            }
        )
        self.assertEqual(_classify_leadership_trend(up, config), "趋势向上")
        self.assertEqual(_classify_leadership_trend(down, config), "趋势向下")
        self.assertEqual(_classify_leadership_trend(flat, config), "趋势平稳")


if __name__ == "__main__":
    unittest.main()
