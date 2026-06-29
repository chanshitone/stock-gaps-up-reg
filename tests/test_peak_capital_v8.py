import unittest

import pandas as pd

from run_peak_capital_v8 import (
    ACTIVE_THEME,
    HOT_REBOUND,
    LEADERSHIP,
    STRONG_LEADERSHIP,
    TREND_STABLE,
    TREND_UP,
    _apply_leadership_gate_to_sizing,
    _coerce_bool,
    _disabled_leadership_result,
    _leadership_allows_entry,
)


class PeakCapitalV8LeadershipTests(unittest.TestCase):
    def test_leadership_gate_matches_plan(self) -> None:
        self.assertTrue(_leadership_allows_entry(STRONG_LEADERSHIP, "\u8d8b\u52bf\u5411\u4e0b"))
        self.assertTrue(_leadership_allows_entry(LEADERSHIP, TREND_UP))
        self.assertTrue(_leadership_allows_entry(LEADERSHIP, TREND_STABLE))
        self.assertFalse(_leadership_allows_entry(LEADERSHIP, "\u8d8b\u52bf\u5411\u4e0b"))
        self.assertTrue(_leadership_allows_entry(ACTIVE_THEME, TREND_UP))
        self.assertTrue(_leadership_allows_entry(HOT_REBOUND, TREND_UP))
        self.assertFalse(_leadership_allows_entry(ACTIVE_THEME, TREND_STABLE))
        self.assertFalse(_leadership_allows_entry("\u666e\u901a / \u4f11\u7720", TREND_UP))

    def test_gate_zeroes_blocked_initial_sizing(self) -> None:
        frame = pd.DataFrame(
            {
                "leadership_buyable": [True, False],
                "shares": [100, 200],
                "actual_cost": [1000.0, 2000.0],
                "exit_proceeds": [1100.0, 2200.0],
                "position_bucket": ["strong", "mid"],
            }
        )

        gated = _apply_leadership_gate_to_sizing(frame)

        self.assertEqual(int(gated.loc[0, "shares"]), 100)
        self.assertEqual(int(gated.loc[1, "shares"]), 0)
        self.assertEqual(float(gated.loc[1, "actual_cost"]), 0.0)
        self.assertEqual(float(gated.loc[1, "exit_proceeds"]), 0.0)
        self.assertEqual(gated.loc[1, "position_bucket"], "mid+leadership_blocked")

    def test_disabled_filter_allows_rows_through(self) -> None:
        frame = pd.DataFrame({"ts_code": ["000001.SZ"], "buy_date": [pd.Timestamp("2025-05-07")]})

        disabled = _disabled_leadership_result(frame)

        self.assertTrue(bool(disabled.loc[0, "leadership_buyable"]))
        self.assertEqual(disabled.loc[0, "leadership_block_reason"], "leadership radar disabled")

    def test_reused_csv_bool_parser_handles_false_strings(self) -> None:
        self.assertTrue(_coerce_bool("True"))
        self.assertTrue(_coerce_bool("1"))
        self.assertFalse(_coerce_bool("False"))
        self.assertFalse(_coerce_bool(""))


if __name__ == "__main__":
    unittest.main()
