import unittest
from decimal import Decimal

from strategy_rules import evaluate_cancel_decision


class TestStrategyRules(unittest.TestCase):
    def test_no_cancel_before_30s_window_even_if_gap_small(self):
        reason, cancel = evaluate_cancel_decision(
            200,
            Decimal("0.0008"),
            force_cancel_seconds=10,
            gap_keep_threshold=Decimal("0.0008"),
            gap_check_start_seconds=30,
        )
        self.assertEqual(reason, "before_gap_window")
        self.assertFalse(cancel)

    def test_no_cancel_before_30s_window_even_if_gap_large(self):
        reason, cancel = evaluate_cancel_decision(
            200,
            Decimal("0.0009"),
            force_cancel_seconds=10,
            gap_keep_threshold=Decimal("0.0008"),
            gap_check_start_seconds=30,
        )
        self.assertEqual(reason, "before_gap_window")
        self.assertFalse(cancel)

    def test_keep_when_gap_le_threshold_and_20s(self):
        reason, cancel = evaluate_cancel_decision(
            20,
            Decimal("0.0007"),
            force_cancel_seconds=10,
            gap_keep_threshold=Decimal("0.0008"),
            gap_check_start_seconds=30,
        )
        self.assertEqual(reason, "keep_by_gap")
        self.assertFalse(cancel)

    def test_cancel_when_gap_gt_threshold_and_20s(self):
        reason, cancel = evaluate_cancel_decision(
            20,
            Decimal("0.0010"),
            force_cancel_seconds=10,
            gap_keep_threshold=Decimal("0.0008"),
            gap_check_start_seconds=30,
        )
        self.assertEqual(reason, "cancel_by_gap")
        self.assertTrue(cancel)

    def test_force_cancel_when_end_10_or_less(self):
        for secs in (10, 9, 0):
            reason, cancel = evaluate_cancel_decision(
                secs,
                Decimal("0.0"),
                force_cancel_seconds=10,
                gap_keep_threshold=Decimal("0.0008"),
                gap_check_start_seconds=30,
            )
            self.assertEqual(reason, "force_cancel")
            self.assertTrue(cancel)

    def test_mapping_math_precision(self):
        ptb_open = Decimal("100.1234")
        binance_open = Decimal("100.0000")
        offset = ptb_open - binance_open

        ptb_now = Decimal("100.3000")
        mapped_ptb = ptb_now - offset
        current = Decimal("100.1700")
        gap = abs(current - mapped_ptb)

        self.assertEqual(offset, Decimal("0.1234"))
        self.assertEqual(mapped_ptb, Decimal("100.1766"))
        self.assertEqual(gap, Decimal("0.0066"))


if __name__ == "__main__":
    unittest.main()
