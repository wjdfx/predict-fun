from __future__ import annotations

from decimal import Decimal
from typing import Optional, Tuple


def evaluate_cancel_decision(
    end_secs: Optional[int],
    gap: Optional[Decimal],
    *,
    force_cancel_seconds: int,
    gap_keep_threshold: Decimal,
    gap_check_start_seconds: int,
) -> Tuple[str, bool]:
    if end_secs is None:
        return "missing_end_time", False
    if end_secs <= force_cancel_seconds:
        return "force_cancel", True
    if end_secs > gap_check_start_seconds:
        return "before_gap_window", False
    if gap is None:
        return "missing_gap", False
    if gap <= gap_keep_threshold:
        return "keep_by_gap", False
    return "cancel_by_gap", True
