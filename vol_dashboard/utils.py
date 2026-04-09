"""
utils.py — Shared formatting helpers.
"""

import numpy as np


def annualise(daily_pct: float) -> float:
    """Convert daily vol percent to annualised vol percent."""
    return daily_pct * (252 ** 0.5)


def fmt_vol(v, decimals=3) -> str:
    """Format a vol value (already in %) as a string, or '—' if NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}%"
