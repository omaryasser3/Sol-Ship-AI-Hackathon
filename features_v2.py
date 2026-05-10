"""
features_v2.py — drop-in replacement for features.py.

Changes vs features.py:
  + is_spike_likely  : binary, 1 when within 2 h (8 steps) of a recent spike
  + load_zscore_lag1 : z-score of load_s1 relative to slot-month profile
    (complements load_zscore_month which uses the current step's slot stats)

All other feature definitions are identical; causality invariant is preserved.
"""
from features import (
    DEFAULT_LAGS, DEFAULT_ROLL_WINDOWS,
    compute_slot_stats, compute_slot_month_stats, compute_slot_dow_stats,
    build_inference_features,
)
import features as _base
import numpy as np
import pandas as pd


def build_features(df, lags=None, roll_windows=None, horizon=1,
                   holiday_set=None, include_pv=True,
                   slot_stats=None, slot_month_stats=None, slot_dow_stats=None):
    """Delegates to features.build_features then appends v2 columns."""
    X, y = _base.build_features(
        df, lags=lags, roll_windows=roll_windows, horizon=horizon,
        holiday_set=holiday_set, include_pv=include_pv,
        slot_stats=slot_stats, slot_month_stats=slot_month_stats,
        slot_dow_stats=slot_dow_stats,
    )

    # ── v2 spike-clustering features (causal — use same load_s1 = load.shift(1)) ──
    load   = df["load_kw"]
    load_s1 = load.shift(1).reindex(X.index)

    # Reconstruct slot-month mean/std on the X rows only (already in X as columns)
    sm_mean = X["slot_month_mean"].values
    sm_std  = np.where(X["slot_month_std"].values > 0.05,
                       X["slot_month_std"].values, 0.05)

    # steps_since_spike is already in X; derive is_spike_likely from it
    if "steps_since_spike" in X.columns:
        X = X.copy()
        X["is_spike_likely"]  = (X["steps_since_spike"] < 8).astype(np.int8)
        X["load_zscore_lag1"] = ((load_s1.values - sm_mean) / sm_std)

    return X, y
