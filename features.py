"""
Feature engineering for the load forecaster.

Builds a tabular feature matrix from the load_kw time series + calendar.
All features are CAUSAL: at predicted timestep t, only data from t-1 and earlier is used.

Three feature families:
  1. Calendar       — universal across sites, drives most of the variance
  2. Lag features   — last-value-ish, the workhorse for short horizons
  3. Rolling stats  — smoothed local mean / variability

Multi-horizon strategy: this builder accepts a `horizon` argument that shifts
the target by `horizon` steps. To produce predictions for steps t+1..t+H,
train H separate models (one per horizon) — see forecaster.train_multi_horizon.
"""
import numpy as np
import pandas as pd
from tariff import assign_tariff_band, get_italian_holidays


# Lags expressed in 15-min steps
DEFAULT_LAGS = [1, 2, 4, 8, 96, 96 * 2, 672]   # 15min, 30min, 1h, 2h, 1d, 2d, 1w
DEFAULT_ROLL_WINDOWS = [4, 16, 96, 672]         # 1h, 4h, 1d, 1w


def build_features(df, lags=None, roll_windows=None, horizon=1,
                   holiday_set=None, include_pv=True):
    """
    Build a feature matrix and target for LOAD forecasting.

    Parameters
    ----------
    df          : pd.DataFrame, indexed by 15-min timestamps, must contain 'load_kw'
                  (and 'pv_kw' if include_pv=True). Other columns optional.
    lags        : list of int, lag steps for load. Default DEFAULT_LAGS.
    roll_windows: list of int, rolling windows (in steps).
    horizon     : int ≥ 1. Target is load_kw shifted by -horizon steps,
                  i.e. y[t] = load_kw[t + horizon].
                  Returned X is rows where y is non-NaN.
    holiday_set : optional set of date objects. If None, computed from df.index years.
    include_pv  : whether to include PV columns as features.

    Returns
    -------
    X : pd.DataFrame, feature matrix
    y : pd.Series, target (load at t+horizon)
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if roll_windows is None:
        roll_windows = DEFAULT_ROLL_WINDOWS

    if "load_kw" not in df.columns:
        raise ValueError("df must contain 'load_kw'")

    out = pd.DataFrame(index=df.index)

    # === Calendar features ===
    ts = df.index
    out["hour"]      = ts.hour
    out["minute"]    = ts.minute
    out["minute_of_day"] = ts.hour * 4 + ts.minute // 15
    out["dow"]       = ts.dayofweek
    out["is_weekend"] = (ts.dayofweek >= 5).astype(int)
    out["month"]     = ts.month
    out["weekofyear"] = ts.isocalendar().week.astype(int).values
    out["dayofyear"] = ts.dayofyear

    if holiday_set is None:
        years = sorted(set(ts.year))
        holiday_set = get_italian_holidays(years)
    out["is_holiday"] = pd.Series(
        [d in holiday_set for d in ts.date], index=ts).astype(int)

    # Italian Ferragosto vacation: Aug 10-20 special flag
    out["is_ferragosto"] = (
        (ts.month == 8) & (ts.day >= 10) & (ts.day <= 20)
    ).astype(int)

    # Cyclic encodings (sin/cos)
    out["hour_sin"]  = np.sin(2 * np.pi * out["minute_of_day"] / 96)
    out["hour_cos"]  = np.cos(2 * np.pi * out["minute_of_day"] / 96)
    out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)

    # === Tariff band as feature ===
    bands = assign_tariff_band(ts, holiday_set=holiday_set)
    out["band_F1"] = (bands == "F1").astype(int)
    out["band_F2"] = (bands == "F2").astype(int)
    out["band_F3"] = (bands == "F3").astype(int)

    # === Lag features (load) ===
    load = df["load_kw"]
    for lag in lags:
        out[f"load_lag_{lag}"] = load.shift(lag)

    # === Rolling statistics (load) — must be shifted by 1 to avoid leakage ===
    for w in roll_windows:
        out[f"load_roll_mean_{w}"] = load.shift(1).rolling(w, min_periods=1).mean()
        out[f"load_roll_std_{w}"]  = load.shift(1).rolling(w, min_periods=2).std()
        out[f"load_roll_max_{w}"]  = load.shift(1).rolling(w, min_periods=1).max()
        out[f"load_roll_min_{w}"]  = load.shift(1).rolling(w, min_periods=1).min()

    # === PV features (PV is observable at time t) ===
    if include_pv and "pv_kw" in df.columns:
        out["pv_kw"]       = df["pv_kw"]
        out["pv_lag_1"]    = df["pv_kw"].shift(1)
        out["pv_lag_96"]   = df["pv_kw"].shift(96)
        out["pv_roll_mean_96"]  = df["pv_kw"].shift(1).rolling(96, min_periods=1).mean()

    # === Scale-invariance helpers (helps cross-site generalization) ===
    if 96 in roll_windows or 672 in roll_windows:
        # Ratio of latest known value to recent mean — shape rather than magnitude
        ref_window = max([w for w in roll_windows if w >= 96], default=672)
        ref_col = f"load_roll_mean_{ref_window}"
        if ref_col in out.columns:
            out["load_ratio_recent"] = out[f"load_lag_1"] / (out[ref_col] + 1e-3)

    # === Target ===
    y = load.shift(-horizon)
    y.name = f"load_t+{horizon}"

    # === Drop rows with NaN in either X or y ===
    valid = (~out.isna().any(axis=1)) & (~y.isna())
    return out[valid], y[valid]


def build_inference_features(history_df, t_now, horizon, lags=None,
                              roll_windows=None, holiday_set=None,
                              include_pv=True):
    """
    Build a SINGLE feature row for inference at time t_now, predicting load[t_now + horizon].

    history_df must contain enough past data (load_kw and pv_kw) up to and
    including t_now to compute all lags and rolling windows.

    Returns
    -------
    pd.DataFrame with a single row (index = t_now), suitable for model.predict().
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if roll_windows is None:
        roll_windows = DEFAULT_ROLL_WINDOWS

    # Compute features over the entire history then take the last row at t_now.
    X_all, _ = build_features(
        history_df, lags=lags, roll_windows=roll_windows,
        horizon=horizon, holiday_set=holiday_set, include_pv=include_pv,
    )
    if t_now not in X_all.index:
        # If t_now is past the last "valid" row (e.g., near year-end),
        # build manually.
        raise KeyError(f"Cannot build features at {t_now} — try lower horizon or check history coverage.")
    return X_all.loc[[t_now]]


if __name__ == "__main__":
    from data_loader import load_dataset

    sheets = load_dataset("data/synthetic_dataset.xlsx")
    df_2024 = sheets["2024"]

    X, y = build_features(df_2024, horizon=1)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"\nFeature columns ({len(X.columns)}):")
    for c in X.columns:
        print(f"  {c}")
    print(f"\nFirst row sample:")
    print(X.iloc[0])
