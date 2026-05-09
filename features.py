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

Per-slot stats (slot_stats parameter):
  Pass a DataFrame with columns ["mean", "std"] indexed by slot 0-95
  (slot = hour*4 + minute//15). When provided (inference path), these
  pre-computed training statistics are used instead of computing from df,
  preventing test-data leakage. Compute with compute_slot_stats().
"""
import numpy as np
import pandas as pd
from tariff import assign_tariff_band, get_italian_holidays


# Lags expressed in 15-min steps
DEFAULT_LAGS = [1, 2, 4, 8, 96, 96 * 2, 672]   # 15min, 30min, 1h, 2h, 1d, 2d, 1w
DEFAULT_ROLL_WINDOWS = [4, 16, 96, 672]         # 1h, 4h, 1d, 1w


def compute_slot_stats(df):
    """
    Compute per-slot (0-95) mean and std of load_kw from a training DataFrame.

    Returns a DataFrame indexed 0-95 with columns ["mean", "std"].
    Always call this on training data only and pass the result to build_features
    during inference, so test-data statistics never influence features.
    """
    slot_key = df.index.hour * 4 + df.index.minute // 15
    stats = df.groupby(slot_key)["load_kw"].agg(["mean", "std"])
    stats.index.name = "slot"
    return stats


def build_features(df, lags=None, roll_windows=None, horizon=1,
                   holiday_set=None, include_pv=True, slot_stats=None):
    """
    Build a feature matrix and target for LOAD forecasting.

    Parameters
    ----------
    df          : pd.DataFrame, indexed by 15-min timestamps, must contain 'load_kw'
                  (and 'pv_kw' if include_pv=True). Other columns optional.
    lags        : list of int, lag steps for load. Default DEFAULT_LAGS.
    roll_windows: list of int, rolling windows (in steps).
    horizon     : int >= 1. Target is load_kw shifted by -horizon steps,
                  i.e. y[t] = load_kw[t + horizon].
                  Returned X is rows where y is non-NaN.
    holiday_set : optional set of date objects. If None, computed from df.index years.
    include_pv  : whether to include PV columns as features.
    slot_stats  : pd.DataFrame indexed 0-95 with ["mean","std"] from compute_slot_stats().
                  If None, stats are computed from df itself (training-only calls only).

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
    ts = df.index
    load = df["load_kw"]

    # === Calendar features ===
    out["hour"]           = ts.hour
    out["minute"]         = ts.minute
    out["minute_of_day"]  = ts.hour * 4 + ts.minute // 15
    out["dow"]            = ts.dayofweek
    out["is_weekend"]     = (ts.dayofweek >= 5).astype(int)
    out["month"]          = ts.month
    out["weekofyear"]     = ts.isocalendar().week.astype(int).values
    out["dayofyear"]      = ts.dayofyear

    if holiday_set is None:
        years = sorted(set(ts.year))
        holiday_set = get_italian_holidays(years)
    out["is_holiday"] = pd.Series(
        [d in holiday_set for d in ts.date], index=ts).astype(int)

    # Italian Ferragosto vacation: Aug 10-20 special flag
    out["is_ferragosto"] = (
        (ts.month == 8) & (ts.day >= 10) & (ts.day <= 20)
    ).astype(int)

    # Seasonal context flags (critical for Sondrio Alpine transition months)
    out["heating_season"]      = ts.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    out["is_transition_month"] = ts.month.isin([4, 9]).astype(int)

    # Cyclic encodings (sin/cos)
    out["hour_sin"]  = np.sin(2 * np.pi * out["minute_of_day"] / 96)
    out["hour_cos"]  = np.cos(2 * np.pi * out["minute_of_day"] / 96)
    out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * out["dayofyear"] / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * out["dayofyear"] / 365)

    # Theoretical solar irradiance proxy (no external data needed)
    hod = ts.hour + ts.minute / 60.0
    doy = ts.dayofyear
    out["solar_proxy"] = np.clip(
        np.sin(np.pi * hod / 24) * np.sin(np.pi * (doy - 80) / 365), 0, None)

    # === Tariff band as feature ===
    bands = assign_tariff_band(ts, holiday_set=holiday_set)
    out["band_F1"] = (bands == "F1").astype(int)
    out["band_F2"] = (bands == "F2").astype(int)
    out["band_F3"] = (bands == "F3").astype(int)

    # === Lag features (load) ===
    for lag in lags:
        out[f"load_lag_{lag}"] = load.shift(lag)

    # === Rolling statistics (load) — shifted by 1 to avoid using current step ===
    load_s1 = load.shift(1)
    for w in roll_windows:
        out[f"load_roll_mean_{w}"] = load_s1.rolling(w, min_periods=1).mean()
        out[f"load_roll_std_{w}"]  = load_s1.rolling(w, min_periods=2).std()
        out[f"load_roll_max_{w}"]  = load_s1.rolling(w, min_periods=1).max()
        out[f"load_roll_min_{w}"]  = load_s1.rolling(w, min_periods=1).min()

    # === Exponential weighted moving averages (causal, shift 1) ===
    out["load_ewma_fast"] = load_s1.ewm(span=4,  adjust=False).mean()   # 1h
    out["load_ewma_mid"]  = load_s1.ewm(span=16, adjust=False).mean()   # 4h
    out["load_ewma_slow"] = load_s1.ewm(span=96, adjust=False).mean()   # 1d

    # === Recent load change (spike onset signal) ===
    out["load_delta_1h"]  = load_s1 - load.shift(5)    # change over last 1h
    out["load_delta_4h"]  = load_s1 - load.shift(17)   # change over last 4h

    # === Per-slot historical statistics (causal — avoids test leakage when slot_stats provided) ===
    slot_key_arr = ts.hour * 4 + ts.minute // 15  # 0-95

    if slot_stats is not None:
        # Inference path: use pre-computed training stats (no leakage)
        slot_mean_s = pd.Series(slot_key_arr, index=df.index).map(slot_stats["mean"])
        slot_std_s  = pd.Series(slot_key_arr, index=df.index).map(slot_stats["std"])
        slot_mean_s.index = df.index
        slot_std_s.index  = df.index
    else:
        # Training-only path: compute from df (standard for full-training calls)
        slot_mean_s = df.groupby(slot_key_arr)["load_kw"].transform("mean")
        slot_std_s  = df.groupby(slot_key_arr)["load_kw"].transform("std")

    out["slot_mean"]     = slot_mean_s.values
    out["slot_std"]      = slot_std_s.values
    out["load_vs_slot"]  = load_s1 / (slot_mean_s + 1e-6)   # deviation from typical

    # === Scale-invariance ratio ===
    if 96 in roll_windows or 672 in roll_windows:
        ref_window = max([w for w in roll_windows if w >= 96], default=672)
        ref_col = f"load_roll_mean_{ref_window}"
        if ref_col in out.columns and "load_lag_1" in out.columns:
            out["load_ratio_recent"] = out["load_lag_1"] / (out[ref_col] + 1e-3)

    # === PV features (PV is observable at time t for h=1 forecast) ===
    if include_pv and "pv_kw" in df.columns:
        pv = df["pv_kw"]
        pv_s1 = pv.shift(1)

        out["pv_kw"]           = pv                                   # current PV (known at t)
        out["pv_lag_1"]        = pv_s1
        out["pv_lag_96"]       = pv.shift(96)                         # yesterday same slot
        out["pv_roll_mean_96"] = pv_s1.rolling(96, min_periods=1).mean()
        out["pv_roll_std_4"]   = pv_s1.rolling(4,  min_periods=1).std()   # cloud cover proxy

        # PV ratio: today vs yesterday same slot (captures cloud/season shift)
        out["pv_vs_yesterday"] = pv_s1 / (pv.shift(97) + 0.01)

        # Cumulative PV generated today so far (solar energy proxy)
        out["pv_cumday"] = pv_s1.groupby(pv_s1.index.date).cumsum()

        # PV × time-of-day interaction (morning vs afternoon solar effect on load)
        out["pv_x_hour_sin"] = pv_s1 * out["hour_sin"]

    # === Target ===
    y = load.shift(-horizon)
    y.name = f"load_t+{horizon}"

    # === Drop rows with NaN in either X or y ===
    valid = (~out.isna().any(axis=1)) & (~y.isna())
    return out[valid], y[valid]


def build_inference_features(history_df, t_now, horizon, lags=None,
                              roll_windows=None, holiday_set=None,
                              include_pv=True, slot_stats=None):
    """
    Build a SINGLE feature row for inference at time t_now, predicting load[t_now + horizon].

    history_df must contain enough past data (load_kw and pv_kw) up to and
    including t_now to compute all lags and rolling windows.

    slot_stats should always be passed here (pre-computed from training data).
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if roll_windows is None:
        roll_windows = DEFAULT_ROLL_WINDOWS

    X_all, _ = build_features(
        history_df, lags=lags, roll_windows=roll_windows,
        horizon=horizon, holiday_set=holiday_set,
        include_pv=include_pv, slot_stats=slot_stats,
    )
    if t_now not in X_all.index:
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
