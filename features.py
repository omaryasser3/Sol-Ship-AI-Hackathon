"""
Feature engineering for the load forecaster — Sondrio residential site.

All features are CAUSAL: at predicted timestep t, only data from t-1 and earlier is used.

Key design choices for this site:
  - Sondrio is Alpine (46°N): load is strongly thermally-driven (heating Oct-Mar,
    appliance-spike dominated in Apr/Sep).
  - Per-(month, slot) statistics are the single most important feature group because
    the daily load profile changes completely between heating and non-heating months.
  - pv_kw / solar_proxy = cloud-cover proxy = free temperature signal without external data.
  - load_lag_0 (current load) captures autocorrelation; slot_month features put it in context.

External stats (slot_stats, slot_month_stats):
  Always compute from TRAINING data and pass to build_features() during inference
  to prevent test-data leakage. Use compute_slot_stats() and compute_slot_month_stats().
"""
import numpy as np
import pandas as pd
from tariff import assign_tariff_band, get_italian_holidays

DEFAULT_LAGS        = [1, 2, 4, 8, 96, 192, 672]
DEFAULT_ROLL_WINDOWS = [4, 16, 96, 672]


# ── stat helpers ──────────────────────────────────────────────────────────────

def compute_slot_stats(df):
    """Per-slot (0-95) mean and std of load_kw. Call on training data only."""
    slot_key = df.index.hour * 4 + df.index.minute // 15
    stats = df.groupby(slot_key)["load_kw"].agg(["mean", "std"])
    stats.index.name = "slot"
    return stats


def compute_slot_month_stats(df):
    """
    Per-(month × slot) mean and std of load_kw. Call on training data only.

    Key for Sondrio: load at 08:00 in January (heating) is ~2.5 kW while
    the same slot in September is ~0.7 kW. Annual slot stats blend these,
    giving a poor baseline. Month-specific stats capture seasonality directly.

    Index: month*100 + slot  (e.g. April 08:00 → 4*100+32 = 432)
    """
    slot_key  = df.index.hour * 4 + df.index.minute // 15
    month_key = df.index.month
    key       = month_key * 100 + slot_key
    stats     = df.groupby(key)["load_kw"].agg(["mean", "std"])
    stats.index.name = "month_slot"
    return stats


def compute_slot_dow_stats(df):
    """
    Per-(day-of-week × slot) mean and std of load_kw. Call on training data only.

    Saturday 08:00 has a different profile from Tuesday 08:00.
    Index: dow*100 + slot  (e.g. Monday 08:00 → 0*100+32 = 32)
    """
    slot_key = df.index.hour * 4 + df.index.minute // 15
    dow_key  = df.index.dayofweek
    key      = dow_key * 100 + slot_key
    stats    = df.groupby(key)["load_kw"].agg(["mean", "std"])
    stats.index.name = "dow_slot"
    return stats


# ── main feature builder ──────────────────────────────────────────────────────

def build_features(df, lags=None, roll_windows=None, horizon=1,
                   holiday_set=None, include_pv=True,
                   slot_stats=None, slot_month_stats=None, slot_dow_stats=None):
    """
    Build feature matrix and target for LOAD forecasting.

    Parameters
    ----------
    df                : DataFrame indexed by 15-min timestamps, must have 'load_kw'.
    lags              : lag steps for load (in 15-min steps).
    roll_windows      : rolling window sizes (in steps).
    horizon           : predict load at t+horizon. Target = load.shift(-horizon).
    holiday_set       : set of date objects; computed if None.
    include_pv        : include PV-derived features.
    slot_stats        : DataFrame from compute_slot_stats()  — annual slot profile.
    slot_month_stats  : DataFrame from compute_slot_month_stats() — monthly slot profile.
    slot_dow_stats    : DataFrame from compute_slot_dow_stats()  — weekday slot profile.

    Returns
    -------
    X : feature DataFrame,  y : target Series
    """
    if lags is None:
        lags = DEFAULT_LAGS
    if roll_windows is None:
        roll_windows = DEFAULT_ROLL_WINDOWS

    if "load_kw" not in df.columns:
        raise ValueError("df must contain 'load_kw'")

    out  = pd.DataFrame(index=df.index)
    ts   = df.index
    load = df["load_kw"]

    # ── calendar ─────────────────────────────────────────────────────────────
    slot_key_arr = ts.hour * 4 + ts.minute // 15   # 0-95
    out["minute_of_day"]  = slot_key_arr
    out["hour"]           = ts.hour
    out["minute"]         = ts.minute
    out["dow"]            = ts.dayofweek
    out["is_weekend"]     = (ts.dayofweek >= 5).astype(int)
    out["month"]          = ts.month
    out["weekofyear"]     = ts.isocalendar().week.astype(int).values
    out["dayofyear"]      = ts.dayofyear

    if holiday_set is None:
        holiday_set = get_italian_holidays(sorted(set(ts.year)))
    out["is_holiday"]    = pd.Series([d in holiday_set for d in ts.date], index=ts).astype(int)
    out["is_ferragosto"] = ((ts.month == 8) & (ts.day >= 10) & (ts.day <= 20)).astype(int)

    # Sondrio-specific season flags
    out["heating_season"]      = ts.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    out["is_transition_month"] = ts.month.isin([4, 9]).astype(int)

    # cyclic encodings
    out["hour_sin"]  = np.sin(2 * np.pi * slot_key_arr / 96)
    out["hour_cos"]  = np.cos(2 * np.pi * slot_key_arr / 96)
    out["dow_sin"]   = np.sin(2 * np.pi * ts.dayofweek / 7)
    out["dow_cos"]   = np.cos(2 * np.pi * ts.dayofweek / 7)
    out["month_sin"] = np.sin(2 * np.pi * (ts.month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (ts.month - 1) / 12)
    out["doy_sin"]   = np.sin(2 * np.pi * ts.dayofyear / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * ts.dayofyear / 365)

    # theoretical solar irradiance proxy (Alpine Sondrio 46°N)
    hod = ts.hour + ts.minute / 60.0
    doy = ts.dayofyear
    solar_proxy = np.clip(
        np.sin(np.pi * hod / 24) * np.sin(np.pi * (doy - 80) / 365), 0, None)
    out["solar_proxy"] = solar_proxy

    # ── tariff band ───────────────────────────────────────────────────────────
    bands = assign_tariff_band(ts, holiday_set=holiday_set)
    out["band_F1"] = (bands == "F1").astype(int)
    out["band_F2"] = (bands == "F2").astype(int)
    out["band_F3"] = (bands == "F3").astype(int)

    # ── lag features ──────────────────────────────────────────────────────────
    for lag in lags:
        out[f"load_lag_{lag}"] = load.shift(lag)

    # ── rolling statistics (all shifted by 1 — strictly causal) ──────────────
    load_s1 = load.shift(1)
    for w in roll_windows:
        out[f"load_roll_mean_{w}"] = load_s1.rolling(w, min_periods=1).mean()
        out[f"load_roll_std_{w}"]  = load_s1.rolling(w, min_periods=2).std()
        out[f"load_roll_max_{w}"]  = load_s1.rolling(w, min_periods=1).max()
        out[f"load_roll_min_{w}"]  = load_s1.rolling(w, min_periods=1).min()

    # exponential weighted moving averages (causal)
    out["load_ewma_fast"] = load_s1.ewm(span=4,  adjust=False).mean()   # 1h
    out["load_ewma_mid"]  = load_s1.ewm(span=16, adjust=False).mean()   # 4h
    out["load_ewma_slow"] = load_s1.ewm(span=96, adjust=False).mean()   # 1d

    # recent load change (spike onset / appliance switch signal)
    out["load_delta_1h"] = load_s1 - load.shift(5)
    out["load_delta_4h"] = load_s1 - load.shift(17)

    # ── annual slot profile (baseline across all months) ─────────────────────
    if slot_stats is not None:
        slot_mean_s = pd.Series(slot_key_arr, index=df.index).map(slot_stats["mean"])
        slot_std_s  = pd.Series(slot_key_arr, index=df.index).map(slot_stats["std"])
        slot_mean_s.index = df.index
        slot_std_s.index  = df.index
    else:
        slot_mean_s = df.groupby(slot_key_arr)["load_kw"].transform("mean")
        slot_std_s  = df.groupby(slot_key_arr)["load_kw"].transform("std")

    out["slot_mean"]    = slot_mean_s.values
    out["slot_std"]     = slot_std_s.values
    out["load_vs_slot"] = load_s1 / (slot_mean_s + 1e-6)

    # ── monthly slot profile — MOST IMPORTANT for Sondrio seasonality ─────────
    # Each (month, slot) bucket gives the expected load for this exact hour in
    # this exact month. Dramatically better baseline than the annual slot mean.
    ms_key = ts.month * 100 + slot_key_arr
    if slot_month_stats is not None:
        sm_mean_s = pd.Series(ms_key, index=df.index).map(slot_month_stats["mean"])
        sm_std_s  = pd.Series(ms_key, index=df.index).map(slot_month_stats["std"])
        sm_mean_s.index = df.index
        sm_std_s.index  = df.index
    else:
        sm_mean_s = df.groupby(ms_key)["load_kw"].transform("mean")
        sm_std_s  = df.groupby(ms_key)["load_kw"].transform("std")

    out["slot_month_mean"] = sm_mean_s.values
    out["slot_month_std"]  = sm_std_s.values
    # deviation from the monthly profile — key signal: is load elevated vs seasonal norm?
    out["load_vs_slot_month"] = load_s1 / (sm_mean_s + 1e-6)
    # z-score relative to monthly profile
    sm_std_safe = sm_std_s.where(sm_std_s > 0.05, other=0.05)
    out["load_zscore_month"]  = (load_s1 - sm_mean_s) / sm_std_safe

    # ── weekday slot profile ──────────────────────────────────────────────────
    dow_key = ts.dayofweek * 100 + slot_key_arr
    if slot_dow_stats is not None:
        sd_mean_s = pd.Series(dow_key, index=df.index).map(slot_dow_stats["mean"])
        sd_std_s  = pd.Series(dow_key, index=df.index).map(slot_dow_stats["std"])
        sd_mean_s.index = df.index
        sd_std_s.index  = df.index
    else:
        sd_mean_s = df.groupby(dow_key)["load_kw"].transform("mean")
        sd_std_s  = df.groupby(dow_key)["load_kw"].transform("std")

    out["slot_dow_mean"]     = sd_mean_s.values
    out["load_vs_slot_dow"]  = load_s1 / (sd_mean_s + 1e-6)

    # ── scale-invariance ratio ────────────────────────────────────────────────
    if "load_lag_1" in out.columns:
        ref_window = max([w for w in roll_windows if w >= 96], default=672)
        ref_col = f"load_roll_mean_{ref_window}"
        if ref_col in out.columns:
            out["load_ratio_recent"] = out["load_lag_1"] / (out[ref_col] + 1e-3)

    # ── PV features + cloud/temperature proxy ─────────────────────────────────
    if include_pv and "pv_kw" in df.columns:
        pv    = df["pv_kw"]
        pv_s1 = pv.shift(1)

        out["pv_kw"]           = pv
        out["pv_lag_1"]        = pv_s1
        out["pv_lag_96"]       = pv.shift(96)
        out["pv_roll_mean_96"] = pv_s1.rolling(96, min_periods=1).mean()
        out["pv_roll_std_4"]   = pv_s1.rolling(4,  min_periods=1).std()

        # cloud-cover / temperature proxy for Sondrio (no external data needed):
        # actual PV / theoretical max — low ratio = overcast = likely colder = higher load
        out["pv_vs_expected"]  = pv / (solar_proxy + 0.01)
        out["pv_vs_yesterday"] = pv_s1 / (pv.shift(97) + 0.01)

        # cumulative PV today so far (proxy for "how sunny has today been")
        out["pv_cumday"]       = pv_s1.groupby(pv_s1.index.date).cumsum()

        # PV × season interaction (heating-season days with low PV → high load)
        out["pv_x_heating"]    = pv_s1 * out["heating_season"]
        out["pv_x_hour_sin"]   = pv_s1 * out["hour_sin"]

        # early-morning load as temperature proxy:
        # yesterday's average load 05:00-07:00 = mostly heating demand → cold proxy
        morning_mask = ((df.index.hour >= 5) & (df.index.hour < 7)).astype(float)
        morning_load = load * morning_mask
        # rolling 8-step (2h) sum yesterday same period = daily heating baseline
        out["morning_load_proxy"] = morning_load.shift(96).rolling(8, min_periods=1).mean()

    # ── sell price features ───────────────────────────────────────────────────
    # Sell price spikes → grid-wide high demand → higher residential load too.
    # Using lagged values only (causal): 1 step ago (15 min), 4 steps (1h), 96 steps (yesterday).
    if "sell_price" in df.columns:
        sp    = df["sell_price"]
        sp_s1 = sp.shift(1)
        out["sell_lag_1"]       = sp_s1
        out["sell_lag_4"]       = sp.shift(4)
        out["sell_lag_96"]      = sp.shift(96)
        out["sell_roll_mean_4"] = sp_s1.rolling(4,  min_periods=1).mean()
        out["sell_roll_std_4"]  = sp_s1.rolling(4,  min_periods=2).std()
        out["sell_roll_mean_96"]= sp_s1.rolling(96, min_periods=1).mean()
        # price momentum: recent vs yesterday baseline (direction of price move)
        out["sell_momentum"]    = sp_s1 - sp.shift(97)

    # ── target ────────────────────────────────────────────────────────────────
    y = load.shift(-horizon)
    y.name = f"load_t+{horizon}"

    valid = (~out.isna().any(axis=1)) & (~y.isna())
    return out[valid], y[valid]


def build_inference_features(history_df, t_now, horizon, lags=None,
                              roll_windows=None, holiday_set=None,
                              include_pv=True, slot_stats=None,
                              slot_month_stats=None, slot_dow_stats=None):
    """Single-row feature build for inference at t_now predicting t_now+horizon."""
    if lags is None:
        lags = DEFAULT_LAGS
    if roll_windows is None:
        roll_windows = DEFAULT_ROLL_WINDOWS

    X_all, _ = build_features(
        history_df, lags=lags, roll_windows=roll_windows, horizon=horizon,
        holiday_set=holiday_set, include_pv=include_pv,
        slot_stats=slot_stats, slot_month_stats=slot_month_stats,
        slot_dow_stats=slot_dow_stats,
    )
    if t_now not in X_all.index:
        raise KeyError(f"Cannot build features at {t_now}")
    return X_all.loc[[t_now]]
