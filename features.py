"""
Feature engineering for the load forecaster.

Builds a tabular feature matrix from the load_kw time series + calendar.
All features are CAUSAL: at predicted timestep t, only data from t-1 and earlier is used.

Feature families:
  1. Calendar       — drives most variance; includes hour×month interaction (38.9% of variance)
  2. Cyclic         — sin/cos encodings for periodicity; includes annual (doy) cycle
  3. Lag features   — workhorse for short horizons; covers 3d/4d/5d/6d gap between 2d and 1w
  4. Rolling stats  — smoothed local mean / variability; pv_std as cloud proxy
  5. Seasonal flags — heating_season, transition months (Apr/Sep = test months!)
  6. Solar proxy    — physics-based irradiance estimate independent of measured PV
  7. Price features — sell_price lags for tariff-aware load shifting signal

Multi-horizon strategy: this builder accepts a `horizon` argument that shifts
the target by `horizon` steps. To produce predictions for steps t+1..t+H,
train H separate models (one per horizon) — see forecaster.train_multi_horizon.
"""
import numpy as np
import pandas as pd
from tariff import assign_tariff_band, get_italian_holidays


# Lags expressed in 15-min steps
# 3d/4d/5d/6d fill the gap between 2d (192) and 1w (672) — important for weekly pattern
DEFAULT_LAGS = [1, 2, 4, 8, 96, 192, 288, 384, 480, 576, 672]
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
    # Annual cycle: dayofyear as cyclic — captures seasonal load curve correctly
    out["doy_sin"]   = np.sin(2 * np.pi * (out["dayofyear"] - 1) / 365)
    out["doy_cos"]   = np.cos(2 * np.pi * (out["dayofyear"] - 1) / 365)

    # hour × month interaction — single strongest predictor (38.9% variance in EDA)
    # Encodes the joint "what time of day AND what season" which drives load profile shape
    out["hour_x_month_ss"] = out["hour_sin"] * out["month_sin"]
    out["hour_x_month_sc"] = out["hour_sin"] * out["month_cos"]
    out["hour_x_month_cs"] = out["hour_cos"] * out["month_sin"]
    out["hour_x_month_cc"] = out["hour_cos"] * out["month_cos"]

    # Seasonal regime flags
    out["heating_season"]    = ts.month.isin([10, 11, 12, 1, 2, 3]).astype(int)
    out["is_transition_month"] = ts.month.isin([4, 9]).astype(int)  # test months; high variance

    # DST transition day flag — instability around clocks-change
    # Spring forward: last Sunday of March; Fall back: last Sunday of October
    is_dst_day = np.zeros(len(ts), dtype=np.int8)
    for yr in set(ts.year):
        # Last Sunday of March
        m3 = pd.Timestamp(f"{yr}-03-31")
        spring = m3 - pd.Timedelta(days=m3.dayofweek + 1) if m3.dayofweek != 6 else m3
        # Last Sunday of October
        m10 = pd.Timestamp(f"{yr}-10-31")
        fall = m10 - pd.Timedelta(days=m10.dayofweek + 1) if m10.dayofweek != 6 else m10
        mask = (ts.date == spring.date()) | (ts.date == fall.date())
        is_dst_day[mask] = 1
    out["is_dst_day"] = is_dst_day

    # Solar irradiance proxy — physics estimate, independent of measured PV
    # sin(π × step_in_day/96) = daytime arch; sin(π × (doy-80)/365) = seasonal amplitude
    # Product clipped to 0 gives "potential PV output" even without a sensor reading
    day_arch = np.sin(np.pi * out["minute_of_day"].values / 96)
    season_amp = np.sin(np.pi * (out["dayofyear"].values - 80) / 365)
    out["solar_proxy"] = np.clip(day_arch * season_amp, 0, None)

    # === Sondrio (Valtellina) location-specific features ===
    # 46.17°N, 9.87°E — Alpine valley, E-W orientation, 307m altitude
    _add_sondrio_features(out, ts)

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

    # Momentum: rate of change over last hour (load_delta_4 = load[t-1] - load[t-5])
    out["load_delta_4"] = load.shift(1) - load.shift(5)

    # === PV features (PV is observable at time t) ===
    if include_pv and "pv_kw" in df.columns:
        out["pv_kw"]       = df["pv_kw"]
        out["pv_lag_1"]    = df["pv_kw"].shift(1)
        out["pv_lag_96"]   = df["pv_kw"].shift(96)
        out["pv_roll_mean_96"]  = df["pv_kw"].shift(1).rolling(96, min_periods=1).mean()
        # 1h PV std: cloud cover proxy — high std = passing clouds = harder to forecast load
        out["pv_roll_std_4"]    = df["pv_kw"].shift(1).rolling(4, min_periods=2).std().fillna(0)

    # === Price features (sell_price signal for load shifting awareness) ===
    if "sell_price" in df.columns:
        out["sell_price_lag_4"]  = df["sell_price"].shift(4)   # 1h ago
        out["sell_price_lag_96"] = df["sell_price"].shift(96)  # yesterday same time

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


def _add_sondrio_features(out, ts):
    """
    Location-specific features for Sondrio, Valtellina (46.17°N, 9.87°E).

    All features derived from calendar only — no external weather feed needed.
    Climatology sources: climate-data.org/Sondrio, Arpa Lombardia historical data.

    Features added (all in-place on `out` DataFrame):
      solar_elevation_deg  — exact sun angle above horizon (physics, ±90°)
      is_above_horizon     — 1 when sun is up (elevation > 0)
      valley_fog_risk      — Dec–Mar mornings: Alpine thermal inversion fog
      foehn_risk           — Dec–Apr: warm dry downslope wind, reduces heating load
      is_harvest_season    — Oct 15–Nov 15: Nebbiolo harvest, local activity spike
      is_ski_tourism       — Dec–Mar: winter tourism demand
      is_summer_hiking     — Jun–Sep: summer tourism demand
      temp_proxy           — estimated temperature (°C) from Sondrio climatology
                             using monthly mean + diurnal cycle
    """
    LAT  = np.radians(46.17)   # Sondrio latitude in radians
    LON  =  9.87               # Sondrio longitude (degrees)
    TZ   =  1                  # UTC+1 (CET); CEST (+2) handled by local timestamp

    doy  = ts.dayofyear.values.astype(float)
    hod  = (ts.hour + ts.minute / 60.0).values   # hour of day as float

    # ── Solar elevation at Sondrio ──────────────────────────────────────────
    # Declination angle (degrees), accurate to ±0.3°
    dec_deg  = 23.45 * np.sin(np.radians(360 / 365 * (doy - 81)))
    dec_rad  = np.radians(dec_deg)

    # Hour angle: solar noon = 0, morning negative, afternoon positive
    # Longitude correction: (LON - 15*TZ) / 15 hours offset from standard meridian
    lon_corr = (LON - 15.0 * TZ) / 15.0          # ≈ −0.342 h for Sondrio CET
    solar_hr  = hod - 12.0 + lon_corr             # hours from true solar noon
    ha_rad    = np.radians(solar_hr * 15.0)        # hour angle in radians

    # Elevation = arcsin(sin(lat)·sin(dec) + cos(lat)·cos(dec)·cos(ha))
    elev_rad = np.arcsin(
        np.sin(LAT) * np.sin(dec_rad)
        + np.cos(LAT) * np.cos(dec_rad) * np.cos(ha_rad)
    )
    elev_deg = np.degrees(elev_rad)

    # Valley shadow correction: E-W valley floor, south-facing panels at 46°N.
    # Low-angle morning/evening sun is blocked by surrounding peaks in winter.
    # Empirical threshold: effective sun access only when elevation > 8° (valley walls ~400m).
    SHADOW_THR = 8.0
    out["solar_elevation_deg"] = elev_deg
    out["is_above_horizon"]    = (elev_deg > 0).astype(int)
    out["effective_solar"]     = np.clip(elev_deg - SHADOW_THR, 0, None)   # 0 when in shadow

    # ── Valley fog risk ─────────────────────────────────────────────────────
    # December–March: high-pressure systems trap cold air; fog persists until ~10-11am
    # Effect: suppresses PV generation and keeps heating load elevated through morning
    month_arr = ts.month.values
    hour_arr  = ts.hour.values
    day_arr   = ts.day.values

    fog_month = np.isin(month_arr, [12, 1, 2, 3])
    fog_hour  = (hour_arr >= 5) & (hour_arr <= 11)
    out["valley_fog_risk"] = (fog_month & fog_hour).astype(int)

    # ── Foehn wind risk ─────────────────────────────────────────────────────
    out["foehn_risk"] = np.isin(month_arr, [12, 1, 2, 3, 4]).astype(int)

    # ── Activity / tourism seasons ──────────────────────────────────────────
    harvest_mask = (
        ((month_arr == 10) & (day_arr >= 15))
        | (month_arr == 11)
        | ((month_arr == 12) & (day_arr <= 15))
    )
    out["is_harvest_season"] = harvest_mask.astype(int)

    out["is_ski_tourism"]   = np.isin(month_arr, [12, 1, 2, 3]).astype(int)
    out["is_summer_hiking"] = np.isin(month_arr, [6, 7, 8, 9]).astype(int)

    # ── Temperature proxy ────────────────────────────────────────────────────
    # Monthly mean temperatures (°C) from Sondrio climate records (Arpa Lombardia avg)
    MONTHLY_MEAN_TEMP = np.array([2.0, 4.0, 8.0, 13.0, 17.0, 21.0,
                                   23.0, 22.0, 18.0, 12.0, 6.0, 2.0])
    # Daily cycle: peak at 14:00 (+5°C above mean), trough at 05:00 (-5°C below mean)
    daily_cycle = 5.0 * np.sin(np.pi * (hod - 5.0) / 9.0)   # half-sine, peak at 14h
    daily_cycle = np.where((hod >= 5) & (hod <= 23), daily_cycle, -5.0)

    month_idx   = (ts.month.values - 1)
    base_temp   = MONTHLY_MEAN_TEMP[month_idx]
    out["temp_proxy"] = base_temp + daily_cycle

    # Heating degree indicator: temp below 15°C drives residential heating load
    out["heating_degree"] = np.clip(15.0 - out["temp_proxy"].values, 0, None)

    # Cooling degree indicator: temp above 22°C drives AC (limited in Sondrio)
    out["cooling_degree"] = np.clip(out["temp_proxy"].values - 22.0, 0, None)


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
