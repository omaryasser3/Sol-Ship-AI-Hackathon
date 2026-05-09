"""
Italian residential Time-of-Use tariff band assignment.

The buy_price column is provided in the dataset, but we re-derive bands
from timestamps for two reasons:
  1. Validation — sanity-check the dataset's buy_price matches our bands.
  2. The MPC sometimes wants to know the band itself (categorical feature).

Bands per Brief Section 2:
  F1: Mon-Fri 08-19 (excl. national holidays)             0.2540 €/kWh
  F2: Mon-Fri 07-08 + 19-23, Sat 07-23 (excl. holidays)   0.2682 €/kWh
  F3: Mon-Sat 00-07 + 23-24, Sundays + holidays all day   0.2440 €/kWh
"""
import pandas as pd
import numpy as np
import holidays
from config import F1_PRICE, F2_PRICE, F3_PRICE


def get_italian_holidays(years):
    """Return a set of date objects for Italian national holidays in given years."""
    it_holidays = holidays.country_holidays("IT", years=years)
    return set(it_holidays.keys())


def assign_tariff_band(timestamps, holiday_set=None):
    """
    Assign F1/F2/F3 band to each timestamp.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex or array-like of timestamps
    holiday_set : set of date objects, optional. If None, computed from the years present.

    Returns
    -------
    pd.Series of band strings: 'F1', 'F2', or 'F3'.
    """
    ts = pd.DatetimeIndex(timestamps)

    if holiday_set is None:
        years = sorted(set(ts.year))
        holiday_set = get_italian_holidays(years)

    dates = ts.date
    is_holiday = pd.Series([d in holiday_set for d in dates], index=ts)
    dow = ts.dayofweek          # 0=Mon ... 6=Sun
    hour = ts.hour

    # Sundays and holidays → F3 all day
    sunday_or_holiday = (dow == 6) | is_holiday.values

    # Saturday (non-holiday)
    saturday = (dow == 5) & ~is_holiday.values

    # Mon-Fri (non-holiday)
    weekday = (dow <= 4) & ~is_holiday.values

    band = np.full(len(ts), "F3", dtype=object)

    # F1: Mon-Fri 08:00-19:00 (excluding holidays)
    f1_mask = weekday & (hour >= 8) & (hour < 19)
    band[f1_mask] = "F1"

    # F2: Mon-Fri 07:00-08:00 + 19:00-23:00, Saturday 07:00-23:00
    weekday_f2 = weekday & (((hour >= 7) & (hour < 8)) | ((hour >= 19) & (hour < 23)))
    saturday_f2 = saturday & (hour >= 7) & (hour < 23)
    f2_mask = weekday_f2 | saturday_f2
    band[f2_mask] = "F2"

    # Everything else stays F3 (Mon-Sat 00-07, Mon-Fri 23-24, Sat 23-24, Sun all day, holidays all day)
    return pd.Series(band, index=ts, name="band")


def band_to_price(band_series):
    """Map band labels to buy prices."""
    price_map = {"F1": F1_PRICE, "F2": F2_PRICE, "F3": F3_PRICE}
    return band_series.map(price_map).astype(float)


def compute_bill(p_grid_kw, buy_price, sell_price, dt_hours=0.25):
    """
    Compute electricity bill from signed grid power.

    Bill = Σ [p_grid > 0] * p_grid * buy * Δt  −  Σ [p_grid < 0] * |p_grid| * sell * Δt

    Per Brief, Section 7 objective.

    Parameters
    ----------
    p_grid_kw  : array-like, signed grid power (positive = import, negative = export)
    buy_price  : array-like, €/kWh per timestep
    sell_price : array-like, €/kWh per timestep
    dt_hours   : timestep in hours (default 0.25)

    Returns
    -------
    bill : float, total bill in €
    """
    p = np.asarray(p_grid_kw, dtype=float)
    b = np.asarray(buy_price, dtype=float)
    s = np.asarray(sell_price, dtype=float)

    import_part = np.where(p > 0, p * b, 0.0).sum() * dt_hours
    export_part = np.where(p < 0, (-p) * s, 0.0).sum() * dt_hours
    return import_part - export_part


if __name__ == "__main__":
    # Smoke test
    ts = pd.date_range("2025-01-01 00:00", "2025-01-08 00:00", freq="15min", inclusive="left")
    bands = assign_tariff_band(ts)
    counts = bands.value_counts()
    print("Tariff band distribution for week of Jan 1-7, 2025:")
    print(counts)
    print(f"\nJan 1, 2025 (Wednesday, but holiday) noon: {bands.loc[pd.Timestamp('2025-01-01 12:00')]}")
    print(f"Jan 2, 2025 (Thursday) noon:                 {bands.loc[pd.Timestamp('2025-01-02 12:00')]}")
    print(f"Jan 4, 2025 (Saturday) 10am:                 {bands.loc[pd.Timestamp('2025-01-04 10:00')]}")
    print(f"Jan 5, 2025 (Sunday) noon:                   {bands.loc[pd.Timestamp('2025-01-05 12:00')]}")
