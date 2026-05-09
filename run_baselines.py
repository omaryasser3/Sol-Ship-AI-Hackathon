"""
Compute Baseline A and Baseline B billing for:
  - Full Year 2025
  - April 2025
  - September 2025

Baseline A: bill using actual p_battery_kw from dataset (existing on-site controller).
Baseline B: zero-intelligence — PV serves load first, no battery, deficit imported, surplus exported.
"""

import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solship"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import holidays as hol_pkg

F1, F2, F3       = 0.2540, 0.2682, 0.2440
ITALIAN_HOLIDAYS = hol_pkg.Italy(years=[2024, 2025])
DT_HOURS         = 0.25

def compute_buy_price(ts):
    dow = ts.dt.dayofweek; h = ts.dt.hour
    hol = ts.dt.date.map(lambda d: d in ITALIAN_HOLIDAYS)
    f3  = (dow==6)|hol|(~(dow==6)&~hol&((h<7)|(h>=23)))
    f2  = (~f3)&(((dow==5)&(h>=7)&(h<23))|(~(dow==5)&~(dow==6)&~hol&((h==7)|((h>=19)&(h<23)))))
    p   = pd.Series(F1, index=ts.index, dtype=float); p[f2]=F2; p[f3]=F3
    return p

# ── load data ─────────────────────────────────────────────────────────────────
raw = pd.read_csv("ENERGY_Hackathon_DataSet(Sheet1).csv", sep=";", decimal=",")
raw["timestamp"] = pd.to_datetime(raw["timestamp"])
raw = (raw.sort_values("timestamp")
          .drop_duplicates(subset="timestamp", keep="first")
          .reset_index(drop=True))
raw = raw.rename(columns={"load_p":"load_kw","pv_p":"pv_kw",
                           "Selling_price_eur_kwh":"sell_price",
                           "battery_p":"p_battery_kw","grid_p":"grid_kw"})
raw["sell_price"] = raw["sell_price"].ffill()
raw["buy_price"]  = compute_buy_price(raw["timestamp"])
idx = pd.date_range(raw["timestamp"].min(), raw["timestamp"].max(), freq="15min")
raw = raw.set_index("timestamp").reindex(idx).ffill()
raw["load_kw"]     = raw["load_kw"].clip(lower=0)
raw["pv_kw"]       = raw["pv_kw"].clip(lower=0)

# ── periods ───────────────────────────────────────────────────────────────────
periods = {
    "Full Year 2025": raw[raw.index.year == 2025],
    "April 2025":     raw[(raw.index.year == 2025) & (raw.index.month == 4)],
    "September 2025": raw[(raw.index.year == 2025) & (raw.index.month == 9)],
}

# ── billing functions ─────────────────────────────────────────────────────────
def bill_baseline_a(df):
    """Use actual grid_kw from dataset directly (recorded P_grid from sensors)."""
    p_imp = df["grid_kw"].clip(lower=0)
    p_exp = (-df["grid_kw"]).clip(lower=0)
    return float((p_imp * df["buy_price"] - p_exp * df["sell_price"]).sum() * DT_HOURS)

def bill_baseline_b(df):
    """Zero-intelligence: no battery. PV serves load, rest imported/exported.
    No grid cap — the brief defines this as pure surplus/deficit with no constraints."""
    net    = df["load_kw"] - df["pv_kw"]
    p_imp  = net.clip(lower=0)
    p_exp  = (-net).clip(lower=0)
    return float((p_imp * df["buy_price"] - p_exp * df["sell_price"]).sum() * DT_HOURS)

# ── compute and print ─────────────────────────────────────────────────────────
print("="*65)
print("BILLING BASELINES")
print("="*65)
print(f"  {'Period':<20s}  {'Baseline A (EUR)':>16s}  {'Baseline B (EUR)':>16s}")
print("-"*65)

rows = []
for period, df in periods.items():
    a = bill_baseline_a(df)
    b = bill_baseline_b(df)
    rows.append({"period": period, "baseline_a_eur": round(a,2), "baseline_b_eur": round(b,2)})
    print(f"  {period:<20s}  {a:>16.2f}  {b:>16.2f}")
    print()

pd.DataFrame(rows).to_csv("baseline_bills.csv", index=False)
print("Saved: baseline_bills.csv")
