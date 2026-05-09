"""
Baselines per Brief Section 5.

Baseline A — Historical Operation
  Use the actual p_battery_kw column from the 2025 dataset.
  Compute the bill that the existing on-site controller produced.
  Our MPC must beat this.

Baseline B — Zero-Intelligence (No battery)
  PV serves load first. Surplus exported, deficit imported. Battery idle.
  Shows the value of having any controllable storage.
"""
import numpy as np
import pandas as pd
from config import GRID_P_MAX_KW, DT_HOURS
from tariff import compute_bill


def compute_baseline_a(df_2025, corruption_window=None):
    """
    Baseline A: replay the historical p_battery_kw signal to compute the bill.

    Parameters
    ----------
    df_2025          : DataFrame with load_kw, pv_kw, buy_price, sell_price, p_battery_kw
    corruption_window: optional (start_ts, end_ts) tuple. If provided, sets
                       p_battery to zero in this range (treats corruption as no-op).
                       This produces a "cleaned Baseline A" — useful when the
                       historical p_battery is unreliable in the corruption span.

    Returns
    -------
    dict with keys: dispatch (DataFrame), bill, total_imports_kwh, total_exports_kwh
    """
    df = df_2025.copy()
    p_battery = df["p_battery_kw"].astype(float).copy()

    # Optional cleaning of corruption window
    if corruption_window is not None:
        start, end = corruption_window
        mask = (df.index >= start) & (df.index <= end)
        p_battery.loc[mask] = 0.0

    # Energy balance: load = pv + p_battery + p_grid → p_grid = load - pv - p_battery
    p_grid = df["load_kw"] - df["pv_kw"] - p_battery

    # Clip to grid limits (in reality, the controller would have respected these,
    # but if dataset records a violation we clip and accept some unmet load)
    p_grid_clipped = p_grid.clip(-GRID_P_MAX_KW, GRID_P_MAX_KW)
    unmet = (p_grid - p_grid_clipped).clip(lower=0)   # positive when import > 6 kW

    bill = compute_bill(
        p_grid_clipped.values,
        df["buy_price"].values,
        df["sell_price"].values,
        dt_hours=DT_HOURS,
    )
    total_import = (p_grid_clipped.clip(lower=0).sum()) * DT_HOURS
    total_export = (-p_grid_clipped.clip(upper=0).sum()) * DT_HOURS

    dispatch = pd.DataFrame({
        "load_kw":   df["load_kw"].values,
        "pv_kw":     df["pv_kw"].values,
        "buy_price": df["buy_price"].values,
        "sell_price": df["sell_price"].values,
        "p_battery": p_battery.values,
        "p_grid":    p_grid_clipped.values,
        "unmet_load": unmet.values,
    }, index=df.index)

    return {
        "dispatch":          dispatch,
        "bill":              float(bill),
        "total_imports_kwh": float(total_import),
        "total_exports_kwh": float(total_export),
        "unmet_load_kwh":    float(unmet.sum() * DT_HOURS),
    }


def compute_baseline_b(df_2025):
    """
    Baseline B: no battery. PV serves load first; surplus exported, deficit imported.

    Battery stays at initial SoC throughout (specifically: never used).
    """
    df = df_2025.copy()
    net_load = df["load_kw"] - df["pv_kw"]   # positive = need import, negative = export

    # Clip to grid limits
    p_grid = net_load.clip(-GRID_P_MAX_KW, GRID_P_MAX_KW)
    unmet = (net_load - p_grid).clip(lower=0)

    bill = compute_bill(
        p_grid.values,
        df["buy_price"].values,
        df["sell_price"].values,
        dt_hours=DT_HOURS,
    )

    total_import = (p_grid.clip(lower=0).sum()) * DT_HOURS
    total_export = (-p_grid.clip(upper=0).sum()) * DT_HOURS

    dispatch = pd.DataFrame({
        "load_kw":   df["load_kw"].values,
        "pv_kw":     df["pv_kw"].values,
        "buy_price": df["buy_price"].values,
        "sell_price": df["sell_price"].values,
        "p_battery": np.zeros(len(df)),
        "p_grid":    p_grid.values,
        "unmet_load": unmet.values,
    }, index=df.index)

    return {
        "dispatch":          dispatch,
        "bill":              float(bill),
        "total_imports_kwh": float(total_import),
        "total_exports_kwh": float(total_export),
        "unmet_load_kwh":    float(unmet.sum() * DT_HOURS),
    }


def savings_summary(controller_bill, baseline_a_bill, baseline_b_bill):
    """Build the standard savings table."""
    return {
        "baseline_a_bill":        round(baseline_a_bill, 2),
        "baseline_b_bill":        round(baseline_b_bill, 2),
        "controller_bill":        round(controller_bill, 2),
        "savings_vs_a_eur":       round(baseline_a_bill - controller_bill, 2),
        "savings_vs_a_pct":       round(100 * (baseline_a_bill - controller_bill) / baseline_a_bill, 2),
        "savings_vs_b_eur":       round(baseline_b_bill - controller_bill, 2),
        "savings_vs_b_pct":       round(100 * (baseline_b_bill - controller_bill) / baseline_b_bill, 2),
    }


if __name__ == "__main__":
    from data_loader import load_dataset

    sheets = load_dataset("data/synthetic_dataset.xlsx")
    df_2025 = sheets["2025"]

    print("=== Baseline A (historical operation) ===")
    a = compute_baseline_a(df_2025)
    print(f"  Bill: €{a['bill']:.2f}")
    print(f"  Imports: {a['total_imports_kwh']:.1f} kWh,  Exports: {a['total_exports_kwh']:.1f} kWh")
    print(f"  Unmet load: {a['unmet_load_kwh']:.3f} kWh")

    print("\n=== Baseline B (no battery) ===")
    b = compute_baseline_b(df_2025)
    print(f"  Bill: €{b['bill']:.2f}")
    print(f"  Imports: {b['total_imports_kwh']:.1f} kWh,  Exports: {b['total_exports_kwh']:.1f} kWh")

    print(f"\nA vs B: A is €{b['bill'] - a['bill']:+.2f} relative to B "
          f"({100 * (b['bill'] - a['bill'])/b['bill']:+.1f}%)")
