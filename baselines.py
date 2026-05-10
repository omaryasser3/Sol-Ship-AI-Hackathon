"""
Baselines per Brief Section 5.

Baseline A — Historical Operation
  Use the actual p_battery_kw column from the 2025 dataset.
  Compute the bill that the existing on-site controller produced.
  Our MPC must beat this.

Baseline B — Zero-Intelligence (No battery)
  PV serves load first. Surplus exported, deficit imported. Battery idle.
  Shows the value of having any controllable storage.

Corruption Detection (Brief Section 3)
  The 2025 p_battery_kw column contains a multi-week window with values
  inconsistent with the site energy balance.  detect_corruption() reconstructs
  the battery SoC from scratch using the exact efficiency formula from the brief
  (√0.90 per direction) and flags every timestep where the cumulative SoC would
  fall below 0 or exceed 1.  Contiguous violation runs are merged into windows;
  the largest window is returned as the primary corruption_window to pass into
  compute_baseline_a().
"""
import numpy as np
import pandas as pd
from config import GRID_P_MAX_KW, DT_HOURS, BATTERY_KWH, ETA_DIR, SOC_INIT
from tariff import compute_bill


def detect_corruption(df_2025, soc_init=SOC_INIT, merge_gap_steps=96,
                      min_window_steps=4):
    """
    Reconstruct battery SoC from p_battery_kw and identify corrupted windows.

    Per Brief Section 3:
      "Reconstruct the battery SoC trajectory from scratch and verify it is
       physically consistent.  Unexplained SoC drift is a sign of corrupted data."

    Method
    ------
    At each timestep t, signed p_battery_kw is split into charge / discharge
    using the brief's sign convention (negative = charging, positive = discharging)
    and the per-direction efficiency η = √0.90 ≈ 0.9487:

        p_ch  = max(-p_battery, 0)          # charging power  (≥ 0)
        p_dis = max( p_battery, 0)          # discharge power (≥ 0)
        ΔE    = (p_ch * η  −  p_dis / η) * Δt
        SoC[t+1] = SoC[t] + ΔE / E_max

    A timestep is flagged as a violation when the UNCLIPPED SoC would go
    below 0 or above 1.  Contiguous violation runs closer than `merge_gap_steps`
    apart are merged into a single window.  Windows shorter than
    `min_window_steps` are discarded as numerical noise.

    Parameters
    ----------
    df_2025          : DataFrame indexed by 15-min timestamps, must contain
                       'p_battery_kw', 'load_kw', 'pv_kw'
    soc_init         : initial SoC at start of the series (default 0.50)
    merge_gap_steps  : merge violation windows separated by fewer steps (default 96 = 1 day)
    min_window_steps : discard windows shorter than this (default 4 = 1 hour)

    Returns
    -------
    dict with keys:
      soc_reconstructed  : pd.Series  — unclipped SoC trajectory (may exceed [0,1])
      soc_clipped        : pd.Series  — SoC clipped to [0, 1] (physical reference)
      violation_mask     : pd.Series  — bool, True at every violation timestep
      windows            : pd.DataFrame — columns: start, end, n_steps, max_drift
                           one row per detected corruption window, sorted by n_steps desc
      primary_window     : (start_ts, end_ts) tuple for the largest window,
                           or None if no corruption detected
      energy_balance_residual : pd.Series — |load − pv − p_battery − p_grid_implied|
                                as a secondary consistency check
    """
    df = df_2025.copy()
    p_bat = df["p_battery_kw"].astype(float).values
    n     = len(p_bat)

    # ── SoC reconstruction ────────────────────────────────────────────────────
    soc_unclipped = np.empty(n + 1)
    soc_unclipped[0] = soc_init

    for t in range(n):
        p_ch  = max(-p_bat[t], 0.0)   # charging:    p_battery < 0
        p_dis = max( p_bat[t], 0.0)   # discharging: p_battery > 0
        delta_e = (p_ch * ETA_DIR - p_dis / ETA_DIR) * DT_HOURS
        soc_unclipped[t + 1] = soc_unclipped[t] + delta_e / BATTERY_KWH

    # Align to df index (SoC after each step = state entering next step)
    soc_series     = pd.Series(soc_unclipped[1:], index=df.index, name="soc_reconstructed")
    soc_clipped    = soc_series.clip(0.0, 1.0).rename("soc_clipped")

    # ── violation mask ─────────────────────────────────────────────────────────
    violation_mask = ((soc_series < -1e-4) | (soc_series > 1.0 + 1e-4))
    violation_mask.name = "violation"

    # ── merge contiguous violation runs ───────────────────────────────────────
    viol_idx = np.where(violation_mask.values)[0]
    windows_raw = []
    if len(viol_idx) > 0:
        run_start = viol_idx[0]
        run_end   = viol_idx[0]
        for i in viol_idx[1:]:
            if i - run_end <= merge_gap_steps:
                run_end = i
            else:
                windows_raw.append((run_start, run_end))
                run_start = i
                run_end   = i
        windows_raw.append((run_start, run_end))

    # Filter short windows and build DataFrame
    rows = []
    for s, e in windows_raw:
        if (e - s + 1) < min_window_steps:
            continue
        ts_start = df.index[s]
        ts_end   = df.index[e]
        soc_slice = soc_series.iloc[s : e + 1]
        max_drift = float(max(
            abs(soc_slice.min()),           # how far below 0
            abs(soc_slice.max() - 1.0),     # how far above 1
        ))
        rows.append({
            "start":     ts_start,
            "end":       ts_end,
            "n_steps":   e - s + 1,
            "duration_days": round((e - s + 1) * DT_HOURS / 24, 1),
            "max_drift": round(max_drift, 4),
        })

    windows_df = (pd.DataFrame(rows)
                  .sort_values("n_steps", ascending=False)
                  .reset_index(drop=True)
                  if rows else pd.DataFrame(
                      columns=["start", "end", "n_steps", "duration_days", "max_drift"]))

    primary_window = None
    if len(windows_df) > 0:
        primary_window = (windows_df.iloc[0]["start"], windows_df.iloc[0]["end"])

    # ── secondary check: energy balance residual ──────────────────────────────
    # p_grid_implied = load - pv - p_battery  (from energy balance)
    # If the sensor is corrupted, this will produce physically impossible grid values.
    p_grid_implied = df["load_kw"] - df["pv_kw"] - df["p_battery_kw"]
    grid_violation = (p_grid_implied.abs() > GRID_P_MAX_KW + 0.1)

    return {
        "soc_reconstructed":        soc_series,
        "soc_clipped":              soc_clipped,
        "violation_mask":           violation_mask,
        "windows":                  windows_df,
        "primary_window":           primary_window,
        "grid_violation_mask":      grid_violation,
        "n_soc_violations":         int(violation_mask.sum()),
        "n_grid_violations":        int(grid_violation.sum()),
    }


def print_corruption_report(result):
    """Print a human-readable corruption detection report."""
    print("\n" + "=" * 60)
    print("CORRUPTION DETECTION REPORT  (Brief Section 3)")
    print("=" * 60)
    print(f"  SoC violations (unclipped outside [0,1]) : {result['n_soc_violations']:,} steps")
    print(f"  Grid violations (|p_grid_implied| > 6 kW): {result['n_grid_violations']:,} steps")

    wdf = result["windows"]
    if len(wdf) == 0:
        print("\n  No corruption windows detected — p_battery_kw appears physically consistent.")
        return

    print(f"\n  Detected {len(wdf)} corruption window(s):\n")
    print(f"  {'#':<4} {'Start':<22} {'End':<22} {'Steps':>6} {'Days':>6} {'MaxDrift':>10}")
    print(f"  {'─'*4} {'─'*22} {'─'*22} {'─'*6} {'─'*6} {'─'*10}")
    for i, row in wdf.iterrows():
        marker = "  ◄ PRIMARY" if i == 0 else ""
        print(f"  {i+1:<4} {str(row['start']):<22} {str(row['end']):<22} "
              f"{row['n_steps']:>6} {row['duration_days']:>6.1f} "
              f"{row['max_drift']:>10.4f}{marker}")

    pw = result["primary_window"]
    print(f"\n  Primary window passed to compute_baseline_a:")
    print(f"    Start : {pw[0]}")
    print(f"    End   : {pw[1]}")
    print(f"    p_battery zeroed in this range → treated as no-op for bill calculation")
    print("=" * 60)


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

    print("=== Corruption Detection ===")
    corruption = detect_corruption(df_2025)
    print_corruption_report(corruption)

    print("\n=== Baseline A (historical operation) ===")
    a = compute_baseline_a(df_2025, corruption_window=corruption["primary_window"])
    print(f"  Bill: €{a['bill']:.2f}")
    print(f"  Imports: {a['total_imports_kwh']:.1f} kWh,  Exports: {a['total_exports_kwh']:.1f} kWh")
    print(f"  Unmet load: {a['unmet_load_kwh']:.3f} kWh")

    print("\n=== Baseline B (no battery) ===")
    b = compute_baseline_b(df_2025)
    print(f"  Bill: €{b['bill']:.2f}")
    print(f"  Imports: {b['total_imports_kwh']:.1f} kWh,  Exports: {b['total_exports_kwh']:.1f} kWh")

    print(f"\nA vs B: A is €{b['bill'] - a['bill']:+.2f} relative to B "
          f"({100 * (b['bill'] - a['bill'])/b['bill']:+.1f}%)")
