"""
main.py — End-to-end pipeline for the Solship Energy AI Hackathon 2026.

Runs in order:
  1. Load & clean data
  2. Forecast April and September 2025 (LightGBM, multi-horizon)
  3. Build Baseline A (historical replay) and Baseline B (no battery)
  4. Run oracle MPC on March 2025 → mandatory dispatch plot
  5. Run MPC controller on April and September using the forecast
  6. Run oracle MPC on April and September (perfect-foresight upper bound)
  7. Compute bills, savings, oracle gap
  8. Generate all required plots + results table

Usage:
    python main.py
    python main.py --skip-forecast   # re-use a cached period_outputs dict (not implemented yet)
"""
import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── import reusable modules ────────────────────────────────────────────────────
from forecasting import (
    load_data, run_period, build_mpc_forecast_matrix,
    SPARSE_HORIZONS, H_MAX,
)
from optimizer  import run_mpc, validate_dispatch
from baselines  import compute_baseline_a, compute_baseline_b, savings_summary
from metrics    import (oracle_gap_analysis, forecast_metrics,
                        print_results_table, compute_dispatch_bill)
from plots      import (plot_march_week3_dispatch, plot_oracle_gap_summary,
                        plot_horizon_sensitivity)
from tariff     import compute_bill
from config     import DT_HOURS, SOC_INIT, HORIZON_OPTIONS

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════
raw = load_data()
df_2025 = raw[raw.index.year == 2025]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Forecast April and September (h=1 + multi-horizon for MPC)
# ══════════════════════════════════════════════════════════════════════════════
PERIODS = [
    (4, "April 2025",     pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30 23:45")),
    (9, "September 2025", pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-30 23:45")),
]

period_outputs = {}
for m_num, label, m_start, m_end in PERIODS:
    period_outputs[label] = run_period(m_num, label, m_start, m_end, raw)

# Build MPC forecast matrices (n_timesteps × 96 columns, one per period)
mpc_matrices = {}
for m_num, label, m_start, m_end in PERIODS:
    po = period_outputs[label]
    mpc_matrices[label] = build_mpc_forecast_matrix(
        po["pred_hub"], po["pred_qnt"], po["best_alpha_hub"],
        common_index=po["common"], H=96,
    )
    print(f"  MPC forecast matrix [{label}]: {mpc_matrices[label].shape}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Baselines (full 2025 — for the annual bill comparison)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASELINES")
print("=" * 60)
bl_a = compute_baseline_a(df_2025)
bl_b = compute_baseline_b(df_2025)
print(f"  Baseline A bill : €{bl_a['bill']:.2f}  "
      f"(imports {bl_a['total_imports_kwh']:.0f} kWh, exports {bl_a['total_exports_kwh']:.0f} kWh)")
print(f"  Baseline B bill : €{bl_b['bill']:.2f}  "
      f"(imports {bl_b['total_imports_kwh']:.0f} kWh, exports {bl_b['total_exports_kwh']:.0f} kWh)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Mandatory March Week 3 dispatch plot
#    Run oracle MPC on March 10–23 (extra context so SoC settles before week 3)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MARCH WEEK 3 — ORACLE DISPATCH (mandatory plot)")
print("=" * 60)
march_ctx = df_2025.loc["2025-03-10":"2025-03-23 23:45"].copy()
march_dispatch = run_mpc(march_ctx, forecasts=None, horizon=96,
                          soc_init=SOC_INIT, verbose_every=400)
ok, rep = validate_dispatch(march_dispatch)
print(f"  Dispatch validation: {'PASS' if ok else 'FAIL'}")
for k, v in rep.items():
    print(f"    {k}: {v}")

march_plot_path = os.path.join(PLOTS_DIR, "march_week3_dispatch.png")
plot_march_week3_dispatch(march_dispatch, out_path=march_plot_path)


# ══════════════════════════════════════════════════════════════════════════════
# 5. MPC controller — April and September using the forecast
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MPC CONTROLLER — FORECAST-BASED")
print("=" * 60)

controller_bills = {}
controller_dispatches = {}

for m_num, label, m_start, m_end in PERIODS:
    print(f"\n  Running MPC [{label}] …")
    df_period = df_2025.loc[m_start:m_end].copy()
    fc_matrix = mpc_matrices[label]

    t0 = time.time()
    dispatch = run_mpc(df_period, forecasts=fc_matrix, horizon=96,
                        soc_init=SOC_INIT, verbose_every=480)
    print(f"  Done in {time.time()-t0:.1f}s")

    ok, rep = validate_dispatch(dispatch)
    print(f"  Validation: {'PASS' if ok else 'FAIL'}  "
          f"SoC [{rep['soc_min']:.3f}, {rep['soc_max']:.3f}]  "
          f"grid_max {rep['p_grid_max']:.2f} kW")

    bill = compute_dispatch_bill(dispatch)
    controller_bills[label] = bill
    controller_dispatches[label] = dispatch
    print(f"  Controller bill [{label}]: €{bill:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Oracle MPC — same periods, actual load as input
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ORACLE MPC — PERFECT FORESIGHT")
print("=" * 60)

oracle_bills = {}
oracle_dispatches = {}

for m_num, label, m_start, m_end in PERIODS:
    print(f"\n  Running oracle MPC [{label}] …")
    df_period = df_2025.loc[m_start:m_end].copy()

    t0 = time.time()
    dispatch = run_mpc(df_period, forecasts=None, horizon=96,
                        soc_init=SOC_INIT, verbose_every=480)
    print(f"  Done in {time.time()-t0:.1f}s")

    bill = compute_dispatch_bill(dispatch)
    oracle_bills[label] = bill
    oracle_dispatches[label] = dispatch
    print(f"  Oracle bill [{label}]: €{bill:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Results table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

for m_num, label, m_start, m_end in PERIODS:
    po = period_outputs[label]
    result_df = po["result"]

    fm = forecast_metrics(result_df["actual_kw"].values,
                           result_df["forecast_kw"].values)

    ctrl_bill   = controller_bills[label]
    oracle_bill = oracle_bills[label]

    # For period-level baselines, compute from the slice
    df_period = df_2025.loc[m_start:m_end].copy()
    bl_a_p = compute_baseline_a(df_period)
    bl_b_p = compute_baseline_b(df_period)

    gap = oracle_gap_analysis(
        controller_dispatches[label],
        oracle_dispatches[label],
        bl_a_p["bill"],
        bl_b_p["bill"],
    )

    print(f"\n  ── {label} ──────────────────────────")
    print(f"  Forecast  NRMSE={fm['nrmse']:.2f}%  RMSE={fm['rmse']:.4f} kW  MAE={fm['mae']:.4f} kW")
    print_results_table(gap)

    # Oracle gap plot
    plot_oracle_gap_summary(
        ctrl_bill, oracle_bill, bl_a_p["bill"],
        out_path=os.path.join(PLOTS_DIR, f"oracle_gap_{label.lower().replace(' ','_')}.png"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8. Extension: horizon sensitivity (H = 4, 24, 96) on April — +5 bonus points
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXTENSION — HORIZON SENSITIVITY")
print("=" * 60)

label_ext = "April 2025"
m_start_ext, m_end_ext = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30 23:45")
df_april = df_2025.loc[m_start_ext:m_end_ext].copy()
fc_april = mpc_matrices[label_ext]

horizon_results = []
for H in HORIZON_OPTIONS:
    print(f"  H={H} ({H//4}h) …")
    t0 = time.time()
    disp = run_mpc(df_april, forecasts=fc_april, horizon=H,
                    soc_init=SOC_INIT, verbose_every=0)
    elapsed = time.time() - t0
    bill = compute_dispatch_bill(disp)
    horizon_results.append({"horizon": H, "bill": bill, "compute_time_s": elapsed})
    print(f"    bill=€{bill:.2f}  time={elapsed:.1f}s")

plot_horizon_sensitivity(
    horizon_results,
    out_path=os.path.join(PLOTS_DIR, "horizon_sensitivity.png"),
)

print("\n" + "=" * 60)
print("ALL DONE — outputs in outputs/plots/")
print("=" * 60)
