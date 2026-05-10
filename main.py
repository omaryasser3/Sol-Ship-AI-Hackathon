"""
main.py — End-to-end pipeline for the Solship Energy AI Hackathon 2026.

Runs in order:
  1. Load & clean data
  2. Forecast March, April and September 2025 (LightGBM, multi-horizon)
     March  → train Jan 2024 – Feb 2025  (needed for mandatory dispatch plot)
     April  → train Jan 2024 – Mar 2025
     Sep    → train Jan 2024 – Aug 2025
  3. Corruption detection — reconstruct SoC from p_battery_kw, flag bad window
     (Brief Section 3 — mandatory; primary window zeroed in Baseline A)
  4. Baselines A (historical, corruption-cleaned) and B (no battery)
  5. Run forecast-based MPC on March 2025 → mandatory dispatch plot
     (uses ML forecast, NOT oracle — brief requires controller's actual dispatch)
  6. Run MPC controller on April and September using the forecast
  7. Run oracle MPC on April and September (perfect-foresight upper bound)
  8. Compute bills, savings, oracle gap
  9. Generate all required plots + results table

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
    # submission export helpers (ensure submission_forecast/ outputs are regenerated)
    compute_metrics as fc_compute_metrics,
    save_metrics_txt as fc_save_metrics_txt,
    make_plots as fc_make_plots,
    save_excel as fc_save_excel,
    OUT_DIR as FC_OUT_DIR,
)
from optimizer  import run_mpc, validate_dispatch
from baselines  import (compute_baseline_a, compute_baseline_b, savings_summary,
                        detect_corruption, print_corruption_report)
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
# 2. Forecast March, April and September (h=1 + multi-horizon for MPC)
#
#    March is included solely to produce the mandatory Week 3 dispatch plot
#    using the real ML forecast (not oracle).  It is NOT a scored period.
#    Train cutoff for March: Jan 2024 – Feb 28 2025 (everything before March).
# ══════════════════════════════════════════════════════════════════════════════
PERIODS = [
    (3, "March 2025",     pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-31 23:45")),
    (4, "April 2025",     pd.Timestamp("2025-04-01"), pd.Timestamp("2025-04-30 23:45")),
    (9, "September 2025", pd.Timestamp("2025-09-01"), pd.Timestamp("2025-09-30 23:45")),
]

# Scored periods only (April + September) — used for results table / oracle gap
SCORED_PERIODS = [p for p in PERIODS if p[0] != 3]

period_outputs = {}
for m_num, label, m_start, m_end in PERIODS:
    period_outputs[label] = run_period(m_num, label, m_start, m_end, raw)

# Ensure submission_forecast outputs (metrics, plots, excel) are generated
# when running via main.py (forecasting.py only writes these when run as script).
for label, po in period_outputs.items():
    try:
        m = fc_compute_metrics(po["result"], label)
        slug = label.lower().replace(" ", "_")
        fc_save_metrics_txt(m, os.path.join(FC_OUT_DIR, f"metrics_{slug}.txt"))
        plot_paths = fc_make_plots(po["result"], m, slug)
        fc_save_excel(po["result"], m, slug, plot_paths[0], plot_paths[1], plot_paths[2])
    except Exception as e:
        print(f"  Warning: failed to export submission files for {label}: {e}")

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
# 3. Corruption detection (Brief Section 3 — mandatory)
#    Reconstruct SoC from p_battery_kw using η=√0.90 per direction.
#    Flag timesteps where unclipped SoC goes outside [0, 1].
#    The primary (largest) corruption window is zeroed in Baseline A.
# ══════════════════════════════════════════════════════════════════════════════
corruption = detect_corruption(df_2025)
print_corruption_report(corruption)

# Save corruption evidence plot (uses plots.py helper)
if corruption["primary_window"] is not None:
    from plots import plot_corruption_evidence
    plot_corruption_evidence(
        timestamps=df_2025.index,
        p_battery=df_2025["p_battery_kw"].values,
        soc_unclipped=corruption["soc_reconstructed"].values,
        soc_clipped=corruption["soc_clipped"].values,
        windows_df=corruption["windows"],
        out_path=os.path.join(PLOTS_DIR, "corruption_evidence.png"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. Baselines (full 2025 — for the annual bill comparison)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("BASELINES")
print("=" * 60)
bl_a = compute_baseline_a(df_2025, corruption_window=corruption["primary_window"])
bl_b = compute_baseline_b(df_2025)
print(f"  Baseline A bill : €{bl_a['bill']:.2f}  "
      f"(imports {bl_a['total_imports_kwh']:.0f} kWh, exports {bl_a['total_exports_kwh']:.0f} kWh)")
print(f"  Baseline B bill : €{bl_b['bill']:.2f}  "
      f"(imports {bl_b['total_imports_kwh']:.0f} kWh, exports {bl_b['total_exports_kwh']:.0f} kWh)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Mandatory March Week 3 dispatch plot
#    Run FORECAST-BASED MPC on full March (context from Mar 1 so SoC settles
#    naturally before Week 3).  Uses the ML forecast — NOT oracle — so the plot
#    genuinely represents what the controller does with imperfect information.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MARCH WEEK 3 — FORECAST-BASED DISPATCH (mandatory plot)")
print("=" * 60)

df_march   = df_2025.loc["2025-03-01":"2025-03-31 23:45"].copy()
fc_march   = mpc_matrices["March 2025"]

# Align: run_mpc will intersect df_march.index with fc_march.index internally
t0 = time.time()
march_dispatch = run_mpc(df_march, forecasts=fc_march, horizon=96,
                          soc_init=SOC_INIT, verbose_every=400)
print(f"  Done in {time.time()-t0:.1f}s")

ok, rep = validate_dispatch(march_dispatch)
print(f"  Dispatch validation: {'PASS' if ok else 'FAIL'}")
for k, v in rep.items():
    print(f"    {k}: {v}")

march_plot_path = os.path.join(PLOTS_DIR, "march_week3_dispatch.png")
# Also run oracle MPC for March to provide an oracle overlay on the mandatory plot
try:
    oracle_march_dispatch = run_mpc(df_march, forecasts=None, horizon=96,
                                     soc_init=SOC_INIT, verbose_every=400)
    plot_march_week3_dispatch(march_dispatch, out_path=march_plot_path,
                               oracle_dispatch_df=oracle_march_dispatch)
except Exception:
    # Fallback: plot without oracle if oracle run fails
    plot_march_week3_dispatch(march_dispatch, out_path=march_plot_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. MPC controller — April and September using the forecast
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MPC CONTROLLER — FORECAST-BASED")
print("=" * 60)

controller_bills = {}
controller_dispatches = {}

for m_num, label, m_start, m_end in SCORED_PERIODS:
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
# 7. Oracle MPC — same periods, actual load as input
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ORACLE MPC — PERFECT FORESIGHT")
print("=" * 60)

oracle_bills = {}
oracle_dispatches = {}

for m_num, label, m_start, m_end in SCORED_PERIODS:
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
# 8. Results table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

for m_num, label, m_start, m_end in SCORED_PERIODS:
    po = period_outputs[label]
    result_df = po["result"]

    fm = forecast_metrics(result_df["actual_kw"].values,
                           result_df["forecast_kw"].values)

    ctrl_bill   = controller_bills[label]
    oracle_bill = oracle_bills[label]

    # For period-level baselines, compute from the slice
    # Use the raw historical `p_battery_kw` for period-level Baseline A so
    # it reflects the actual on-site controller (do NOT zero the corruption
    # window here — that cleaning is for full-year reporting only).
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
# 9. Extension: horizon sensitivity (H = 4, 24, 96) on April — +5 bonus points
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
