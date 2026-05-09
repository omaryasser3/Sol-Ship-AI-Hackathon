"""
Plotting utilities.

The MANDATORY plot per Brief Section 7:
  Dispatch plot for Week 3 of March 2025 — load, PV, P_battery, P_grid, SoC.

Also helpful EDA plots for the slide deck.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from config import F1_PRICE, F2_PRICE, F3_PRICE
from tariff import assign_tariff_band


# Solship-themed palette: deep navy for primary, teal for accent, amber for warnings
COLORS = {
    "load":      "#1E2761",   # Navy
    "pv":        "#F4A300",   # Amber (sun)
    "battery":   "#02C39A",   # Mint (the smart one)
    "grid":      "#990011",   # Cherry (cost)
    "soc":       "#0891B2",   # Teal
    "F1":        "#FCF6F5",   # near-white (normal)
    "F2":        "#FEEAEA",   # pinkish (expensive)
    "F3":        "#E8F4F8",   # light blue (cheap)
}


def plot_march_week3_dispatch(dispatch_df, out_path="plots/march_week3.png",
                               title="March Week 3, 2025 — Dispatch Plot",
                               year=2025):
    """
    The mandatory dispatch plot. 5 panels (sharing x-axis):
      1. Load + PV
      2. P_battery (signed: negative = charging, positive = discharging)
      3. P_grid (signed: positive = import, negative = export)
      4. SoC (0..1)
      5. Tariff bands (background-shaded along time axis)

    Week 3 of March 2025 = Monday March 17 → Sunday March 23 inclusive.
    (Italian week conventions: Mon-Sun.)
    """
    week_start = pd.Timestamp(f"{year}-03-17 00:00")
    week_end   = pd.Timestamp(f"{year}-03-24 00:00")
    sub = dispatch_df.loc[(dispatch_df.index >= week_start) &
                           (dispatch_df.index < week_end)].copy()

    if len(sub) == 0:
        # Fallback to first week if March data not available (synthetic data may not cover it)
        sub = dispatch_df.iloc[:96 * 7].copy()
        title = title.replace("March Week 3, 2025", f"First Week ({sub.index[0].date()} → {sub.index[-1].date()})")

    bands = assign_tariff_band(sub.index)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1.5, 1.5, 1.2]})

    # --- Panel 1: Load + PV ---
    ax = axes[0]
    _shade_tariff_bands(ax, sub.index, bands)
    ax.plot(sub.index, sub["load_kw"], color=COLORS["load"], lw=1.5, label="Load")
    ax.fill_between(sub.index, 0, sub["pv_kw"], color=COLORS["pv"], alpha=0.4, label="PV")
    ax.set_ylabel("Power (kW)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- Panel 2: P_battery (signed) ---
    ax = axes[1]
    _shade_tariff_bands(ax, sub.index, bands)
    pb = sub["p_battery"].values
    ax.fill_between(sub.index, 0, np.where(pb > 0, pb, 0),
                     color="#02C39A", alpha=0.7, label="Discharging")
    ax.fill_between(sub.index, 0, np.where(pb < 0, pb, 0),
                     color="#80CBC4", alpha=0.9, label="Charging")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("P_battery (kW)\n+ discharge, − charge")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: P_grid (signed) ---
    ax = axes[2]
    _shade_tariff_bands(ax, sub.index, bands)
    pg = sub["p_grid"].values
    ax.fill_between(sub.index, 0, np.where(pg > 0, pg, 0),
                     color="#990011", alpha=0.6, label="Import")
    ax.fill_between(sub.index, 0, np.where(pg < 0, pg, 0),
                     color="#EE6C4D", alpha=0.6, label="Export")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("P_grid (kW)\n+ import, − export")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: SoC ---
    ax = axes[3]
    _shade_tariff_bands(ax, sub.index, bands)
    ax.plot(sub.index, sub["soc"], color=COLORS["soc"], lw=1.8, label="SoC")
    ax.axhline(0, color="grey", lw=0.5, linestyle="--")
    ax.axhline(1, color="grey", lw=0.5, linestyle="--")
    ax.fill_between(sub.index, 0, sub["soc"], color=COLORS["soc"], alpha=0.15)
    ax.set_ylabel("SoC")
    ax.set_xlabel("Time")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # X-axis formatting (daily ticks)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0, ha="center")

    # Tariff legend overlay
    ax_legend = axes[0]
    from matplotlib.patches import Patch
    handles, labels = ax_legend.get_legend_handles_labels()
    handles += [
        Patch(facecolor=COLORS["F1"], edgecolor="grey", label=f"F1 (€{F1_PRICE:.4f})"),
        Patch(facecolor=COLORS["F2"], edgecolor="grey", label=f"F2 (€{F2_PRICE:.4f})"),
        Patch(facecolor=COLORS["F3"], edgecolor="grey", label=f"F3 (€{F3_PRICE:.4f})"),
    ]
    ax_legend.legend(handles=handles, loc="upper right", framealpha=0.95, fontsize=8, ncol=2)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _shade_tariff_bands(ax, timestamps, bands):
    """Shade the background of a plot panel by tariff band."""
    # Find contiguous runs of the same band
    band_array = bands.values
    if len(band_array) == 0:
        return
    runs = []
    cur_band = band_array[0]
    cur_start = timestamps[0]
    for i in range(1, len(band_array)):
        if band_array[i] != cur_band:
            runs.append((cur_band, cur_start, timestamps[i]))
            cur_band = band_array[i]
            cur_start = timestamps[i]
    runs.append((cur_band, cur_start, timestamps[-1] + pd.Timedelta("15min")))

    for band, start, end in runs:
        color = COLORS.get(band, "white")
        ax.axvspan(start, end, color=color, alpha=0.5, zorder=0)


def plot_forecast_vs_actual(y_true, y_pred, out_path="plots/forecast_vs_actual.png",
                              title="Forecast vs. Actual (1-week sample)",
                              n_steps=672):
    """Plot first n_steps of forecast vs actual."""
    fig, ax = plt.subplots(figsize=(14, 4.5))
    idx = y_true.index[:n_steps]
    ax.plot(idx, y_true.iloc[:n_steps], color=COLORS["load"], lw=1.4, label="Actual")
    ax.plot(idx, y_pred.iloc[:n_steps], color=COLORS["battery"], lw=1.0, alpha=0.85,
            label="Forecast")
    ax.set_ylabel("Load (kW)")
    ax.set_xlabel("Time")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_corruption_evidence(timestamps, p_battery, soc_unclipped, soc_clipped,
                               windows_df, out_path="plots/corruption_evidence.png"):
    """Plot showing corruption window detection."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1.2, 2]})

    ax = axes[0]
    ax.plot(timestamps, p_battery, color=COLORS["battery"], lw=0.5)
    ax.set_ylabel("p_battery (kW)\nas recorded")
    ax.grid(True, alpha=0.3)
    ax.set_title("p_battery_kw — historical record + detected corruption windows",
                  fontsize=12, fontweight="bold")

    ax = axes[1]
    ax.plot(timestamps, soc_clipped, color=COLORS["soc"], lw=1.2,
            label="SoC (clipped to [0, 1])")
    ax.plot(timestamps, soc_unclipped, color="#990011", lw=1.0, alpha=0.7,
            label="SoC (unclipped — physics)")
    ax.axhline(0, color="grey", linestyle="--", lw=0.5)
    ax.axhline(1, color="grey", linestyle="--", lw=0.5)
    ax.set_ylabel("SoC")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Highlight corruption windows
    for _, row in windows_df.iterrows():
        for ax_i in axes:
            ax_i.axvspan(row["start"], row["end"], color="red", alpha=0.15)
            ax_i.axvline(row["start"], color="red", lw=0.7, linestyle=":")
            ax_i.axvline(row["end"],   color="red", lw=0.7, linestyle=":")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_horizon_sensitivity(horizon_results, out_path="plots/horizon_sensitivity.png"):
    """Bar/line plot of bill vs horizon for the extension."""
    horizons = [r["horizon"] for r in horizon_results]
    bills = [r["bill"] for r in horizon_results]
    times = [r["compute_time_s"] for r in horizon_results]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(horizons, bills, "o-", color=COLORS["grid"], lw=2, markersize=10,
             label="Bill (€)")
    ax2.plot(horizons, times, "s--", color=COLORS["battery"], lw=1.5, markersize=8,
             label="Compute time (s)")

    ax1.set_xlabel("Horizon H (15-min steps)")
    ax1.set_ylabel("2025 bill (€)", color=COLORS["grid"])
    ax2.set_ylabel("Compute time (s)", color=COLORS["battery"])
    ax1.tick_params(axis="y", labelcolor=COLORS["grid"])
    ax2.tick_params(axis="y", labelcolor=COLORS["battery"])

    ax1.set_xticks(horizons)
    ax1.set_xticklabels([f"{h}\n({h//4}h)" for h in horizons])
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Horizon Sensitivity — Bill vs Compute Time",
                   fontsize=12, fontweight="bold")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_oracle_gap_summary(controller_bill, oracle_bill, baseline_a_bill,
                              out_path="plots/oracle_gap.png"):
    """Bar comparison: Baseline A vs Controller (with forecast) vs Oracle."""
    labels = ["Baseline A\n(historical)", "Controller\n(forecast)", "Oracle\n(perfect)"]
    bills = [baseline_a_bill, controller_bill, oracle_bill]
    colors = [COLORS["grid"], COLORS["battery"], COLORS["soc"]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, bills, color=colors, edgecolor="black", lw=0.5)
    for bar, bill in zip(bars, bills):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                 f"€{bill:.0f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Annual bill (€)")
    ax.set_title("Bill Comparison — Forecast vs Oracle",
                  fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate oracle gap
    gap_eur = controller_bill - oracle_bill
    saved_potential = baseline_a_bill - oracle_bill
    if saved_potential > 0:
        gap_pct = 100 * gap_eur / saved_potential
        ax.text(0.5, 0.95,
                 f"Oracle gap: €{gap_eur:.0f} ({gap_pct:.1f}% of oracle savings potential)",
                 transform=ax.transAxes, ha="center",
                 fontsize=10, style="italic",
                 bbox=dict(facecolor="white", edgecolor="grey", alpha=0.9))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    from data_loader import load_dataset
    from optimizer import run_mpc

    sheets = load_dataset("data/ENERGY_Hackathon_DataSet(Sheet1).csv")
    df_2025 = sheets["2025"]

    # Run oracle MPC on a few weeks around March
    march_start = pd.Timestamp("2025-03-10")
    march_end   = pd.Timestamp("2025-03-26")
    df_march = df_2025.loc[march_start:march_end].copy()
    dispatch = run_mpc(df_march, forecasts=None, horizon=96, verbose_every=400)

    plot_march_week3_dispatch(dispatch, out_path="plots/march_week3_test.png")
