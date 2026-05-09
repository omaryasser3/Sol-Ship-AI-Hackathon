"""
Real-data EDA — ENERGY_Hackathon_DataSet(Sheet1).csv
=====================================================
Validates the real dataset for:
  1. Parse quality (semicolons, European decimals)
  2. Schema integrity & cadence gaps
  3. Value range / physical plausibility
  4. Battery SoC corruption detection in 2025
  5. Deep-dive on April 2025 & September 2025 (the test months)

All plots saved to outputs/plots/eda_real/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from pathlib import Path

# ---------- constants (mirrors config.py) ----------
BATTERY_KWH   = 16.0
ETA_DIR       = np.sqrt(0.90)
DT_HOURS      = 0.25
SOC_INIT      = 0.50
BATTERY_P_MAX = 8.0
GRID_P_MAX    = 6.0
PV_MAX_KWP    = 9.0

OUT_DIR = Path("outputs/plots/eda_real")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════
def load_real_csv(path="data/ENERGY_Hackathon_DataSet(Sheet1).csv"):
    """
    Parse the real hackathon CSV.
    - Separator: semicolon
    - Decimal:   comma  (European locale)
    - Columns:   battery_p, grid_p, load_p, pv_p, timestamp, Selling_price_eur_kwh
    """
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        parse_dates=["timestamp"],
    )
    # Canonical rename
    df = df.rename(columns={
        "battery_p":           "p_battery_kw",
        "grid_p":              "p_grid_kw",
        "load_p":              "load_kw",
        "pv_p":                "pv_kw",
        "Selling_price_eur_kwh": "sell_price",
    })
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


# ══════════════════════════════════════════════════════════════════
# 2. BASIC INTEGRITY CHECKS
# ══════════════════════════════════════════════════════════════════
def check_integrity(df, label):
    print(f"\n{'='*65}")
    print(f"  INTEGRITY — {label}")
    print(f"{'='*65}")
    print(f"  Rows:       {len(df):,}")
    print(f"  Date range: {df.index.min()}  →  {df.index.max()}")

    # Cadence
    diffs = df.index.to_series().diff().dropna()
    expected = pd.Timedelta("15min")
    bad_gaps = diffs[diffs != expected]
    print(f"  Cadence: 15-min expected. Non-15-min gaps: {len(bad_gaps)}")
    if len(bad_gaps):
        print("    First 10:")
        for ts, gap in bad_gaps.head(10).items():
            print(f"      {ts}  →  gap = {gap}")

    # Missing / NaN
    print(f"\n  NaN counts:")
    for col in df.columns:
        n = df[col].isna().sum()
        if n:
            print(f"    {col}: {n} NaN")
        else:
            print(f"    {col}: 0 NaN  ✓")

    # Physical ranges
    print(f"\n  Physical range checks:")
    _range_check(df, "load_kw",      lo=0,    hi=None,        unit="kW")
    _range_check(df, "pv_kw",        lo=0,    hi=PV_MAX_KWP,  unit="kW")
    _range_check(df, "p_battery_kw", lo=-BATTERY_P_MAX, hi=BATTERY_P_MAX, unit="kW")
    _range_check(df, "p_grid_kw",    lo=-GRID_P_MAX,    hi=GRID_P_MAX,    unit="kW")
    _range_check(df, "sell_price",   lo=0,    hi=None,        unit="€/kWh")


def _range_check(df, col, lo, hi, unit):
    if col not in df.columns:
        return
    vals = df[col]
    lo_viol = (vals < lo).sum() if lo is not None else 0
    hi_viol = (vals > hi).sum() if hi is not None else 0
    flag = "✗" if (lo_viol or hi_viol) else "✓"
    print(f"    {flag} {col}: min={vals.min():.4f} max={vals.max():.4f} {unit}  "
          f"|  <{lo}: {lo_viol}" + (f"  >{hi}: {hi_viol}" if hi is not None else ""))


def print_descriptive_stats(df, label):
    print(f"\n{'='*65}")
    print(f"  DESCRIPTIVE STATS — {label}")
    print(f"{'='*65}")
    cols = [c for c in ["load_kw", "pv_kw", "p_battery_kw", "p_grid_kw", "sell_price"]
            if c in df.columns]
    stats = df[cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
    print(stats.round(4).to_string())


# ══════════════════════════════════════════════════════════════════
# 3. SoC RECONSTRUCTION & CORRUPTION DETECTION
# ══════════════════════════════════════════════════════════════════
def reconstruct_soc(p_battery_kw, soc_init=SOC_INIT, clip=False):
    p = np.asarray(p_battery_kw, dtype=float)
    soc = np.empty(len(p) + 1)
    soc[0] = soc_init
    for t in range(len(p)):
        de = (-p[t] * ETA_DIR if p[t] < 0 else -p[t] / ETA_DIR) * DT_HOURS
        s = soc[t] + de / BATTERY_KWH
        soc[t + 1] = np.clip(s, 0, 1) if clip else s
    return soc


def detect_corruption_windows(ts, p_battery, soc_init=SOC_INIT,
                               drift_thr=1e-3, min_steps=96):
    soc_u = reconstruct_soc(p_battery, soc_init=soc_init, clip=False)[1:]
    soc_c = reconstruct_soc(p_battery, soc_init=soc_init, clip=True)[1:]
    div   = soc_u - soc_c
    rate  = np.abs(np.diff(div, prepend=0))
    flag  = rate > drift_thr

    rows, in_run, run_start = [], False, None
    for i, v in enumerate(flag):
        if v and not in_run:
            in_run, run_start = True, i
        elif not v and in_run:
            in_run = False
            L = i - run_start
            if L >= min_steps:
                rows.append(_win_row(run_start, i - 1, ts, div))
    if in_run and (len(flag) - run_start) >= min_steps:
        rows.append(_win_row(run_start, len(flag) - 1, ts, div))
    return pd.DataFrame(rows), soc_u, soc_c


def _win_row(s, e, ts, div):
    return dict(start=ts[s], end=ts[e],
                length_steps=e - s + 1, length_days=(e - s + 1) / 96,
                max_divergence=float(div[s:e + 1].max()),
                min_divergence=float(div[s:e + 1].min()))


# ══════════════════════════════════════════════════════════════════
# 4. PLOT HELPERS
# ══════════════════════════════════════════════════════════════════
PALETTE = dict(load="#1E2761", pv="#F4A300", battery="#02C39A",
               grid="#990011", soc="#0891B2", price="#6B21A8")


def _save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _fmt_monthly_x(ax, df_slice):
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


# ══════════════════════════════════════════════════════════════════
# 5. OVERVIEW PLOTS (full 2024 + 2025)
# ══════════════════════════════════════════════════════════════════
def plot_full_overview(df_2024, df_2025):
    """Monthly mean of load, PV, and price for both years."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)

    for ax, col, color, label in zip(
        axes,
        ["load_kw", "pv_kw", "sell_price"],
        [PALETTE["load"], PALETTE["pv"], PALETTE["price"]],
        ["Load (kW)", "PV (kW)", "Sell price (€/kWh)"]
    ):
        for df, year, ls in [(df_2024, "2024 train", "-"), (df_2025, "2025 test", "--")]:
            m = df[col].resample("D").mean()
            ax.plot(m.index, m.values, lw=0.8, alpha=0.5, color=color, linestyle=ls,
                    label=f"{year}")
            # 7-day rolling mean
            ax.plot(m.index, m.rolling(7).mean(), lw=2, color=color, linestyle=ls,
                    label=f"{year} 7-day avg")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.25)

        # Shade April and September in 2025
        for mo, mo_color in [(4, "blue"), (9, "orange")]:
            for yr in [2024, 2025]:
                start = pd.Timestamp(f"{yr}-{mo:02d}-01")
                end   = pd.Timestamp(f"{yr}-{mo:02d}-01") + pd.offsets.MonthEnd(0)
                ax.axvspan(start, end + pd.Timedelta("1D"),
                           color=mo_color, alpha=0.08, zorder=0)

    axes[0].set_title("Full Dataset Overview — Daily Mean (2024 train / 2025 test)\n"
                       "Blue shading = April, Orange shading = September",
                       fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    _save(fig, "01_full_overview.png")


def plot_monthly_boxplots(df_2024, df_2025):
    """Box plots of daily-peak load per month, 2024 vs 2025."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    for ax, df, year in [(axes[0], df_2024, "2024"), (axes[1], df_2025, "2025")]:
        daily_peak = df["load_kw"].resample("D").max()
        monthly    = [daily_peak[daily_peak.index.month == m].values for m in range(1, 13)]
        bp = ax.boxplot(monthly, patch_artist=True, medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(PALETTE["load"] if year == "2024" else "#E11D48")
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"])
        ax.set_title(f"{year} — Daily Peak Load by Month", fontweight="bold")
        ax.set_ylabel("Peak load (kW)")
        ax.grid(True, axis="y", alpha=0.3)
        # Highlight April and September
        for m_idx in [4, 9]:
            ax.axvspan(m_idx - 0.5, m_idx + 0.5,
                       color="gold", alpha=0.35, zorder=0, label="Test months" if m_idx == 4 else None)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _save(fig, "02_monthly_boxplots_load.png")


def plot_price_overview(df_2024, df_2025):
    """Sell price heatmap by hour-of-day × month."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, df, year in [(axes[0], df_2024, "2024"), (axes[1], df_2025, "2025")]:
        pivot = df.copy()
        pivot["month"] = pivot.index.month
        pivot["hour"]  = pivot.index.hour
        heat = pivot.groupby(["hour", "month"])["sell_price"].mean().unstack()
        im = ax.imshow(heat.values, aspect="auto", origin="lower",
                       cmap="plasma", interpolation="nearest")
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        ax.set_xticks(range(12))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"], rotation=45, ha="right")
        ax.set_title(f"{year} — Mean Sell Price (€/kWh) by Hour × Month", fontweight="bold")
        ax.set_ylabel("Hour of day")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="€/kWh")
    plt.tight_layout()
    _save(fig, "03_price_heatmap.png")


def plot_pv_load_scatter(df_2024, df_2025):
    """PV vs Load scatter by year and month."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, year, color in [
        (axes[0], df_2024, "2024", PALETTE["load"]),
        (axes[1], df_2025, "2025", "#E11D48"),
    ]:
        ax.scatter(df["pv_kw"], df["load_kw"], s=0.3, alpha=0.15, c=color, rasterized=True)
        ax.set_xlabel("PV (kW)")
        ax.set_ylabel("Load (kW)")
        ax.set_title(f"{year} — PV vs Load (all 15-min samples)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, PV_MAX_KWP + 0.5)
    plt.tight_layout()
    _save(fig, "04_pv_load_scatter.png")


# ══════════════════════════════════════════════════════════════════
# 6. CORRUPTION EVIDENCE PLOT
# ══════════════════════════════════════════════════════════════════
def plot_corruption(df_2025, windows, soc_u, soc_c):
    ts = df_2025.index
    p  = df_2025["p_battery_kw"].values

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True,
                              gridspec_kw={"height_ratios": [1.5, 2, 1]})

    # Panel 1: p_battery
    ax = axes[0]
    ax.plot(ts, p, color=PALETTE["battery"], lw=0.4, rasterized=True)
    ax.set_ylabel("p_battery (kW)\n+dis / −ch")
    ax.set_title("2025 Battery Power + SoC Reconstruction — Corruption Detection",
                 fontsize=12, fontweight="bold")
    ax.axhline(BATTERY_P_MAX,  color="grey", lw=0.7, ls="--", label=f"+{BATTERY_P_MAX} kW limit")
    ax.axhline(-BATTERY_P_MAX, color="grey", lw=0.7, ls="--", label=f"−{BATTERY_P_MAX} kW limit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Panel 2: SoC comparison
    ax = axes[1]
    ax.plot(ts, soc_c, color=PALETTE["soc"],  lw=0.8, label="SoC clipped [0,1]", rasterized=True)
    ax.plot(ts, soc_u, color="#990011", lw=0.6, alpha=0.8, label="SoC unclipped (physics)", rasterized=True)
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axhline(1, color="grey", lw=0.5, ls="--")
    ax.set_ylabel("SoC")
    ax.set_ylim(-0.3, 1.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.25)

    # Panel 3: divergence
    div = soc_u - soc_c
    ax = axes[2]
    ax.fill_between(ts, 0, div, color="#990011", alpha=0.6, label="|divergence| = corruption signal")
    ax.set_ylabel("SoC divergence")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Shade corruption windows
    if len(windows):
        for _, row in windows.iterrows():
            for ax_i in axes:
                ax_i.axvspan(row["start"], row["end"], color="red", alpha=0.18, zorder=0)
                ax_i.axvline(row["start"], color="red", lw=1.0, ls=":")
                ax_i.axvline(row["end"],   color="red", lw=1.0, ls=":")

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    _save(fig, "05_corruption_detection_2025.png")


# ══════════════════════════════════════════════════════════════════
# 7. DEEP-DIVE: APRIL 2025
# ══════════════════════════════════════════════════════════════════
def plot_april_deep_dive(df_2025, windows):
    apr = df_2025.loc["2025-04"].copy()
    if len(apr) == 0:
        print("  [WARN] No April 2025 data found — skipping deep-dive.")
        return

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5, 1]})

    # Panel 1: Load + PV
    ax = axes[0]
    ax.plot(apr.index, apr["load_kw"], color=PALETTE["load"], lw=1.0, label="Load")
    ax.fill_between(apr.index, 0, apr["pv_kw"], color=PALETTE["pv"], alpha=0.5, label="PV")
    ax.set_ylabel("Power (kW)")
    ax.set_title("April 2025 — Deep Dive (test month)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Annotate outliers in load
    load_mean = apr["load_kw"].mean()
    load_std  = apr["load_kw"].std()
    spikes = apr[apr["load_kw"] > load_mean + 3 * load_std]
    if len(spikes):
        ax.scatter(spikes.index, spikes["load_kw"], color="red", zorder=5,
                   s=30, label=f"Load spikes (>μ+3σ, n={len(spikes)})")
        ax.legend(fontsize=9)

    # Panel 2: Battery
    ax = axes[1]
    pb = apr["p_battery_kw"].values
    ax.fill_between(apr.index, 0, np.where(pb > 0, pb, 0),
                     color=PALETTE["battery"], alpha=0.7, label="Discharge")
    ax.fill_between(apr.index, 0, np.where(pb < 0, pb, 0),
                     color="#80CBC4", alpha=0.9, label="Charge")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(BATTERY_P_MAX,  color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(-BATTERY_P_MAX, color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("p_battery (kW)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Battery spikes
    bat_hi = apr[apr["p_battery_kw"] > BATTERY_P_MAX]
    bat_lo = apr[apr["p_battery_kw"] < -BATTERY_P_MAX]
    if len(bat_hi) + len(bat_lo):
        for v in [bat_hi, bat_lo]:
            ax.scatter(v.index, v["p_battery_kw"], color="red", zorder=5, s=40)
        ax.set_title(f"⚠ {len(bat_hi)} over-limit discharge, {len(bat_lo)} over-limit charge",
                     fontsize=9, color="red")

    # Panel 3: Grid
    ax = axes[2]
    pg = apr["p_grid_kw"].values
    ax.fill_between(apr.index, 0, np.where(pg > 0, pg, 0),
                     color=PALETTE["grid"], alpha=0.6, label="Import")
    ax.fill_between(apr.index, 0, np.where(pg < 0, pg, 0),
                     color="#EE6C4D", alpha=0.6, label="Export")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(GRID_P_MAX,  color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(-GRID_P_MAX, color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("p_grid (kW)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # Grid constraint violations
    grid_viol = apr[(apr["p_grid_kw"].abs() > GRID_P_MAX)]
    if len(grid_viol):
        ax.scatter(grid_viol.index, grid_viol["p_grid_kw"], color="red", zorder=5, s=40)
        ax.set_title(f"⚠ {len(grid_viol)} grid constraint violations (|p_grid| > {GRID_P_MAX} kW)",
                     fontsize=9, color="red")

    # Panel 4: Sell price
    ax = axes[3]
    ax.plot(apr.index, apr["sell_price"], color=PALETTE["price"], lw=0.8)
    ax.set_ylabel("Sell price (€/kWh)")
    ax.grid(True, alpha=0.25)
    # Annotate price spikes
    price_mean = apr["sell_price"].mean()
    price_std  = apr["sell_price"].std()
    price_spikes = apr[apr["sell_price"] > price_mean + 3 * price_std]
    if len(price_spikes):
        ax.scatter(price_spikes.index, price_spikes["sell_price"],
                   color="red", zorder=5, s=30, label=f"Price spikes (n={len(price_spikes)})")
        ax.legend(fontsize=9)

    # Panel 5: Daily energy balance
    ax = axes[4]
    daily_load = apr["load_kw"].resample("D").sum() * DT_HOURS
    daily_pv   = apr["pv_kw"].resample("D").sum() * DT_HOURS
    ax.bar(daily_load.index, daily_load.values, width=0.4, color=PALETTE["load"],
           align="center", label="Load (kWh/day)", alpha=0.8)
    ax.bar(daily_pv.index, daily_pv.values, width=0.4, color=PALETTE["pv"],
           align="edge", label="PV (kWh/day)", alpha=0.8)
    ax.set_ylabel("Energy (kWh/day)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Shade corruption windows that fall in April
    if len(windows):
        for _, row in windows.iterrows():
            if row["start"] <= apr.index[-1] and row["end"] >= apr.index[0]:
                for ax_i in axes:
                    ax_i.axvspan(max(row["start"], apr.index[0]),
                                 min(row["end"],   apr.index[-1]),
                                 color="red", alpha=0.15, zorder=0)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    _save(fig, "06_april_2025_deep_dive.png")


def plot_april_hourly_heatmap(df_2025):
    """Heatmap: day-of-month × hour for load, PV, battery in April 2025."""
    apr = df_2025.loc["2025-04"].copy()
    if len(apr) == 0:
        return
    apr["day"]  = apr.index.day
    apr["hour"] = apr.index.hour + apr.index.minute / 60

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, col, cmap, label in [
        (axes[0], "load_kw",      "Blues",  "Load (kW)"),
        (axes[1], "pv_kw",        "YlOrRd", "PV (kW)"),
        (axes[2], "p_battery_kw", "RdBu_r", "Battery (kW)"),
    ]:
        pivot = apr.pivot_table(values=col, index="hour", columns="day", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap=cmap, interpolation="nearest")
        ax.set_title(f"April 2025 — {label}", fontweight="bold")
        ax.set_xlabel("Day of month")
        ax.set_ylabel("Hour of day")
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    plt.suptitle("April 2025 — Hourly Heatmaps (test month)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "07_april_2025_heatmaps.png")


# ══════════════════════════════════════════════════════════════════
# 8. DEEP-DIVE: SEPTEMBER 2025
# ══════════════════════════════════════════════════════════════════
def plot_september_deep_dive(df_2025, windows):
    sep = df_2025.loc["2025-09"].copy()
    if len(sep) == 0:
        print("  [WARN] No September 2025 data found — skipping deep-dive.")
        return

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1.5, 1.5, 1.5, 1]})

    # Panel 1: Load + PV
    ax = axes[0]
    ax.plot(sep.index, sep["load_kw"], color=PALETTE["load"], lw=1.0, label="Load")
    ax.fill_between(sep.index, 0, sep["pv_kw"], color=PALETTE["pv"], alpha=0.5, label="PV")
    ax.set_ylabel("Power (kW)")
    ax.set_title("September 2025 — Deep Dive (test month)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    # Annotate outliers
    load_mean = sep["load_kw"].mean()
    load_std  = sep["load_kw"].std()
    spikes = sep[sep["load_kw"] > load_mean + 3 * load_std]
    if len(spikes):
        ax.scatter(spikes.index, spikes["load_kw"], color="red", zorder=5,
                   s=30, label=f"Load spikes (>μ+3σ, n={len(spikes)})")
        ax.legend(fontsize=9)

    # Panel 2: Battery
    ax = axes[1]
    pb = sep["p_battery_kw"].values
    ax.fill_between(sep.index, 0, np.where(pb > 0, pb, 0),
                     color=PALETTE["battery"], alpha=0.7, label="Discharge")
    ax.fill_between(sep.index, 0, np.where(pb < 0, pb, 0),
                     color="#80CBC4", alpha=0.9, label="Charge")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(BATTERY_P_MAX,  color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.axhline(-BATTERY_P_MAX, color="grey", lw=0.7, ls="--", alpha=0.7)
    ax.set_ylabel("p_battery (kW)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    bat_hi = sep[sep["p_battery_kw"] > BATTERY_P_MAX]
    bat_lo = sep[sep["p_battery_kw"] < -BATTERY_P_MAX]
    if len(bat_hi) + len(bat_lo):
        for v in [bat_hi, bat_lo]:
            ax.scatter(v.index, v["p_battery_kw"], color="red", zorder=5, s=40)
        ax.set_title(f"⚠ {len(bat_hi)} over-limit discharge, {len(bat_lo)} over-limit charge",
                     fontsize=9, color="red")

    # Panel 3: Grid
    ax = axes[2]
    pg = sep["p_grid_kw"].values
    ax.fill_between(sep.index, 0, np.where(pg > 0, pg, 0),
                     color=PALETTE["grid"], alpha=0.6, label="Import")
    ax.fill_between(sep.index, 0, np.where(pg < 0, pg, 0),
                     color="#EE6C4D", alpha=0.6, label="Export")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("p_grid (kW)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    grid_viol = sep[(sep["p_grid_kw"].abs() > GRID_P_MAX)]
    if len(grid_viol):
        ax.scatter(grid_viol.index, grid_viol["p_grid_kw"], color="red", zorder=5, s=40)
        ax.set_title(f"⚠ {len(grid_viol)} grid violations", fontsize=9, color="red")

    # Panel 4: Sell price
    ax = axes[3]
    ax.plot(sep.index, sep["sell_price"], color=PALETTE["price"], lw=0.8)
    ax.set_ylabel("Sell price (€/kWh)")
    ax.grid(True, alpha=0.25)
    price_mean = sep["sell_price"].mean()
    price_std  = sep["sell_price"].std()
    price_spikes = sep[sep["sell_price"] > price_mean + 3 * price_std]
    if len(price_spikes):
        ax.scatter(price_spikes.index, price_spikes["sell_price"],
                   color="red", zorder=5, s=30, label=f"Price spikes (n={len(price_spikes)})")
        ax.legend(fontsize=9)

    # Panel 5: Daily energy balance
    ax = axes[4]
    daily_load = sep["load_kw"].resample("D").sum() * DT_HOURS
    daily_pv   = sep["pv_kw"].resample("D").sum() * DT_HOURS
    ax.bar(daily_load.index, daily_load.values, width=0.4, color=PALETTE["load"],
           align="center", label="Load (kWh/day)", alpha=0.8)
    ax.bar(daily_pv.index, daily_pv.values, width=0.4, color=PALETTE["pv"],
           align="edge", label="PV (kWh/day)", alpha=0.8)
    ax.set_ylabel("Energy (kWh/day)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    if len(windows):
        for _, row in windows.iterrows():
            if row["start"] <= sep.index[-1] and row["end"] >= sep.index[0]:
                for ax_i in axes:
                    ax_i.axvspan(max(row["start"], sep.index[0]),
                                 min(row["end"],   sep.index[-1]),
                                 color="red", alpha=0.15, zorder=0)

    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=0, ha="center")
    plt.tight_layout()
    _save(fig, "08_september_2025_deep_dive.png")


def plot_september_hourly_heatmap(df_2025):
    sep = df_2025.loc["2025-09"].copy()
    if len(sep) == 0:
        return
    sep["day"]  = sep.index.day
    sep["hour"] = sep.index.hour + sep.index.minute / 60

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, col, cmap, label in [
        (axes[0], "load_kw",      "Blues",  "Load (kW)"),
        (axes[1], "pv_kw",        "YlOrRd", "PV (kW)"),
        (axes[2], "p_battery_kw", "RdBu_r", "Battery (kW)"),
    ]:
        pivot = sep.pivot_table(values=col, index="hour", columns="day", aggfunc="mean")
        im = ax.imshow(pivot.values, aspect="auto", origin="lower",
                       cmap=cmap, interpolation="nearest")
        ax.set_title(f"September 2025 — {label}", fontweight="bold")
        ax.set_xlabel("Day of month")
        ax.set_ylabel("Hour of day")
        ax.set_yticks(range(0, 24, 2))
        ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)
    plt.suptitle("September 2025 — Hourly Heatmaps (test month)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "09_september_2025_heatmaps.png")


# ══════════════════════════════════════════════════════════════════
# 9. SIDE-BY-SIDE: APRIL vs SEPTEMBER vs SAME MONTH IN 2024
# ══════════════════════════════════════════════════════════════════
def plot_test_months_vs_train(df_2024, df_2025):
    """Compare April/September 2025 (test) against 2024 same-month profiles."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey="row")

    for row_i, month, month_name in [(0, 4, "April"), (1, 9, "September")]:
        d24 = df_2024[df_2024.index.month == month]
        d25 = df_2025[df_2025.index.month == month]

        for col_i, (col, label, color24, color25) in enumerate([
            ("load_kw", "Load (kW)",   "#1E2761", "#E11D48"),
            ("pv_kw",   "PV (kW)",     "#F4A300", "#16A34A"),
        ]):
            ax = axes[row_i][col_i]
            # Average hourly profile
            prof24 = d24.groupby(d24.index.hour)[col].agg(["mean", "std"])
            prof25 = d25.groupby(d25.index.hour)[col].agg(["mean", "std"])

            hours = range(24)
            ax.plot(hours, prof24["mean"], color=color24, lw=2, label=f"2024 mean")
            ax.fill_between(hours,
                            prof24["mean"] - prof24["std"],
                            prof24["mean"] + prof24["std"],
                            color=color24, alpha=0.2)
            ax.plot(hours, prof25["mean"], color=color25, lw=2, ls="--", label=f"2025 mean")
            ax.fill_between(hours,
                            prof25["mean"] - prof25["std"],
                            prof25["mean"] + prof25["std"],
                            color=color25, alpha=0.2)

            ax.set_title(f"{month_name} — {label}", fontweight="bold")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel(label)
            ax.set_xticks(range(0, 24, 2))
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Hourly Profiles: 2024 Train vs 2025 Test  |  April & September\n"
                 "Shaded bands = ±1 std", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "10_test_months_vs_train_profiles.png")


# ══════════════════════════════════════════════════════════════════
# 10. ENERGY BALANCE CHECK
# ══════════════════════════════════════════════════════════════════
def energy_balance_check(df, label):
    """
    Energy balance: load = pv + grid + battery (net)
    Residual = load - pv - grid - battery should be ~0.
    """
    print(f"\n{'='*65}")
    print(f"  ENERGY BALANCE CHECK — {label}")
    print(f"{'='*65}")
    residual = df["load_kw"] - df["pv_kw"] - df["p_grid_kw"] - df["p_battery_kw"]
    print(f"  residual = load − pv − grid − battery")
    print(f"  mean   = {residual.mean():.5f} kW")
    print(f"  std    = {residual.std():.5f} kW")
    print(f"  max    = {residual.max():.5f} kW")
    print(f"  min    = {residual.min():.5f} kW")
    print(f"  |res| > 0.1 kW: {(residual.abs() > 0.1).sum()} rows")
    print(f"  |res| > 1.0 kW: {(residual.abs() > 1.0).sum()} rows")
    return residual


def plot_energy_balance(df_2024, df_2025):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    for ax, df, year, color in [
        (axes[0], df_2024, "2024", PALETTE["load"]),
        (axes[1], df_2025, "2025", "#E11D48"),
    ]:
        res = df["load_kw"] - df["pv_kw"] - df["p_grid_kw"] - df["p_battery_kw"]
        daily_res = res.resample("D").mean()
        ax.plot(daily_res.index, daily_res.values, color=color, lw=0.8, alpha=0.7)
        ax.fill_between(daily_res.index, 0, daily_res.values, color=color, alpha=0.3)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{year} — Daily Mean Energy Balance Residual "
                     f"(load − pv − grid − battery)", fontweight="bold")
        ax.set_ylabel("Residual (kW)")
        ax.grid(True, alpha=0.25)

        # Shade April and September
        for mo, mc in [(4, "blue"), (9, "orange")]:
            start = pd.Timestamp(f"{year}-{mo:02d}-01")
            end   = start + pd.offsets.MonthEnd(0) + pd.Timedelta("1D")
            ax.axvspan(start, end, color=mc, alpha=0.12, zorder=0)

    plt.tight_layout()
    _save(fig, "11_energy_balance_residual.png")


# ══════════════════════════════════════════════════════════════════
# 11. SUMMARY TEXT REPORT
# ══════════════════════════════════════════════════════════════════
def print_test_month_summary(df_2025, windows):
    for month, name in [(4, "APRIL"), (9, "SEPTEMBER")]:
        d = df_2025[df_2025.index.month == month]
        if len(d) == 0:
            continue
        print(f"\n{'='*65}")
        print(f"  TEST MONTH SUMMARY — {name} 2025")
        print(f"{'='*65}")
        print(f"  Rows:          {len(d):,}  ({len(d)/96:.0f} days × 96 steps)")
        print(f"  Load:          mean={d['load_kw'].mean():.3f}  max={d['load_kw'].max():.3f}  "
              f"std={d['load_kw'].std():.3f} kW")
        print(f"  PV:            mean={d['pv_kw'].mean():.3f}  max={d['pv_kw'].max():.3f} kW")
        print(f"  Battery:       mean={d['p_battery_kw'].mean():.3f}  "
              f"min={d['p_battery_kw'].min():.3f}  max={d['p_battery_kw'].max():.3f} kW")
        print(f"  Grid:          mean={d['p_grid_kw'].mean():.3f}  "
              f"min={d['p_grid_kw'].min():.3f}  max={d['p_grid_kw'].max():.3f} kW")
        print(f"  Sell price:    mean={d['sell_price'].mean():.5f}  "
              f"min={d['sell_price'].min():.5f}  max={d['sell_price'].max():.5f} €/kWh")
        print(f"  Bat violations (|p|>{BATTERY_P_MAX}): {(d['p_battery_kw'].abs() > BATTERY_P_MAX).sum()}")
        print(f"  Grid violations (|p|>{GRID_P_MAX}): {(d['p_grid_kw'].abs() > GRID_P_MAX).sum()}")
        # Corruption windows overlapping this month
        start_m = d.index[0]
        end_m   = d.index[-1]
        if len(windows):
            overlap = windows[(windows["start"] <= end_m) & (windows["end"] >= start_m)]
            if len(overlap):
                print(f"  ⚠  Corruption windows in this month:")
                for _, row in overlap.iterrows():
                    print(f"      {row['start']} → {row['end']}  "
                          f"({row['length_days']:.1f} days, "
                          f"max_div={row['max_divergence']:.3f})")
            else:
                print(f"  ✓  No corruption windows detected in this month.")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "█"*65)
    print("  REAL-DATA EDA — ENERGY_Hackathon_DataSet(Sheet1).csv")
    print("█"*65)

    # --- Load ---
    CSV = "data/ENERGY_Hackathon_DataSet(Sheet1).csv"
    print(f"\nLoading: {CSV}")
    df = load_real_csv(CSV)
    print(f"  Total rows: {len(df):,}  |  Columns: {list(df.columns)}")

    # --- Split by year ---
    df_2024 = df[df.index.year == 2024].copy()
    df_2025 = df[df.index.year == 2025].copy()
    print(f"  2024 (train): {len(df_2024):,} rows  "
          f"({df_2024.index.min().date()} → {df_2024.index.max().date()})")
    print(f"  2025 (test):  {len(df_2025):,} rows  "
          f"({df_2025.index.min().date()} → {df_2025.index.max().date()})")

    # --- Integrity ---
    check_integrity(df_2024, "2024 (TRAINING)")
    check_integrity(df_2025, "2025 (TEST)")

    # --- Descriptive stats ---
    print_descriptive_stats(df_2024, "2024 (TRAINING)")
    print_descriptive_stats(df_2025, "2025 (TEST)")

    # --- Energy balance ---
    _ = energy_balance_check(df_2024, "2024 (TRAINING)")
    _ = energy_balance_check(df_2025, "2025 (TEST)")

    # --- Corruption detection on 2025 battery ---
    print(f"\n{'='*65}")
    print("  CORRUPTION DETECTION — 2025 p_battery_kw")
    print(f"{'='*65}")
    windows, soc_u, soc_c = detect_corruption_windows(
        df_2025.index, df_2025["p_battery_kw"].values,
        drift_thr=1e-3, min_steps=96,
    )
    print(f"  SoC (unclipped): min={soc_u.min():.3f}  max={soc_u.max():.3f}")
    print(f"  Steps with SoC < 0: {(soc_u < 0).sum()}")
    print(f"  Steps with SoC > 1: {(soc_u > 1).sum()}")
    if len(windows):
        print(f"\n  ⚠  Detected {len(windows)} corruption window(s):")
        print(windows[["start", "end", "length_days", "max_divergence", "min_divergence"]].to_string(index=False))
    else:
        print("  ✓  No corruption windows detected.")

    # --- Test month summaries ---
    print_test_month_summary(df_2025, windows)

    # --- Plots ---
    print(f"\n{'='*65}")
    print("  GENERATING PLOTS → {OUT_DIR}")
    print(f"{'='*65}")

    print("\n[1/11] Full year overview...")
    plot_full_overview(df_2024, df_2025)

    print("[2/11] Monthly boxplots...")
    plot_monthly_boxplots(df_2024, df_2025)

    print("[3/11] Price heatmaps...")
    plot_price_overview(df_2024, df_2025)

    print("[4/11] PV vs Load scatter...")
    plot_pv_load_scatter(df_2024, df_2025)

    print("[5/11] Corruption evidence...")
    plot_corruption(df_2025, windows, soc_u, soc_c)

    print("[6/11] April 2025 deep-dive...")
    plot_april_deep_dive(df_2025, windows)

    print("[7/11] April 2025 hourly heatmaps...")
    plot_april_hourly_heatmap(df_2025)

    print("[8/11] September 2025 deep-dive...")
    plot_september_deep_dive(df_2025, windows)

    print("[9/11] September 2025 hourly heatmaps...")
    plot_september_hourly_heatmap(df_2025)

    print("[10/11] Test months vs train profiles...")
    plot_test_months_vs_train(df_2024, df_2025)

    print("[11/11] Energy balance residual...")
    plot_energy_balance(df_2024, df_2025)

    print(f"\n{'█'*65}")
    print(f"  EDA complete — {len(list(OUT_DIR.glob('*.png')))} plots saved to {OUT_DIR}")
    print(f"{'█'*65}\n")
