"""
main_full_pipeline.py  —  Sol-Ship Energy AI Hackathon 2026
===========================================================
Full end-to-end pipeline producing ALL required and bonus outputs.

Required outputs (per brief):
  ✅ NRMSE on 2025          — 2024-trained LightGBM, h=1 forecast
  ✅ Baseline A bill         — historical replay, SoC-clamped to [0,1]
  ✅ Baseline B bill         — no battery
  ✅ Controller bill         — rolling MPC H=96 with causal forecasts
  ✅ Oracle bill             — rolling MPC H=96 with actual load (perfect foresight)
  ✅ Savings vs A/B          — absolute + percentage
  ✅ Oracle gap %            — (controller − oracle)/(baseline_A − oracle) × 100
  ✅ March Week 3 mandatory  — dispatch plot Mar 17-23 2025
  ✅ Extension               — H ∈ {4, 24, 96} bill, savings, compute time

Causality enforcement:
  Load      → multi-horizon LightGBM trained on 2024 ONLY
  PV        → lag-96 persistence (yesterday same 15-min slot) — fully causal
  sell_price → lag-96 persistence — fully causal
  buy_price  → deterministic Italian ToU schedule (known in advance) ✓

Run:
    python main_full_pipeline.py
"""
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = "data/ENERGY_Hackathon_DataSet(Sheet1).csv"
OUT_DIR   = Path("outputs")
PLOT_DIR  = OUT_DIR / "plots" / "pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── System constants (from config.py) ─────────────────────────────────────────
from config import (
    BATTERY_KWH, BATTERY_P_MAX_KW, ETA_DIR, GRID_P_MAX_KW,
    SOC_INIT, DT_HOURS, HORIZON_OPTIONS,
)

H_MAX = 96   # maximum MPC look-ahead (24 h)

# ── LightGBM hyper-parameters (conservative — favours cross-site gen.) ────────
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "verbose": -1,
    "num_threads": -1, "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 80, "feature_fraction": 0.85,
    "bagging_fraction": 0.85, "bagging_freq": 5, "lambda_l2": 0.5,
}
SPARSE_H = [1, 2, 4, 8, 16, 24, 48, 72, 96]
LAGS     = [1, 2, 4, 8, 96, 192, 672]
ROLLS    = [4, 16, 96, 672]


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _nrmse(yt, yp):
    m = float(np.mean(yt))
    return float(np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2))) / m * 100 \
        if m > 1e-9 else float("nan")


def _bill_from_dispatch(dispatch):
    """Compute bill from a dispatch DataFrame with p_imp, p_exp, buy_price, sell_price."""
    return float(
        ((dispatch["p_imp"] * dispatch["buy_price"])
         - (dispatch["p_exp"] * dispatch["sell_price"])).sum() * DT_HOURS
    )


# ════════════════════════════════════════════════════════════════════════════════
# Phase 1 — Train forecaster (2024 only, 75 / 25 split)
# ════════════════════════════════════════════════════════════════════════════════

def train_forecaster_2024(df_2024):
    import lightgbm as lgb
    from features import build_features

    cut     = df_2024.index[int(0.75 * len(df_2024))]
    df_tr   = df_2024.loc[:cut]
    df_va   = df_2024.loc[cut:]
    print(f"  Train: {df_tr.index.min().date()} → {df_tr.index.max().date()} "
          f"({len(df_tr):,} rows)")
    print(f"  Val  : {df_va.index.min().date()} → {df_va.index.max().date()} "
          f"({len(df_va):,} rows)")

    # Log1p stabilises spike variance
    def log_df(d):
        d2 = d.copy(); d2["load_kw"] = np.log1p(d2["load_kw"]); return d2

    df_tr_log = log_df(df_tr)
    df_va_log = log_df(df_va)
    buf        = df_tr_log.iloc[-max(LAGS):]
    df_va_buf  = pd.concat([buf, df_va_log])

    models = {}
    for h in SPARSE_H:
        X_tr, y_tr = build_features(df_tr_log, lags=LAGS, roll_windows=ROLLS, horizon=h)
        X_va, y_va = build_features(df_va_buf, lags=LAGS, roll_windows=ROLLS, horizon=h)
        fc  = list(X_tr.columns)
        X_va = X_va.loc[X_va.index.intersection(df_va.index), fc]
        y_va = y_va.loc[y_va.index.intersection(df_va.index)]

        dtr   = lgb.Dataset(X_tr.values, y_tr.values, feature_name=fc)
        dva   = lgb.Dataset(X_va.values, y_va.values, reference=dtr)
        model = lgb.train(
            LGB_PARAMS, dtr, num_boost_round=1500,
            valid_sets=[dtr, dva], valid_names=["tr", "va"],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        models[h] = (model, fc)
        print(f"    h={h:3d}  best_iter={model.best_iteration:4d}")
    return models


# ════════════════════════════════════════════════════════════════════════════════
# Phase 2 — Generate full-2025 forecast matrix + NRMSE
# ════════════════════════════════════════════════════════════════════════════════

def generate_2025_forecasts(models, df_2024, df_2025):
    from features import build_features

    # Include 2024 as lag context so Jan-1-2025 gets valid features
    def log_cat(a, b):
        c = pd.concat([a, b]).copy(); c["load_kw"] = np.log1p(c["load_kw"]); return c

    buf_log  = log_cat(df_2024, df_2025)
    sparse_h = sorted(models.keys())
    pred_d   = {}

    for h in sparse_h:
        model, fc = models[h]
        X, _  = build_features(buf_log, lags=LAGS, roll_windows=ROLLS, horizon=h)
        X     = X.loc[X.index.intersection(df_2025.index), fc]
        raw   = model.predict(X.values)
        pred_d[h] = pd.Series(np.expm1(raw).clip(min=0), index=X.index)

    # Align to common index across all horizons
    common = pred_d[sparse_h[0]].index
    for h in sparse_h[1:]:
        common = common.intersection(pred_d[h].index)

    sparse_arr  = np.array(sparse_h)
    sparse_pred = np.column_stack([pred_d[h].loc[common].values for h in sparse_h])
    target_h    = np.arange(1, H_MAX + 1)
    n           = len(common)
    mat         = np.zeros((n, H_MAX))
    for i in range(n):
        mat[i] = np.interp(target_h, sparse_arr, sparse_pred[i])
    mat = np.clip(mat, 0, None)

    load_fc  = pd.DataFrame(mat, index=common, columns=[f"h{h}" for h in target_h])
    actual_h1 = df_2025["load_kw"].reindex(common).values
    nrmse_h1  = _nrmse(actual_h1, load_fc["h1"].values)
    return load_fc, nrmse_h1


# ════════════════════════════════════════════════════════════════════════════════
# Phase 3 — Causal PV and sell_price forecast matrices  (lag-96)
# ════════════════════════════════════════════════════════════════════════════════

def build_causal_forecast_matrices(df_2024, df_2025, H_max=96):
    """
    At 2025 step t, horizon h: forecast = value at t+h−96 (yesterday same slot).
    Dec-2024 context prepended so Jan-1-2025 has a valid lag-96.

    pv_context[96 + t] = df_2025["pv_kw"][t]
    lag-96 of target 2025[t+h] = pv_context[96 + t + h − 96] = pv_context[t + h]
    """
    n          = len(df_2025)
    pv_ctx     = np.concatenate([df_2024["pv_kw"].values[-96:],
                                  df_2025["pv_kw"].values])
    sell_ctx   = np.concatenate([df_2024["sell_price"].values[-96:],
                                  df_2025["sell_price"].values])

    pv_fc   = np.zeros((n, H_max))
    sell_fc = np.zeros((n, H_max))

    for h in range(1, H_max + 1):        # h = 1 .. H_max
        col_pv   = pv_ctx  [h: h + n]
        col_sell = sell_ctx[h: h + n]
        if len(col_pv) < n:              # safety pad (shouldn't happen)
            col_pv   = np.concatenate([col_pv,   np.full(n - len(col_pv),   col_pv[-1]  )])
            col_sell = np.concatenate([col_sell, np.full(n - len(col_sell), col_sell[-1])])
        pv_fc  [:, h - 1] = np.clip(col_pv[:n],  0, None)
        sell_fc[:, h - 1] = col_sell[:n]

    return pv_fc, sell_fc


# ════════════════════════════════════════════════════════════════════════════════
# Phase 4 — Rolling-horizon MPC runner  (oracle and controller)
# ════════════════════════════════════════════════════════════════════════════════

def run_mpc_full_year(df_2025, load_fc_df, pv_fc_mat, sell_fc_mat,
                      H, soc_init=SOC_INIT, oracle=False, label="MPC"):
    """
    Roll MPC over the full year.

    oracle=True  → uses actual load / PV / sell as perfect-foresight inputs
    oracle=False → causal: load from LightGBM, PV/sell from lag-96 persistence
    buy_price    → always from the deterministic ToU schedule (causal by design)
    """
    from optimizer import MPCController

    controller = MPCController(horizon=H)

    n_2025     = len(df_2025)
    load_act   = df_2025["load_kw"].values
    pv_act     = df_2025["pv_kw"].values
    buy_arr    = df_2025["buy_price"].values
    sell_arr   = df_2025["sell_price"].values

    # Index alignment
    if oracle:
        idx = df_2025.index
    else:
        idx = df_2025.index.intersection(load_fc_df.index)

    pos_map = {ts: pos for pos, ts in enumerate(df_2025.index)}

    rows    = []
    soc     = float(soc_init)
    t_start = time.time()

    for step_i, ts in enumerate(idx):
        t   = pos_map[ts]
        end = min(t + H, n_2025)
        ahl = end - t

        # ── Build horizon inputs ────────────────────────────────────────────
        if oracle:
            load_hat = load_act [t: t + ahl]
            pv_hat   = pv_act   [t: t + ahl]
            sell_hat = sell_arr [t: t + ahl]
        else:
            load_hat = load_fc_df.loc[ts].values[:ahl]
            pv_hat   = pv_fc_mat [t, :ahl]
            sell_hat = sell_fc_mat[t, :ahl]

        buy_hat = buy_arr[t: t + ahl]   # deterministic ToU — causal ✓

        # ── Solve LP ────────────────────────────────────────────────────────
        sol = controller.solve_one(load_hat, pv_hat, buy_hat, sell_hat, soc)

        if sol is None:
            net = max(load_act[t] - pv_act[t], 0.0)
            sol = {"p_ch": 0.0, "p_dis": 0.0,
                   "p_imp": min(net, GRID_P_MAX_KW),
                   "p_exp": min(max(pv_act[t] - load_act[t], 0.0), GRID_P_MAX_KW),
                   "soc_planned_next": soc}

        p_ch  = sol["p_ch"]
        p_dis = sol["p_dis"]

        # ── Execute against ACTUAL load / PV ────────────────────────────────
        net_actual = load_act[t] - pv_act[t]
        residual   = net_actual - p_dis + p_ch

        if residual >= 0.0:
            p_imp = min(residual, GRID_P_MAX_KW)
            p_exp = 0.0
            unmet = max(residual - p_imp, 0.0)
        else:
            p_exp = min(-residual, GRID_P_MAX_KW)
            p_imp = 0.0
            unmet = 0.0

        # Scale back charging if grid limit still violated
        if unmet > 1e-6 and p_ch > 0.0:
            cut       = min(unmet, p_ch)
            p_ch     -= cut
            residual  -= cut
            p_imp      = min(residual, GRID_P_MAX_KW)
            unmet      = max(residual - p_imp, 0.0)

        # ── SoC update (physically from executed decision) ──────────────────
        delta = (ETA_DIR * p_ch - p_dis / ETA_DIR) * DT_HOURS / BATTERY_KWH
        soc   = float(np.clip(soc + delta, 0.0, 1.0))

        rows.append({
            "load_kw":    float(load_act[t]),
            "pv_kw":      float(pv_act[t]),
            "buy_price":  float(buy_arr[t]),
            "sell_price": float(sell_arr[t]),
            "p_ch":       p_ch,
            "p_dis":      p_dis,
            "p_battery":  p_dis - p_ch,        # + = discharging
            "p_imp":      p_imp,
            "p_exp":      p_exp,
            "p_grid":     p_imp - p_exp,        # + = importing
            "soc":        soc,
            "energy_kwh": soc * BATTERY_KWH,
            "unmet_load": unmet,
        })

        if (step_i + 1) % 2000 == 0:
            elapsed = time.time() - t_start
            pct     = 100 * (step_i + 1) / len(idx)
            print(f"    [{label}] {step_i+1}/{len(idx)} ({pct:.0f}%)  "
                  f"elapsed {elapsed:.0f}s  SoC={soc:.2f}")

    elapsed = time.time() - t_start
    print(f"    [{label}] done — {len(rows)} steps in {elapsed:.1f}s")
    return pd.DataFrame(rows, index=idx[: len(rows)])


# ════════════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════════════

def validate_dispatch(df, label):
    eb  = (df["load_kw"] - df["pv_kw"] - df["p_battery"] - df["p_grid"]).abs()
    soc_v = ((df["soc"] < -1e-4) | (df["soc"] > 1 + 1e-4)).sum()
    bat_v = (df["p_battery"].abs() > BATTERY_P_MAX_KW + 1e-4).sum()
    grd_v = (df["p_grid"].abs()    > GRID_P_MAX_KW    + 1e-4).sum()
    print(f"  [{label}] Validation:")
    print(f"    Energy balance max residual : {eb.max():.4f} kW  "
          f"({'OK' if eb.max() < 0.1 else 'WARN'})")
    print(f"    SoC violations              : {soc_v}")
    print(f"    Battery power violations    : {bat_v}")
    print(f"    Grid power violations       : {grd_v}")
    print(f"    Unmet load total            : {df['unmet_load'].sum() * DT_HOURS:.3f} kWh")


# ════════════════════════════════════════════════════════════════════════════════
# March Week 3 mandatory plot  (5-panel, tariff-shaded)
# ════════════════════════════════════════════════════════════════════════════════

def plot_march_week3(dispatch, out_path):
    from tariff import assign_tariff_band

    w_start = pd.Timestamp("2025-03-17 00:00")
    w_end   = pd.Timestamp("2025-03-24 00:00")
    sub     = dispatch.loc[(dispatch.index >= w_start) & (dispatch.index < w_end)].copy()
    if len(sub) == 0:
        print("  [WARN] No March Week 3 data found — skipping mandatory plot.")
        return

    bands = assign_tariff_band(sub.index)
    BCLR  = {"F1": "#FCF6F5", "F2": "#FEEAEA", "F3": "#E8F4F8"}

    def shade(ax):
        ba = bands.values
        runs, cb, cs = [], ba[0], sub.index[0]
        for i in range(1, len(ba)):
            if ba[i] != cb:
                runs.append((cb, cs, sub.index[i])); cb, cs = ba[i], sub.index[i]
        runs.append((cb, cs, sub.index[-1] + pd.Timedelta("15min")))
        for b, s, e in runs:
            ax.axvspan(s, e, color=BCLR.get(b, "white"), alpha=0.5, zorder=0)

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1.4, 1.4, 1.2, 1.0]})
    fig.suptitle(
        "March Week 3, 2025 — MPC Controller Dispatch  [MANDATORY]\n"
        "Load · PV · P_battery · P_grid · SoC  |  tariff bands shaded",
        fontsize=13, fontweight="bold",
    )

    # 1 — Load + PV
    ax = axes[0]; shade(ax)
    ax.plot(sub.index, sub["load_kw"], color="#1E2761", lw=1.5, label="Load")
    ax.fill_between(sub.index, 0, sub["pv_kw"], color="#F4A300", alpha=0.45, label="PV")
    ax.set_ylabel("Power (kW)"); ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    # 2 — P_battery
    ax = axes[1]; shade(ax)
    pb = sub["p_battery"].values
    ax.fill_between(sub.index, 0, np.where(pb > 0, pb, 0),
                    color="#02C39A", alpha=0.75, label="Discharge (+)")
    ax.fill_between(sub.index, 0, np.where(pb < 0, pb, 0),
                    color="#80CBC4", alpha=0.85, label="Charge (−)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("P_battery (kW)"); ax.legend(loc="upper right", fontsize=9)
    ax.axhline( BATTERY_P_MAX_KW, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-BATTERY_P_MAX_KW, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.grid(alpha=0.3)

    # 3 — P_grid
    ax = axes[2]; shade(ax)
    pg = sub["p_grid"].values
    ax.fill_between(sub.index, 0, np.where(pg > 0, pg, 0),
                    color="#990011", alpha=0.6, label="Import (+)")
    ax.fill_between(sub.index, 0, np.where(pg < 0, pg, 0),
                    color="#EE6C4D", alpha=0.6, label="Export (−)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("P_grid (kW)"); ax.legend(loc="upper right", fontsize=9)
    ax.axhline( GRID_P_MAX_KW, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(-GRID_P_MAX_KW, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.grid(alpha=0.3)

    # 4 — SoC
    ax = axes[3]; shade(ax)
    ax.plot(sub.index, sub["soc"], color="#0891B2", lw=1.8, label="SoC")
    ax.fill_between(sub.index, 0, sub["soc"], color="#0891B2", alpha=0.15)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("SoC [0–1]")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    # 5 — Stored energy kWh
    ax = axes[4]; shade(ax)
    ax.plot(sub.index, sub["energy_kwh"], color="#7B5EA7", lw=1.6, label="Stored energy")
    ax.fill_between(sub.index, 0, sub["energy_kwh"], color="#7B5EA7", alpha=0.15)
    ax.set_ylim(-0.5, BATTERY_KWH + 0.5); ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    # X-axis
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    axes[-1].set_xlabel("Date (March 2025)")

    # Constraint annotation
    min_soc = sub["soc"].min(); max_soc = sub["soc"].max()
    max_bat = sub["p_battery"].abs().max(); max_grid = sub["p_grid"].abs().max()
    axes[3].text(0.01, 0.05,
                 f"min SoC={min_soc:.2f}  max SoC={max_soc:.2f}",
                 transform=axes[3].transAxes, fontsize=8, color="#0891B2")
    axes[2].text(0.01, 0.05,
                 f"|P_grid|_max={max_grid:.2f} kW (limit ±6 kW)",
                 transform=axes[2].transAxes, fontsize=8, color="#990011")
    axes[1].text(0.01, 0.05,
                 f"|P_bat|_max={max_bat:.2f} kW (limit ±8 kW)",
                 transform=axes[1].transAxes, fontsize=8, color="#02C39A")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════════
# Extension plot
# ════════════════════════════════════════════════════════════════════════════════

def plot_extension(ext_results, bl_a_bill, bl_b_bill, oracle_bill, out_path):
    ext  = sorted(ext_results, key=lambda r: r["H"])
    Hs   = [r["H"] for r in ext]
    bills   = [r["bill"] for r in ext]
    savings = [r["savings_vs_a"] for r in ext]
    times   = [r["time_s"] for r in ext]
    xlabels = [f"H={h}\n({h//4}h look-ahead)" for h in Hs]
    x = np.arange(len(Hs))

    CLRS = ["#E74C3C", "#F39C12", "#2ECC71"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Extension — MPC Horizon Sensitivity  H ∈ {4, 24, 96}",
                 fontsize=13, fontweight="bold")

    # Panel 1: Bill
    ax = axes[0]
    bars = ax.bar(x, bills, color=CLRS, edgecolor="white", linewidth=1.2)
    ax.axhline(bl_a_bill, color="black",   lw=1.4, ls="--", label=f"Baseline A: €{bl_a_bill:.1f}")
    ax.axhline(bl_b_bill, color="#8E44AD", lw=1.4, ls=":",  label=f"Baseline B: €{bl_b_bill:.1f}")
    ax.axhline(oracle_bill, color="#27AE60", lw=1.4, ls="-.", label=f"Oracle: €{oracle_bill:.1f}")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Annual Bill (€)"); ax.set_title("Bill vs Horizon")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, b in enumerate(bills):
        ax.text(i, b + 1, f"€{b:.1f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 2: Savings vs Baseline A
    ax = axes[1]
    ax.bar(x, savings, color=CLRS, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Savings vs Baseline A (€)"); ax.set_title("Savings vs Horizon")
    ax.grid(axis="y", alpha=0.3)
    for i, s in enumerate(savings):
        ax.text(i, s + 0.5, f"€{s:.1f}", ha="center", fontsize=10, fontweight="bold")

    # Panel 3: Compute time
    ax = axes[2]
    ax.bar(x, times, color=["#3498DB", "#9B59B6", "#1ABC9C"],
           edgecolor="white", linewidth=1.2)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Compute Time (s)"); ax.set_title("Compute Time vs Horizon")
    ax.grid(axis="y", alpha=0.3)
    for i, t in enumerate(times):
        ax.text(i, t + 1, f"{t:.0f}s", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════════
# SoC overview plot for full year
# ════════════════════════════════════════════════════════════════════════════════

def plot_soc_overview(ctrl_dispatch, oracle_dispatch, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.suptitle("Full 2025 SoC Trajectory — Controller vs Oracle", fontsize=13, fontweight="bold")

    ax1.plot(ctrl_dispatch.index,   ctrl_dispatch["soc"],   color="#0891B2", lw=0.6,
             alpha=0.8, label=f"Controller (bill=€{_bill_from_dispatch(ctrl_dispatch):.2f})")
    ax1.plot(oracle_dispatch.index, oracle_dispatch["soc"],  color="#27AE60", lw=0.6,
             alpha=0.8, label=f"Oracle     (bill=€{_bill_from_dispatch(oracle_dispatch):.2f})")
    ax1.set_ylim(-0.05, 1.05); ax1.set_ylabel("SoC [0–1]")
    ax1.legend(loc="upper right", fontsize=9); ax1.grid(alpha=0.25)

    ax2.plot(ctrl_dispatch.index,   ctrl_dispatch["energy_kwh"],   color="#0891B2", lw=0.6, alpha=0.8)
    ax2.plot(oracle_dispatch.index, oracle_dispatch["energy_kwh"],  color="#27AE60", lw=0.6, alpha=0.8)
    ax2.set_ylim(-0.5, BATTERY_KWH + 0.5); ax2.set_ylabel("Stored Energy (kWh)")
    ax2.set_xlabel("Date"); ax2.grid(alpha=0.25)

    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def main():
    from data_loader import load_dataset
    from baselines import compute_baseline_a, compute_baseline_b

    print("=" * 65)
    print("  SOL-SHIP ENERGY AI HACKATHON — FULL PIPELINE")
    print("=" * 65)

    # ── [0] Load data ─────────────────────────────────────────────────────────
    print("\n[0/9] Loading dataset …")
    sheets  = load_dataset(DATA_PATH)
    df_2024 = sheets["2024"]
    df_2025 = sheets["2025"]
    print(f"  2024: {len(df_2024):,} rows  |  2025: {len(df_2025):,} rows")

    # ── [1] Train forecaster (2024 only) ─────────────────────────────────────
    print("\n[1/9] Training multi-horizon LightGBM (2024 only, 9 horizons) …")
    t0     = time.time()
    models = train_forecaster_2024(df_2024)
    print(f"  Training done in {time.time() - t0:.1f}s")

    # ── [2] Full-2025 forecast + NRMSE ────────────────────────────────────────
    print("\n[2/9] Generating 2025 forecasts + NRMSE …")
    load_fc, nrmse_2025 = generate_2025_forecasts(models, df_2024, df_2025)
    print(f"  Forecast NRMSE on 2025 (h=1): {nrmse_2025:.2f}%")

    # ── [3] Causal PV + sell_price forecast matrices ──────────────────────────
    print("\n[3/9] Building lag-96 causal forecast matrices (PV + sell_price) …")
    pv_fc_mat, sell_fc_mat = build_causal_forecast_matrices(df_2024, df_2025, H_max=H_MAX)
    print(f"  Shape: ({len(df_2025)}, {H_MAX}) each")

    # ── [4] Baselines A and B ─────────────────────────────────────────────────
    print("\n[4/9] Computing Baseline A (historical replay, SoC-clamped) …")
    bl_a = compute_baseline_a(df_2025)
    print(f"  Baseline A bill: €{bl_a['bill']:.2f}")

    print("\n[5/9] Computing Baseline B (no battery) …")
    bl_b = compute_baseline_b(df_2025)
    print(f"  Baseline B bill: €{bl_b['bill']:.2f}")

    # ── [5] Oracle MPC (H=96) ─────────────────────────────────────────────────
    print("\n[6/9] Running Oracle MPC (H=96, perfect foresight) …")
    t0 = time.time()
    oracle_disp = run_mpc_full_year(
        df_2025, None, None, None, H=96,
        soc_init=SOC_INIT, oracle=True, label="Oracle",
    )
    oracle_time = time.time() - t0
    oracle_bill = _bill_from_dispatch(oracle_disp)
    print(f"  Oracle bill: €{oracle_bill:.2f}  ({oracle_time:.0f}s)")
    validate_dispatch(oracle_disp, "Oracle")
    oracle_disp.to_csv(OUT_DIR / "dispatch_2025_oracle.csv")

    # ── [6] Controller MPC (H=96) ─────────────────────────────────────────────
    print("\n[7/9] Running Controller MPC (H=96, causal forecasts) …")
    t0 = time.time()
    ctrl_disp = run_mpc_full_year(
        df_2025, load_fc, pv_fc_mat, sell_fc_mat, H=96,
        soc_init=SOC_INIT, oracle=False, label="Controller H=96",
    )
    ctrl_time_96 = time.time() - t0
    ctrl_bill_96 = _bill_from_dispatch(ctrl_disp)
    print(f"  Controller bill (H=96): €{ctrl_bill_96:.2f}  ({ctrl_time_96:.0f}s)")
    validate_dispatch(ctrl_disp, "Controller H=96")
    ctrl_disp.to_csv(OUT_DIR / "dispatch_2025_controller_h96.csv")

    # ── [7] March Week 3 mandatory plot ───────────────────────────────────────
    print("\n[8/9] Generating mandatory March Week 3 dispatch plot …")
    mw3_path = PLOT_DIR / "march_week3_dispatch.png"
    plot_march_week3(ctrl_disp, mw3_path)

    # ── [8] Extension — H ∈ {4, 24, 96} ──────────────────────────────────────
    print("\n[9/9] Extension: horizon sensitivity sweep …")
    ext_results = [{"H": 96, "bill": ctrl_bill_96,
                    "time_s": ctrl_time_96,
                    "savings_vs_a": bl_a["bill"] - ctrl_bill_96}]
    print(f"  H= 96  bill=€{ctrl_bill_96:.2f}  "
          f"savings=€{bl_a['bill'] - ctrl_bill_96:.2f}  time={ctrl_time_96:.0f}s")

    for H in [h for h in HORIZON_OPTIONS if h != 96]:
        print(f"  Running Controller MPC H={H} …")
        t0 = time.time()
        d_h = run_mpc_full_year(
            df_2025, load_fc, pv_fc_mat, sell_fc_mat, H=H,
            soc_init=SOC_INIT, oracle=False, label=f"H={H}",
        )
        t_h   = time.time() - t0
        bill_h = _bill_from_dispatch(d_h)
        sav_h  = bl_a["bill"] - bill_h
        ext_results.append({"H": H, "bill": bill_h, "time_s": t_h, "savings_vs_a": sav_h})
        print(f"  H={H:2d}  bill=€{bill_h:.2f}  savings=€{sav_h:.2f}  time={t_h:.0f}s")
        d_h.to_csv(OUT_DIR / f"dispatch_2025_controller_h{H}.csv")

    ext_results.sort(key=lambda r: r["H"])

    # ── Results table ──────────────────────────────────────────────────────────
    sav_vs_a    = bl_a["bill"] - ctrl_bill_96
    sav_vs_a_pct = 100 * sav_vs_a / bl_a["bill"] if bl_a["bill"] != 0 else 0.0
    sav_vs_b    = bl_b["bill"] - ctrl_bill_96
    sav_vs_b_pct = 100 * sav_vs_b / bl_b["bill"] if bl_b["bill"] != 0 else 0.0

    oracle_sav  = bl_a["bill"] - oracle_bill
    ctrl_sav_from_a = bl_a["bill"] - ctrl_bill_96
    gap_eur     = oracle_sav - ctrl_sav_from_a       # = ctrl_bill - oracle_bill
    gap_pct     = 100 * gap_eur / oracle_sav if oracle_sav > 0 else float("nan")

    print("\n" + "=" * 65)
    print("  RESULTS TABLE  (Brief Section 4 — Required)")
    print("=" * 65)
    rows_table = [
        ("Baseline A — historical operation",   f"€{bl_a['bill']:.2f}"),
        ("Baseline B — zero-intelligence",       f"€{bl_b['bill']:.2f}"),
        ("Our controller  (rolling MPC, H=96)",  f"€{ctrl_bill_96:.2f}"),
        ("", ""),
        ("Savings vs. Baseline A",               f"€{sav_vs_a:+.2f}  ({sav_vs_a_pct:+.2f}%)"),
        ("Savings vs. Baseline B",               f"€{sav_vs_b:+.2f}  ({sav_vs_b_pct:+.2f}%)"),
        ("", ""),
        ("Oracle bill  (perfect foresight)",     f"€{oracle_bill:.2f}"),
        ("Oracle gap",                           f"€{gap_eur:.2f}  ({gap_pct:.2f}%)"),
        ("", ""),
        ("Forecast NRMSE on 2025  (h=1)",        f"{nrmse_2025:.2f}%"),
    ]
    for label, val in rows_table:
        if label:
            print(f"  {label:42s}{val:>20s}")
        else:
            print()
    print("=" * 65)

    if gap_pct < 10:
        interp = "Forecast excellent — bottleneck is the optimizer, not the forecast."
    elif gap_pct < 25:
        interp = "Sweet spot — forecast decent; targeted improvements would help."
    elif gap_pct < 40:
        interp = "Forecast is the bottleneck — significant savings left on the table."
    else:
        interp = "Forecast has systematic bias the optimizer repeatedly bets on."
    print(f"\n  Oracle-gap interpretation: {interp}")

    print("\n" + "=" * 65)
    print("  EXTENSION — HORIZON SENSITIVITY")
    print("=" * 65)
    print(f"  {'H':>5}  {'Look-ahead':>10}  {'Bill (€)':>10}  "
          f"{'Savings vs A (€)':>18}  {'Time (s)':>10}")
    print("  " + "-" * 60)
    for r in ext_results:
        la = f"{r['H'] // 4}h"
        print(f"  {r['H']:>5}  {la:>10}  {r['bill']:>10.2f}  "
              f"{r['savings_vs_a']:>18.2f}  {r['time_s']:>10.0f}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_extension(ext_results, bl_a["bill"], bl_b["bill"], oracle_bill,
                   PLOT_DIR / "extension_horizon_sensitivity.png")
    plot_soc_overview(ctrl_disp, oracle_disp,
                      PLOT_DIR / "soc_full_year_overview.png")

    print(f"\nAll outputs → {OUT_DIR}/")
    print(f"  dispatch_2025_oracle.csv")
    print(f"  dispatch_2025_controller_h96.csv")
    for r in ext_results:
        if r["H"] != 96:
            print(f"  dispatch_2025_controller_h{r['H']}.csv")
    print(f"  plots/pipeline/march_week3_dispatch.png   ← MANDATORY")
    print(f"  plots/pipeline/extension_horizon_sensitivity.png")
    print(f"  plots/pipeline/soc_full_year_overview.png")


if __name__ == "__main__":
    main()
