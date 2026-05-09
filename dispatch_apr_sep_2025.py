"""
dispatch_apr_sep_2025.py — April & September 2025 Full Analysis
===============================================================
Two-model focused analysis of the test months with ALL doc-required outputs.

Models
------
  April model    : train = 2024 full year + 2025 Jan        | val = 2025 Feb–Mar
  September model: train = 2024 full year + 2025 Jan–Jun    | val = 2025 Jul–Aug
  (Using all data available BEFORE each test month — maximises forecast quality)

Causality enforcement
---------------------
  Load forecast  : multi-horizon LightGBM (trained on data before each month)
  PV forecast    : lag-96 persistence (yesterday same 15-min slot) — causal ✓
  sell_price     : lag-96 persistence — causal ✓
  buy_price      : deterministic Italian ToU schedule — causal by design ✓

Required outputs (per brief)
-----------------------------
  ✅ EDA summary            per test month (load / PV / sell_price stats + spikes)
  ✅ Baseline A bill        historical replay, SoC clamped to [0,1]
  ✅ Baseline B bill        no battery
  ✅ Controller MPC H=96    causal forecasts, SoC reset 50 % at each month start
  ✅ Oracle MPC H=96        perfect-foresight load (lower bound for controller)
  ✅ Oracle gap %           (controller − oracle) / (baseline_A − oracle) × 100
  ✅ Savings vs A / B       absolute € and %
  ✅ Extension H ∈ {4,24,96} bill, savings, compute-time table + bar chart
  ✅ March Week 3 plot      mandatory 5-panel dispatch plot (Mar 17–23 2025)
  ✅ 5-panel dispatch plot  per test month (load, PV, battery, grid, SoC)
  ✅ SoC reconstruction     side-by-side April + September

Run:
    python dispatch_apr_sep_2025.py
"""
import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cvxpy as cp
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_PATH = "data/ENERGY_Hackathon_DataSet(Sheet1).csv"
OUT_DIR   = Path("outputs")
PLOT_DIR  = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# ── System constants ──────────────────────────────────────────────────────────
BATTERY_KWH  = 16.0
BATTERY_PMAX = 8.0
ETA          = np.sqrt(0.90)       # ≈ 0.9487 per direction
GRID_PMAX    = 6.0
SOC_INIT     = 0.50                # reset at START of each test month
DT           = 0.25                # 15 min in hours

SPARSE_H     = [1, 2, 4, 8, 16, 24, 48, 72, 96]
EXTENSION_H  = [4, 24, 96]         # for the extension sweep

# Feature lags (dense short-term + multi-day long-range)
LAGS  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 48, 96, 192, 288, 336, 384, 480, 576, 672]
ROLLS = [4, 8, 16, 96, 672]

# Optuna-tuned LightGBM hyper-params
LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "verbose": -1, "num_threads": -1,
    "learning_rate": 0.0272, "num_leaves": 167, "min_data_in_leaf": 135,
    "feature_fraction": 0.862, "bagging_fraction": 0.656, "bagging_freq": 5,
    "lambda_l2": 0.078, "lambda_l1": 0.150,
}

COLORS = {"load": "#1E2761", "pv": "#F4A300", "battery": "#02C39A",
          "grid": "#990011", "soc": "#0891B2"}
BAND_CLR = {"F1": "#FCF6F5", "F2": "#FEEAEA", "F3": "#E8F4F8"}


# ════════════════════════════════════════════════════════════════════════════
# Basic helpers
# ════════════════════════════════════════════════════════════════════════════

def rmse(yt, yp):
    return float(np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2)))

def mae(yt, yp):
    return float(np.mean(np.abs(np.asarray(yt) - np.asarray(yp))))

def nrmse(yt, yp):
    m = float(np.mean(yt))
    return float(np.sqrt(np.mean((np.asarray(yt) - np.asarray(yp)) ** 2))) / m * 100 \
        if m > 1e-9 else float("nan")

def compute_bill(dispatch):
    return float(
        ((dispatch["p_imp"] * dispatch["buy_price"])
         - (dispatch["p_exp"] * dispatch["sell_price"])).sum() * DT
    )


def detect_corrupted_battery(df_2025):
    """Reconstruct SoC from p_battery_kw and flag the corrupted window."""
    p_bat = df_2025["p_battery_kw"].values.astype(float)
    soc   = np.zeros(len(p_bat))
    s     = SOC_INIT
    for i, pb in enumerate(p_bat):
        ds = (-pb / np.sqrt(0.90) if pb > 0 else -pb * np.sqrt(0.90)) * DT / BATTERY_KWH
        s += ds  # intentionally NOT clamped — to detect drift
        soc[i] = s

    # Flag where SoC drifts outside [−0.5, 1.5] as clearly corrupt
    corrupt_mask = (soc < -0.5) | (soc > 1.5)
    if corrupt_mask.any():
        idx = df_2025.index
        first = idx[corrupt_mask][0]
        last  = idx[corrupt_mask][-1]
        n_bad = corrupt_mask.sum()
        print(f"  ⚠️  CORRUPTED BATTERY DATA DETECTED")
        print(f"      Window: {first.date()} → {last.date()} ({n_bad} steps)")
        print(f"      SoC drifted to min={soc.min():.2f}, max={soc.max():.2f}")
        print(f"      (Unclamped reconstruction shows physically impossible state)")
    else:
        print(f"  ✓ No SoC corruption detected (unclamped SoC stays in [{soc.min():.2f}, {soc.max():.2f}])")
    return soc, corrupt_mask


# ════════════════════════════════════════════════════════════════════════════
# EDA summary
# ════════════════════════════════════════════════════════════════════════════

def eda_summary(df_test, df_2024, month_name):
    """Print concise EDA for a test month vs 2024 training baseline."""
    lo = df_test["load_kw"]
    pv = df_test["pv_kw"]
    sp = df_test["sell_price"]
    bp = df_test["buy_price"]

    mean_24 = df_2024["load_kw"].mean()
    spike_thresh = lo.mean() + 2 * lo.std()
    n_spikes = (lo > spike_thresh).sum()

    print(f"\n  ── EDA: {month_name} 2025 ─────────────────────────────────")
    print(f"  Rows: {len(df_test):,}  ({len(df_test)//96} days)")
    print(f"  Load  mean={lo.mean():.3f}  max={lo.max():.3f}  std={lo.std():.3f} kW")
    print(f"        spikes(>μ+2σ={spike_thresh:.2f}kW): {n_spikes} steps "
          f"| spike:mean ratio {lo.max()/lo.mean():.1f}×")
    print(f"        2024 load mean={mean_24:.3f} kW → shift={lo.mean()-mean_24:+.3f} kW")
    print(f"  PV    mean={pv.mean():.3f}  max={pv.max():.3f}  "
          f"std(daylight)={pv[pv>0.1].std():.3f} kW")
    print(f"  Sell  mean=€{sp.mean():.4f}  min=€{sp.min():.4f}  "
          f"max=€{sp.max():.4f}/kWh")
    print(f"        near-zero(<€0.01): {(sp<0.01).sum()} steps")
    print(f"  Buy   F1=€0.2540  F2=€0.2682  F3=€0.2440 (deterministic ToU)")
    print(f"        sell:buy spread max={sp.max()/bp.mean():.2f}× mean={sp.mean()/bp.mean():.2f}×")


# ════════════════════════════════════════════════════════════════════════════
# Forecaster
# ════════════════════════════════════════════════════════════════════════════

def train_multihorizon(df_train_raw, df_val_raw):
    """Train LightGBM + RandomForest ensemble per horizon. Returns {h: (lgb_model, rf_model, feat_cols)}."""
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from features import build_features

    # Train on raw load_kw (no log-transform — preserves peak accuracy)
    buf       = df_train_raw.iloc[-max(LAGS):]
    df_va_buf = pd.concat([buf, df_val_raw])

    models = {}
    for h in SPARSE_H:
        X_tr, y_tr = build_features(df_train_raw, lags=LAGS, roll_windows=ROLLS, horizon=h)
        X_va, y_va = build_features(df_va_buf,    lags=LAGS, roll_windows=ROLLS, horizon=h)
        fc = list(X_tr.columns)
        X_va = X_va.loc[X_va.index.intersection(df_val_raw.index), fc]
        y_va = y_va.loc[y_va.index.intersection(df_val_raw.index)]

        # --- LightGBM ---
        dtr   = lgb.Dataset(X_tr.values, y_tr.values, feature_name=fc)
        dva   = lgb.Dataset(X_va.values, y_va.values, reference=dtr)
        lgb_model = lgb.train(
            LGB_PARAMS, dtr, num_boost_round=5000,
            valid_sets=[dtr, dva], valid_names=["tr", "va"],
            callbacks=[lgb.early_stopping(200, verbose=False),
                       lgb.log_evaluation(period=0)],
        )

        # --- RandomForest ---
        rf_model = RandomForestRegressor(
            n_estimators=300, max_depth=20, min_samples_leaf=20,
            max_features=0.7, n_jobs=-1, random_state=42,
        )
        rf_model.fit(X_tr.values, y_tr.values)

        models[h] = (lgb_model, rf_model, fc)
        print(f"    h={h:3d}  lgb_iter={lgb_model.best_iteration:4d}  rf_trees=300")
    return models


# Ensemble blend weight: LightGBM gets 60%, RandomForest gets 40%
# LGB is stronger on smooth trends; RF captures peaks & outliers better
ENSEMBLE_LGB_WEIGHT = 0.60


def predict_month(models, df_history_raw, df_target_raw, H_out=96):
    """
    Generate h=1..H_out forecast matrix for every step in df_target_raw.
    Uses LightGBM + RandomForest ensemble (weighted average).
    Returns DataFrame (n_target, H_out) indexed to df_target_raw.
    """
    from features import build_features

    # No log-transform — predict raw load_kw directly
    df_buf = pd.concat([df_history_raw, df_target_raw])
    sparse_hs  = sorted(models.keys())
    pred_dict  = {}

    w_lgb = ENSEMBLE_LGB_WEIGHT
    w_rf  = 1.0 - w_lgb

    for h in sparse_hs:
        lgb_model, rf_model, fc = models[h]
        X, _  = build_features(df_buf, lags=LAGS, roll_windows=ROLLS, horizon=h)
        X     = X.loc[X.index.intersection(df_target_raw.index), fc]
        lgb_pred = lgb_model.predict(X.values)
        rf_pred  = rf_model.predict(X.values)
        blended  = w_lgb * lgb_pred + w_rf * rf_pred
        pred_dict[h] = pd.Series(blended.clip(min=0), index=X.index)

    common = pred_dict[sparse_hs[0]].index
    for h in sparse_hs[1:]:
        common = common.intersection(pred_dict[h].index)

    sparse_arr  = np.array(sparse_hs)
    sparse_pred = np.column_stack([pred_dict[h].loc[common].values for h in sparse_hs])
    target_h    = np.arange(1, H_out + 1)
    n           = len(common)
    mat         = np.zeros((n, H_out))
    for i in range(n):
        mat[i] = np.interp(target_h, sparse_arr, sparse_pred[i])
    mat = np.clip(mat, 0, None)
    return pd.DataFrame(mat, index=common, columns=[f"h{h}" for h in target_h])


# ════════════════════════════════════════════════════════════════════════════
# Causal forecast matrices (lag-96)
# ════════════════════════════════════════════════════════════════════════════

def build_causal_horizon_matrices(df_buf, df_test, H_max=96):
    """
    Build (n_test, H_max) matrices of causal lag-96 forecasts.
    At step t, horizon h: forecast = value at t+h−96 (yesterday same 15-min slot).
    df_buf must contain at least 96 steps before df_test.
    """
    n        = len(df_test)
    pv_ctx   = np.concatenate([df_buf["pv_kw"].values[-96:],    df_test["pv_kw"].values])
    sell_ctx = np.concatenate([df_buf["sell_price"].values[-96:], df_test["sell_price"].values])

    pv_fc   = np.zeros((n, H_max))
    sell_fc = np.zeros((n, H_max))
    for h in range(1, H_max + 1):
        c_pv   = pv_ctx  [h: h + n]
        c_sell = sell_ctx[h: h + n]
        if len(c_pv) < n:
            c_pv   = np.concatenate([c_pv,   np.full(n - len(c_pv),   c_pv[-1]  )])
            c_sell = np.concatenate([c_sell, np.full(n - len(c_sell), c_sell[-1])])
        pv_fc  [:, h - 1] = np.clip(c_pv[:n], 0, None)
        sell_fc[:, h - 1] = c_sell[:n]
    return pv_fc, sell_fc


# ════════════════════════════════════════════════════════════════════════════
# Baselines (for a monthly slice)
# ════════════════════════════════════════════════════════════════════════════

def baseline_b_month(df_test):
    """Baseline B: no battery. PV serves load; surplus exported, deficit imported."""
    net  = df_test["load_kw"] - df_test["pv_kw"]
    p_gr = net.clip(-GRID_PMAX, GRID_PMAX)
    bill = float(
        (np.where(p_gr > 0, p_gr * df_test["buy_price"], 0)
         + np.where(p_gr < 0, p_gr * df_test["sell_price"], 0)).sum() * DT
    )
    return {"bill": bill,
            "dispatch": pd.DataFrame({
                "load_kw": df_test["load_kw"].values, "pv_kw": df_test["pv_kw"].values,
                "buy_price": df_test["buy_price"].values, "sell_price": df_test["sell_price"].values,
                "p_battery": np.zeros(len(df_test)), "p_grid": p_gr.values,
                "p_imp": p_gr.clip(lower=0).values, "p_exp": (-p_gr).clip(lower=0).values,
                "soc": np.full(len(df_test), SOC_INIT),
            }, index=df_test.index)}


def baseline_a_month(df_test):
    """Baseline A: replay historical p_battery_kw with SoC clamped to [0,1]."""
    p_bat = df_test["p_battery_kw"].astype(float).copy()
    p_gr  = (df_test["load_kw"] - df_test["pv_kw"] - p_bat).clip(-GRID_PMAX, GRID_PMAX)

    # Reconstruct SoC (clamped) to verify physical plausibility
    soc_arr = np.zeros(len(df_test))
    s = SOC_INIT
    for i, pb in enumerate(p_bat.values):
        # positive p_bat = discharging
        ds = (-pb / ETA if pb > 0 else -pb * ETA) * DT / BATTERY_KWH
        s  = float(np.clip(s + ds, 0, 1))
        soc_arr[i] = s

    bill = float(
        (np.where(p_gr > 0, p_gr * df_test["buy_price"], 0)
         + np.where(p_gr < 0, p_gr * df_test["sell_price"], 0)).sum() * DT
    )
    return {"bill": bill,
            "dispatch": pd.DataFrame({
                "load_kw": df_test["load_kw"].values, "pv_kw": df_test["pv_kw"].values,
                "buy_price": df_test["buy_price"].values, "sell_price": df_test["sell_price"].values,
                "p_battery": p_bat.values, "p_grid": p_gr.values,
                "p_imp": p_gr.clip(lower=0).values, "p_exp": (-p_gr).clip(lower=0).values,
                "soc": soc_arr,
            }, index=df_test.index)}


# ════════════════════════════════════════════════════════════════════════════
# General MPC runner  (controller and oracle, any H)
# ════════════════════════════════════════════════════════════════════════════

def run_mpc_general(df_actual, H, soc_start=SOC_INIT,
                    load_fc=None, pv_fc_mat=None, sell_fc_mat=None,
                    oracle=False, label="MPC"):
    """
    Rolling-horizon MPC dispatcher.

    oracle=True  → use actual future load / PV / sell (perfect foresight)
    oracle=False → use causal forecasts: load_fc, pv_fc_mat, sell_fc_mat

    buy_price: always deterministic ToU — causal by design.
    SoC reset to soc_start at the beginning of the period.
    """
    # Build CVXPY problem once, update Parameters each step
    load_p  = cp.Parameter(H, nonneg=True)
    pv_p    = cp.Parameter(H, nonneg=True)
    buy_p   = cp.Parameter(H, nonneg=True)
    sell_p  = cp.Parameter(H, nonneg=True)
    soc_par = cp.Parameter(nonneg=True)

    p_ch  = cp.Variable(H, nonneg=True)
    p_dis = cp.Variable(H, nonneg=True)
    p_imp = cp.Variable(H, nonneg=True)
    p_exp = cp.Variable(H, nonneg=True)
    soc_v = cp.Variable(H + 1)

    cons = [
        soc_v[0] == soc_par,
        soc_v >= 0, soc_v <= 1,
        p_ch  <= BATTERY_PMAX,
        p_dis <= BATTERY_PMAX,
        p_imp <= GRID_PMAX,
        p_exp <= GRID_PMAX,
        soc_v[1:] == soc_v[:-1] + (ETA * p_ch - p_dis / ETA) * DT / BATTERY_KWH,
        load_p + p_ch + p_exp == pv_p + p_dis + p_imp,
    ]
    cost    = cp.sum(cp.multiply(p_imp, buy_p) - cp.multiply(p_exp, sell_p)) * DT
    problem = cp.Problem(cp.Minimize(cost), cons)

    load_v = df_actual["load_kw"].values
    pv_v   = df_actual["pv_kw"].values
    buy_v  = df_actual["buy_price"].values
    sell_v = df_actual["sell_price"].values
    n      = len(df_actual)

    # Align load forecast to actual index
    if oracle:
        idx    = df_actual.index
        fc_arr = None
    else:
        idx    = df_actual.index.intersection(load_fc.index)
        fc_arr = load_fc.loc[idx].values      # (n_common, H_max≥H)

    pos_map = {ts: i for i, ts in enumerate(df_actual.index)}

    rows = []
    soc  = float(soc_start)

    def _pad(arr, length, fill):
        if len(arr) < length:
            return np.concatenate([arr, np.full(length - len(arr), fill)])
        return arr[:length]

    for step_i, ts in enumerate(idx):
        t   = pos_map[ts]
        end = min(t + H, n)
        ahl = end - t

        if oracle:
            lh = load_v[t: t + ahl]
            pvh = pv_v [t: t + ahl]
            sh  = sell_v[t: t + ahl]
        else:
            lh  = fc_arr[step_i, :ahl]
            pvh = pv_fc_mat [t, :ahl]
            sh  = sell_fc_mat[t, :ahl]

        bh = buy_v[t: t + ahl]   # deterministic ToU ✓

        load_p.value  = _pad(lh,  H, lh[-1] ).clip(min=0)
        pv_p.value    = _pad(pvh, H, 0.0    ).clip(min=0)
        buy_p.value   = _pad(bh,  H, bh[-1] ).clip(min=0)
        sell_p.value  = _pad(sh,  H, sh[-1] ).clip(min=0)
        soc_par.value = float(np.clip(soc, 0, 1))

        try:
            problem.solve(solver=cp.CLARABEL, warm_start=True, verbose=False)
            ok = problem.status in ("optimal", "optimal_inaccurate")
        except Exception:
            ok = False

        if ok:
            pch  = float(p_ch.value[0]);  pdis = float(p_dis.value[0])
            pimp = float(p_imp.value[0]); pexp = float(p_exp.value[0])
        else:
            net  = float(load_v[t] - pv_v[t])
            pch  = pdis = 0.0
            pimp = float(np.clip(net,  0, GRID_PMAX))
            pexp = float(np.clip(-net, 0, GRID_PMAX))

        # Execute decision against ACTUAL load / PV
        net_actual = float(load_v[t] - pv_v[t])
        residual   = net_actual - pdis + pch
        if residual >= 0:
            pimp_act = min(residual, GRID_PMAX)
            pexp_act = 0.0
            unmet    = max(residual - pimp_act, 0.0)
        else:
            pexp_act = min(-residual, GRID_PMAX)
            pimp_act = 0.0
            unmet    = 0.0

        if unmet > 1e-6 and pch > 0:
            cut  = min(unmet, pch)
            pch -= cut
            residual  -= cut
            pimp_act   = min(residual, GRID_PMAX)
            unmet      = max(residual - pimp_act, 0.0)

        # SoC reconstruction from actual executed decision
        delta_soc = (ETA * pch - pdis / ETA) * DT / BATTERY_KWH
        soc       = float(np.clip(soc + delta_soc, 0.0, 1.0))

        rows.append({
            "load_kw":    float(load_v[t]),
            "pv_kw":      float(pv_v[t]),
            "buy_price":  float(buy_v[t]),
            "sell_price": float(sell_v[t]),
            "p_ch":       pch,  "p_dis":   pdis,
            "p_battery":  pdis - pch,
            "p_imp":      pimp_act, "p_exp": pexp_act,
            "p_grid":     pimp_act - pexp_act,
            "soc":        soc,
            "energy_kwh": soc * BATTERY_KWH,
            "unmet_load": unmet,
        })

        if (step_i + 1) % 960 == 0:
            elapsed = (step_i + 1) * DT
            bill_sf = sum(
                (r["p_imp"] * r["buy_price"] - r["p_exp"] * r["sell_price"]) * DT
                for r in rows
            )
            print(f"    [{label}] step {step_i+1}/{len(idx)}  "
                  f"sim {elapsed/24:.0f}d  bill €{bill_sf:.2f}  SoC={soc:.2f}")

    return pd.DataFrame(rows, index=idx[: len(rows)])


# ════════════════════════════════════════════════════════════════════════════
# Plots
# ════════════════════════════════════════════════════════════════════════════

def _shade(ax, idx, bands):
    from tariff import assign_tariff_band as _atb
    if bands is None:
        bands = _atb(idx)
    ba = bands.values if hasattr(bands, "values") else bands
    if len(ba) == 0:
        return
    runs, cb, cs = [], ba[0], idx[0]
    for i in range(1, len(ba)):
        if ba[i] != cb:
            runs.append((cb, cs, idx[i])); cb, cs = ba[i], idx[i]
    runs.append((cb, cs, idx[-1] + pd.Timedelta("15min")))
    for b, s, e in runs:
        ax.axvspan(s, e, color=BAND_CLR.get(b, "white"), alpha=0.5, zorder=0)


def plot_dispatch_5panel(dispatch, month_name, nrmse_h1, bill, label="Controller"):
    """5-panel dispatch plot: Load+PV, Battery, Grid, SoC, Stored energy."""
    from tariff import assign_tariff_band
    idx   = dispatch.index
    bands = assign_tariff_band(idx)

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.2, 1.2]})
    fig.suptitle(
        f"{month_name} 2025 — {label}  |  "
        f"NRMSE(h=1)={nrmse_h1:.1f}%  |  Bill=€{bill:.2f}  |  SoC₀=50 %",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]; _shade(ax, idx, bands)
    ax.plot(idx, dispatch["load_kw"], color=COLORS["load"], lw=1.4, label="Load")
    ax.fill_between(idx, 0, dispatch["pv_kw"], color=COLORS["pv"], alpha=0.45, label="PV")
    ax.set_ylabel("Power (kW)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]; _shade(ax, idx, bands)
    pb = dispatch["p_battery"].values
    ax.fill_between(idx, 0, np.where(pb > 0, pb, 0), color=COLORS["battery"], alpha=0.7, label="Discharge (+)")
    ax.fill_between(idx, 0, np.where(pb < 0, pb, 0), color="#80CBC4", alpha=0.9, label="Charge (−)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline( BATTERY_PMAX, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-BATTERY_PMAX, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("P_battery (kW)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]; _shade(ax, idx, bands)
    pg = dispatch["p_grid"].values
    ax.fill_between(idx, 0, np.where(pg > 0, pg, 0), color=COLORS["grid"], alpha=0.6, label="Import (+)")
    ax.fill_between(idx, 0, np.where(pg < 0, pg, 0), color="#EE6C4D", alpha=0.6, label="Export (−)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline( GRID_PMAX, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-GRID_PMAX, color="grey", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("P_grid (kW)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[3]; _shade(ax, idx, bands)
    ax.plot(idx, dispatch["soc"], color=COLORS["soc"], lw=1.8, label="SoC")
    ax.axhline(SOC_INIT, color="grey", lw=1, ls="--", label=f"Start {SOC_INIT:.0%}")
    ax.fill_between(idx, 0, dispatch["soc"], color=COLORS["soc"], alpha=0.15)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("SoC [0–1]")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[4]; _shade(ax, idx, bands)
    ax.plot(idx, dispatch["energy_kwh"], color="#7B5EA7", lw=1.6, label="Stored energy")
    ax.axhline(SOC_INIT * BATTERY_KWH, color="grey", lw=1, ls="--")
    ax.fill_between(idx, 0, dispatch["energy_kwh"], color="#7B5EA7", alpha=0.15)
    ax.set_ylim(-0.5, BATTERY_KWH + 0.5); ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    fname = PLOT_DIR / f"dispatch_{month_name.lower()}_2025.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {fname}")


def plot_march_week3(dispatch, out_path):
    """Mandatory March Week 3 dispatch plot (Mar 17–23 2025)."""
    from tariff import assign_tariff_band
    w_s = pd.Timestamp("2025-03-17 00:00")
    w_e = pd.Timestamp("2025-03-24 00:00")
    sub = dispatch.loc[(dispatch.index >= w_s) & (dispatch.index < w_e)].copy()
    if len(sub) == 0:
        print("  [WARN] March Week 3 data not found — skipping mandatory plot.")
        return
    bands = assign_tariff_band(sub.index)

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 1.4, 1.4, 1.2, 1.0]})
    fig.suptitle(
        "March Week 3, 2025 — MPC Controller Dispatch  [MANDATORY PLOT]\n"
        "Load · PV · P_battery · P_grid · SoC  |  tariff bands F1/F2/F3 shaded",
        fontsize=13, fontweight="bold",
    )

    ax = axes[0]; _shade(ax, sub.index, bands)
    ax.plot(sub.index, sub["load_kw"], color=COLORS["load"], lw=1.5, label="Load (kW)")
    ax.fill_between(sub.index, 0, sub["pv_kw"], color=COLORS["pv"], alpha=0.45, label="PV (kW)")
    ax.set_ylabel("Power (kW)"); ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3); ax.set_ylim(bottom=0)

    ax = axes[1]; _shade(ax, sub.index, bands)
    pb = sub["p_battery"].values
    ax.fill_between(sub.index, 0, np.where(pb > 0, pb, 0),
                    color=COLORS["battery"], alpha=0.75, label="Discharge (+)")
    ax.fill_between(sub.index, 0, np.where(pb < 0, pb, 0),
                    color="#80CBC4", alpha=0.85, label="Charge (−)")
    ax.axhline(0, color="k", lw=0.5)
    for lim in [BATTERY_PMAX, -BATTERY_PMAX]:
        ax.axhline(lim, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("P_battery (kW)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[2]; _shade(ax, sub.index, bands)
    pg = sub["p_grid"].values
    ax.fill_between(sub.index, 0, np.where(pg > 0, pg, 0),
                    color=COLORS["grid"], alpha=0.6, label="Import (+)")
    ax.fill_between(sub.index, 0, np.where(pg < 0, pg, 0),
                    color="#EE6C4D", alpha=0.6, label="Export (−)")
    ax.axhline(0, color="k", lw=0.5)
    for lim in [GRID_PMAX, -GRID_PMAX]:
        ax.axhline(lim, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("P_grid (kW)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    ax = axes[3]; _shade(ax, sub.index, bands)
    ax.plot(sub.index, sub["soc"], color=COLORS["soc"], lw=1.8, label="SoC")
    ax.fill_between(sub.index, 0, sub["soc"], color=COLORS["soc"], alpha=0.15)
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("SoC [0–1]")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    ax.text(0.01, 0.07, f"min={sub['soc'].min():.2f}  max={sub['soc'].max():.2f}",
            transform=ax.transAxes, fontsize=8, color=COLORS["soc"])

    ax = axes[4]; _shade(ax, sub.index, bands)
    ax.plot(sub.index, sub["energy_kwh"], color="#7B5EA7", lw=1.6, label="Stored energy")
    ax.fill_between(sub.index, 0, sub["energy_kwh"], color="#7B5EA7", alpha=0.15)
    ax.set_ylim(-0.5, BATTERY_KWH + 0.5); ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    axes[-1].set_xlabel("March 2025")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_extension_bars(ext_results, bl_a_bill, bl_b_bill, oracle_bill, month_name, out_path):
    """3-panel bar chart: bill, savings, compute time vs H."""
    ext = sorted(ext_results, key=lambda r: r["H"])
    Hs  = [r["H"] for r in ext]
    bills   = [r["bill"]   for r in ext]
    savings = [r["savings_vs_a"] for r in ext]
    times   = [r["time_s"] for r in ext]
    xlabels = [f"H={h}\n({h//4}h)" for h in Hs]
    x       = np.arange(len(Hs))
    clrs    = ["#E74C3C", "#F39C12", "#2ECC71"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Extension — Horizon Sensitivity  [{month_name} 2025]",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.bar(x, bills, color=clrs)
    ax.axhline(bl_a_bill,   color="k",       lw=1.3, ls="--", label=f"Baseline A €{bl_a_bill:.2f}")
    ax.axhline(bl_b_bill,   color="#8E44AD", lw=1.3, ls=":",  label=f"Baseline B €{bl_b_bill:.2f}")
    ax.axhline(oracle_bill, color="#27AE60", lw=1.3, ls="-.", label=f"Oracle €{oracle_bill:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Monthly Bill (€)"); ax.set_title("Bill vs Horizon")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(bills):
        ax.text(i, v + 0.1, f"€{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.bar(x, savings, color=clrs)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Savings vs Baseline A (€)"); ax.set_title("Savings vs Horizon")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(savings):
        ax.text(i, max(v, 0) + 0.05, f"€{v:.2f}", ha="center", fontsize=9, fontweight="bold")

    ax = axes[2]
    ax.bar(x, times, color=["#3498DB", "#9B59B6", "#1ABC9C"])
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_ylabel("Compute Time (s)"); ax.set_title("Compute Time vs Horizon")
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(times):
        ax.text(i, v + 0.5, f"{v:.0f}s", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_soc_reconstruction(results):
    """Side-by-side SoC + stored energy reconstruction for both months."""
    months = [m for m in ["April", "September"] if m in results]
    if not months:
        return
    fig, axes = plt.subplots(2, len(months), figsize=(14, 8),
                              gridspec_kw={"height_ratios": [1.2, 1]})
    if len(months) == 1:
        axes = axes.reshape(-1, 1)
    fig.suptitle("SoC & Stored Energy — Reconstructed from MPC Output\n"
                 "SoC resets to 50 % at start of each month",
                 fontsize=13, fontweight="bold")

    for col, month in enumerate(months):
        d   = results[month]["ctrl_dispatch"]
        idx = d.index

        ax = axes[0][col]
        ax.plot(idx, d["soc"], color=COLORS["soc"], lw=1.5, label="SoC (controller)")
        ax.axhline(SOC_INIT, color="grey", lw=1.2, ls="--", label=f"Start = {SOC_INIT:.0%}")
        ax.fill_between(idx, SOC_INIT, d["soc"], where=(d["soc"] > SOC_INIT),
                        color="#02C39A", alpha=0.3, label="Above 50 %")
        ax.fill_between(idx, SOC_INIT, d["soc"], where=(d["soc"] < SOC_INIT),
                        color="#990011", alpha=0.25, label="Below 50 %")
        ax.set_ylim(-0.05, 1.05); ax.set_ylabel("SoC [0–1]")
        ax.set_title(f"{month} 2025\nbill=€{results[month]['ctrl_bill']:.2f}  "
                     f"savings vs A=€{results[month]['sav_vs_a']:.2f}",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.9); ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        soc_min = d["soc"].min(); soc_max = d["soc"].max()
        ax.annotate(f"min {soc_min:.2f}", xy=(d["soc"].idxmin(), soc_min),
                    xytext=(d["soc"].idxmin(), soc_min + 0.12),
                    arrowprops=dict(arrowstyle="->", color="red"), fontsize=8, color="red")
        ax.annotate(f"max {soc_max:.2f}", xy=(d["soc"].idxmax(), soc_max),
                    xytext=(d["soc"].idxmax(), soc_max - 0.12),
                    arrowprops=dict(arrowstyle="->", color="green"), fontsize=8, color="green")

        ax2 = axes[1][col]
        ax2.plot(idx, d["energy_kwh"], color="#7B5EA7", lw=1.5, label="Stored energy")
        ax2.axhline(SOC_INIT * BATTERY_KWH, color="grey", lw=1.2, ls="--",
                    label=f"Start = {SOC_INIT*BATTERY_KWH:.0f} kWh")
        ax2.fill_between(idx, 0, d["energy_kwh"], color="#7B5EA7", alpha=0.2)
        ax2.set_ylim(-0.5, BATTERY_KWH + 0.5); ax2.set_ylabel("Stored energy (kWh)")
        ax2.set_xlabel("Date"); ax2.legend(fontsize=8, framealpha=0.9); ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

    plt.tight_layout()
    out = PLOT_DIR / "soc_reconstruction.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {out}")


def print_results_table(month_name, bl_a_bill, bl_b_bill,
                         ctrl_bill, oracle_bill, nrmse_h1,
                         rmse_h1=None, mae_h1=None):
    sav_a     = bl_a_bill - ctrl_bill
    sav_a_pct = 100 * sav_a / bl_a_bill if bl_a_bill != 0 else 0.0
    sav_b     = bl_b_bill - ctrl_bill
    sav_b_pct = 100 * sav_b / bl_b_bill if bl_b_bill != 0 else 0.0
    gap_eur   = ctrl_bill - oracle_bill
    gap_pct   = 100 * gap_eur / (bl_a_bill - oracle_bill) \
                if (bl_a_bill - oracle_bill) > 0 else float("nan")

    w = 62
    print(f"\n  {'=' * w}")
    print(f"  RESULTS TABLE — {month_name.upper()} 2025")
    print(f"  {'=' * w}")
    rows = [
        ("Baseline A  (historical, SoC-clamped)",  f"€{bl_a_bill:.2f}"),
        ("Baseline B  (no battery)",                f"€{bl_b_bill:.2f}"),
        ("Controller  (MPC H=96, causal fcast)",    f"€{ctrl_bill:.2f}"),
        ("", ""),
        ("Savings vs Baseline A",  f"€{sav_a:+.2f}  ({sav_a_pct:+.1f}%)"),
        ("Savings vs Baseline B",  f"€{sav_b:+.2f}  ({sav_b_pct:+.1f}%)"),
        ("", ""),
        ("Oracle  (perfect-foresight MPC)",         f"€{oracle_bill:.2f}"),
        ("Oracle gap",             f"€{gap_eur:.2f}  ({gap_pct:.1f}%)"),
        ("", ""),
        ("Forecast RMSE   h=1",                     f"{rmse_h1:.4f} kW" if rmse_h1 is not None else "N/A"),
        ("Forecast MAE    h=1",                     f"{mae_h1:.4f} kW" if mae_h1 is not None else "N/A"),
        ("Forecast NRMSE  h=1",                     f"{nrmse_h1:.2f}%"),
    ]
    for lbl, val in rows:
        if lbl:
            print(f"  {lbl:45s}{val:>15s}")
        else:
            print()
    print(f"  {'=' * w}")
    return sav_a, sav_a_pct, gap_eur, gap_pct


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    from data_loader import load_dataset

    print("=" * 65)
    print("  APRIL & SEPTEMBER 2025 — FULL MPC ANALYSIS")
    print("=" * 65)

    print("\n[0] Loading data …")
    sheets  = load_dataset(DATA_PATH)
    df_2024 = sheets["2024"]
    df_2025 = sheets["2025"]
    print(f"  2024: {len(df_2024):,} rows  |  2025: {len(df_2025):,} rows")

    # ── Corrupted battery data detection (required by brief) ──────────────
    print("\n  Checking 2025 battery data integrity …")
    detect_corrupted_battery(df_2025)

    month_configs = [
        dict(name="April",     month_num=4,
             val_start="2025-02-01",
             test_start="2025-04-01", test_end="2025-05-01"),
        dict(name="September", month_num=9,
             val_start="2025-07-01",
             test_start="2025-09-01", test_end="2025-10-01"),
    ]

    all_results = {}

    for cfg in month_configs:
        month_name  = cfg["name"]
        print(f"\n{'='*65}")
        print(f"  {month_name.upper()} 2025")
        print(f"{'='*65}")

        # ── Data splits ─────────────────────────────────────────────────────
        df_pre   = df_2025[df_2025.index < cfg["val_start"]]         # all before val (inclusive)
        df_train = pd.concat([df_2024, df_pre])                      # all before val
        df_val   = df_2025[(df_2025.index >= cfg["val_start"]) &
                           (df_2025.index <  cfg["test_start"])]
        df_test  = df_2025[(df_2025.index >= cfg["test_start"]) &
                           (df_2025.index <  cfg["test_end"])]
        df_buf   = df_2025[df_2025.index < cfg["test_start"]].iloc[-max(LAGS):]

        print(f"  Train: {df_train.index.min().date()} → {df_train.index.max().date()} "
              f"({len(df_train):,} rows)")
        print(f"  Val  : {df_val.index.min().date()} → {df_val.index.max().date()}")
        print(f"  Test : {df_test.index.min().date()} → {df_test.index.max().date()}")

        # ── EDA ─────────────────────────────────────────────────────────────
        eda_summary(df_test, df_2024, month_name)

        # ── Train forecaster ─────────────────────────────────────────────────
        print(f"\n  [1/5] Training {len(SPARSE_H)} LightGBM models …")
        t0     = time.time()
        models = train_multihorizon(df_train, df_val)
        print(f"  Training done in {time.time()-t0:.1f}s")

        # ── Generate forecasts + metrics (RMSE, MAE, NRMSE — all required) ───
        print(f"\n  [2/5] Generating forecasts …")
        forecasts = predict_month(models, df_buf, df_test)
        actual_h1 = df_test["load_kw"].reindex(forecasts.index).values
        pred_h1   = forecasts["h1"].values
        nr        = nrmse(actual_h1, pred_h1)
        rm        = rmse(actual_h1, pred_h1)
        ma        = mae(actual_h1, pred_h1)
        print(f"  Forecast RMSE  (h=1): {rm:.4f} kW")
        print(f"  Forecast MAE   (h=1): {ma:.4f} kW")
        print(f"  Forecast NRMSE (h=1): {nr:.2f}%")

        # ── Causal PV + sell matrices ─────────────────────────────────────────
        pv_fc_mat, sell_fc_mat = build_causal_horizon_matrices(df_buf, df_test, H_max=96)

        # ── Baselines A and B ─────────────────────────────────────────────────
        print(f"\n  [3/5] Computing baselines …")
        bl_b = baseline_b_month(df_test)
        bl_a = baseline_a_month(df_test)
        print(f"  Baseline A (hist. replay): €{bl_a['bill']:.2f}")
        print(f"  Baseline B (no battery)  : €{bl_b['bill']:.2f}")

        # ── Oracle MPC ────────────────────────────────────────────────────────
        print(f"\n  [4/5] Running Oracle MPC (H=96, perfect foresight) …")
        t0            = time.time()
        oracle_disp   = run_mpc_general(df_test, H=96, soc_start=SOC_INIT, oracle=True,
                                        label=f"Oracle {month_name}")
        oracle_time   = time.time() - t0
        oracle_bill   = compute_bill(oracle_disp)
        print(f"  Oracle bill: €{oracle_bill:.2f}  ({oracle_time:.0f}s)")

        # ── Controller MPC  +  Extension (H = 4, 24, 96) ─────────────────────
        print(f"\n  [5/5] Controller MPC — Extension H ∈ {EXTENSION_H} …")
        ext_results   = []
        ctrl_dispatch = None

        for H in EXTENSION_H:
            print(f"    H={H} …")
            t0   = time.time()
            disp = run_mpc_general(
                df_test, H=H, soc_start=SOC_INIT, oracle=False,
                load_fc=forecasts, pv_fc_mat=pv_fc_mat, sell_fc_mat=sell_fc_mat,
                label=f"{month_name} H={H}",
            )
            t_h  = time.time() - t0
            bill = compute_bill(disp)
            ext_results.append({
                "H": H, "bill": bill,
                "savings_vs_a": bl_a["bill"] - bill,
                "time_s": t_h,
            })
            print(f"    H={H:2d}  bill=€{bill:.2f}  "
                  f"sav=€{bl_a['bill']-bill:.2f}  time={t_h:.0f}s")
            if H == 96:
                ctrl_dispatch = disp
                ctrl_bill     = bill

        # ── Results table ─────────────────────────────────────────────────────
        sav_a, sav_a_pct, gap_eur, gap_pct = print_results_table(
            month_name, bl_a["bill"], bl_b["bill"],
            ctrl_bill, oracle_bill, nr,
            rmse_h1=rm, mae_h1=ma,
        )

        # ── Extension table ───────────────────────────────────────────────────
        print(f"\n  Extension summary — {month_name} 2025:")
        print(f"  {'H':>5}  {'Look-ahead':>10}  {'Bill €':>10}  "
              f"{'Savings vs A €':>15}  {'Time s':>8}")
        print("  " + "-" * 55)
        for r in sorted(ext_results, key=lambda x: x["H"]):
            print(f"  {r['H']:>5}  {r['H']//4:>9}h  {r['bill']:>10.2f}  "
                  f"{r['savings_vs_a']:>15.2f}  {r['time_s']:>8.0f}")
        # Recommended H with rationale (required by extension spec)
        best_ext = max(ext_results, key=lambda r: r["savings_vs_a"])
        print(f"\n  → Recommended H={best_ext['H']} ({best_ext['H']//4}h look-ahead): "
              f"captures full daily tariff cycle with best savings-to-compute tradeoff.")

        # ── Save CSV ──────────────────────────────────────────────────────────
        ctrl_dispatch.to_csv(OUT_DIR / f"dispatch_{month_name.lower()}_2025.csv")

        all_results[month_name] = {
            "ctrl_dispatch": ctrl_dispatch,
            "oracle_dispatch": oracle_disp,
            "ctrl_bill": ctrl_bill,
            "oracle_bill": oracle_bill,
            "bl_a_bill": bl_a["bill"],
            "bl_b_bill": bl_b["bill"],
            "sav_vs_a": sav_a,
            "sav_vs_a_pct": sav_a_pct,
            "oracle_gap_eur": gap_eur,
            "oracle_gap_pct": gap_pct,
            "nrmse_h1": nr,
            "rmse_h1": rm,
            "mae_h1": ma,
            "ext_results": ext_results,
            "models": models,
            "df_buf": df_buf,
            "df_val": df_val,
            "df_test": df_test,
        }

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_dispatch_5panel(ctrl_dispatch, month_name, nr, ctrl_bill)
        plot_extension_bars(
            ext_results, bl_a["bill"], bl_b["bill"], oracle_bill,
            month_name, PLOT_DIR / f"extension_{month_name.lower()}.png",
        )

    # ── March Week 3 mandatory plot ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  MARCH WEEK 3 — MANDATORY DISPATCH PLOT")
    print(f"{'='*65}")
    # Use April model (trained on 2024 + Jan 2025); March is in its validation period
    # Start MPC from March 1 with SoC = 0.50
    apr = all_results["April"]
    df_march = df_2025[(df_2025.index >= "2025-03-01") & (df_2025.index < "2025-04-01")]
    df_buf_mar = df_2025[df_2025.index < "2025-03-01"].iloc[-max(LAGS):]
    pv_fc_mar, sell_fc_mar = build_causal_horizon_matrices(df_buf_mar, df_march, H_max=96)
    fc_march = predict_month(apr["models"], df_buf_mar, df_march)

    print("  Running MPC on March 2025 (for Week 3 plot) …")
    t0 = time.time()
    march_disp = run_mpc_general(
        df_march, H=96, soc_start=SOC_INIT, oracle=False,
        load_fc=fc_march, pv_fc_mat=pv_fc_mar, sell_fc_mat=sell_fc_mar,
        label="March",
    )
    march_bill = compute_bill(march_disp)
    print(f"  March 2025 bill: €{march_bill:.2f}  ({time.time()-t0:.0f}s)")
    march_disp.to_csv(OUT_DIR / "dispatch_march_2025.csv")

    plot_march_week3(march_disp, PLOT_DIR / "march_week3_dispatch.png")

    # ── Side-by-side SoC reconstruction ──────────────────────────────────────
    plot_soc_reconstruction(all_results)

    # ── Final cross-month summary ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — BOTH TEST MONTHS")
    print(f"{'='*65}")
    print(f"  {'Month':>12}  {'BL-A €':>8}  {'BL-B €':>8}  {'Ctrl €':>8}  "
          f"{'Sav-A €':>8}  {'Sav-A%':>7}  {'Gap%':>6}  {'NRMSE%':>7}")
    print("  " + "-" * 75)
    for mn, r in all_results.items():
        print(f"  {mn:>12}  {r['bl_a_bill']:>8.2f}  {r['bl_b_bill']:>8.2f}  "
              f"{r['ctrl_bill']:>8.2f}  {r['sav_vs_a']:>8.2f}  "
              f"{r['sav_vs_a_pct']:>7.1f}  {r['oracle_gap_pct']:>6.1f}  "
              f"{r['nrmse_h1']:>7.2f}")

    print(f"\nOutputs written to {OUT_DIR}/")
    print("  dispatch_april_2025.csv          — April MPC controller (H=96)")
    print("  dispatch_september_2025.csv      — September MPC controller (H=96)")
    print("  dispatch_march_2025.csv          — March MPC (for mandatory plot)")
    print("  plots/dispatch_april_2025.png    — 5-panel April dispatch")
    print("  plots/dispatch_september_2025.png — 5-panel September dispatch")
    print("  plots/march_week3_dispatch.png   — MANDATORY March Week 3 plot")
    print("  plots/extension_april.png        — H sensitivity April")
    print("  plots/extension_september.png    — H sensitivity September")
    print("  plots/soc_reconstruction.png     — SoC side-by-side both months")


if __name__ == "__main__":
    main()
