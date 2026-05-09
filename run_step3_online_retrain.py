"""
Online-retraining MPC: Huber+Q, H=192, monthly model refresh.

Each month of 2025 uses models retrained on ALL data available BEFORE that month:
  Jan: train Jan-Oct 2024,  val Nov-Dec 2024
  Feb: train Jan-Nov 2024,  val Dec 2024 - Jan 2025
  Mar: train Jan-Dec 2024,  val Jan-Feb 2025
  Apr: train Jan-Dec 2024 + Jan 2025,     val Feb-Mar 2025
  ...
  Dec: train Jan-Dec 2024 + Jan-Sep 2025, val Oct-Nov 2025

Better seasonal fit → lower forecast error → lower bill.
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solship"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays as hol_pkg

from features   import build_features
from forecaster import nrmse as _nrmse, rmse as _rmse, mae as _mae
from optimizer  import validate_dispatch
from config     import (SOC_INIT, DT_HOURS, BATTERY_KWH, BATTERY_P_MAX_KW,
                        GRID_P_MAX_KW, ETA_DIR)

BASELINE_A = 1_218.97
BASELINE_B = 1_601.09
ORACLE     = 1_194.90
BILL_H192  = 1_289.94   # previous best (H=192 fixed models)

F1, F2, F3       = 0.2540, 0.2682, 0.2440
ITALIAN_HOLIDAYS = hol_pkg.Italy(years=[2024, 2025])
ENHANCED_LAGS    = [0, 1, 2, 3, 4, 5, 6, 7, 8, 48, 96, 192, 336, 672]
ENHANCED_ROLLS   = [4, 8, 16, 96, 672]
SPARSE_HORIZONS  = [1, 2, 4, 8, 16, 24, 48, 72, 96, 144, 192]
H_MAX            = 192

LGB_PARAMS = {
    "verbose":-1, "num_threads":-1,
    "learning_rate":0.0272, "num_leaves":167,
    "min_data_in_leaf":135, "feature_fraction":0.862,
    "bagging_fraction":0.656, "bagging_freq":5,
    "lambda_l2":0.078, "lambda_l1":0.150,
}

log1p  = np.log1p
expm1c = lambda x: np.expm1(x).clip(min=0)


# ── load data ──────────────────────────────────────────────────────────────────
print("="*60)
print("1. LOADING DATA")
print("="*60)

def compute_buy_price(ts):
    dow = ts.dt.dayofweek; h = ts.dt.hour
    hol = ts.dt.date.map(lambda d: d in ITALIAN_HOLIDAYS)
    f3  = (dow==6)|hol|(~(dow==6)&~hol&((h<7)|(h>=23)))
    f2  = (~f3)&(((dow==5)&(h>=7)&(h<23))|(~(dow==5)&~(dow==6)&~hol&((h==7)|((h>=19)&(h<23)))))
    p   = pd.Series(F1, index=ts.index, dtype=float); p[f2]=F2; p[f3]=F3
    return p

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
raw["load_kw"] = raw["load_kw"].clip(lower=0)
raw["pv_kw"]   = raw["pv_kw"].clip(lower=0)

df2024 = raw[raw.index.year == 2024]
df2025 = raw[raw.index.year == 2025]

print(f"  2024 rows: {len(df2024):,}")
print(f"  2025 rows: {len(df2025):,}")
print(f"  H_MAX : {H_MAX} steps (48h lookahead)")
print(f"  Sparse horizons: {SPARSE_HORIZONS}")

def log_df(df):
    d = df.copy(); d["load_kw"] = log1p(d["load_kw"]); return d

def make_Xy(df_buf, df_target, horizon, feat_cols=None):
    X, y = build_features(df_buf, lags=ENHANCED_LAGS,
                           roll_windows=ENHANCED_ROLLS, horizon=horizon)
    idx = df_target.index.intersection(X.index)
    X, y = X.loc[idx], y.loc[idx]
    if feat_cols is not None:
        X = X[feat_cols]
    return X, y


# ── MPC controller (same as run_step3_improved.py) ───────────────────────────
class ImprovedMPCController:
    def __init__(self, horizon):
        self.H = horizon
        self.eta = ETA_DIR
        self.dt = DT_HOURS
        self.e_max = BATTERY_KWH
        self.p_bat_max = BATTERY_P_MAX_KW
        self.p_grid_max = GRID_P_MAX_KW   # full 6 kW planning cap

        self.load_hat   = cp.Parameter(horizon, nonneg=True)
        self.pv_hat     = cp.Parameter(horizon, nonneg=True)
        self.buy_price  = cp.Parameter(horizon, nonneg=True)
        self.sell_price = cp.Parameter(horizon, nonneg=True)
        self.soc_init   = cp.Parameter(nonneg=True)

        self.p_ch  = cp.Variable(horizon, nonneg=True)
        self.p_dis = cp.Variable(horizon, nonneg=True)
        self.p_imp = cp.Variable(horizon, nonneg=True)
        self.p_exp = cp.Variable(horizon, nonneg=True)
        self.soc   = cp.Variable(horizon + 1)

        cons = [
            self.soc[0] == self.soc_init,
            self.p_ch  <= self.p_bat_max,
            self.p_dis <= self.p_bat_max,
            self.p_imp <= self.p_grid_max,
            self.p_exp <= self.p_grid_max,
            self.soc >= 0,
            self.soc <= 1,
            self.soc[1:] == (self.soc[:-1] +
                (self.eta * self.p_ch - self.p_dis / self.eta) * self.dt / self.e_max),
            (self.load_hat + self.p_ch + self.p_exp ==
             self.pv_hat + self.p_dis + self.p_imp),
        ]

        cost = cp.sum(
            cp.multiply(self.p_imp, self.buy_price)
            - cp.multiply(self.p_exp, self.sell_price)
        ) * self.dt

        self.problem = cp.Problem(cp.Minimize(cost), cons)

    def solve_one(self, load_hat_arr, pv_hat_arr, buy_arr, sell_arr, soc_now,
                  warm_start=True):
        H = self.H

        def pad(arr):
            if len(arr) < H:
                return np.concatenate([arr, np.full(H - len(arr), arr[-1])])
            return arr[:H]

        self.load_hat.value   = np.maximum(pad(load_hat_arr), 0.0)
        self.pv_hat.value     = np.maximum(pad(pv_hat_arr),   0.0)
        self.buy_price.value  = np.maximum(pad(buy_arr),      0.0)
        self.sell_price.value = np.maximum(pad(sell_arr),     0.0)
        self.soc_init.value   = float(np.clip(soc_now, 0.0, 1.0))

        try:
            self.problem.solve(solver=cp.HIGHS, warm_start=warm_start, verbose=False)
        except Exception:
            return None

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            return None

        return {
            "p_ch":  float(self.p_ch.value[0]),
            "p_dis": float(self.p_dis.value[0]),
            "p_imp": float(self.p_imp.value[0]),
            "p_exp": float(self.p_exp.value[0]),
            "soc_planned_next": float(self.soc.value[1]),
        }


def run_mpc_segment(df_actual_seg, forecasts_seg, soc_init, horizon=H_MAX,
                    ctrl=None, verbose=False):
    """Run MPC over one segment (one month). Returns dispatch DataFrame."""
    common_idx   = df_actual_seg.index.intersection(forecasts_seg.index)
    n            = len(common_idx)
    load_actual  = df_actual_seg.loc[common_idx, "load_kw"].values
    pv_actual    = df_actual_seg.loc[common_idx, "pv_kw"].values
    buy_arr      = df_actual_seg.loc[common_idx, "buy_price"].values
    sell_arr     = df_actual_seg.loc[common_idx, "sell_price"].values
    forecast_arr = forecasts_seg.loc[common_idx].values

    if ctrl is None:
        ctrl = ImprovedMPCController(horizon=horizon)

    rows = []
    soc  = float(soc_init)

    for t in range(n):
        end = min(t + horizon, n)
        L   = end - t
        if L < 1:
            break

        load_hat = forecast_arr[t, :L]
        pv_h     = pv_actual[t:end]
        buy_h    = buy_arr[t:end]
        sell_h   = sell_arr[t:end]

        sol = ctrl.solve_one(load_hat, pv_h, buy_h, sell_h, soc)

        if sol is None:
            net = max(load_actual[t] - pv_actual[t], 0)
            sol = {"p_ch": 0.0, "p_dis": 0.0,
                   "p_imp": min(net, GRID_P_MAX_KW),
                   "p_exp": max(pv_actual[t] - load_actual[t], 0),
                   "soc_planned_next": soc}

        p_ch  = sol["p_ch"]
        p_dis = sol["p_dis"]

        # Execute against actual: keep battery decision, rebalance grid
        net_actual = load_actual[t] - pv_actual[t]
        residual   = net_actual - p_dis + p_ch

        if residual >= 0:
            p_imp = min(residual, GRID_P_MAX_KW)
            p_exp = 0.0
            unmet = residual - p_imp
        else:
            p_exp = min(-residual, GRID_P_MAX_KW)
            p_imp = 0.0
            unmet = 0.0

        # Fix 1: scale back charging
        if unmet > 1e-6 and p_ch > 0:
            reduction = min(unmet, p_ch)
            p_ch  -= reduction
            residual -= reduction
            p_imp = min(residual, GRID_P_MAX_KW)
            unmet = max(residual - p_imp, 0)

        # Fix 2: boost discharge if still unmet (load genuinely > 6 kW)
        if unmet > 1e-6:
            headroom_bat = BATTERY_P_MAX_KW - p_dis
            max_from_soc = soc * BATTERY_KWH / ETA_DIR / DT_HOURS
            extra_dis = min(unmet, headroom_bat, max_from_soc)
            if extra_dis > 1e-6:
                p_dis    += extra_dis
                residual -= extra_dis
                p_imp = min(residual, GRID_P_MAX_KW)
                unmet = max(residual - p_imp, 0)

        de  = (p_ch * ETA_DIR - p_dis / ETA_DIR) * DT_HOURS
        soc = float(np.clip(soc + de / BATTERY_KWH, 0, 1))

        rows.append({
            "load_kw":   float(load_actual[t]),
            "pv_kw":     float(pv_actual[t]),
            "buy_price": float(buy_arr[t]),
            "sell_price":float(sell_arr[t]),
            "p_ch":      p_ch,
            "p_dis":     p_dis,
            "p_battery": p_dis - p_ch,
            "p_imp":     p_imp,
            "p_exp":     p_exp,
            "p_grid":    p_imp - p_exp,
            "soc":       soc,
            "unmet_load":unmet,
        })

    return pd.DataFrame(rows, index=common_idx[:len(rows)]), soc


# ── main loop: one model per month ───────────────────────────────────────────
print("\n" + "="*60)
print("2. ONLINE RETRAINING — 12 MONTHLY PASSES")
print("="*60)

all_dispatches = []
soc_carry = SOC_INIT
monthly_nrmse = []

# For aggregate metric collection
all_y_true = []   # h=1 actuals across all 2025
all_y_pred = []   # h=1 predictions across all 2025
apr_y_true = None; apr_y_pred = None
sep_y_true = None; sep_y_pred = None

ctrl = ImprovedMPCController(horizon=H_MAX)   # reuse one controller

t_total = time.time()

for m in range(1, 13):
    month_start = pd.Timestamp(f"2025-{m:02d}-01")
    month_end   = (month_start + pd.offsets.MonthEnd(0)).replace(hour=23, minute=45)
    month_name  = month_start.strftime("%B")

    # ── determine train / val cutoffs ────────────────────────────────────────
    val_end   = month_start - pd.Timedelta("15min")
    val_start = (month_start - pd.DateOffset(months=2)).replace(day=1)

    # Guard: val cannot go before start of 2024
    val_start = max(val_start, pd.Timestamp("2024-01-01"))
    train_end = val_start - pd.Timedelta("15min")
    train_start = pd.Timestamp("2024-01-01")

    df_train_m = raw[(raw.index >= train_start) & (raw.index <= train_end)]
    df_val_m   = raw[(raw.index >= val_start)   & (raw.index <= val_end)]
    df_test_m  = raw[(raw.index >= month_start) & (raw.index <= month_end)]

    print(f"\n  [{m:02d}/{month_name}]  "
          f"train={len(df_train_m):,} rows ({train_start.date()} - {train_end.date()})  "
          f"val={len(df_val_m):,}  test={len(df_test_m):,}")

    t_m = time.time()

    # ── train 9×2 models ─────────────────────────────────────────────────────
    df_tr_log  = log_df(df_train_m)
    df_va_buf  = log_df(pd.concat([df_train_m.iloc[-max(ENHANCED_LAGS):], df_val_m]))

    X_tr1, y_tr1 = make_Xy(df_tr_log, df_train_m, 1)
    feat_cols    = list(X_tr1.columns)

    models_hub = {}; models_qnt = {}

    for h in SPARSE_HORIZONS:
        X_tr, y_tr = make_Xy(df_tr_log, df_train_m, h, feat_cols)
        X_va, y_va = make_Xy(df_va_buf, df_val_m,   h, feat_cols)

        dtrain = lgb.Dataset(X_tr.values, y_tr.values, feature_name=feat_cols, free_raw_data=False)
        dval   = lgb.Dataset(X_va.values, y_va.values, feature_name=feat_cols,
                              reference=dtrain, free_raw_data=False)
        cbs = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]

        models_hub[h] = lgb.train(
            {**LGB_PARAMS,"objective":"huber","metric":"huber","alpha":0.9},
            dtrain, num_boost_round=2000, valid_sets=[dval], valid_names=["val"], callbacks=cbs)
        models_qnt[h] = lgb.train(
            {**LGB_PARAMS,"objective":"quantile","metric":"quantile","alpha":0.50},
            dtrain, num_boost_round=2000, valid_sets=[dval], valid_names=["val"], callbacks=cbs)

    print(f"    Training: {time.time()-t_m:.1f}s", end="  |  ")

    # ── generate forecast matrix for this month ───────────────────────────────
    # Lag buffer = all data up to start of this month
    buf_data = raw[raw.index < month_start]
    df_te_buf = log_df(pd.concat([buf_data, df_test_m]))

    pred_hub = {}; pred_qnt = {}
    for h in SPARSE_HORIZONS:
        X, _ = make_Xy(df_te_buf, df_test_m, h, feat_cols)
        pred_hub[h] = pd.Series(expm1c(models_hub[h].predict(X.values)), index=X.index)
        pred_qnt[h] = pd.Series(expm1c(models_qnt[h].predict(X.values)), index=X.index)

    # Common index across all horizons within this month
    common = pred_hub[SPARSE_HORIZONS[0]].index
    for h in SPARSE_HORIZONS[1:]:
        common = common.intersection(pred_hub[h].index)
    common = common[(common >= month_start) & (common <= month_end)]

    if len(common) == 0:
        print(f"    WARNING: no common forecast index for {month_name}, skipping")
        continue

    sparse_arr = np.array(SPARSE_HORIZONS)
    hub_mat = np.column_stack([pred_hub[h].loc[common].values for h in SPARSE_HORIZONS])
    qnt_mat = np.column_stack([pred_qnt[h].loc[common].values for h in SPARSE_HORIZONS])
    blend   = 0.5 * hub_mat + 0.5 * qnt_mat

    target_h = np.arange(1, H_MAX + 1)
    mat = np.zeros((len(common), H_MAX))
    for i in range(len(common)):
        mat[i] = np.interp(target_h, sparse_arr, blend[i])
    mat = np.clip(mat, 0, None)

    forecasts_m = pd.DataFrame(mat, index=common,
                                columns=[f"h{h}" for h in target_h])

    # h=1 metrics for this month
    y_true_h1 = df_test_m["load_kw"].reindex(common).dropna()
    y_pred_h1 = forecasts_m["h1"].reindex(y_true_h1.index)
    h1_nrmse  = _nrmse(y_true_h1, y_pred_h1)
    monthly_nrmse.append(h1_nrmse)
    print(f"h=1 NRMSE={h1_nrmse:.1f}%", end="  |  ")

    # Accumulate for aggregate metrics
    all_y_true.append(y_true_h1)
    all_y_pred.append(y_pred_h1)
    if m == 4:
        apr_y_true = y_true_h1.copy(); apr_y_pred = y_pred_h1.copy()
    if m == 9:
        sep_y_true = y_true_h1.copy(); sep_y_pred = y_pred_h1.copy()

    # ── run MPC for this month ────────────────────────────────────────────────
    df_actual_m = df_test_m.reindex(forecasts_m.index)
    dispatch_m, soc_carry = run_mpc_segment(df_actual_m, forecasts_m,
                                             soc_init=soc_carry, ctrl=ctrl)

    bill_m = float(
        (dispatch_m["p_imp"] * dispatch_m["buy_price"] -
         dispatch_m["p_exp"] * dispatch_m["sell_price"]).sum() * DT_HOURS
    )
    unmet_m = dispatch_m["unmet_load"].sum() * DT_HOURS
    print(f"bill=EUR {bill_m:7.2f}  unmet={unmet_m:.3f} kWh  "
          f"total={time.time()-t_m:.1f}s")

    all_dispatches.append(dispatch_m)

print(f"\n  Total elapsed: {time.time()-t_total:.1f}s")


# ── aggregate annual results ──────────────────────────────────────────────────
print("\n" + "="*60)
print("3. ANNUAL RESULTS")
print("="*60)

dispatch_full = pd.concat(all_dispatches).sort_index()

ok, rep = validate_dispatch(dispatch_full)
bill = float(
    (dispatch_full["p_imp"] * dispatch_full["buy_price"] -
     dispatch_full["p_exp"] * dispatch_full["sell_price"]).sum() * DT_HOURS
)

print(f"\n  Validation : {'PASS' if ok else 'FAIL'}")
print(f"  Unmet load : {rep['total_unmet_load_kwh']:.4f} kWh")
print(f"  Annual bill: EUR {bill:.2f}")
print(f"  vs Baseline A ({BASELINE_A:.2f}): {bill-BASELINE_A:+.2f} EUR  "
      f"({'BEAT!' if bill < BASELINE_A else 'did NOT beat'})")
print(f"  vs Baseline B ({BASELINE_B:.2f}): {bill-BASELINE_B:+.2f} EUR")
print(f"  vs Oracle    ({ORACLE:.2f}):   {bill-ORACLE:+.2f} EUR")
print(f"  vs H=192 fixed models: {BILL_H192-bill:+.2f} EUR improvement")
saving_vs_B = (BASELINE_B - bill) / BASELINE_B * 100
print(f"  Saving vs B: {saving_vs_B:.1f}%")
print(f"  Avg h=1 NRMSE: {np.mean(monthly_nrmse):.2f}%")

# ── forecast metrics for 3 periods ───────────────────────────────────────────
print("\n" + "="*60)
print("4. FORECAST METRICS (Online Retrain — h=1)")
print("="*60)

yt_full = pd.concat(all_y_true)
yp_full = pd.concat(all_y_pred)

metrics_rows = []
for label, yt, yp in [
    ("Full Year 2025", yt_full,    yp_full),
    ("April 2025",     apr_y_true, apr_y_pred),
    ("September 2025", sep_y_true, sep_y_pred),
]:
    nrmse_v = _nrmse(yt, yp)
    rmse_v  = _rmse(yt, yp)
    mae_v   = _mae(yt, yp)
    print(f"  {label:<20s}  NRMSE={nrmse_v:.2f}%  RMSE={rmse_v:.4f} kW  MAE={mae_v:.4f} kW")
    metrics_rows.append({"period": label, "nrmse_pct": round(nrmse_v,2),
                         "rmse_kw": round(rmse_v,4), "mae_kw": round(mae_v,4)})

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("step3_online_forecast_metrics.csv", index=False)
print("\n  Saved: step3_online_forecast_metrics.csv")


# ── plots ─────────────────────────────────────────────────────────────────────
print("\nGenerating plots...")

# Plot 1: Bill comparison
fig, ax = plt.subplots(figsize=(11, 5))
labels = ["Baseline B\n(no battery)", "Baseline A\n(historical)",
          "H=96\nfixed", "H=192\nfixed", "H=192\nonline\nretrain", "Oracle"]
bills  = [BASELINE_B, BASELINE_A, 1303.28, BILL_H192, bill, ORACLE]
colors = ["#d62728", "#ff7f0e", "#9467bd", "#17becf", "#2ca02c", "#8c564b"]
bars   = ax.bar(labels, bills, color=colors, alpha=0.85, edgecolor="white")
ax.axhline(BASELINE_A, ls="--", color="#ff7f0e", lw=1.4, alpha=0.8,
           label=f"Baseline A = EUR {BASELINE_A:.2f}")
for bar, b in zip(bars, bills):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
            f"EUR {b:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Annual Bill (EUR)")
ax.set_title("Full-Year 2025: Online Retraining MPC vs All Approaches")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(bills)*1.15)
plt.tight_layout()
plt.savefig("step3_online_bill_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: step3_online_bill_comparison.png")

# Plot 2: Monthly NRMSE + monthly bill
monthly_bills = []
for disp in all_dispatches:
    b = float((disp["p_imp"]*disp["buy_price"] - disp["p_exp"]*disp["sell_price"]).sum() * DT_HOURS)
    monthly_bills.append(b)

months_str = [pd.Timestamp(f"2025-{m:02d}-01").strftime("%b") for m in range(1, len(all_dispatches)+1)]

fig, axes = plt.subplots(2, 1, figsize=(12, 7))
axes[0].bar(months_str, monthly_nrmse, color="#2ca02c", alpha=0.8, edgecolor="white")
axes[0].set_ylabel("h=1 NRMSE (%)"); axes[0].set_title("Monthly h=1 NRMSE — Online Retrain")
axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(monthly_nrmse):
    axes[0].text(i, v+0.3, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

axes[1].bar(months_str, monthly_bills, color="#17becf", alpha=0.8, edgecolor="white")
axes[1].set_ylabel("Monthly Bill (EUR)"); axes[1].set_title("Monthly Bill")
axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(monthly_bills):
    axes[1].text(i, v+0.5 if v >= 0 else v-8, f"{v:.0f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=8)

plt.tight_layout()
plt.savefig("step3_online_monthly_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: step3_online_monthly_breakdown.png")

# Plot 3: Full-year SoC
fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
fig.suptitle("Online Retrain MPC (H=192) — Full Year 2025", fontsize=12, fontweight="bold")
ax = axes[0]
ax.fill_between(dispatch_full.index, 0, dispatch_full["pv_kw"], color="gold", alpha=0.5, label="PV")
ax.plot(dispatch_full.index, dispatch_full["load_kw"], color="navy", lw=0.4, alpha=0.7, label="Load")
ax.set_ylabel("Power (kW)"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(dispatch_full.index, dispatch_full["soc"]*100, color="#2ca02c", lw=0.5, label="SoC")
ax.fill_between(dispatch_full.index, 0, dispatch_full["soc"]*100, color="#2ca02c", alpha=0.15)
ax.set_ylim(0, 105); ax.set_ylabel("SoC (%)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig("step3_online_fullyear_soc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: step3_online_fullyear_soc.png")

# ── final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"  Baseline B (no battery) : EUR {BASELINE_B:.2f}")
print(f"  Baseline A (historical) : EUR {BASELINE_A:.2f}  <-- must beat this")
print(f"  MPC H=96  fixed models  : EUR 1303.28")
print(f"  MPC H=192 fixed models  : EUR {BILL_H192:.2f}")
print(f"  MPC H=192 online retrain: EUR {bill:.2f}  "
      f"({'BEAT Baseline A!' if bill < BASELINE_A else f'EUR {bill-BASELINE_A:.2f} above A'})")
print(f"  Oracle H=96             : EUR {ORACLE:.2f}")
print(f"  Online vs fixed H=192   : EUR {BILL_H192-bill:+.2f} improvement")
print(f"  Avg monthly h=1 NRMSE   : {np.mean(monthly_nrmse):.2f}%")
print(f"  Saving vs Baseline B    : {saving_vs_B:.1f}%")
print(f"  Unmet load              : {rep['total_unmet_load_kwh']:.4f} kWh")
