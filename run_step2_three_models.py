"""
Step 2 — Three additional models vs enhanced LightGBM baseline:
  Model 1: Stacked Ensemble  (LightGBM + XGBoost + CatBoost, averaged)
  Model 2: Two-Stage         (binary idle/active classifier → two regressors)
  Model 3: Huber + Quantile  (robust loss, spike-resistant)

Baseline to beat: NRMSE = 47.49%  (Step 2 Enhanced)
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "solship"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays as hol_pkg

from features import build_features
from forecaster import rmse as _rmse, mae as _mae, nrmse as _nrmse

# ── Best params from Optuna (Step 2 Enhanced) ────────────────────────────────
BEST_LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "verbose": -1, "num_threads": -1,
    "learning_rate": 0.0272, "num_leaves": 167, "min_data_in_leaf": 135,
    "feature_fraction": 0.862, "bagging_fraction": 0.656, "bagging_freq": 5,
    "lambda_l2": 0.078, "lambda_l1": 0.150,
}
ENHANCED_LAGS  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 48, 96, 192, 336, 672]
ENHANCED_ROLLS = [4, 8, 16, 96, 672]
IDLE_THRESHOLD = 0.5    # kW — below this = idle regime
BASELINE_NRMSE = 47.49  # enhanced LightGBM from Step 2

F1, F2, F3 = 0.2540, 0.2682, 0.2440
ITALIAN_HOLIDAYS = hol_pkg.Italy(years=[2024, 2025])


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════
def load_clean_data():
    def buy_price(ts):
        dow = ts.dt.dayofweek; h = ts.dt.hour
        hol = ts.dt.date.map(lambda d: d in ITALIAN_HOLIDAYS)
        f3 = (dow==6)|hol|(~(dow==6)&~hol&((h<7)|(h>=23)))
        f2 = (~f3)&(((dow==5)&(h>=7)&(h<23))|(~(dow==5)&~(dow==6)&~hol&((h==7)|((h>=19)&(h<23)))))
        p = pd.Series(F1, index=ts.index, dtype=float); p[f2]=F2; p[f3]=F3
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
    raw["buy_price"]  = buy_price(raw["timestamp"])
    idx = pd.date_range(raw["timestamp"].min(), raw["timestamp"].max(), freq="15min")
    raw = raw.set_index("timestamp").reindex(idx).ffill()
    raw["load_kw"] = raw["load_kw"].clip(lower=0)
    raw["pv_kw"]   = raw["pv_kw"].clip(lower=0)
    return raw

print("Loading data...")
raw    = load_clean_data()
df2024 = raw[raw.index.year == 2024]
df2025 = raw[raw.index.year == 2025]
df_train = df2024[df2024.index <= "2024-10-31 23:45"]
df_val   = df2024[df2024.index >= "2024-11-01"]

# log helpers
log1p  = np.log1p
expm1c = lambda x: np.expm1(x).clip(min=0)

# Build feature matrices (log-transformed, with lag buffer)
def make_Xy(df_buf, df_tgt, horizon):
    X, y = build_features(df_buf, lags=ENHANCED_LAGS,
                           roll_windows=ENHANCED_ROLLS, horizon=horizon)
    idx = df_tgt.index.intersection(X.index)
    return X.loc[idx], y.loc[idx]

df_tr_log  = df_train.assign(load_kw=log1p(df_train["load_kw"]))
df_va_buf  = pd.concat([df_train.iloc[-max(ENHANCED_LAGS):], df_val]).assign(
                load_kw=lambda d: log1p(d["load_kw"]))
df_te_buf  = pd.concat([df2024, df2025]).assign(load_kw=lambda d: log1p(d["load_kw"]))

X_tr, y_tr = make_Xy(df_tr_log,  df_train, 1)
X_va, y_va = make_Xy(df_va_buf,  df_val,   1)
X_te, y_te = make_Xy(df_te_buf,  df2025,   1)
feat_cols   = list(X_tr.columns)

print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
print(f"Features: {len(feat_cols)}\n")

# Helper: evaluate on test
def score(y_true_log, y_pred_log, label):
    yt = expm1c(y_true_log)
    yp = expm1c(y_pred_log)
    r  = _rmse(yt, yp); m = _mae(yt, yp); n = _nrmse(yt, yp)
    print(f"  {label:<35s}  RMSE={r:.4f}  MAE={m:.4f}  NRMSE={n:.2f}%")
    return r, m, n, yt, yp


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — STACKED ENSEMBLE (LightGBM + XGBoost + CatBoost)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL 1 — STACKED ENSEMBLE")
print("=" * 60)
t0 = time.time()

# 1a. LightGBM (best Optuna params, retrained on all 2024)
df_2024_log = df2024.assign(load_kw=log1p(df2024["load_kw"]))
X_24, y_24  = make_Xy(df_2024_log, df2024, 1)

dtrain_lgb = lgb.Dataset(X_24.values, y_24.values, feature_name=feat_cols)
lgb_model  = lgb.train(
    BEST_LGB_PARAMS, dtrain_lgb,
    num_boost_round=400,
    callbacks=[lgb.log_evaluation(0)],
)
pred_lgb = lgb_model.predict(X_te[feat_cols].values)
print(f"  LightGBM trained in {time.time()-t0:.1f}s")

# 1b. XGBoost
t1 = time.time()
xgb_params = {
    "objective": "reg:squarederror", "eval_metric": "rmse",
    "learning_rate": 0.05, "max_depth": 7, "n_estimators": 600,
    "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "reg_alpha": 0.1, "verbosity": 0, "n_jobs": -1,
}
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_24.values, y_24.values,
              eval_set=[(X_va[feat_cols].values, y_va.values)],
              verbose=False)
pred_xgb = xgb_model.predict(X_te[feat_cols].values)
print(f"  XGBoost  trained in {time.time()-t1:.1f}s")

# 1c. CatBoost
t1 = time.time()
cat_model = cb.CatBoostRegressor(
    iterations=600, learning_rate=0.05, depth=7,
    l2_leaf_reg=3, subsample=0.8, colsample_bylevel=0.8,
    loss_function="RMSE", verbose=0, thread_count=-1,
)
cat_model.fit(X_24.values, y_24.values,
              eval_set=(X_va[feat_cols].values, y_va.values),
              early_stopping_rounds=80)
pred_cat = cat_model.predict(X_te[feat_cols].values)
print(f"  CatBoost trained in {time.time()-t1:.1f}s")

# Simple average ensemble
pred_ensemble = (pred_lgb + pred_xgb + pred_cat) / 3.0

print(f"\nResults (all on 2025 test):")
r1_lgb, m1_lgb, n1_lgb, yt1, yp1_lgb = score(y_te.values, pred_lgb,      "  LightGBM alone")
r1_xgb, m1_xgb, n1_xgb, _,  yp1_xgb = score(y_te.values, pred_xgb,      "  XGBoost alone")
r1_cat, m1_cat, n1_cat, _,  yp1_cat  = score(y_te.values, pred_cat,      "  CatBoost alone")
r1,     m1,     n1,     _,  yp1      = score(y_te.values, pred_ensemble,  "  ENSEMBLE (average)")
print(f"  Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — TWO-STAGE (Classify idle/active → separate regressors)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 2 — TWO-STAGE CLASSIFIER + REGRESSOR")
print("=" * 60)
t0 = time.time()

# Build un-transformed features for stage-1 classifier
X_tr_raw, y_tr_raw = make_Xy(df_train, df_train, 1)
X_te_raw, y_te_raw = make_Xy(pd.concat([df2024, df2025]), df2025, 1)

y_tr_idle = (expm1c(y_tr.values) < IDLE_THRESHOLD).astype(int)  # 1 = idle
y_te_idle_true = (expm1c(y_te.values) < IDLE_THRESHOLD).astype(int)

# Stage 1: binary classifier (idle vs active)
clf_params = {**BEST_LGB_PARAMS, "objective": "binary", "metric": "binary_logloss"}
dtrain_clf = lgb.Dataset(X_tr_raw.values, y_tr_idle, feature_name=feat_cols)
dval_clf   = lgb.Dataset(X_va[feat_cols].values,
                          (expm1c(y_va.values) < IDLE_THRESHOLD).astype(int),
                          reference=dtrain_clf)
clf = lgb.train(
    clf_params, dtrain_clf, num_boost_round=500,
    valid_sets=[dval_clf],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)],
)
prob_idle_tr  = clf.predict(X_tr_raw.values)
prob_idle_te  = clf.predict(X_te_raw[feat_cols].values)
pred_idle_te  = (prob_idle_te > 0.5).astype(int)

idle_acc = (pred_idle_te == y_te_idle_true).mean() * 100
print(f"  Classifier accuracy on test: {idle_acc:.1f}%  "
      f"(idle={y_te_idle_true.mean()*100:.1f}% of test slots)")

# Stage 2a: regressor for IDLE regime (log-transformed)
idle_mask_tr  = prob_idle_tr > 0.5
active_mask_tr = ~idle_mask_tr

X_idle_tr  = X_tr.values[idle_mask_tr]
y_idle_tr  = y_tr.values[idle_mask_tr]
X_act_tr   = X_tr.values[active_mask_tr]
y_act_tr   = y_tr.values[active_mask_tr]

print(f"  Train split  idle: {idle_mask_tr.sum():,}  active: {active_mask_tr.sum():,}")

idle_reg = lgb.train(
    {**BEST_LGB_PARAMS},
    lgb.Dataset(X_idle_tr, y_idle_tr, feature_name=feat_cols),
    num_boost_round=300, callbacks=[lgb.log_evaluation(0)],
)
act_reg = lgb.train(
    {**BEST_LGB_PARAMS},
    lgb.Dataset(X_act_tr, y_act_tr, feature_name=feat_cols),
    num_boost_round=400, callbacks=[lgb.log_evaluation(0)],
)

# Stage 2: predict on test
pred_idle_reg = idle_reg.predict(X_te[feat_cols].values)
pred_act_reg  = act_reg.predict(X_te[feat_cols].values)

# Combine: soft blend using classifier probability
pred_two_stage = (prob_idle_te * pred_idle_reg
                  + (1 - prob_idle_te) * pred_act_reg)

print(f"\nResults:")
r2, m2, n2, _, yp2 = score(y_te.values, pred_two_stage, "  Two-Stage model")
print(f"  Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — HUBER LOSS + QUANTILE ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("MODEL 3 — HUBER LOSS + QUANTILE ENSEMBLE")
print("=" * 60)
t0 = time.time()

# 3a. Huber loss (robust to spikes)
huber_params = {**BEST_LGB_PARAMS, "objective": "huber", "alpha": 0.9,
                "metric": "huber"}
huber_model = lgb.train(
    huber_params, lgb.Dataset(X_24.values, y_24.values, feature_name=feat_cols),
    num_boost_round=400, callbacks=[lgb.log_evaluation(0)],
)
pred_huber = huber_model.predict(X_te[feat_cols].values)
print(f"  Huber model trained in {time.time()-t0:.1f}s")

# 3b. Quantile ensemble (q=0.4, 0.5, 0.6) — median is robust to outliers
preds_quantile = []
for q in [0.4, 0.5, 0.6]:
    t1 = time.time()
    q_params = {**BEST_LGB_PARAMS, "objective": "quantile", "alpha": q,
                "metric": "quantile"}
    q_model = lgb.train(
        q_params, lgb.Dataset(X_24.values, y_24.values, feature_name=feat_cols),
        num_boost_round=400, callbacks=[lgb.log_evaluation(0)],
    )
    preds_quantile.append(q_model.predict(X_te[feat_cols].values))
    print(f"  Quantile q={q} trained in {time.time()-t1:.1f}s")

pred_quantile_avg = np.mean(preds_quantile, axis=0)

# Blend: 50% Huber + 50% Quantile ensemble
pred_robust = 0.5 * pred_huber + 0.5 * pred_quantile_avg

print(f"\nResults:")
r3h, m3h, n3h, _, yp3h = score(y_te.values, pred_huber,        "  Huber alone")
r3q, m3q, n3q, _, yp3q = score(y_te.values, pred_quantile_avg, "  Quantile ensemble (avg)")
r3,  m3,  n3,  _, yp3  = score(y_te.values, pred_robust,       "  Huber + Quantile blend")
print(f"  Total time: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════════
# FINAL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL COMPARISON — ALL MODELS ON 2025 TEST")
print("=" * 60)
results = [
    ("Persistence (floor)",          53.38, None),
    ("Enhanced LightGBM (baseline)", BASELINE_NRMSE, None),
    ("Model 1 — Ensemble",           n1,  yp1),
    ("Model 2 — Two-Stage",          n2,  yp2),
    ("Model 3 — Huber+Quantile",     n3,  yp3),
]
for name, nrmse_val, _ in results:
    delta = nrmse_val - BASELINE_NRMSE
    bar   = "#" * int(abs(delta) * 5)
    sign  = "+" if delta > 0 else "-"
    print(f"  {name:<35s}  NRMSE={nrmse_val:.2f}%  ({sign}{abs(delta):.2f}% vs baseline)")

best_nrmse = min(n1, n2, n3)
best_name  = ["Ensemble", "Two-Stage", "Huber+Quantile"][[n1,n2,n3].index(best_nrmse)]
print(f"\n  Best model: {best_name}  NRMSE={best_nrmse:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS — one per model
# ═══════════════════════════════════════════════════════════════════════════════
sample_idx = X_te.index[:96*14]   # first 2 weeks of 2025

def plot_forecast(idx, y_true, y_pred, title, fname, color):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(idx, y_true[:len(idx)], color="black", lw=1.3, label="Actual load")
    ax.plot(idx, y_pred[:len(idx)], color=color,   lw=1.0, label="Predicted", alpha=0.85)
    ax.set_title(title, fontsize=13)
    ax.set_ylabel("Load (kW)"); ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)

print("\nGenerating plots...")

plot_forecast(
    sample_idx, yt1, yp1,
    f"Model 1 — Ensemble (LGB+XGB+CatBoost) | NRMSE={n1:.2f}% vs baseline {BASELINE_NRMSE}%",
    "plot_M1_ensemble_forecast.png", "steelblue"
)
plot_forecast(
    sample_idx, yt1, yp2,
    f"Model 2 — Two-Stage Classifier+Regressor | NRMSE={n2:.2f}% vs baseline {BASELINE_NRMSE}%",
    "plot_M2_twostage_forecast.png", "mediumseagreen"
)
plot_forecast(
    sample_idx, yt1, yp3,
    f"Model 3 — Huber + Quantile Blend | NRMSE={n3:.2f}% vs baseline {BASELINE_NRMSE}%",
    "plot_M3_huber_quantile_forecast.png", "darkorange"
)

# Summary comparison bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names   = ["Persistence\n(floor)", "Enhanced\nLightGBM\n(baseline)",
           "M1\nEnsemble", "M2\nTwo-Stage", "M3\nHuber+\nQuantile"]
nrmses  = [53.38, BASELINE_NRMSE, n1, n2, n3]
colors  = ["gray", "gray", "steelblue", "mediumseagreen", "darkorange"]
bars = ax.bar(names, nrmses, color=colors, alpha=0.85, width=0.5)
ax.axhline(BASELINE_NRMSE, color="red", linestyle="--", lw=1.5, label=f"Baseline {BASELINE_NRMSE}%")
for bar, val in zip(bars, nrmses):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3,
            f"{val:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("NRMSE (%)", fontsize=11)
ax.set_title("Model Comparison — 2025 Test NRMSE", fontsize=13)
ax.set_ylim(0, 60)
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig("plot_M0_comparison_summary.png", dpi=150, bbox_inches="tight")
print("Saved: plot_M0_comparison_summary.png")
plt.close(fig)
