"""
forecasting_v3.py — exact same logic as forecasting_v2.py, different dataset.

Dataset : 2nd_DataSet.xlsx  (2024-11-25 to 2025-03-31)
Train   : 2024-11-25 to 2025-02-28
ES val  : last 4 weeks of train
Test    : 2025-03-01 to 2025-03-31  (last month)

Uses features_v2 (is_spike_likely, load_zscore_lag1) — same as v2.
Flags:
  ENABLE_TWEEDIE_HEAD      = True
  ENABLE_Q95_HEAD          = False
  ENABLE_SEASON_MODEL      = False
  ENABLE_AFFINE_CALIBRATION = True
  CAL_A_MIN = 0.5 / CAL_A_MAX = 1.8  (same as v2)

Outputs -> submission_forecast_v3/
"""

import sys, os, warnings, time
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import holidays as hol_pkg
import openpyxl

# ── same feature patching as v2 ────────────────────────────────────────────────
import features_v2
import forecasting as _fc
_fc.build_features = features_v2.build_features

from forecasting import (
    ENHANCED_LAGS, ENHANCED_ROLLS, SPARSE_HORIZONS, H_MAX,
    LGB_PARAMS, SEEDS, BLEND_GRID_STEP,
    _train_seeds, _predict_seeds, _weight_grid,
    log1p, expm1c, _nrmse,
    compute_slot_stats, compute_slot_month_stats, compute_slot_dow_stats,
    build_mpc_forecast_matrix,
)
from optimizer import run_mpc, validate_dispatch
from baselines import detect_corruption, compute_baseline_a, compute_baseline_b
from tariff import compute_bill
from config import HORIZON_DEFAULT, DT_HOURS

# ── constants (identical to v2) ────────────────────────────────────────────────
OUT_DIR          = "submission_forecast_v3"
ITALIAN_HOLIDAYS = hol_pkg.Italy(years=[2024, 2025])

ENABLE_TWEEDIE_HEAD       = True
ENABLE_RAW_RMSE_HEAD      = False
ENABLE_Q95_HEAD           = False
ENABLE_SEASON_MODEL       = False
ENABLE_AFFINE_CALIBRATION = True
CAL_MIN_NRMSE_GAIN        = 0.0
CAL_A_MIN                 = 0.5
CAL_A_MAX                 = 1.8

os.makedirs(OUT_DIR, exist_ok=True)

# ── helpers (identical to v2) ──────────────────────────────────────────────────
F1, F2, F3 = 0.2540, 0.2682, 0.2440

def compute_buy_price(ts):
    dow = ts.dt.dayofweek; h = ts.dt.hour
    hol = ts.dt.date.map(lambda d: d in ITALIAN_HOLIDAYS)
    f3  = (dow == 6) | hol | (~(dow == 6) & ~hol & ((h < 7) | (h >= 23)))
    f2  = (~f3) & (((dow == 5) & (h >= 7) & (h < 23)) |
                   (~(dow == 5) & ~(dow == 6) & ~hol & ((h == 7) | ((h >= 19) & (h < 23)))))
    p   = pd.Series(F1, index=ts.index, dtype=float); p[f2] = F2; p[f3] = F3
    return p

def log_df(df):
    d = df.copy(); d["load_kw"] = log1p(d["load_kw"]); return d

def make_Xy(df_buf, df_target, horizon, feat_cols=None,
            slot_stats=None, slot_month_stats=None, slot_dow_stats=None):
    X, y = features_v2.build_features(
        df_buf, lags=ENHANCED_LAGS, roll_windows=ENHANCED_ROLLS, horizon=horizon,
        slot_stats=slot_stats, slot_month_stats=slot_month_stats, slot_dow_stats=slot_dow_stats)
    idx  = df_target.index.intersection(X.index)
    X, y = X.loc[idx], y.loc[idx]
    if feat_cols is not None:
        X = X[feat_cols]
    return X, y


# ── load 2nd_DataSet.xlsx ──────────────────────────────────────────────────────
def load_data(path="ENERGY_Hackathon_DataSet(Sheet1).csv"):
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    candidate_paths = [
        path,
        "ENERGY_Hackathon_DataSet(Sheet1).csv",
        os.path.join(os.path.dirname(__file__), "ENERGY_Hackathon_DataSet(Sheet1).csv"),
        os.path.join(os.path.dirname(__file__), "data", "ENERGY_Hackathon_DataSet(Sheet1).csv"),
    ]
    resolved = next((p for p in candidate_paths if os.path.exists(p)), None)
    if resolved is None:
        raise FileNotFoundError(
            "Could not find ENERGY_Hackathon_DataSet(Sheet1).csv in expected locations"
        )

    df = pd.read_csv(resolved, sep=";", decimal=",")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="first")
            .reset_index(drop=True))
    df = df.rename(columns={"load_p": "load_kw", "pv_p": "pv_kw",
                             "Selling_price_eur_kwh": "sell_price",
                             "battery_p": "p_battery_kw", "grid_p": "grid_kw"})
    if "sell_price" in df.columns:
        df["sell_price"] = df["sell_price"].ffill()
    
    df["buy_price"]  = compute_buy_price(df["timestamp"])
    idx = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="15min")
    df  = df.set_index("timestamp").reindex(idx).ffill()
    df["load_kw"] = df["load_kw"].clip(lower=0)
    if "pv_kw" in df.columns:
        df["pv_kw"]   = df["pv_kw"].clip(lower=0)
    print(f"  Rows: {len(df):,}  ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ── run_period — identical logic to v2 ────────────────────────────────────────
def run_period(month_num, label, month_start, month_end, raw):
    print(f"\n{'='*60}")
    print(f"PERIOD [v3]: {label}  ({month_start.date()} - {month_end.date()})")
    print(f"{'='*60}")
    t0 = time.time()

    train_end = month_start - pd.Timedelta("15min")
    df_train  = raw[raw.index <= train_end]
    df_test   = raw[(raw.index >= month_start) & (raw.index <= month_end)]

    # ES val: last 8 weeks (56 days) of training data — same as v2
    es_split  = train_end - pd.Timedelta(days=56)
    df_tr_es  = df_train[df_train.index <= es_split]
    df_val_es = df_train[df_train.index >  es_split]

    print(f"  Train : {df_train.index[0].date()} - {df_train.index[-1].date()}  ({len(df_train):,} rows)")
    print(f"  ES val: {df_val_es.index[0].date()} - {df_val_es.index[-1].date()}  ({len(df_val_es):,} rows)")
    print(f"  Test  : {df_test.index[0].date()} - {df_test.index[-1].date()}  ({len(df_test):,} rows)")

    train_ss  = compute_slot_stats(df_train)
    train_sms = compute_slot_month_stats(df_train)
    train_sds = compute_slot_dow_stats(df_train)
    es_ss     = compute_slot_stats(df_tr_es)
    es_sms    = compute_slot_month_stats(df_tr_es)
    es_sds    = compute_slot_dow_stats(df_tr_es)

    def _Xy(df_buf, df_tgt, h, fc=None, es=False):
        ss  = es_ss  if es else train_ss
        sms = es_sms if es else train_sms
        sds = es_sds if es else train_sds
        return make_Xy(df_buf, df_tgt, h, fc, slot_stats=ss,
                       slot_month_stats=sms, slot_dow_stats=sds)

    df_tr_log    = log_df(df_train)
    df_tr_es_log = log_df(df_tr_es)
    buf_tail     = max(ENHANCED_LAGS)
    df_va_buf    = log_df(pd.concat([df_tr_es.iloc[-buf_tail:], df_val_es]))

    X_tr1, _ = _Xy(df_tr_log, df_train, 1)
    feat_cols = list(X_tr1.columns)

    active_heads = ["hub", "qnt"]
    if ENABLE_TWEEDIE_HEAD:  active_heads.append("twe")
    if ENABLE_Q95_HEAD:      active_heads.append("q95")
    if ENABLE_SEASON_MODEL:  active_heads.append("sea")
    if ENABLE_RAW_RMSE_HEAD: active_heads.append("raw")

    print(f"  Training {len(SPARSE_HORIZONS)} horizons x {len(active_heads)} heads x {len(SEEDS)} seeds ...")
    t_tr = time.time()
    models_hub = {}; models_qnt = {}; models_twe = {}; models_raw = {}

    for h in SPARSE_HORIZONS:
        X_es, y_es = _Xy(df_tr_es_log, df_tr_es, h, feat_cols, es=True)
        X_va, y_va = _Xy(df_va_buf,    df_val_es, h, feat_cols, es=True)
        y_es_raw   = np.expm1(y_es.values).clip(min=0)
        y_va_raw   = np.expm1(y_va.values).clip(min=0)

        d_es_log = lgb.Dataset(X_es.values, y_es.values, feature_name=feat_cols, free_raw_data=False)
        d_va_log = lgb.Dataset(X_va.values, y_va.values, feature_name=feat_cols,
                               reference=d_es_log, free_raw_data=False)
        d_es_raw = lgb.Dataset(X_es.values, y_es_raw, feature_name=feat_cols, free_raw_data=False)
        d_va_raw = lgb.Dataset(X_va.values, y_va_raw, feature_name=feat_cols,
                               reference=d_es_raw, free_raw_data=False)
        cbs = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)]

        m_es_hub = lgb.train(
            {**LGB_PARAMS, "objective": "huber", "metric": "huber", "alpha": 0.9},
            d_es_log, num_boost_round=2000, valid_sets=[d_va_log], callbacks=cbs)
        best_log = max(int(m_es_hub.best_iteration * 1.2), 50)

        best_twe = best_raw = None
        if ENABLE_TWEEDIE_HEAD:
            m_es_twe = lgb.train(
                {**LGB_PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "rmse"},
                d_es_raw, num_boost_round=2000, valid_sets=[d_va_raw], callbacks=cbs)
            best_twe = max(int(m_es_twe.best_iteration * 1.2), 50)
        if ENABLE_RAW_RMSE_HEAD:
            m_es_raw = lgb.train(
                {**LGB_PARAMS, "objective": "regression", "metric": "rmse"},
                d_es_raw, num_boost_round=2000, valid_sets=[d_va_raw], callbacks=cbs)
            best_raw = max(int(m_es_raw.best_iteration * 1.2), 50)

        X_tr, y_tr = _Xy(df_tr_log, df_train, h, feat_cols)
        y_tr_raw   = np.expm1(y_tr.values).clip(min=0)

        models_hub[h] = _train_seeds(
            {**LGB_PARAMS, "objective": "huber",    "metric": "huber",    "alpha": 0.9},
            X_tr.values, y_tr.values, best_log, feat_cols)
        models_qnt[h] = _train_seeds(
            {**LGB_PARAMS, "objective": "quantile", "metric": "quantile", "alpha": 0.90},
            X_tr.values, y_tr.values, best_log, feat_cols)
        if ENABLE_TWEEDIE_HEAD:
            models_twe[h] = _train_seeds(
                {**LGB_PARAMS, "objective": "tweedie", "tweedie_variance_power": 1.5, "metric": "rmse"},
                X_tr.values, y_tr_raw, best_twe, feat_cols)
        if ENABLE_RAW_RMSE_HEAD:
            models_raw[h] = _train_seeds(
                {**LGB_PARAMS, "objective": "regression", "metric": "rmse"},
                X_tr.values, y_tr_raw, best_raw, feat_cols)

    print(f"  Training done in {time.time()-t_tr:.1f}s")

    # ── blend search ───────────────────────────────────────────────────────────
    blend_weights = {}
    val_pred_h1 = y_val_h1 = None

    for h in SPARSE_HORIZONS:
        X_v, y_v = _Xy(df_va_buf, df_val_es, h, feat_cols, es=True)
        vp = {
            "hub": expm1c(_predict_seeds(models_hub[h], X_v.values)),
            "qnt": expm1c(_predict_seeds(models_qnt[h], X_v.values)),
        }
        if ENABLE_TWEEDIE_HEAD:  vp["twe"] = np.clip(_predict_seeds(models_twe[h], X_v.values), 0, None)
        if ENABLE_RAW_RMSE_HEAD: vp["raw"] = np.clip(_predict_seeds(models_raw[h], X_v.values), 0, None)
        y_act = np.expm1(y_v.values).clip(min=0)

        best_n, best_w = float("inf"), {"hub": 1.0, "qnt": 0.0}
        for w in _weight_grid(active_heads):
            pred = sum(w[k] * vp[k] for k in active_heads).clip(min=0)
            n = _nrmse(y_act, pred)
            if n < best_n:
                best_n, best_w = n, w
        blend_weights[h] = best_w
        if h == 1:
            wtxt = "  ".join(f"{k}={best_w.get(k,0):.2f}" for k in active_heads)
            print(f"  h=1 blend  {wtxt}  (val NRMSE={best_n:.2f}%)")
            val_pred_h1 = sum(best_w[k] * vp[k] for k in active_heads).clip(min=0)
            y_val_h1    = y_act

    # ── affine calibration (identical to v2: bias-only fallback, widened bounds) ─
    n_pre = _nrmse(y_val_h1, val_pred_h1)
    a_cal, b_cal = 1.0, 0.0
    if ENABLE_AFFINE_CALIBRATION:
        A = np.column_stack([val_pred_h1, np.ones_like(val_pred_h1)])
        sol, *_ = np.linalg.lstsq(A, y_val_h1, rcond=None)
        a_r, b_r = float(sol[0]), float(sol[1])
        n_aff  = _nrmse(y_val_h1, np.clip(a_r * val_pred_h1 + b_r, 0, None))
        b_bias = float(np.mean(y_val_h1 - val_pred_h1))
        n_bias = _nrmse(y_val_h1, np.clip(val_pred_h1 + b_bias, 0, None))
        if n_bias <= n_aff:
            a_r, b_r, n_aff = 1.0, b_bias, n_bias
        if (CAL_A_MIN <= a_r <= CAL_A_MAX) and (n_pre - n_aff >= CAL_MIN_NRMSE_GAIN):
            a_cal, b_cal = a_r, b_r
            print(f"  Calibration  a={a_cal:.3f}  b={b_cal:+.3f}  (val {n_pre:.2f}% -> {n_aff:.2f}%)")
        else:
            print(f"  Calibration: identity (a={a_r:.3f} b={b_r:+.3f}  val {n_pre:.2f}% -> {n_aff:.2f}%)")
    else:
        print("  Calibration: disabled")

    best_alpha_hub = blend_weights[1].get("hub", 0.0)
    best_alpha_qnt = blend_weights[1].get("qnt", 0.0)

    # ── test predictions ───────────────────────────────────────────────────────
    buf_data  = raw[raw.index < month_start]
    pad_idx   = pd.date_range(df_test.index[-1] + pd.Timedelta("15min"), periods=H_MAX, freq="15min")
    pad_df    = pd.concat([df_test.iloc[[-1]]] * H_MAX); pad_df.index = pad_idx
    df_te_buf = log_df(pd.concat([buf_data, df_test, pad_df]))

    def _Xy_te(h, fc=None):
        return make_Xy(df_te_buf, df_test, h, fc,
                       slot_stats=train_ss, slot_month_stats=train_sms, slot_dow_stats=train_sds)

    pred_hub = {}; pred_qnt = {}; pred_twe = {}; pred_raw = {}
    for h in SPARSE_HORIZONS:
        X, _ = _Xy_te(h, feat_cols)
        pred_hub[h] = pd.Series(expm1c(_predict_seeds(models_hub[h], X.values)), index=X.index)
        pred_qnt[h] = pd.Series(expm1c(_predict_seeds(models_qnt[h], X.values)), index=X.index)
        if ENABLE_TWEEDIE_HEAD:
            pred_twe[h] = pd.Series(np.clip(_predict_seeds(models_twe[h], X.values), 0, None), index=X.index)
        if ENABLE_RAW_RMSE_HEAD:
            pred_raw[h] = pd.Series(np.clip(_predict_seeds(models_raw[h], X.values), 0, None), index=X.index)

    common = pred_hub[SPARSE_HORIZONS[0]].index
    for h in SPARSE_HORIZONS[1:]:
        common = common.intersection(pred_hub[h].index)
    common = common[(common >= month_start) & (common <= month_end)]

    w1 = blend_weights[1]
    h1_blend = (w1.get("hub", 0.0) * pred_hub[1].loc[common].values +
                w1.get("qnt", 0.0) * pred_qnt[1].loc[common].values)
    if ENABLE_TWEEDIE_HEAD:  h1_blend += w1.get("twe", 0.0) * pred_twe[1].loc[common].values
    if ENABLE_RAW_RMSE_HEAD: h1_blend += w1.get("raw", 0.0) * pred_raw[1].loc[common].values
    h1_blend = np.clip(a_cal * h1_blend.clip(min=0) + b_cal, 0, None)

    forecast_h1 = pd.Series(h1_blend, index=common, name="forecast_kw")
    actual_h1   = df_test["load_kw"].shift(-1).reindex(common).dropna()
    forecast_h1 = forecast_h1.reindex(actual_h1.index)

    result = pd.DataFrame({"actual_kw": actual_h1.values, "forecast_kw": forecast_h1.values},
                          index=actual_h1.index)
    result["error_kw"]     = result["forecast_kw"] - result["actual_kw"]
    result["abs_error_kw"] = result["error_kw"].abs()
    result["sq_error_kw2"] = result["error_kw"] ** 2
    mask_nz = result["actual_kw"] > 0.01
    result["pct_error"] = np.nan
    result.loc[mask_nz, "pct_error"] = (result.loc[mask_nz, "abs_error_kw"] /
                                         result.loc[mask_nz, "actual_kw"] * 100)

    print(f"  Forecast timesteps: {len(result):,}")
    print(f"  Total elapsed: {time.time()-t0:.1f}s")

    # feature importance
    fi_arr = np.mean([m.feature_importance("gain") for m in models_hub[1]], axis=0)
    fi_lgb = pd.Series(fi_arr, index=feat_cols).sort_values(ascending=False)
    slug = label.lower().replace(" ", "_")
    fi_txt = os.path.join(OUT_DIR, f"feature_importance_v3_{slug}.txt")
    with open(fi_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Feature importance [v3] - {label}\n{'='*60}\n")
        fh.write("  Blend: " + "  ".join(f"{k}={best_w.get(k,0):.2f}" for k in active_heads) + "\n\n")
        fh.write(f"  {'Rank':<5} {'Feature':<35} {'LGB gain':>12}\n{'-'*55}\n")
        for rank, feat in enumerate(fi_lgb.index[:30], 1):
            fh.write(f"  {rank:<5} {feat:<35} {fi_lgb[feat]:>12,.0f}\n")
    print(f"  Saved: {fi_txt}")
    print(f"\n  Top 10 features ({label}):")
    for feat, val in fi_lgb.head(10).items():
        print(f"    {feat:<35s}  {val:,.0f}")

    return {
        "result": result, "pred_hub": pred_hub, "pred_qnt": pred_qnt,
        "pred_twe": pred_twe, "blend_weights": blend_weights,
        "a_cal": a_cal, "b_cal": b_cal,
        "best_alpha_hub": best_alpha_hub, "best_alpha_qnt": best_alpha_qnt,
        "feat_cols": feat_cols, "common": common,
    }


# ── metrics, plots, main ───────────────────────────────────────────────────────
def compute_metrics(result, label):
    actual   = result["actual_kw"].values
    forecast = result["forecast_kw"].values
    rmse  = float(np.sqrt(np.mean((actual - forecast) ** 2)))
    mae   = float(np.mean(np.abs(actual - forecast)))
    nrmse = rmse / float(np.mean(actual)) * 100 if np.mean(actual) > 1e-9 else float("nan")
    bias  = float(np.mean(forecast - actual))
    mape  = float(result["pct_error"].mean())
    return {"label": label, "nrmse": nrmse, "rmse": rmse, "mae": mae,
            "bias": bias, "mape": mape, "mean_actual": float(np.mean(actual))}

def save_metrics(m, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*55}\n  FORECAST METRICS [v3] - {m['label']}\n{'='*55}\n")
        f.write(f"  Mean actual load : {m['mean_actual']:.4f} kW\n")
        f.write(f"  MAE              : {m['mae']:.4f} kW\n")
        f.write(f"  RMSE             : {m['rmse']:.4f} kW\n")
        f.write(f"  MAPE             : {m['mape']:.2f} %\n")
        f.write(f"  NRMSE            : {m['nrmse']:.2f} %\n")
        f.write(f"  Bias             : {m['bias']:+.4f} kW\n")
        f.write(f"{'='*55}\n")
    print(f"  Saved: {path}")

def make_plot(result, m, slug):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                              gridspec_kw={"height_ratios": [3, 1.5]})
    fig.suptitle(f"Forecast vs Actual - {m['label']}  [v3]",
                 fontsize=13, fontweight="bold")
    ts = result.index
    axes[0].plot(ts, result["actual_kw"],   lw=0.9, color="#1f77b4", label="Actual",   alpha=0.9)
    axes[0].plot(ts, result["forecast_kw"], lw=0.7, color="#ff7f0e", label="Forecast", alpha=0.85)
    axes[0].set_ylabel("Load (kW)")
    axes[0].set_title(f"NRMSE={m['nrmse']:.2f}%  RMSE={m['rmse']:.4f} kW  "
                       f"MAE={m['mae']:.4f} kW  Bias={m['bias']:+.3f} kW",
                       fontsize=9, color="#333")
    axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.25)
    err = result["error_kw"].values
    axes[1].fill_between(ts, err, 0, where=(err >= 0), color="#ff7f0e", alpha=0.5, label="Over")
    axes[1].fill_between(ts, err, 0, where=(err <  0), color="#1f77b4", alpha=0.5, label="Under")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("Error (kW)"); axes[1].legend(fontsize=9); axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"plot_v3_{slug}_forecast.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")
    return path

def make_daily_nrmse_plot(result, m, slug):
    overall_mean = m['mean_actual']
    overall_nrmse = m['nrmse']
    
    daily_rmse = result.groupby(result.index.day).apply(
        lambda df: np.sqrt(np.mean((df["actual_kw"] - df["forecast_kw"]) ** 2))
    )
    daily_nrmse = (daily_rmse / overall_mean) * 100
    
    fig, ax = plt.subplots(figsize=(16, 5))
    
    days = daily_nrmse.index.values
    nrmse_vals = daily_nrmse.values
    
    colors = ['tab:red' if val > overall_nrmse else 'tab:blue' for val in nrmse_vals]
    bars = ax.bar(days, nrmse_vals, color=colors, zorder=3)
    
    ax.axhline(overall_nrmse, color='black', linestyle='--', zorder=4, 
               label=f'Overall NRMSE = {overall_nrmse:.2f}%')
    
    for bar, val in zip(bars, nrmse_vals):
        if val >= 55:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{int(round(val))}%', 
                    ha='center', va='bottom', color='tab:red', fontweight='bold', fontsize=9)

    ax.set_title(f"Per-Day NRMSE — {m['label']}\nRed bars = above overall NRMSE ({overall_nrmse:.2f}%)", 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel("Daily NRMSE [%]\n(denominator = whole-month mean load)", fontsize=10)
    ax.set_xlabel("Day of month", fontsize=10)
    ax.set_xticks(days)
    ax.set_ylim(0, max(60, max(nrmse_vals) + 5))
    
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, zorder=0)
    
    fig.tight_layout()
    path = os.path.join(OUT_DIR, f"plot_v3_{slug}_daily_nrmse.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path

def save_comprehensive_excel(result, m, slug, label,
                              dispatch, oracle_dispatch,
                              base_a, base_b,
                              controller_bill, oracle_bill,
                              savings_vs_a_eur, savings_vs_a_pct,
                              oracle_gap_pct):
    """
    Single Excel workbook with 4 sheets:
      1. Forecast       — actual vs forecast + per-step errors
      2. MPC Dispatch   — optimizer behaviour every 15-min step
      3. All Strategies — controller / oracle / baseline A & B side-by-side
      4. Summary        — forecast metrics + bills + savings
    """
    path = os.path.join(OUT_DIR, f"submission_v3_{slug}.xlsx")

    # ── Sheet 1: Forecast vs Actual ──────────────────────────────────────────
    df_fc = pd.DataFrame({
        "Timestamp":        [f"{ts.month}/{ts.day}/{ts.year} {ts.hour}:{ts.minute:02d}"
                             for ts in result.index],
        "Actual Load (kW)":    result["actual_kw"].round(4).values,
        "Forecast Load (kW)":  result["forecast_kw"].round(4).values,
        "Error (kW)":          result["error_kw"].round(4).values,
        "Abs Error (kW)":      result["abs_error_kw"].round(4).values,
        "% Error":             result["pct_error"].round(2).values,
    })

    # ── Sheet 2: MPC Dispatch (controller step-by-step) ─────────────────────
    df_disp = pd.DataFrame({
        "Timestamp":          [str(ts) for ts in dispatch.index],
        "Load (kW)":           dispatch["load_kw"].round(4).values,
        "PV (kW)":             dispatch["pv_kw"].round(4).values,
        "P_Battery (kW)":      dispatch["p_battery"].round(4).values,
        "P_Charge (kW)":       dispatch["p_ch"].round(4).values,
        "P_Discharge (kW)":    dispatch["p_dis"].round(4).values,
        "P_Grid (kW)":         dispatch["p_grid"].round(4).values,
        "P_Import (kW)":       dispatch["p_imp"].round(4).values,
        "P_Export (kW)":       dispatch["p_exp"].round(4).values,
        "SoC":                 dispatch["soc"].round(4).values,
        "Buy Price (€/kWh)":   dispatch["buy_price"].round(4).values,
        "Sell Price (€/kWh)":  dispatch["sell_price"].round(4).values,
    })

    # ── Sheet 3: All Strategies side-by-side ─────────────────────────────────
    idx = dispatch.index
    ba_d = base_a["dispatch"].reindex(idx)
    bb_d = base_b["dispatch"].reindex(idx)
    or_d = oracle_dispatch.reindex(idx)

    df_strat = pd.DataFrame({
        "Timestamp":              [str(ts) for ts in idx],
        "Load (kW)":               dispatch["load_kw"].round(4).values,
        "PV (kW)":                 dispatch["pv_kw"].round(4).values,
        # Controller (MPC + forecast)
        "MPC P_Battery (kW)":      dispatch["p_battery"].round(4).values,
        "MPC P_Grid (kW)":         dispatch["p_grid"].round(4).values,
        "MPC SoC":                 dispatch["soc"].round(4).values,
        # Oracle (perfect foresight)
        "Oracle P_Battery (kW)":   or_d["p_battery"].round(4).values,
        "Oracle P_Grid (kW)":      or_d["p_grid"].round(4).values,
        "Oracle SoC":              or_d["soc"].round(4).values,
        # Baseline A (historical replay)
        "BaseA P_Battery (kW)":    ba_d["p_battery"].round(4).values,
        "BaseA P_Grid (kW)":       ba_d["p_grid"].round(4).values,
        # Baseline B (no battery)
        "BaseB P_Grid (kW)":       bb_d["p_grid"].round(4).values,
    })

    # ── Sheet 4: Summary ─────────────────────────────────────────────────────
    savings_vs_b_eur = base_b["bill"] - controller_bill
    oracle_gap_str   = f"{oracle_gap_pct:.2f}" if not np.isnan(oracle_gap_pct) else "n/a"
    df_summary = pd.DataFrame({
        "Item": [
            "Period",
            "",
            "FORECAST METRICS",
            "Mean Actual Load (kW)", "RMSE (kW)", "MAE (kW)", "MAPE (%)", "NRMSE (%)", "Bias (kW)",
            "",
            "BILLS (€)",
            "Controller (MPC + forecast)",
            "Baseline A (historical)",
            "Baseline B (no battery)",
            "Oracle (perfect foresight)",
            "",
            "SAVINGS",
            "Savings vs Baseline A (€)",
            "Savings vs Baseline A (%)",
            "Savings vs Baseline B (€)",
            "Oracle Gap (%)",
        ],
        "Value": [
            label,
            "",
            "",
            round(m["mean_actual"], 4), round(m["rmse"], 4), round(m["mae"], 4),
            round(m["mape"], 2), round(m["nrmse"], 2), round(m["bias"], 4),
            "",
            "",
            round(controller_bill, 2),
            round(base_a["bill"], 2),
            round(base_b["bill"], 2),
            round(oracle_bill, 2),
            "",
            "",
            round(savings_vs_a_eur, 2),
            round(savings_vs_a_pct, 2),
            round(savings_vs_b_eur, 2),
            oracle_gap_str,
        ],
    })

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df_fc.to_excel(    writer, sheet_name="Forecast",           index=False)
        df_disp.to_excel(  writer, sheet_name="MPC Dispatch",       index=False)
        df_strat.to_excel( writer, sheet_name="All Strategies",     index=False)
        df_summary.to_excel(writer, sheet_name="Summary",           index=False)

    print(f"  Saved comprehensive Excel: {path}")
    return path


def make_dispatch_plot(dispatch_df, out_path, title, actual_df=None):
    """
    Week dispatch plot: Load/PV, P_battery, P_grid, SoC.
    actual_df : optional Baseline-A dispatch for the same window — overlaid as
                dashed lines so controller vs historical is directly comparable.
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2.0, 1.4, 1.4, 1.2]})

    # ── Panel 1: Load + PV ───────────────────────────────────────────────────
    axes[0].plot(dispatch_df.index, dispatch_df["load_kw"],
                 lw=1.3, color="#1f77b4", label="Actual Load")
    axes[0].fill_between(dispatch_df.index, 0, dispatch_df["pv_kw"],
                         color="#f4a300", alpha=0.45, label="PV")
    axes[0].set_ylabel("kW")
    axes[0].set_title(title, fontsize=12, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(alpha=0.3)

    # ── Panel 2: P_battery — MPC (filled) + Baseline A (dashed) ─────────────
    pb = dispatch_df["p_battery"].values
    axes[1].fill_between(dispatch_df.index, 0, np.where(pb > 0, pb, 0),
                         color="#02C39A", alpha=0.75, label="MPC Discharge")
    axes[1].fill_between(dispatch_df.index, 0, np.where(pb < 0, pb, 0),
                         color="#80CBC4", alpha=0.85, label="MPC Charge")
    if actual_df is not None:
        pb_act = actual_df["p_battery"].reindex(dispatch_df.index).values
        axes[1].plot(dispatch_df.index, pb_act,
                     lw=1.1, color="#333333", ls="--", alpha=0.7, label="Baseline A (actual)")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_ylabel("P_battery (kW)")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    # ── Panel 3: P_grid — MPC (filled) + Baseline A (dashed) ────────────────
    pg = dispatch_df["p_grid"].values
    axes[2].fill_between(dispatch_df.index, 0, np.where(pg > 0, pg, 0),
                         color="#990011", alpha=0.65, label="MPC Import")
    axes[2].fill_between(dispatch_df.index, 0, np.where(pg < 0, pg, 0),
                         color="#EE6C4D", alpha=0.65, label="MPC Export")
    if actual_df is not None:
        pg_act = actual_df["p_grid"].reindex(dispatch_df.index).values
        axes[2].plot(dispatch_df.index, pg_act,
                     lw=1.1, color="#333333", ls="--", alpha=0.7, label="Baseline A (actual)")
    axes[2].axhline(0, color="black", lw=0.6)
    axes[2].set_ylabel("P_grid (kW)")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(alpha=0.3)

    # ── Panel 4: SoC ─────────────────────────────────────────────────────────
    axes[3].plot(dispatch_df.index, dispatch_df["soc"],
                 lw=1.7, color="#0891B2", label="MPC SoC")
    axes[3].fill_between(dispatch_df.index, 0, dispatch_df["soc"],
                         color="#0891B2", alpha=0.15)
    axes[3].axhline(0, color="gray", lw=0.5, ls="--")
    axes[3].axhline(1, color="gray", lw=0.5, ls="--")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].set_ylabel("SoC")
    axes[3].set_xlabel("Timestamp")
    axes[3].legend(loc="upper right", fontsize=9)
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    raw = load_data()

    # Agreed evaluation protocol:
    # 1) April 2025: train on all data before April, test full April
    # 2) September 2025: train on all data before September, test full September
    periods = [
        (4, "April 2025",     pd.Timestamp("2025-04-01 00:00"), pd.Timestamp("2025-04-30 23:45")),
        (9, "September 2025", pd.Timestamp("2025-09-01 00:00"), pd.Timestamp("2025-09-30 23:45")),
    ]

    raw_2025 = raw.loc[raw.index.year == 2025].copy()
    corruption = detect_corruption(raw_2025)
    primary_window = corruption.get("primary_window", None)

    monthly_rows = []
    week_focus_start = pd.Timestamp("2025-09-15 00:00")
    week_focus_end   = pd.Timestamp("2025-09-21 23:45")

    for month_num, label, month_start, month_end in periods:
        out = run_period(month_num, label, month_start, month_end, raw)
        m = compute_metrics(out["result"], label)

        print(f"\n{'='*55}")
        print(f"  FINAL METRICS [v3] - {label}")
        print(f"{'='*55}")
        print(f"  MAE   : {m['mae']:.4f} kW")
        print(f"  RMSE  : {m['rmse']:.4f} kW")
        print(f"  NRMSE : {m['nrmse']:.2f} %")
        print(f"  Bias  : {m['bias']:+.4f} kW")
        print(f"{'='*55}")

        slug = label.lower().replace(" ", "_")
        save_metrics(m, os.path.join(OUT_DIR, f"metrics_v3_{slug}.txt"))
        make_plot(out["result"], m, slug)
        make_daily_nrmse_plot(out["result"], m, slug)
        # Excel is saved after dispatch + billing (see save_comprehensive_excel below)

        # Forecast matrix for MPC dispatch over the FULL month
        forecast_matrix = build_mpc_forecast_matrix(
            out["pred_hub"],
            out["pred_qnt"],
            out["best_alpha_hub"],
            out["common"],
            H=HORIZON_DEFAULT,
            pred_twe=out["pred_twe"],
            blend_weights=out["blend_weights"],
            a_cal=out["a_cal"],
            b_cal=out["b_cal"],
        )

        month_actual = raw.loc[(raw.index >= month_start) & (raw.index <= month_end),
                               ["load_kw", "pv_kw", "buy_price", "sell_price"]].copy()
        month_forecast = forecast_matrix.loc[forecast_matrix.index.intersection(month_actual.index)]

        print(f"\nRunning MPC dispatch on full month: {label} ...")
        dispatch = run_mpc(month_actual, forecasts=month_forecast, horizon=HORIZON_DEFAULT, verbose_every=0)
        oracle_dispatch = run_mpc(month_actual, forecasts=None, horizon=HORIZON_DEFAULT, verbose_every=0)
        ok, rep = validate_dispatch(dispatch)

        month_hist = raw.loc[(raw.index >= month_start) & (raw.index <= month_end),
                             ["load_kw", "pv_kw", "buy_price", "sell_price", "p_battery_kw"]].copy()
        # Use raw p_battery without corruption window zeroing.
        # The 2025 SoC reconstruction shows chronic drift (94.8% of steps SoC < 0),
        # meaning primary_window covers the entire year — passing it would zero out ALL
        # battery activity for both test months, collapsing Baseline A into Baseline B.
        # Energy balance is perfect in the dataset (residual ≈ 0), so
        #   p_grid = load − pv − p_battery = actual recorded grid power.
        # This gives the genuine historical bill.
        base_a = compute_baseline_a(month_hist, corruption_window=None)
        base_b = compute_baseline_b(month_hist)

        controller_bill = compute_bill(
            dispatch["p_grid"].values,
            dispatch["buy_price"].values,
            dispatch["sell_price"].values,
            dt_hours=DT_HOURS,
        )
        oracle_bill = compute_bill(
            oracle_dispatch["p_grid"].values,
            oracle_dispatch["buy_price"].values,
            oracle_dispatch["sell_price"].values,
            dt_hours=DT_HOURS,
        )

        bug_flag = "OK"
        if base_a["bill"] < oracle_bill:
            bug_flag = "CHECK: baseline A better than oracle"

        savings_vs_a_eur = base_a["bill"] - controller_bill
        savings_vs_a_pct = 100.0 * savings_vs_a_eur / max(base_a["bill"], 1e-6)
        oracle_potential  = base_a["bill"] - oracle_bill
        oracle_gap_pct    = (100.0 * (controller_bill - oracle_bill) / oracle_potential
                             if oracle_potential > 1e-6 else float("nan"))

        print(f"\n  --- Bill Summary: {label} ---")
        print(f"  Controller bill  : €{controller_bill:.2f}")
        print(f"  Baseline A bill  : €{base_a['bill']:.2f}")
        print(f"  Baseline B bill  : €{base_b['bill']:.2f}")
        print(f"  Oracle bill      : €{oracle_bill:.2f}")
        print(f"  Savings vs A     : €{savings_vs_a_eur:+.2f}  ({savings_vs_a_pct:+.1f}%)")
        if not np.isnan(oracle_gap_pct):
            print(f"  Oracle gap       : {oracle_gap_pct:.1f}%  (lower = closer to perfect)")
        else:
            print(f"  Oracle gap       : n/a  (baseline A ≤ oracle)")
        print(f"  Oracle check     : {bug_flag}")

        monthly_rows.append({
            "period":              label,
            "forecast_nrmse_pct":  round(m["nrmse"], 2),
            "controller_bill_eur": round(controller_bill, 2),
            "baseline_a_bill_eur": round(base_a["bill"], 2),
            "baseline_b_bill_eur": round(base_b["bill"], 2),
            "oracle_bill_eur":     round(oracle_bill, 2),
            "savings_vs_a_eur":    round(savings_vs_a_eur, 2),
            "savings_vs_a_pct":    round(savings_vs_a_pct, 2),
            "oracle_gap_pct":      round(oracle_gap_pct, 2) if not np.isnan(oracle_gap_pct) else "n/a",
            "oracle_check":        bug_flag,
        })

        # Comprehensive Excel (all 4 sheets in one file)
        save_comprehensive_excel(
            out["result"], m, slug, label,
            dispatch, oracle_dispatch,
            base_a, base_b,
            controller_bill, oracle_bill,
            savings_vs_a_eur, savings_vs_a_pct, oracle_gap_pct,
        )

        # Save month-level dispatch and comparison
        dispatch_csv = os.path.join(OUT_DIR, f"dispatch_v3_{slug}.csv")
        dispatch[["load_kw", "pv_kw", "p_battery", "p_grid", "soc"]].to_csv(dispatch_csv, index=True)
        print(f"  Saved: {dispatch_csv}")

        compare_fig = os.path.join(OUT_DIR, f"bill_compare_v3_{slug}.png")
        plt.figure(figsize=(7, 4.5))
        labels = ["Controller", "Baseline A", "Baseline B", "Oracle"]
        vals = [controller_bill, base_a["bill"], base_b["bill"], oracle_bill]
        cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#17becf"]
        bars = plt.bar(labels, vals, color=cols, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v * 1.01, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        plt.ylabel("Bill (€)")
        plt.title(f"Monthly Bill Comparison — {label}")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(compare_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {compare_fig}")

        # Save presentation week plots if week overlaps this month
        if (week_focus_start >= month_start) and (week_focus_end <= month_end):
            _wk = lambda df: df.loc[(df.index >= week_focus_start) & (df.index <= week_focus_end)].copy()
            wk_dispatch  = _wk(dispatch)
            wk_oracle    = _wk(oracle_dispatch)
            wk_base_a    = _wk(base_a["dispatch"])
            wk_base_b    = _wk(base_b["dispatch"])

            wk_csv = os.path.join(OUT_DIR, f"dispatch_week_v3_{slug}.csv")
            wk_dispatch[["load_kw", "pv_kw", "p_battery", "p_grid", "soc"]].to_csv(wk_csv, index=True)
            print(f"  Saved: {wk_csv}")

            # Dispatch plot with Baseline A overlaid as dashed lines
            make_dispatch_plot(
                wk_dispatch,
                os.path.join(OUT_DIR, f"dispatch_plot_week_v3_{slug}.png"),
                title=f"Dispatch Week — {label} (2025-09-15 to 2025-09-21)",
                actual_df=wk_base_a,
            )

            # Weekly bill comparison — Controller / Oracle / Baseline A / Baseline B
            wk_comp_fig = os.path.join(OUT_DIR, f"bill_compare_week_v3_{slug}.png")
            wk_controller_bill = compute_bill(wk_dispatch["p_grid"].values,
                                              wk_dispatch["buy_price"].values,
                                              wk_dispatch["sell_price"].values, dt_hours=DT_HOURS)
            wk_oracle_bill     = compute_bill(wk_oracle["p_grid"].values,
                                              wk_oracle["buy_price"].values,
                                              wk_oracle["sell_price"].values, dt_hours=DT_HOURS)
            wk_base_a_bill     = compute_bill(wk_base_a["p_grid"].values,
                                              wk_base_a["buy_price"].values,
                                              wk_base_a["sell_price"].values, dt_hours=DT_HOURS)
            wk_base_b_bill     = compute_bill(wk_base_b["p_grid"].values,
                                              wk_base_b["buy_price"].values,
                                              wk_base_b["sell_price"].values, dt_hours=DT_HOURS)
            plt.figure(figsize=(8, 4.4))
            wk_labels = ["Controller", "Oracle", "Baseline A", "Baseline B"]
            wk_vals   = [wk_controller_bill, wk_oracle_bill, wk_base_a_bill, wk_base_b_bill]
            wk_cols   = ["#1f77b4", "#17becf", "#ff7f0e", "#2ca02c"]
            bars = plt.bar(wk_labels, wk_vals, color=wk_cols, edgecolor="white", linewidth=0.6)
            for b, v in zip(bars, wk_vals):
                y_pos = v + abs(v) * 0.02 if v >= 0 else v - abs(v) * 0.06
                plt.text(b.get_x() + b.get_width() / 2, y_pos, f"{v:.2f}",
                         ha="center", va="bottom", fontsize=9)
            plt.axhline(0, color="black", lw=0.6)
            plt.ylabel("Bill (€)")
            plt.title("Week Bill Comparison — 2025-09-15 to 2025-09-21")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(wk_comp_fig, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {wk_comp_fig}")

        summary_path = os.path.join(OUT_DIR, f"dispatch_summary_v3_{slug}.txt")
        with open(summary_path, "w", encoding="utf-8") as fh:
            fh.write(f"Dispatch summary [v3] - {label}\n")
            fh.write("=" * 60 + "\n")
            fh.write(f"Month window: {month_start} -> {month_end}\n")
            fh.write(f"Dispatch rows: {len(dispatch):,}\n")
            fh.write(f"Validation pass: {ok}\n")
            for k, v in rep.items():
                fh.write(f"{k}: {v}\n")
            fh.write("\nBills (€):\n")
            fh.write(f"  Controller (forecast + MPC) : {controller_bill:.2f}\n")
            fh.write(f"  Baseline A (historical)     : {base_a['bill']:.2f}\n")
            fh.write(f"  Baseline B (no battery)     : {base_b['bill']:.2f}\n")
            fh.write(f"  Oracle (perfect foresight)  : {oracle_bill:.2f}\n")
            fh.write(f"\nSavings vs Baseline A : {savings_vs_a_eur:+.2f} € ({savings_vs_a_pct:+.1f}%)\n")
            fh.write(f"Savings vs Baseline B : {base_b['bill'] - controller_bill:+.2f} €\n")
            if not np.isnan(oracle_gap_pct):
                fh.write(f"Oracle gap            : {oracle_gap_pct:.1f}%\n")
            else:
                fh.write("Oracle gap            : n/a (baseline A ≤ oracle)\n")
            fh.write(f"Oracle check          : {bug_flag}\n")
        print(f"  Saved: {summary_path}")

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_csv = os.path.join(OUT_DIR, "monthly_comparison_v3.csv")
    monthly_df.to_csv(monthly_csv, index=False)
    print(f"\n  Saved: {monthly_csv}")

    # Cross-period savings chart (April vs September)
    savings_fig = os.path.join(OUT_DIR, "savings_compare_v3.png")
    x = np.arange(len(monthly_df))
    width = 0.22
    plt.figure(figsize=(9, 5))
    plt.bar(x - width, monthly_df["baseline_a_bill_eur"], width=width, label="Baseline A", color="#ff7f0e")
    plt.bar(x,         monthly_df["controller_bill_eur"], width=width, label="Controller", color="#1f77b4")
    plt.bar(x + width, monthly_df["oracle_bill_eur"], width=width, label="Oracle", color="#17becf")
    plt.xticks(x, monthly_df["period"].values)
    plt.ylabel("Bill (€)")
    plt.title("Savings Comparison Across Target Months")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(savings_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {savings_fig}")

    print(f"\n  All outputs -> {OUT_DIR}/")
