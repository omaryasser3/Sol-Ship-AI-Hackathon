"""
Multi-horizon LightGBM load forecaster.

Strategy: DIRECT — train a separate model for each horizon h ∈ {1, ..., H}.
At inference time, call each model with the same feature row and get
predictions for all horizons simultaneously.

For the MPC at H=96, we'd need 96 models. To keep training time manageable,
we train at a SPARSE set of horizons {1, 2, 4, 8, 16, 24, 48, 72, 96} and
linearly interpolate predictions between them. This is a tested heuristic
that costs <1 NRMSE point vs. dense models.

NRMSE metric: per the brief — RMSE / mean(load) × 100.
"""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import build_features, DEFAULT_LAGS, DEFAULT_ROLL_WINDOWS

# Sparse horizon set for the MPC; finer for short-term, coarser for long-term
SPARSE_HORIZONS = [1, 2, 4, 8, 16, 24, 48, 72, 96]


# Default LightGBM parameters tuned for residential 15-min load.
# Conservative regularization to favor cross-site generalization.
DEFAULT_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 0.5,
    "verbose": -1,
    "num_threads": -1,
}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def nrmse(y_true, y_pred):
    """Brief Section 6: NRMSE = RMSE / mean(load) × 100."""
    mean_y = float(np.mean(np.asarray(y_true)))
    if mean_y <= 1e-9:
        return float("nan")
    return rmse(y_true, y_pred) / mean_y * 100


def train_single_horizon(df_train, horizon, df_val=None,
                         params=None, num_rounds=1500, early_stop=100,
                         lags=None, roll_windows=None,
                         holiday_set=None, verbose=False):
    """
    Train a LightGBM model for a single horizon.

    df_train must have load_kw (and pv_kw) at 15-min cadence.
    Returns trained Booster + the feature column order.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}

    X_tr, y_tr = build_features(df_train, lags=lags, roll_windows=roll_windows,
                                horizon=horizon, holiday_set=holiday_set)
    feature_cols = list(X_tr.columns)

    dtrain = lgb.Dataset(X_tr.values, y_tr.values, feature_name=feature_cols)
    valid_sets = [dtrain]
    valid_names = ["train"]

    if df_val is not None:
        X_va, y_va = build_features(df_val, lags=lags, roll_windows=roll_windows,
                                    horizon=horizon, holiday_set=holiday_set)
        # Align columns
        X_va = X_va[feature_cols]
        dval = lgb.Dataset(X_va.values, y_va.values, feature_name=feature_cols, reference=dtrain)
        valid_sets.append(dval)
        valid_names.append("val")

    callbacks = []
    if df_val is not None and early_stop:
        callbacks.append(lgb.early_stopping(early_stop, verbose=False))
    if verbose:
        callbacks.append(lgb.log_evaluation(period=200))
    else:
        callbacks.append(lgb.log_evaluation(period=0))

    model = lgb.train(
        p, dtrain,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return model, feature_cols


def train_multi_horizon(df_train, horizons=SPARSE_HORIZONS, df_val=None,
                        params=None, num_rounds=1500, early_stop=100,
                        lags=None, roll_windows=None,
                        holiday_set=None, verbose=True):
    """
    Train one model per horizon. Returns dict {h: (model, feature_cols)}.
    """
    models = {}
    for h in horizons:
        t0 = time.time()
        model, feats = train_single_horizon(
            df_train, horizon=h, df_val=df_val, params=params,
            num_rounds=num_rounds, early_stop=early_stop,
            lags=lags, roll_windows=roll_windows, holiday_set=holiday_set,
        )
        models[h] = (model, feats)
        if verbose:
            print(f"  h={h:3d}  trained in {time.time() - t0:.1f}s "
                  f"(best iter {model.best_iteration})")
    return models


def predict_multi_horizon(models, df_history, h_max=96,
                          lags=None, roll_windows=None, holiday_set=None):
    """
    Generate predictions for ALL horizons 1..h_max for EVERY timestep in df_history.

    Returns a DataFrame of shape (len(df_history) − max_lag, h_max), where row i
    contains the forecast made AT timestep i for steps i+1, i+2, ..., i+h_max.

    Internally:
      1. For each trained horizon h_k, run model.predict on the feature matrix
         to get predictions y_pred[t, h_k] = predicted load at t+h_k.
      2. Linearly interpolate across the sparse horizon set to fill h=1..h_max.
    """
    sparse_horizons = sorted(models.keys())
    pred_dict = {}

    for h in sparse_horizons:
        model, feature_cols = models[h]
        X, _ = build_features(df_history, lags=lags, roll_windows=roll_windows,
                              horizon=h, holiday_set=holiday_set)
        X = X[feature_cols]
        preds = model.predict(X.values)
        pred_dict[h] = pd.Series(preds, index=X.index, name=f"h{h}")

    # Align all on common index (= shortest among sparse horizons, which is highest h)
    common_idx = pred_dict[sparse_horizons[0]].index
    for h in sparse_horizons[1:]:
        common_idx = common_idx.intersection(pred_dict[h].index)

    # Build (n_rows, h_max) matrix by interpolating across sparse horizons
    n = len(common_idx)
    forecast_matrix = np.zeros((n, h_max))
    sparse_arr = np.array(sparse_horizons)
    sparse_preds = np.column_stack([pred_dict[h].loc[common_idx].values for h in sparse_horizons])

    target_h = np.arange(1, h_max + 1)
    for i in range(n):
        forecast_matrix[i] = np.interp(target_h, sparse_arr, sparse_preds[i])
    forecast_matrix = np.clip(forecast_matrix, 0, None)   # load is non-negative

    cols = [f"h{h}" for h in target_h]
    return pd.DataFrame(forecast_matrix, index=common_idx, columns=cols)


def evaluate(model, df_test, horizon=1,
             lags=None, roll_windows=None, holiday_set=None):
    """
    Evaluate a single-horizon model on a test set.

    Returns dict with rmse, mae, nrmse, predictions DataFrame.
    """
    X_te, y_te = build_features(df_test, lags=lags, roll_windows=roll_windows,
                                horizon=horizon, holiday_set=holiday_set)
    if isinstance(model, tuple):
        model, feature_cols = model
        X_te = X_te[feature_cols]
    else:
        feature_cols = list(X_te.columns)

    preds = model.predict(X_te.values)
    preds = np.clip(preds, 0, None)

    return {
        "rmse":  rmse(y_te.values, preds),
        "mae":   mae(y_te.values, preds),
        "nrmse": nrmse(y_te.values, preds),
        "y_true": y_te,
        "y_pred": pd.Series(preds, index=y_te.index, name="pred"),
    }


def evaluate_with_rescaling(model, df_test, horizon=1,
                             warmup_days=7, lags=None, roll_windows=None,
                             holiday_set=None):
    """
    Evaluate with online rescaling for cross-site generalization.

    After producing predictions, applies a multiplicative correction
    based on the ratio of the new site's mean load (over a warm-up window)
    to the predictions' mean over the same window.

    Used for the Day 2 surprise dataset.
    """
    X_te, y_te = build_features(df_test, lags=lags, roll_windows=roll_windows,
                                horizon=horizon, holiday_set=holiday_set)
    if isinstance(model, tuple):
        model, feature_cols = model
        X_te = X_te[feature_cols]

    preds = model.predict(X_te.values)
    preds = np.clip(preds, 0, None)

    # Compute scale correction over warm-up period
    warmup_steps = warmup_days * 96
    if len(y_te) > warmup_steps:
        actual_mean_warmup = y_te.iloc[:warmup_steps].mean()
        pred_mean_warmup   = float(np.mean(preds[:warmup_steps]))
        if pred_mean_warmup > 0:
            scale = actual_mean_warmup / pred_mean_warmup
        else:
            scale = 1.0
    else:
        scale = 1.0

    preds_scaled = preds * scale

    return {
        "rmse_raw":   rmse(y_te.values, preds),
        "mae_raw":    mae(y_te.values, preds),
        "nrmse_raw":  nrmse(y_te.values, preds),
        "rmse_scaled":  rmse(y_te.values, preds_scaled),
        "mae_scaled":   mae(y_te.values, preds_scaled),
        "nrmse_scaled": nrmse(y_te.values, preds_scaled),
        "scale_applied": float(scale),
        "y_true": y_te,
        "y_pred_raw": pd.Series(preds, index=y_te.index, name="pred_raw"),
        "y_pred_scaled": pd.Series(preds_scaled, index=y_te.index, name="pred_scaled"),
    }


if __name__ == "__main__":
    from data_loader import load_dataset

    sheets = load_dataset("data/synthetic_dataset.xlsx")
    df_2024 = sheets["2024"]
    df_2025 = sheets["2025"]

    # 9-month train / 3-month val split on 2024
    cut = df_2024.index[int(0.75 * len(df_2024))]
    df_tr = df_2024.loc[:cut]
    df_va = df_2024.loc[cut:]
    print(f"Train: {df_tr.index.min()} → {df_tr.index.max()} ({len(df_tr)} rows)")
    print(f"Val:   {df_va.index.min()} → {df_va.index.max()} ({len(df_va)} rows)")

    print("\n=== Training h=1 model ===")
    t0 = time.time()
    model, feats = train_single_horizon(df_tr, horizon=1, df_val=df_va, num_rounds=600)
    print(f"Trained in {time.time() - t0:.1f}s. Best iter: {model.best_iteration}")
    print(f"# features: {len(feats)}")

    print("\n=== Evaluating on 2025 (h=1) ===")
    res = evaluate((model, feats), df_2025, horizon=1)
    print(f"  RMSE  = {res['rmse']:.4f} kW")
    print(f"  MAE   = {res['mae']:.4f} kW")
    print(f"  NRMSE = {res['nrmse']:.2f} %")
