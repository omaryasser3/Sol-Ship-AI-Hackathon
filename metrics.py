"""
Metrics & oracle gap analysis.

Per Brief Section 4 (Required Results):
  - Compute 2025 bills under Baseline A, Baseline B, controller
  - Absolute and percentage savings vs. each baseline
  - Oracle gap: same controller with actual 2025 load as input
"""
import numpy as np
import pandas as pd
from baselines import savings_summary
from tariff import compute_bill
from config import DT_HOURS


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    mean_actual = y_true.mean()
    return rmse(y_true, y_pred) / mean_actual * 100 if mean_actual > 1e-9 else float("nan")


def compute_dispatch_bill(dispatch_df):
    """Bill from a dispatch DataFrame (output of MPC or baseline)."""
    return compute_bill(
        dispatch_df["p_grid"].values,
        dispatch_df["buy_price"].values,
        dispatch_df["sell_price"].values,
        dt_hours=DT_HOURS,
    )


def oracle_gap_analysis(controller_dispatch, oracle_dispatch,
                         baseline_a_bill, baseline_b_bill):
    """
    Compute the oracle gap.

    The oracle = same MPC formulation, but uses the ACTUAL 2025 load as
    the "forecast" (perfect foresight). Difference between controller_bill
    and oracle_bill is the cost of forecast error.
    """
    controller_bill = compute_dispatch_bill(controller_dispatch)
    oracle_bill     = compute_dispatch_bill(oracle_dispatch)

    sav = savings_summary(controller_bill, baseline_a_bill, baseline_b_bill)

    oracle_savings_eur = baseline_a_bill - oracle_bill
    forecast_savings_eur = baseline_a_bill - controller_bill
    gap_eur = oracle_savings_eur - forecast_savings_eur
    gap_pct = (100 * gap_eur / oracle_savings_eur) if oracle_savings_eur > 0 else 0.0

    return {
        **sav,
        "oracle_bill":              round(oracle_bill, 2),
        "oracle_savings_vs_a_eur":  round(oracle_savings_eur, 2),
        "oracle_gap_eur":           round(gap_eur, 2),
        "oracle_gap_pct":           round(gap_pct, 2),
        "interpretation": _interpret_gap(gap_pct),
    }


def _interpret_gap(gap_pct):
    if gap_pct < 10:
        return "Forecast is excellent; further improvement requires a different optimizer."
    elif gap_pct < 25:
        return "Sweet spot: forecast is decent. Targeted forecast improvements would help."
    elif gap_pct < 40:
        return "Forecast is the bottleneck — significant savings left on the table."
    else:
        return "Forecast has systematic bias the optimizer is repeatedly betting wrongly on."


def forecast_metrics(y_true, y_pred):
    """Return RMSE, MAE, NRMSE per the brief's Section 6 definitions."""
    return {
        "rmse":  round(rmse(y_true, y_pred),  4),
        "mae":   round(mae(y_true, y_pred),   4),
        "nrmse": round(nrmse(y_true, y_pred), 2),
    }


def build_results_table(metrics_dict):
    """Format the canonical results table for the slide deck."""
    rows = [
        ("Baseline A — historical operation",  f"€{metrics_dict['baseline_a_bill']:.2f}"),
        ("Baseline B — zero-intelligence",      f"€{metrics_dict['baseline_b_bill']:.2f}"),
        ("Our controller (rolling MPC, H=96)",  f"€{metrics_dict['controller_bill']:.2f}"),
        ("",                                     ""),
        ("Savings vs. Baseline A",               f"€{metrics_dict['savings_vs_a_eur']:+.2f}  ({metrics_dict['savings_vs_a_pct']:+.2f}%)"),
        ("Savings vs. Baseline B",               f"€{metrics_dict['savings_vs_b_eur']:+.2f}  ({metrics_dict['savings_vs_b_pct']:+.2f}%)"),
        ("",                                     ""),
        ("Oracle bill (perfect foresight)",      f"€{metrics_dict['oracle_bill']:.2f}"),
        ("Oracle gap",                           f"€{metrics_dict['oracle_gap_eur']:.2f}  ({metrics_dict['oracle_gap_pct']:.2f}%)"),
    ]
    return rows


def print_results_table(metrics_dict):
    print("\n" + "=" * 65)
    print(" RESULTS TABLE — Brief Section 4")
    print("=" * 65)
    for label, value in build_results_table(metrics_dict):
        if label:
            print(f"  {label:42s}{value:>20s}")
        else:
            print()
    print("=" * 65)
    if "interpretation" in metrics_dict:
        print(f"\nOracle-gap interpretation: {metrics_dict['interpretation']}")
