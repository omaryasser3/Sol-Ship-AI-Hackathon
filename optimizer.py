"""
Rolling-horizon MPC battery dispatch controller (LP).

Per the brief (Section 7), the controller must:
  - Operate causally (only past+horizon-window info, no future leak)
  - Solve a horizon-window optimization problem
  - Execute only the FIRST decision, then advance one step
  - Repeat for every 15-min interval of 2025

Mathematical formulation (LP, no integers needed — see below):
  minimise  Σ (p_imp[h] · buy[h] − p_exp[h] · sell[h]) · Δt
  subject to
      load_hat[h] + p_ch[h] + p_exp[h] = pv[h] + p_dis[h] + p_imp[h]   (energy balance)
      s[h+1] = s[h] + (η · p_ch[h] − p_dis[h] / η) · Δt / E_max         (SoC dynamics)
      0 ≤ s[h+1] ≤ 1
      0 ≤ p_ch[h]  ≤ P_bat_max
      0 ≤ p_dis[h] ≤ P_bat_max
      0 ≤ p_imp[h] ≤ P_grid_max
      0 ≤ p_exp[h] ≤ P_grid_max
      s[0] = current measured SoC

Why pure LP (no MILP):
  - With buy > sell always (Italian residential: ~0.25 vs. ~0.05–0.12),
    the optimum NEVER sets p_imp and p_exp simultaneously positive.
  - With η < 1, simultaneous charge+discharge wastes energy that costs more
    to re-buy than it earns; the optimum sets at most one positive.
  → Mutual exclusivity is implicit in the cost structure. No binaries needed.

Implementation:
  - Build the CVXPY problem ONCE with cp.Parameter for time-varying inputs.
  - Each rolling step: update parameter values, call prob.solve() with warm_start.
  - HiGHS LP solver — fast, free, BSD-licensed.
"""
import time
import numpy as np
import pandas as pd
import cvxpy as cp
from config import (
    BATTERY_KWH, BATTERY_P_MAX_KW, ETA_DIR, GRID_P_MAX_KW,
    SOC_INIT, DT_HOURS,
)


class MPCController:
    """
    Rolling-horizon battery dispatch via CVXPY LP.

    Build once, solve many times.
    """

    def __init__(self, horizon, eta_dir=ETA_DIR, dt=DT_HOURS,
                 e_max=BATTERY_KWH, p_bat_max=BATTERY_P_MAX_KW,
                 p_grid_max=GRID_P_MAX_KW):
        self.H = horizon
        self.eta = eta_dir
        self.dt = dt
        self.e_max = e_max
        self.p_bat_max = p_bat_max
        self.p_grid_max = p_grid_max

        # === Parameters (updated each rolling step) ===
        self.load_hat   = cp.Parameter(horizon, nonneg=True)
        self.pv_hat     = cp.Parameter(horizon, nonneg=True)
        self.buy_price  = cp.Parameter(horizon, nonneg=True)
        self.sell_price = cp.Parameter(horizon, nonneg=True)
        self.soc_init   = cp.Parameter(nonneg=True)

        # === Decision variables ===
        self.p_ch  = cp.Variable(horizon, nonneg=True)   # charging power (≥ 0)
        self.p_dis = cp.Variable(horizon, nonneg=True)   # discharging power (≥ 0)
        self.p_imp = cp.Variable(horizon, nonneg=True)   # grid import (≥ 0)
        self.p_exp = cp.Variable(horizon, nonneg=True)   # grid export (≥ 0)
        self.soc   = cp.Variable(horizon + 1)            # SoC, 0..1

        # === Constraints ===
        cons = []

        # Initial SoC
        cons.append(self.soc[0] == self.soc_init)

        # Power bounds
        cons.append(self.p_ch  <= self.p_bat_max)
        cons.append(self.p_dis <= self.p_bat_max)
        cons.append(self.p_imp <= self.p_grid_max)
        cons.append(self.p_exp <= self.p_grid_max)

        # SoC bounds
        cons.append(self.soc >= 0)
        cons.append(self.soc <= 1)

        # SoC dynamics (vectorized)
        # s[h+1] = s[h] + (η · p_ch[h] − p_dis[h]/η) · dt / e_max
        cons.append(
            self.soc[1:] ==
            self.soc[:-1] +
            (self.eta * self.p_ch - self.p_dis / self.eta) * self.dt / self.e_max
        )

        # Energy balance: load + p_ch + p_exp = pv + p_dis + p_imp
        cons.append(
            self.load_hat + self.p_ch + self.p_exp ==
            self.pv_hat + self.p_dis + self.p_imp
        )

        # === Objective ===
        cost = cp.sum(
            cp.multiply(self.p_imp, self.buy_price)
            - cp.multiply(self.p_exp, self.sell_price)
        ) * self.dt

        self.problem = cp.Problem(cp.Minimize(cost), cons)

    def solve_one(self, load_hat_arr, pv_hat_arr, buy_arr, sell_arr, soc_now,
                  warm_start=True):
        """
        Update parameters and solve. Returns the optimal first-step decision.
        """
        H = self.H
        if len(load_hat_arr) < H:
            # Pad with last value if forecast horizon is short (end of year)
            load_hat_arr = np.concatenate(
                [load_hat_arr, np.full(H - len(load_hat_arr), load_hat_arr[-1])])
        if len(pv_hat_arr) < H:
            pv_hat_arr = np.concatenate(
                [pv_hat_arr, np.full(H - len(pv_hat_arr), pv_hat_arr[-1])])
        if len(buy_arr) < H:
            buy_arr = np.concatenate(
                [buy_arr, np.full(H - len(buy_arr), buy_arr[-1])])
        if len(sell_arr) < H:
            sell_arr = np.concatenate(
                [sell_arr, np.full(H - len(sell_arr), sell_arr[-1])])

        self.load_hat.value   = np.maximum(load_hat_arr[:H], 0.0)
        self.pv_hat.value     = np.maximum(pv_hat_arr[:H], 0.0)
        self.buy_price.value  = np.maximum(buy_arr[:H], 0.0)
        self.sell_price.value = np.maximum(sell_arr[:H], 0.0)
        self.soc_init.value   = float(np.clip(soc_now, 0.0, 1.0))

        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=warm_start, verbose=False)
        except Exception as e:
            print(f"[MPC] Solver error: {e}")
            return None

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            print(f"[MPC] Non-optimal status: {self.problem.status}")
            return None

        return {
            "p_ch":  float(self.p_ch.value[0]),
            "p_dis": float(self.p_dis.value[0]),
            "p_imp": float(self.p_imp.value[0]),
            "p_exp": float(self.p_exp.value[0]),
            "soc_planned_next": float(self.soc.value[1]),
            "objective": float(self.problem.value),
        }


def run_mpc(df_actual, forecasts, horizon, soc_init=SOC_INIT, verbose_every=2880):
    """
    Run rolling-horizon MPC over an entire time series.

    Parameters
    ----------
    df_actual  : DataFrame with columns load_kw, pv_kw, buy_price, sell_price,
                 indexed by 15-min timestamps. This is the GROUND TRUTH used only
                 for execution (SoC update, actual grid balance) — never fed to the LP.
    forecasts  : DataFrame shape (n, H_max). Row t = ML load forecasts made AT t for
                 h=1..H steps ahead. If None → oracle mode (actual load as perfect forecast).
    horizon    : int, MPC horizon length in 15-min steps
    soc_init   : initial SoC at start of run
    verbose_every : print progress every N steps

    Returns
    -------
    DataFrame with one row per timestep of df_actual:
        load, pv, buy_price, sell_price, p_ch, p_dis, p_battery, p_imp, p_exp, p_grid,
        soc, soc_planned, energy_balance_residual, solve_time_ms
    """
    n = len(df_actual)
    H = horizon

    controller = MPCController(horizon=H)

    load_actual = df_actual["load_kw"].values
    pv_actual   = df_actual["pv_kw"].values
    buy_arr     = df_actual["buy_price"].values
    sell_arr    = df_actual["sell_price"].values

    if forecasts is not None:
        # Align forecasts to df_actual index — only keep timesteps for which we have a forecast
        common_idx = df_actual.index.intersection(forecasts.index)
        n = len(common_idx)
        load_actual = df_actual.loc[common_idx, "load_kw"].values
        pv_actual   = df_actual.loc[common_idx, "pv_kw"].values
        buy_arr     = df_actual.loc[common_idx, "buy_price"].values
        sell_arr    = df_actual.loc[common_idx, "sell_price"].values
        forecast_arr = forecasts.loc[common_idx].values    # (n, H_max)
        idx_out = common_idx
    else:
        forecast_arr = None
        idx_out = df_actual.index

    # Causal PV and sell-price forecasts: lag-96 (same slot yesterday).
    # At decision time t we only know values up to t — future PV and market
    # prices are not observable. buy_price is ToU (deterministic from clock) so
    # the full horizon IS legitimately known.
    pv_fc   = np.concatenate([pv_actual[:96],   pv_actual[:-96]])   # pv_fc[t] = pv_actual[t-96]
    sell_fc = np.concatenate([sell_arr[:96],     sell_arr[:-96]])    # sell_fc[t] = sell_arr[t-96]

    # Storage
    rows = []
    soc = float(soc_init)

    t_start = time.time()
    last_print = t_start

    for t in range(n):
        # Build the H-step forecast horizon ENDING AT t+H (exclusive)
        end = min(t + H, n)
        actual_horizon_len = end - t
        if actual_horizon_len < 1:
            break

        buy_h  = buy_arr[t:t + actual_horizon_len]   # deterministic ToU — always known

        if forecast_arr is not None:
            # Controller: causal forecasts — no future actual values allowed.
            # LP h=0 is the CURRENT step t — use the actual observed load (measurable
            # at decision time). LP h=1..H-1 use the ML h=1..H-1 ahead forecasts.
            # This aligns with oracle mode (which starts at load_actual[t]) so the
            # oracle gap is not inflated by a spurious 1-step timing offset.
            if actual_horizon_len == 1:
                load_hat = np.array([load_actual[t]])
            else:
                load_hat = np.concatenate(
                    [[load_actual[t]], forecast_arr[t, : actual_horizon_len - 1]]
                )
            pv_h     = pv_fc  [t:t + actual_horizon_len]   # lag-96 causal PV
            sell_h   = sell_fc[t:t + actual_horizon_len]   # lag-96 causal sell price
        else:
            # Oracle: perfect foresight for load, PV, and sell price.
            # All three use actual values so the LP finds the true minimum-cost schedule.
            # This guarantees Oracle ≤ Baseline A by construction.
            load_hat = load_actual[t:t + actual_horizon_len]
            pv_h     = pv_actual  [t:t + actual_horizon_len]
            sell_h   = sell_arr   [t:t + actual_horizon_len]

        ts0 = time.time()
        sol = controller.solve_one(load_hat, pv_h, buy_h, sell_h, soc)
        solve_ms = (time.time() - ts0) * 1000

        if sol is None:
            # Fallback: do nothing — just import what's needed
            net = max(load_actual[t] - pv_actual[t], 0)
            p_imp = min(net, GRID_P_MAX_KW)
            p_exp = max(pv_actual[t] - load_actual[t], 0)
            p_exp = min(p_exp, GRID_P_MAX_KW)
            sol = {"p_ch": 0.0, "p_dis": 0.0, "p_imp": p_imp, "p_exp": p_exp,
                   "soc_planned_next": soc, "objective": float("nan")}

        # Execute first decision against ACTUAL load and PV
        # The optimizer planned with load_hat; we need to balance against load_actual.
        # Strategy: keep the planned battery decision, recompute grid to balance.
        # Clip LP outputs to remove interior-point numerical noise (solver returns
        # tiny positives like 1e-8 for variables that should be exactly 0).
        _TOL = 1e-6
        p_ch_actual  = max(0.0, sol["p_ch"]  - _TOL)
        p_dis_actual = max(0.0, sol["p_dis"] - _TOL)
        net_actual = load_actual[t] - pv_actual[t]   # positive = need power
        # Energy balance: net_actual = p_dis - p_ch + p_imp - p_exp
        # → p_imp - p_exp = net_actual - p_dis + p_ch
        residual = net_actual - p_dis_actual + p_ch_actual
        if residual >= 0:
            p_imp_actual = min(residual, GRID_P_MAX_KW)
            p_exp_actual = 0.0
            unmet = residual - p_imp_actual   # > 0 means infeasible (load not served)
        else:
            p_exp_actual = min(-residual, GRID_P_MAX_KW)
            p_imp_actual = 0.0
            unmet = 0.0   # surplus that can't be exported is curtailed

        # If unmet, scale back charging to free up grid capacity
        if unmet > 1e-6 and p_ch_actual > 0:
            # Reduce charging by `unmet` (extra grid headroom we need)
            reduction = min(unmet, p_ch_actual)
            p_ch_actual -= reduction
            residual -= reduction   # less to import
            p_imp_actual = min(residual, GRID_P_MAX_KW)
            unmet = max(residual - p_imp_actual, 0)

        # Update SoC: both terms active simultaneously (handles residual LP noise).
        # Charging raises SoC by p_ch*η, discharging lowers it by p_dis/η.
        de  = (p_ch_actual * ETA_DIR - p_dis_actual / ETA_DIR) * DT_HOURS
        soc = float(np.clip(soc + de / BATTERY_KWH, 0, 1))

        p_battery_signed = p_dis_actual - p_ch_actual    # positive = discharge
        p_grid_signed    = p_imp_actual - p_exp_actual   # positive = import

        rows.append({
            "load_kw":     float(load_actual[t]),
            "pv_kw":       float(pv_actual[t]),
            "buy_price":   float(buy_arr[t]),    # actual ToU buy price at t
            "sell_price":  float(sell_arr[t]),   # actual market sell price at t
            "p_ch":        p_ch_actual,
            "p_dis":       p_dis_actual,
            "p_battery":   p_battery_signed,
            "p_imp":       p_imp_actual,
            "p_exp":       p_exp_actual,
            "p_grid":      p_grid_signed,
            "soc":         soc,
            "unmet_load":  unmet,
            "solve_ms":    solve_ms,
        })

        if verbose_every and (t + 1) % verbose_every == 0:
            elapsed = time.time() - t_start
            eta = elapsed / (t + 1) * (n - t - 1)
            print(f"  step {t+1:6d}/{n}  elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s  "
                  f"avg solve {np.mean([r['solve_ms'] for r in rows[-100:]]):.1f}ms")

    print(f"  MPC complete: {n} steps in {time.time() - t_start:.1f}s "
          f"(avg {(time.time() - t_start)/max(n,1)*1000:.1f}ms/step)")
    return pd.DataFrame(rows, index=idx_out[:len(rows)])


def validate_dispatch(dispatch_df, tol=1e-4):
    """
    Sanity-check a dispatch DataFrame against all hard constraints.

    Returns (ok: bool, report: dict).
    """
    rep = {}
    df = dispatch_df

    # Energy balance: load = pv + p_battery + p_grid
    bal_residual = df["load_kw"] - df["pv_kw"] - df["p_battery"] - df["p_grid"]
    rep["energy_balance_max_residual"] = float(bal_residual.abs().max())

    # No simultaneous charge + discharge
    simul_chdis = ((df["p_ch"] > tol) & (df["p_dis"] > tol)).sum()
    rep["simul_charge_discharge_steps"] = int(simul_chdis)

    # No simultaneous import + export
    simul_impexp = ((df["p_imp"] > tol) & (df["p_exp"] > tol)).sum()
    rep["simul_import_export_steps"] = int(simul_impexp)

    # SoC bounds
    rep["soc_min"] = float(df["soc"].min())
    rep["soc_max"] = float(df["soc"].max())
    rep["soc_violations"] = int(((df["soc"] < -tol) | (df["soc"] > 1 + tol)).sum())

    # Power bounds
    rep["p_battery_max"] = float(df["p_battery"].abs().max())
    rep["p_grid_max"]    = float(df["p_grid"].abs().max())
    rep["p_battery_violations"] = int((df["p_battery"].abs() > BATTERY_P_MAX_KW + tol).sum())
    rep["p_grid_violations"]    = int((df["p_grid"].abs()    > GRID_P_MAX_KW    + tol).sum())

    # Unmet load
    rep["total_unmet_load_kwh"] = float(df["unmet_load"].sum() * DT_HOURS)

    ok = (
        rep["energy_balance_max_residual"] < 1e-3 and
        rep["simul_charge_discharge_steps"] == 0 and
        rep["simul_import_export_steps"] == 0 and
        rep["soc_violations"] == 0 and
        rep["p_battery_violations"] == 0 and
        rep["p_grid_violations"] == 0 and
        rep["total_unmet_load_kwh"] < 0.01
    )
    return ok, rep


if __name__ == "__main__":
    from data_loader import load_dataset
    from forecaster import train_single_horizon

    sheets = load_dataset("data/synthetic_dataset.xlsx")
    df_2024 = sheets["2024"]
    df_2025 = sheets["2025"]

    # Quick test: oracle MPC on first 7 days of 2025
    test_window = df_2025.iloc[:96 * 7].copy()
    print(f"=== Oracle MPC on {len(test_window)} steps (1 week) ===")
    dispatch = run_mpc(test_window, forecasts=None, horizon=96, verbose_every=192)
    ok, rep = validate_dispatch(dispatch)
    print(f"\nValidation: {'PASS' if ok else 'FAIL'}")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    print(f"\nDispatch summary:")
    print(f"  Total grid imports:  {dispatch['p_imp'].sum() * DT_HOURS:.2f} kWh")
    print(f"  Total grid exports:  {dispatch['p_exp'].sum() * DT_HOURS:.2f} kWh")
    print(f"  Total bill:          €{((dispatch['p_imp'] * dispatch['buy_price']).sum() - (dispatch['p_exp'] * dispatch['sell_price']).sum()) * DT_HOURS:.2f}")
