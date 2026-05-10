"""
Solship Hackathon 2026 — Configuration
All constants taken verbatim from the Participant Brief v4.0.
Single source of truth — every other module imports from here.
"""
import numpy as np

# ============================================================
# System specifications (Brief, Section 2)
# ============================================================
PV_CAPACITY_KWP   = 9.0      # Solar PV peak rating
BATTERY_KWH       = 16.0     # Usable battery capacity
BATTERY_P_MAX_KW  = 8.0      # Symmetric charge / discharge limit
BATTERY_RTE       = 0.90     # Round-trip efficiency
ETA_DIR           = np.sqrt(BATTERY_RTE)   # Per-direction efficiency = √0.9 ≈ 0.9487
GRID_P_MAX_KW     = 6.0      # Grid import / export ceiling
SOC_INIT          = 0.50     # Initial SoC at start of 2025

# ============================================================
# Time discretization
# ============================================================
DT_HOURS          = 0.25     # 15-minute intervals
STEPS_PER_HOUR    = 4
STEPS_PER_DAY     = 96
STEPS_PER_WEEK    = 672

# ============================================================
# Italian Time-of-Use tariff (Brief, Section 2)
# ============================================================
F1_PRICE          = 0.2540   # €/kWh, Mon-Fri 08-19 (excl. holidays)
F2_PRICE          = 0.2682   # €/kWh, Mon-Fri 07-08 + 19-23, Sat 07-23 (excl. holidays)
F3_PRICE          = 0.2440   # €/kWh, Mon-Sat 00-07 + 23-24, Sundays + holidays all day

# ============================================================
# Forecasting / MPC defaults
# ============================================================
HORIZON_DEFAULT   = 96       # 24h MPC horizon
HORIZON_OPTIONS   = [4, 24, 96]   # Required for Extension 1 (1h, 6h, 24h)

# ============================================================
# Italian timezone (CET/CEST per brief)
# Holidays come from the `holidays` package via tariff.py
# ============================================================
TIMEZONE          = "Europe/Rome"

# ============================================================
# Sign convention (Brief, Section 1) — for assertions only
# Internally we split into nonneg pairs (p_ch/p_dis, p_imp/p_exp)
# and reconstruct signed quantities for output.
# ============================================================
#   p_battery > 0  →  discharging
#   p_battery < 0  →  charging
#   p_grid    > 0  →  importing
#   p_grid    < 0  →  exporting
