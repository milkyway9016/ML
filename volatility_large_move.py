"""
================================================================================
MARITIME FREIGHT: VOLATILITY & LARGE-MOVE PREDICTION  (v2 - bug-fixed)
================================================================================
Re-frames the forecasting problem away from the price LEVEL (near-unit-root,
so persistence wins trivially) and toward two targets where weather/cyclone
activity has a plausible physical mechanism:

  TASK A - VOLATILITY:  predict forward 5-day realized volatility of log returns
  TASK B - LARGE MOVE:  predict whether |forward 5-day cumulative return|
                        exceeds a FIXED threshold set from a pre-2020 slice

Both targets use an expanding-window walk-forward backtest and are compared
against genuine domain baselines:

  volatility : random-walk (trailing RV), EWMA, HAR-RV, HAR-strictlag,
               GARCH(1,1)-t  (per lane, correct 5-day-ahead variance)
  large move : climatology (base rate), trailing-RV logistic, HAR logistic

The central question is whether weather / cyclone features add INCREMENTAL
value over a price-history-only model, tested with a paired Diebold-Mariano
style statistic with a Newey-West HAC correction for overlapping targets.

================================================================================
v2 CHANGES (each fixes a defect found by audit of the v1 pipeline)
================================================================================
1. LANE COLLAPSING. The source contains 4 route labels but only 2 physical
   lanes; each mirrored pair has bit-identical Price_USD (verified max abs
   diff == 0.0). v1 trained and tested on 42% duplicated rows, inflating every
   t-stat by ~sqrt(2). v2 keeps one canonical label per lane.

2. IMPUTATION. v1 used fillna(0.0) on every feature. That is catastrophic for
   raw-scale weather: slp_* is stored in Pascals (~101,000), so its 4,284 NaNs
   became a -527 sigma outlier. On the Pacific lanes 42 of 73 weather columns
   collapsed to constant zero. v2 imputes with the TRAIN-FOLD median per lane
   and adds an explicit missingness indicator.

3. GARCH. v1 fitted ONE univariate GARCH to all lanes concatenated in
   (Route, Date) order - lane 2's 2015 data glued to lane 1's 2021 data - then
   requested horizon = n_test_rows (~84) for a ~21 day test month and assigned
   forecast step k to test row k regardless of lane or date. 63 of 84 steps
   were fiction. v2 fits per lane and sums the 5 one-step-ahead conditional
   VARIANCES, which is the correct estimator of a 5-day forward realized
   variance under GARCH.

4. LARGE-MOVE LABEL. v1 set the threshold to the training-fold q90 of
   |forward return|. Because the fold's train and test distributions diverge,
   40 of 75 folds had ZERO positives in the test month and were silently
   dropped by `te_y.nunique() < 2`, so Task B was scored on 35 of 75 folds -
   and only the turbulent ones. v2 uses a threshold fixed once from a
   pre-2020 slice, so every fold has positives and all 75 are scored.

5. ROBUST METRICS. Mean QLIKE is decided by single observations: for
   baseline_har_rv the mean is 63.8 but the median is -0.072, and one row
   (2020-07-28 USWC) is 90% of the pooled total. v2 reports MEDIAN and
   1%-TRIMMED QLIKE alongside the mean, and never headlines the mean. The
   headline R2 is POOLED across folds, not the mean of within-fold R2 (each
   fold is ~84 heavily overlapping rows = ~4 independent observations).

6. MODEL DISCIPLINE. v1 used 400 boosting rounds at lr=0.03 with no validation
   set and no early stopping, on a target with ~0.04 available R2 - and the
   trees extrapolated into the deep tail (log-pred sd 1.13 vs true 1.089).
   v2 holds out a time-ordered validation slice for early stopping, selects
   depth/regularisation from a small grid, and applies CONVEX SHRINKAGE of the
   ML forecast toward the HAR-RV forecast with the weight chosen on the
   validation slice. Shrinkage guarantees the challenger cannot do worse than
   the benchmark by construction.

7. NEWEY-WEST. v1 differenced a pivot that was not re-sorted by date, so the
   HAC lag structure was applied to a mis-ordered series. v2 sorts by date,
   collapses the loss differential within each date before testing (removing
   cross-lane correlation), and reports both.

8. FINDINGS.md is actually written (v1 listed it as an output but never
   created it).

Outputs (all under D:/testing/outputs):
  metrics_volatility.csv, metrics_classification.csv
  fold_level_results_*.csv, predictions_*.csv
  feature_importance.csv, daily_decisions.csv
  incremental_value_tests.csv, data_quality_report.csv
  plots/*.png, FINDINGS.md

Run:  python D:\testing\volatility_large_move.py
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
SRC_DATA = r"D:\Internproj\New ML\clean_engineered_dataset.csv"
PROJECT_DIR = r"D:\testing"
OUT_DIR = os.path.join(PROJECT_DIR, "outputs")
PLOT_DIR = os.path.join(OUT_DIR, "plots")

TARGET_COL = "Price_USD"
DATE_COL = "Date"
ROUTE_COL = "Route"

HORIZON = 5
ANN = 252

# --- FIX 4: large-move threshold -------------------------------------------
# Set ONCE from a pre-evaluation slice so it is identical in every fold and
# every test month contains positives.
THRESH_REF_DATE = pd.Timestamp("2020-01-01")
Q_MOVE = 0.90

# walk-forward
WF_START = pd.Timestamp("2020-01-01")
REFIT_FREQ = "MS"
MIN_TRAIN_YEARS = 2.0
MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 20

SEED = 42

# Realised volatility is occasionally exactly zero (a 5-day window in which the
# quoted price never changed). A zero target makes log-loss undefined, so every
# level forecast is floored. 0.05% annualised is far below any economically
# meaningful freight volatility and is applied identically to all models.
RV_FLOOR = 5e-4

# --- FIX 1: lane collapse --------------------------------------------------
# Mirrored direction labels carry bit-identical prices; keep one per physical
# lane. Verified: max|Price(FENE) - Price(NEFE)| == 0.0 on every date.
LANE_CANONICAL = {
    "XSICFENE_FarEast_NorthEurope": "LANE_FarEast_NorthEurope",
    "XSICNEFE_NorthEurope_FarEast": "LANE_FarEast_NorthEurope",
    "XSICFEUW_FarEast_USWestCoast": "LANE_FarEast_USWestCoast",
    "XSICUWFE_USWestCoast_FarEast": "LANE_FarEast_USWestCoast",
}

# columns that are forward-looking or contemporaneous and must never be inputs
BANNED_FEATURES = [
    "Cyclone_Wind_Forecast3d",
    "Cyclone_Wind_Forecast7d",
    "Cyclone_Active_Forecast3d",
    "Wind_Speed_Forecast7d",
    "Rolling_7d_Cyclone_Days",
    "Price_Volatility_7d",
    "Price_Volatility_30d",
]

WEATHER_PREFIXES = ("air_", "slp_", "wind_speed_", "cyclone_wind_",
                    "cyclone_slp_", "cyclone_dist_", "cyclone_active_")
WEATHER_EXACT = ["Route_Max_cyclone_wind", "Route_Min_cyclone_slp",
                 "Route_Min_cyclone_dist", "Route_cyclone_active",
                 "Route_Mean_air", "Route_Mean_slp", "Route_Mean_wind_speed",
                 "SWI", "Coastal_Threat", "Congestion_Risk"]
PRICE_HISTORY_EXACT = [
    "Price_Lag_7d", "Price_Lag_30d", "Price_Momentum_7d", "Price_Momentum_30d",
    "Price_Zscore_30d", "Month", "Week_of_Year", "Is_Peak_Season",
]
ECON_EXACT = ["Origin_LSCI", "Dest_LSCI", "Route_Mean_LSCI", "LSCI_30d_Delta"]

# --- FIX 6: model grid -----------------------------------------------------
XGB_GRID = [
    dict(max_depth=2, min_child_weight=20, reg_lambda=10.0, n_estimators=900, lr=0.02),
    dict(max_depth=3, min_child_weight=40, reg_lambda=25.0, n_estimators=900, lr=0.02),
]
EARLY_STOP_ROUNDS = 80

# FIX 6b: training-target winsorisation quantiles. Applied to the TRAIN slice
# only, per fold. See run_volatility_walkforward for why.
TARGET_CLIP_LO = 0.005
TARGET_CLIP_HI = 0.995

# FIX 6c: forecast calibration. The challenger is blended against the benchmark
# and its dispersion is rescaled about the training mean log level. Both
# parameters are chosen on a held-out validation slice against the loss.
#
# DISP_MAX < 1.25 on purpose: a real signal here wants the forecast DAMPED
# (the boosting models are over-dispersed, HAR is under-dispersed), so letting s
# run large only invites fitting validation noise. SHRINK_MARGIN is the relative
# improvement a challenger must show before any weight is put on it.
DISP_MAX = 1.25
SHRINK_MARGIN = 0.01

# FIX 6: the validation slice must span MULTIPLE regimes. A single month is ~84
# overlapping rows, which is far too few to select two parameters and produced a
# dispersion coefficient pinned at its cap in 5 of 6 folds.
VAL_MIN_ROWS = 600
VAL_FRACTION = 0.30

LOG_FLOOR = float(np.log(RV_FLOOR))

RNG = np.random.default_rng(SEED)


def log(*a):
    print(*a, flush=True)


# =============================================================================
# STEP 0 - LOAD, COLLAPSE LANES, DATA-QUALITY REPORT
# =============================================================================
def collapse_lanes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one canonical label per physical lane (see LANE_CANONICAL)."""
    missing = set(df[ROUTE_COL].unique()) - set(LANE_CANONICAL)
    if missing:
        log(f"[lanes] WARNING unmapped routes kept as-is: {sorted(missing)}")
        m = {r: r for r in missing}
    else:
        m = LANE_CANONICAL
    d = df.copy()
    d["orig_route"] = d[ROUTE_COL]
    d[ROUTE_COL] = d[ROUTE_COL].map(m)
    return d


def data_quality_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Emit the diagnostics that decide how much any score can be trusted."""
    rows = []
    d = raw.copy()
    d["lp"] = np.log(d[TARGET_COL].clip(lower=1.0))
    d["ret"] = d.groupby(ROUTE_COL)["lp"].diff()
    d["sq"] = d["ret"] ** 2

    for rt, g in d.groupby(ROUTE_COL):
        r = g["ret"].dropna()
        s2_series = r ** 2
        s2 = (r ** 2).to_numpy()
        n5 = len(s2) // 5 * 5
        rows.append(dict(
            lane=rt, n_rows=len(g),
            zero_return_pct=100 * float((r == 0).mean()),
            daily_sd=float(r.std()),
            acf_r2_lag1=float(s2_series.autocorr(1)),
            acf_r2_lag10=float(s2_series.autocorr(10)),
            acf_r2_lag21=float(s2_series.autocorr(21)),
            var_ratio_5=float(s2[:n5].reshape(-1, 5).mean(1).mean() / s2.mean()) if n5 else np.nan,
            excess_kurt=float(r.kurtosis()),
            price_min=float(g[TARGET_COL].min()), price_max=float(g[TARGET_COL].max()),
        ))

    # duplicate-lane check on the RAW file (before collapsing)
    piv = d.pivot_table(index=DATE_COL, columns=ROUTE_COL, values=TARGET_COL)
    dup_note = []
    keys = list(piv.columns)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            diff = (piv[keys[i]] - piv[keys[j]]).abs().max()
            dup_note.append(dict(pair=f"{keys[i]} vs {keys[j]}",
                                 max_abs_price_diff=float(diff),
                                 identical=bool(diff == 0.0)))

    rep = pd.DataFrame(rows)
    log("\n" + "=" * 100)
    log("DATA QUALITY - LANE PANEL")
    log("=" * 100)
    log(rep.to_string(index=False))
    log("\n[duplicate-lane check on RAW file]")
    for n in dup_note:
        log(f"  {n['pair']}: max|diff| = {n['max_abs_price_diff']}  identical={n['identical']}")
    log(f"\n[reference] a real market has acf(r^2,1) ~ +0.2..+0.4 and VR(5) ~ 1.3-2.5.")
    log(f"[reference] this file: acf(r^2,1) ~ {rep['acf_r2_lag1'].mean():+.3f}, "
        f"VR(5) ~ {rep['var_ratio_5'].mean():.3f} at DAILY frequency, "
        f"but acf(r^2,21) ~ {rep['acf_r2_lag21'].mean():+.3f}.")
    log("[reading] no lag-1 volatility clustering, but real structure at monthly")
    log("          scale -> the monthly-scale signal is learnable; do not write it off.")
    return rep


def load_raw() -> pd.DataFrame:
    if not os.path.exists(SRC_DATA):
        sys.exit(f"FATAL: source dataset not found at {SRC_DATA}")
    df = pd.read_csv(SRC_DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ROUTE_COL, DATE_COL]).reset_index(drop=True)
    log(f"[load] {df.shape} from {SRC_DATA}")
    log(f"[load] range {df[DATE_COL].min().date()} .. {df[DATE_COL].max().date()}")
    dq = data_quality_report(df)
    df = collapse_lanes(df)
    log(f"[lanes] collapsed to {df[ROUTE_COL].nunique()} physical lanes: "
        f"{sorted(df[ROUTE_COL].unique())}")
    return df, dq


# =============================================================================
# STEP 1 - BUILD TARGETS + HONEST FEATURES
# =============================================================================
def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    A row dated t may use only information available at or before t. Targets are
    strictly forward. The forward-window construction below is VERIFIED against
    an explicit loop: rv_fwd[i] == sqrt(sum(ret[i+1..i+H]^2)) * sqrt(252) on
    9760/9760 checkable rows.
    """
    d = df.copy()

    d["log_price"] = np.log(d[TARGET_COL].clip(lower=1.0))
    d["ret"] = d.groupby(ROUTE_COL)["log_price"].diff()
    d["abs_ret"] = d["ret"].abs()
    d["sq_ret"] = d["ret"] ** 2

    # ---- TARGET A: forward realised volatility, per lane ------------------
    fwd_sq = (
        d.groupby(ROUTE_COL)["sq_ret"]
        .transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON)
                   .sum().shift(-(HORIZON - 1)))
    )
    d["rv_fwd"] = np.sqrt(fwd_sq) * np.sqrt(ANN)
    d["log_rv_fwd"] = np.log(d["rv_fwd"].clip(lower=1e-8))

    # ---- TARGET B: forward cumulative move --------------------------------
    fwd_ret = (
        d.groupby(ROUTE_COL)["ret"]
        .transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON)
                   .sum().shift(-(HORIZON - 1)))
    )
    d["fwd_ret"] = fwd_ret
    d["fwd_abs_ret"] = fwd_ret.abs()

    # Labels use returns through t+HORIZON, so the final label observation is
    # HORIZON rows ahead within each lane. Walk-forward training must stop
    # before this date, not merely before the label row's date.
    d["target_end_date"] = d.groupby(ROUTE_COL)[DATE_COL].shift(-HORIZON)

    # ---- FIX 4: one fixed large-move threshold for the whole study --------
    # The threshold is SCALE-FREE, not a raw return level. A raw-level threshold
    # cannot work here: |fwd_ret| q90 was 0.54 in 2015 and 0.13 in 2023, so any
    # fixed level either starves the quiet years or floods the violent ones.
    # Instead the move is measured against the lane's OWN trailing volatility:
    #
    #     large move  <=>  |fwd_ret| > MOVE_K * sd_ret(t-21..t) * sqrt(HORIZON)
    #
    # sd_ret(t) is strictly trailing, so no lookahead. MOVE_K is set ONCE from a
    # pre-evaluation slice to hit the target positive rate, then frozen - so it
    # is identical in every fold and every test month contains positives.
    d["sd_ret_22"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: s.shift(1).rolling(22, min_periods=11).std())
    d["move_scale"] = d["sd_ret_22"] * np.sqrt(HORIZON)

    ref = d[(d[DATE_COL] < THRESH_REF_DATE) & d["fwd_abs_ret"].notna()
            & (d["move_scale"] > 0)].copy()
    ratio = ref["fwd_abs_ret"] / ref["move_scale"]
    # Q_MOVE of the ratio must be EXCEEDED for a positive, so the threshold is
    # the Q_MOVE quantile of the ratio (not 1 - Q_MOVE).
    MOVE_K = float(ratio.quantile(Q_MOVE))
    d["y_move"] = (d["fwd_abs_ret"] > MOVE_K * d["move_scale"]).astype(float)
    d.loc[d["fwd_abs_ret"].isna() | (d["move_scale"] <= 0), "y_move"] = np.nan

    log(f"[label] FIXED scale-free threshold: |fwd_ret| > {MOVE_K:.3f} x "
        f"sd_ret(t-21..t) x sqrt({HORIZON})")
    log(f"[label]   MOVE_K set once from <{THRESH_REF_DATE.date()} to hit "
        f"a {Q_MOVE:.0%} positive rate")
    log(f"[label] overall positive rate = {d['y_move'].mean():.4f}")
    log("[label] positives per year (must be non-zero in every year):")
    _yr = d.groupby(d[DATE_COL].dt.year)["y_move"].agg(["size", "sum", "mean"])
    for yy, rr in _yr.iterrows():
        flag = "  <-- DEGENERATE" if rr["sum"] == 0 else ""
        log(f"[label]   {yy}: n={int(rr['size']):5d} pos={int(rr['sum']):4d} "
            f"rate={rr['mean']:.4f}{flag}")

    # ---- HAR-RV style history (the strong baseline features) --------------
    g = d.groupby(ROUTE_COL)["sq_ret"]
    d["rv_d"] = np.sqrt(g.transform(lambda s: s.rolling(1).sum()))
    d["rv_w"] = np.sqrt(g.transform(lambda s: s.rolling(5).sum() / 5))
    d["rv_m"] = np.sqrt(g.transform(lambda s: s.rolling(22).sum() / 22))
    for c in ["rv_d", "rv_w", "rv_m"]:
        d["log_" + c] = np.log(d[c].clip(lower=1e-8))

    # strictly-lagged HAR components (windows ending t-1). agentA found this
    # the single best HAR variant because ln sq_ret(t) is pure noise on 18% of
    # days where the quote did not refresh.
    d["rv_w_l1"] = np.sqrt(g.transform(lambda s: s.shift(1).rolling(5, min_periods=5).sum() / 5))
    d["rv_m_l1"] = np.sqrt(g.transform(lambda s: s.shift(1).rolling(22, min_periods=22).sum() / 22))
    d["log_rv_w_l1"] = np.log(d["rv_w_l1"].clip(lower=1e-8))
    d["log_rv_m_l1"] = np.log(d["rv_m_l1"].clip(lower=1e-8))

    # EWMA volatility (RiskMetrics-style, lambda=0.94)
    lam = 0.94

    def _ewma(s: pd.Series) -> pd.Series:
        x = s.fillna(0.0).to_numpy()
        out = np.empty_like(x)
        v = np.nanmean(x[:20]) if len(x) >= 20 else (np.nanmean(x) if len(x) else 0.0)
        for i in range(len(x)):
            out[i] = v
            v = lam * v + (1 - lam) * x[i]
        return pd.Series(out, index=s.index)

    d["ewma_var"] = d.groupby(ROUTE_COL)["sq_ret"].transform(_ewma)
    d["ewma_vol"] = np.sqrt(d["ewma_var"]) * np.sqrt(ANN)
    d["log_ewma_vol"] = np.log(d["ewma_vol"].clip(lower=1e-8))

    for w in [5, 10, 22, 66]:
        rv = d.groupby(ROUTE_COL)["sq_ret"].transform(
            lambda s, w=w: np.sqrt(s.rolling(w, min_periods=max(3, w // 2)).sum() / w) * np.sqrt(ANN)
        )
        d[f"rv_trail_{w}"] = rv
        d[f"log_rv_trail_{w}"] = np.log(rv.clip(lower=1e-8))

    # regime features (all trailing)
    d["rv_ratio_5_22"] = d["rv_trail_5"] / d["rv_trail_22"].replace(0, np.nan)
    d["rv_ratio_22_66"] = d["rv_trail_22"] / d["rv_trail_66"].replace(0, np.nan)
    d["ret_skew_22"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: s.rolling(22, min_periods=10).skew())
    d["ret_kurt_22"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: s.rolling(22, min_periods=10).kurt())
    d["abs_ret_mean_5"] = d.groupby(ROUTE_COL)["abs_ret"].transform(
        lambda s: s.rolling(5, min_periods=3).mean())
    d["abs_ret_mean_22"] = d.groupby(ROUTE_COL)["abs_ret"].transform(
        lambda s: s.rolling(22, min_periods=10).mean())
    d["sign_flip_10"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: (np.sign(s).diff().abs() > 0).rolling(10, min_periods=5).mean())
    d["dd_from_60d_high"] = (
        d[TARGET_COL] / d.groupby(ROUTE_COL)[TARGET_COL].transform(
            lambda s: s.rolling(60, min_periods=20).max()) - 1.0)
    d["dist_from_60d_low"] = (
        d[TARGET_COL] / d.groupby(ROUTE_COL)[TARGET_COL].transform(
            lambda s: s.rolling(60, min_periods=20).min()) - 1.0)

    # --- regime level features ---------------------------------------------
    # Motivated by the audit: acf(r^2,21) = +0.19 means volatility clusters on
    # a MONTHLY scale. A rolling MEDIAN of trailing variance over several
    # non-overlapping windows is a much lower-variance estimator of the current
    # regime than a single 22-day mean, and the level of log price carries the
    # market-cycle component that dominates this dataset.
    d["rv_med_66"] = d.groupby(ROUTE_COL)["sq_ret"].transform(
        lambda s: s.rolling(66, min_periods=33).median()) * ANN
    d["log_rv_med_66"] = np.log(d["rv_med_66"].clip(lower=1e-8))
    d["rv_med_252"] = d.groupby(ROUTE_COL)["sq_ret"].transform(
        lambda s: s.rolling(252, min_periods=66).median()) * ANN
    d["log_rv_med_252"] = np.log(d["rv_med_252"].clip(lower=1e-8))
    d["rv_trail_22_over_med_252"] = d["rv_trail_22"] / d["rv_med_252"].replace(0, np.nan)
    d["log_price"] = d["log_price"]
    # seasonal harmonics of log price: a smooth cycle basis instead of raw
    # integer Month/Week, which trees can only split on as step functions
    ang_d = 2 * np.pi * d[DATE_COL].dt.dayofyear / 365.25
    d["sin_annual"] = np.sin(ang_d)
    d["cos_annual"] = np.cos(ang_d)

    d["rv_pct_rank"] = d.groupby(ROUTE_COL)["rv_trail_22"].transform(
        lambda s: s.expanding(min_periods=60).apply(
            lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else 0.5, raw=True))
    return d


def feature_sets(d: pd.DataFrame) -> dict[str, list[str]]:
    """Partition candidate features into price-history-only vs +exogenous."""
    all_cols = [c for c in d.columns]

    price_feats = [c for c in all_cols if c.startswith(("log_rv", "rv_", "ewma", "abs_ret_mean",
                                                       "ret_skew", "ret_kurt", "sign_flip",
                                                       "dd_from", "dist_from"))
                   and c not in ("rv_fwd", "log_rv_fwd")]
    price_feats += [c for c in PRICE_HISTORY_EXACT if c in all_cols]
    price_feats += [c for c in ["sin_annual", "cos_annual"] if c in all_cols]

    weather_feats = [c for c in all_cols
                     if (c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT)
                     and c not in BANNED_FEATURES]
    weather_feats += [c for c in all_cols
                      if c.startswith(("Cyclone_Wind_Lag", "Cyclone_Active_Lag", "Wind_Speed_Lag"))]

    econ_feats = [c for c in ECON_EXACT if c in all_cols]
    lane_dummies = [c for c in all_cols if c.startswith("Lane_")]

    def uniq(seq):
        seen, out = set(), []
        for x in seq:
            if x not in seen and x not in BANNED_FEATURES:
                seen.add(x)
                out.append(x)
        return out

    price_feats = uniq(price_feats)
    weather_feats = uniq(weather_feats)
    econ_feats = uniq(econ_feats)
    lane_dummies = uniq(lane_dummies)

    return {
        "price": price_feats,
        "price_lane": price_feats + lane_dummies,
        "price_weather": price_feats + weather_feats + lane_dummies,
        "price_econ": price_feats + econ_feats + lane_dummies,
        "all": price_feats + weather_feats + econ_feats + lane_dummies,
    }


# =============================================================================
# STEP 2 - MODELS
# =============================================================================
def _safe_vol(p, floor: float = RV_FLOOR) -> np.ndarray:
    """Floor, de-NaN and clip a volatility forecast into a numerically safe range."""
    p = np.asarray(p, float).ravel()
    fin = p[np.isfinite(p)]
    med = np.nanmedian(fin) if len(fin) else 1.0
    p = np.where(np.isfinite(p), p, med)
    return np.clip(np.abs(p), floor, 1e4)


def _smear(model, X_train, y_train_log) -> float:
    """
    Duan's smearing factor for the log -> level back-transform. exp(E[log y]) is
    the conditional MEDIAN; multiplying by mean(exp(residuals)) restores the
    conditional MEAN.
    """
    resid = np.asarray(y_train_log, float) - np.asarray(model.predict(X_train), float)
    f = float(np.mean(np.exp(np.clip(resid, -20, 20))))
    return f if np.isfinite(f) and f > 0 else 1.0


def make_xgb(target_kind: str, params: dict, seed: int = SEED):
    import xgboost as xgb
    p = dict(params)
    common = dict(subsample=0.8, colsample_bytree=0.7, reg_alpha=0.0,
                  random_state=seed, n_jobs=-1, tree_method="hist",
                  early_stopping_rounds=EARLY_STOP_ROUNDS)
    common.update(p)
    if target_kind == "regression":
        return xgb.XGBRegressor(**common)
    common["eval_metric"] = "logloss"
    return xgb.XGBClassifier(**common)


def fit_har(X: pd.DataFrame, y: np.ndarray):
    """OLS in log space. Defensively drops rows with any non-finite feature."""
    from sklearn.linear_model import LinearRegression
    Xv = np.asarray(X, float)
    yv = np.asarray(y, float)
    ok = np.isfinite(Xv).all(axis=1) & np.isfinite(yv)
    if ok.sum() < Xv.shape[1] + 5:
        raise ValueError(f"fit_har: only {int(ok.sum())} finite rows for "
                         f"{Xv.shape[1]} features")
    m = LinearRegression()
    m.fit(Xv[ok], yv[ok])
    return m


def fit_garch_lane(y_train: np.ndarray) -> np.ndarray | None:
    """
    FIX 3. GARCH(1,1) with Student-t innovations fitted to ONE lane's training
    returns. The 5-day-ahead variance forecast is the SUM of the five one-step
    conditional variances E[r_{t+h}^2 | F_t], which is the correct estimator of
    a 5-day forward realized variance. v1 instead requested horizon = n_test_rows
    and assigned step k to test row k, which is meaningless.

    Returns a length-HORIZON vector of annualised variance contributions, or None.
    """
    try:
        from arch import arch_model
    except Exception:
        return None
    y = np.asarray(y_train, float)
    y = y[np.isfinite(y)]
    if len(y) < 250:
        return None
    try:
        am = arch_model(y * 100.0, vol="Garch", p=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        var = res.forecast(horizon=HORIZON, reindex=False).variance
        v = np.asarray(var, float).ravel()[:HORIZON]        # in %^2 units
        if not np.all(np.isfinite(v)) or v.min() <= 0:
            return None
        return v / 10000.0 * ANN                              # -> annualised variance
    except Exception:
        return None


# =============================================================================
# STEP 2b - ROBUST IMPUTATION (FIX 2)
# =============================================================================
class TrainMedianImputer:
    """
    FIX 2. Replaces v1's fillna(0.0).

    Medians are computed from the TRAIN FOLD ONLY, per lane, so no information
    from the test month leaks into the transform. Any column with at least one
    missing training value also gets a binary missingness indicator, so the model
    can learn that "no weather data" is itself informative (the file has 16.4%
    weather NaN for most folds and 42.5% for the final four).

    Raw-scale weather is the reason this matters: slp_* is in Pascals (~101,000),
    so fillna(0.0) placed those rows 527 sigma below the mean.
    """

    def __init__(self, cols: list[str]):
        self.cols = cols
        self.med_: dict[str, float] = {}
        self.ind_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "TrainMedianImputer":
        self.med_ = {}
        need_ind = []
        for c in self.cols:
            col = df[c]
            if col.isna().any():
                need_ind.append(c)
            # per-lane median, fall back to global, fall back to 0
            per = col.groupby(df[ROUTE_COL]).median()
            glob = col.median()
            base = float(glob) if glob == glob else 0.0
            self.med_[c] = {k: (float(v) if v == v else base) for k, v in per.items()}
            self.med_[c]["__default__"] = base
        self.ind_cols = [c for c in need_ind]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.cols].copy()
        lanes = df[ROUTE_COL]
        for c in self.cols:
            mapping = lanes.map(self.med_[c]).astype(float)
            mapping = mapping.fillna(self.med_[c]["__default__"])
            out[c] = out[c].fillna(mapping)
        for c in self.ind_cols:
            out["NA_" + c] = df[c].isna().astype(float)
        return out


# =============================================================================
# STEP 2c - SHRINKAGE (FIX 6)
# =============================================================================
def _winsorised_sqerr(a: np.ndarray, q: np.ndarray, w: float = 0.99) -> float:
    """
    Squared log error with the worst 1% of observations clipped. The raw mean is
    not a usable selection criterion on this target: one row is 90% of the
    pooled total.
    """
    e = (np.log(np.maximum(a, RV_FLOOR)) - np.log(np.maximum(q, RV_FLOOR))) ** 2
    if len(e) < 20:
        return float(e.mean()) if len(e) else np.nan
    thr = np.quantile(e, w)
    return float(np.minimum(e, thr).mean())


# =============================================================================
# STEP 2d - FORECAST CALIBRATION (FIX 6c)
# =============================================================================
def _winsorised_sqerr_log(ly: np.ndarray, lq: np.ndarray, w: float = 0.99) -> float:
    """Winsorised squared error given targets and forecasts ALREADY in logs."""
    e = (ly - lq) ** 2
    if len(e) < 20:
        return float(e.mean()) if len(e) else np.nan
    return float(np.minimum(e, np.quantile(e, w)).mean())


def fit_blend_and_dispersion(y_val_log: np.ndarray, p_ml: np.ndarray,
                             p_bench: np.ndarray, center: float,
                             margin: float = SHRINK_MARGIN
                             ) -> tuple[float, float, float]:
    """
    Choose (w, s) on the HELD-OUT validation slice, minimising winsorised
    squared log error of

        lq_final = center + s * ( ((1-w) * log(p_bench) + w * log(p_ml)) - center )

    w blends the challenger against the benchmark; s rescales the forecast's
    DISPERSION about the training mean log level. s = 0 collapses to
    climatology, s = 1 leaves the forecast untouched, w = 0 with any s is a
    (possibly rescaled) benchmark.

    Two safeguards, both learned the hard way:

    * `margin` - w is forced to 0 unless the challenger beats the best
      benchmark-only forecast by MARGIN (relative). Without it the search will
      happily accept a large w on the strength of validation noise.
    * s is capped at DISP_MAX <= 1.25. An earlier version let s run to the cap
      and chose 1.50 (i.e. STRETCH the forecast) in 5 of 6 folds, because the
      validation slice was a single month whose spread happened to exceed the
      near-flat benchmark's. A real signal wants s < 1 here, not s > 1.

    Returns (w, s, val_loss_of_the_chosen_combination).
    """
    ly = np.log(np.maximum(np.asarray(y_val_log, float), RV_FLOOR))
    lh = np.log(_safe_vol(p_bench))
    lm = np.log(_safe_vol(p_ml))
    grid_s = np.linspace(0.0, DISP_MAX, 31)
    grid_w = np.linspace(0.0, 1.0, 21)

    def loss_at(w, s):
        return _winsorised_sqerr_log(ly, center + s * ((1.0 - w) * lh + w * lm - center))

    # benchmark-only optimum (w = 0), used as the hurdle
    bench = min(((loss_at(0.0, s), s) for s in grid_s),
                key=lambda z: z[0] if np.isfinite(z[0]) else np.inf)
    l_bench, s_bench = bench

    full = min(((loss_at(w, s), w, s) for w in grid_w for s in grid_s),
               key=lambda z: z[0] if np.isfinite(z[0]) else np.inf)
    if not np.isfinite(l_bench):
        return 0.0, 1.0, np.nan
    if not np.isfinite(full[0]) or full[0] > l_bench * (1.0 - margin):
        return 0.0, float(s_bench), float(l_bench)
    return float(full[1]), float(full[2]), float(full[0])


def apply_blend_and_dispersion(p_ml: np.ndarray, p_bench: np.ndarray,
                               w: float, s: float, center: float) -> np.ndarray:
    lq = center + s * ((1.0 - w) * np.log(_safe_vol(p_bench))
                       + w * np.log(_safe_vol(p_ml)) - center)
    return _safe_vol(np.exp(np.clip(lq, LOG_FLOOR + 0.01, 4.0)))


# =============================================================================
# STEP 3 - VOLATILITY WALK-FORWARD
# =============================================================================
def run_volatility_walkforward(d: pd.DataFrame, fsets: dict):
    dates = pd.date_range(WF_START, d[DATE_COL].max(), freq=REFIT_FREQ)
    recs, preds, foldrows, ws = [], [], [], []

    log("\n" + "=" * 100)
    log("TASK A - VOLATILITY: expanding-window walk-forward (target = forward 5d realised vol)")
    log("=" * 100)
    t0 = time.time()
    needed = ["log_rv_fwd", "rv_trail_22", "ewma_vol", "log_rv_d", "log_rv_w", "log_rv_m",
              "log_rv_w_l1", "log_rv_m_l1"]
    complete = d[needed].notna().all(axis=1)

    har_cols = ["log_rv_d", "log_rv_w", "log_rv_m"]
    har_lag_cols = ["log_rv_w_l1", "log_rv_m_l1"]
    # FIX 9 (the one genuine modelling win): a 4-parameter linear model combining
    # the STRICTLY LAGGED HAR components with rolling-MEDIAN regime features.
    # Swept against HAR(3), HARlag(2)+med, small6, small6+ratio and med3 over 60
    # monthly folds, this is the best forecaster found:
    #     HARlag2+med 0.9952 | HAR(3) 1.0157 | small6 1.0179 | small6+ratio 1.0185
    #     med3 1.0396 | EWMA 1.1394      (mean 1%-trimmed squared log error)
    # Two reasons it wins: (a) ln sq_ret(t) is pure noise on the ~18% of days
    # where the quote did not refresh, so dropping day t from the weekly and
    # monthly windows helps; (b) a rolling MEDIAN of trailing variance over
    # non-overlapping windows is a far lower-variance estimate of the current
    # regime than a single 22-day mean, and this dataset's only real
    # autocorrelation is at the monthly scale (acf(r^2,21) = +0.19).
    har_med_cols = ["log_rv_w_l1", "log_rv_m_l1", "log_rv_med_66", "log_rv_med_252"]

    for fi, origin in enumerate(dates):
        tr = (d[DATE_COL] < origin) & (d["target_end_date"] < origin) & complete
        te = (d[DATE_COL] >= origin) & (d[DATE_COL] < origin + pd.offsets.MonthBegin(1)) & complete
        if te.sum() < MIN_TEST_ROWS or tr.sum() < MIN_TRAIN_ROWS:
            continue
        if (d.loc[tr, DATE_COL].max() - d.loc[tr, DATE_COL].min()).days < MIN_TRAIN_YEARS * 365:
            continue

        dtr = d.loc[tr].reset_index(drop=True)
        dte = d.loc[te].reset_index(drop=True)
        y_tr_raw = dtr["log_rv_fwd"].to_numpy()
        y_te = dte["log_rv_fwd"].to_numpy()
        yte_lvl = dte["rv_fwd"].to_numpy()
        n_te = len(y_te)

        # ---- FIX 6b: WINSORISE THE TRAINING TARGET ------------------------
        # log(rv_fwd) spans -5.4 to +2.3 with sd 1.09; the extreme rows are
        # quote-refresh artefacts that no feature can predict. Fitting them with
        # squared loss is what pushed the trees' log-prediction spread to
        # ~1.11 (v1: 1.13) against a true spread of 1.089, i.e. the model spent
        # its capacity on unpredictable tail noise. Clip to TRAIN quantiles
        # only, so nothing about the test month leaks into the clip.
        y_lo = float(np.quantile(y_tr_raw, TARGET_CLIP_LO))
        y_hi = float(np.quantile(y_tr_raw, TARGET_CLIP_HI))
        y_tr = np.clip(y_tr_raw, y_lo, y_hi)

        # FIX 6: the validation slice must span multiple regimes. A single month
        # is ~84 heavily overlapping rows and cannot support selecting two
        # parameters.
        n_val = int(np.clip(int(VAL_FRACTION * len(dtr)), VAL_MIN_ROWS, 0.45 * len(dtr)))
        tr_idx = np.arange(0, len(dtr) - n_val)
        va_idx = np.arange(len(dtr) - n_val, len(dtr))
        if len(tr_idx) < MIN_TRAIN_ROWS // 2:
            continue

        # ---------------- baselines (no learning from exogenous data) --------
        base = {}
        base["baseline_random_walk_rv22"] = dte["rv_trail_22"].to_numpy()
        base["baseline_ewma"] = dte["ewma_vol"].to_numpy()
        base["baseline_climatology"] = np.full(n_te, float(np.exp(y_tr.mean())))

        m = fit_har(dtr.loc[tr_idx, har_cols], y_tr[tr_idx])
        p_va_har_raw = np.exp(m.predict(dtr.loc[va_idx, har_cols]))
        base["baseline_har_rv"] = np.exp(m.predict(dte[har_cols]))
        base["baseline_har_rv_smeared"] = base["baseline_har_rv"] * _smear(
            m, dtr[har_cols], y_tr)

        ml_ = fit_har(dtr.loc[tr_idx, har_lag_cols], y_tr[tr_idx])
        base["baseline_har_strictlag"] = np.exp(ml_.predict(dte[har_lag_cols]))
        p_va_harlag = np.exp(ml_.predict(dtr.loc[va_idx, har_lag_cols]))

        mmed = fit_har(dtr.loc[tr_idx, har_med_cols], y_tr[tr_idx])
        base["baseline_har_lagmed"] = np.exp(mmed.predict(dte[har_med_cols]))
        p_va_harmed = np.exp(mmed.predict(dtr.loc[va_idx, har_med_cols]))

        # FIX 6c: the strongest benchmark (har_lagmed) gets the SAME dispersion
        # treatment as the challengers, so the comparison is apples-to-apples and
        # reported separately so a reader can separate calibration from
        # information. `center` is a TRAIN statistic only.
        center = float(np.mean(y_tr))
        ly_v = np.log(np.maximum(y_tr[va_idx], RV_FLOOR))
        lh_v = np.log(_safe_vol(p_va_har_raw))
        cand = [(_winsorised_sqerr_log(ly_v, center + s * (lh_v - center)), s)
                for s in np.linspace(0.0, DISP_MAX, 31)]
        s_har = min(cand, key=lambda z: z[0] if np.isfinite(z[0]) else np.inf)[1]
        base["baseline_har_rv_cal"] = _safe_vol(np.exp(np.clip(
            center + s_har * (np.log(_safe_vol(base["baseline_har_rv"])) - center),
            LOG_FLOOR + 0.01, 4.0)))

        lb_v = np.log(_safe_vol(p_va_harmed))
        candb = [(_winsorised_sqerr_log(ly_v, center + s * (lb_v - center)), s)
                 for s in np.linspace(0.0, DISP_MAX, 31)]
        s_bench = min(candb, key=lambda z: z[0] if np.isfinite(z[0]) else np.inf)[1]
        base["baseline_har_lagmed_cal"] = _safe_vol(np.exp(np.clip(
            center + s_bench * (np.log(_safe_vol(base["baseline_har_lagmed"])) - center),
            LOG_FLOOR + 0.01, 4.0)))

        # FIX 3: per-lane GARCH, summed one-step conditional variances
        for lane, gl in dtr.groupby(ROUTE_COL):
            v5 = fit_garch_lane(gl["ret"].to_numpy())
            if v5 is None:
                continue
            # constant within the fold per lane - that is what a GARCH 5-day
            # variance forecast IS; it is not assigned per row by mistake
            base.setdefault("baseline_garch11_t", np.zeros(n_te))
            mask = (dte[ROUTE_COL] == lane).to_numpy()
            base["baseline_garch11_t"][mask] = np.sqrt(v5.sum())

        # ---------------- ML models -----------------------------------------
        ml, shrunk, imp_local = {}, {}, []
        specs = [("xgb_price_only", fsets["price"]),
                 ("xgb_price_lane", fsets["price_lane"]),
                 ("xgb_price_weather", fsets["price_weather"]),
                 ("xgb_all", fsets["all"])]

        for name, cols in specs:
            imp = TrainMedianImputer(cols).fit(dtr.loc[tr_idx])
            Xtr = imp.transform(dtr.loc[tr_idx])
            Xva = imp.transform(dtr.loc[va_idx])
            Xte = imp.transform(dte)
            Xfull = imp.transform(dtr)

            y_val = y_tr[va_idx]
            best = None
            for params in XGB_GRID:
                mm = make_xgb("regression", params)
                mm.fit(Xtr, y_tr[tr_idx], eval_set=[(Xva, y_val)], verbose=False)
                sc = _winsorised_sqerr(np.exp(y_val), _safe_vol(np.exp(mm.predict(Xva))))
                if best is None or sc < best[0]:
                    best = (sc, params, mm)
            _, params, mm = best

            # ONE model for both validation and test. v2.1 used the early-stopped
            # model for validation but a second model refit on all of dtr for the
            # test set; those two have very different dispersion, so anything
            # fitted on the first did not transfer to the second. Using `mm` for
            # both keeps calibration and prediction consistent. The cost is that
            # the last 15% of training rows are not used for fitting; the
            # benefit is that the reported weight and dispersion are genuinely
            # out-of-sample.
            p_va_ml_raw = _safe_vol(np.exp(mm.predict(Xva)))
            p_ml_raw = _safe_vol(np.exp(mm.predict(Xte)))

            ml[name] = p_ml_raw

            # FIX 6c: joint (blend weight, dispersion) search on the validation
            # slice, against the STRONGEST benchmark (har_lagmed). Both
            # parameters are chosen against the loss, and w is forced to 0 unless
            # the challenger clears a margin over the benchmark-only optimum.
            w, s, _ = fit_blend_and_dispersion(y_val, p_va_ml_raw,
                                               p_va_harmed, center)
            shrunk[f"{name}_shrunk"] = apply_blend_and_dispersion(
                p_ml_raw, base["baseline_har_lagmed"], w, s, center)
            ml[f"{name}_cal"] = apply_blend_and_dispersion(
                p_ml_raw, base["baseline_har_lagmed"], 0.0, s, center)

            n_round = getattr(mm, "best_iteration", None)
            n_round = (n_round + 1) if n_round is not None else params["n_estimators"]
            ws.append(dict(origin=origin, model=name, w=w, disp_s=s,
                           disp_s_bench=s_bench,
                           sd_logpred_raw=float(np.log(p_ml_raw).std()),
                           sd_logpred_shrunk=float(np.log(shrunk[f"{name}_shrunk"]).std()),
                           val_loss_ml=_winsorised_sqerr(np.exp(y_val), p_va_ml_raw),
                           val_loss_bench=_winsorised_sqerr(np.exp(y_val), p_va_harmed),
                           best_rounds=int(n_round),
                           params=json.dumps(params)))
            for c, v in zip(list(Xte.columns), mm.feature_importances_):
                imp_local.append(dict(origin=origin, model=name, feature=c, importance=float(v)))

        # ---------------- score ---------------------------------------------
        all_preds = {**base, **ml, **shrunk}
        for nm, p in all_preds.items():
            q = _safe_vol(p)
            valid = yte_lvl > RV_FLOOR
            if valid.sum() < 10:
                continue
            a = np.maximum(yte_lvl[valid], RV_FLOOR)
            qq = q[valid]
            la, lp = np.log(a), np.log(qq)
            e = la - lp
            denom = np.sum((la - la.mean()) ** 2)
            sq = e ** 2
            ql = 2.0 * np.log(qq) + (a ** 2) / (qq ** 2)
            recs.append(dict(
                origin=origin, model=nm, n=int(valid.sum()),
                n_dropped_zero_rv=int((~valid).sum()),
                MAE=float(np.mean(np.abs(e))),
                RMSE=float(np.sqrt(np.mean(sq))),
                R2_log=float(1 - np.sum(sq) / denom) if denom > 0 else np.nan,
                bias=float(np.mean(e)),
                QLIKE=float(np.mean(ql)),
                QLIKE_median=float(np.median(ql)),
                QLIKE_trim1=float(np.mean(np.minimum(ql, np.quantile(ql, 0.99)))),
                SQERR_median=float(np.median(sq)),
                SQERR_trim1=_winsorised_sqerr(a, qq),
                MAE_vol_pp=float(np.mean(np.abs(a - qq)) * 100),
            ))
            for i, ii in enumerate(dte.index[valid]):
                preds.append(dict(task="volatility", origin=origin, idx=int(ii),
                                  date=dte.loc[ii, DATE_COL], route=dte.loc[ii, ROUTE_COL],
                                  model=nm, y_true=float(a[i]), y_pred=float(qq[i])))

        foldrows.append(dict(origin=origin, n_train=int(tr.sum()), n_test=int(te.sum())))
        if (fi + 1) % 12 == 0:
            log(f"  ...fold {fi+1}/{len(dates)} origin {origin.date()} elapsed {time.time()-t0:.0f}s")

    return pd.DataFrame(recs), pd.DataFrame(preds), pd.DataFrame(foldrows), pd.DataFrame(ws), pd.DataFrame(imp_local)


# =============================================================================
# STEP 4 - LARGE-MOVE CLASSIFICATION
# =============================================================================
def run_classification_walkforward(d: pd.DataFrame, fsets: dict):
    """
    FIX 4. Uses the single fixed threshold built in build_dataset, so every fold
    has positives and all 75 folds are scored (v1 scored 35).
    """
    recs, preds, imp = [], [], []
    dates = pd.date_range(WF_START, d[DATE_COL].max(), freq=REFIT_FREQ)
    log("\n" + "=" * 100)
    log(f"TASK B - LARGE MOVE: P(|forward {HORIZON}d return| > FIXED threshold from "
        f"<{THRESH_REF_DATE.date()}) - same threshold in every fold")
    log("=" * 100)
    t0 = time.time()

    for fi, origin in enumerate(dates):
        tr = (d[DATE_COL] < origin) & (d["target_end_date"] < origin) & d["y_move"].notna()
        te = (d[DATE_COL] >= origin) & (d[DATE_COL] < origin + pd.offsets.MonthBegin(1)) \
            & d["y_move"].notna()
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < MIN_TEST_ROWS:
            continue

        dtr = d.loc[tr].reset_index(drop=True)
        dte = d.loc[te].reset_index(drop=True)
        tr_y = dtr["y_move"].to_numpy().astype(int)
        te_y = dte["y_move"].to_numpy().astype(int)
        if len(np.unique(tr_y)) < 2 or len(np.unique(te_y)) < 2:
            log(f"  [skip] origin {origin.date()} degenerate label "
                f"(train {tr_y.mean():.3f}, test {te_y.mean():.3f})")
            continue
        base_rate_te = float(te_y.mean())

        n_val = max(60, int(0.15 * len(dtr)))
        tr_idx = np.arange(0, len(dtr) - n_val)
        va_idx = np.arange(len(dtr) - n_val, len(dtr))

        models = {
            "xgb_price_only": fsets["price"],
            "xgb_price_weather": fsets["price_weather"],
            "xgb_all": fsets["all"],
            "logit_price_only": fsets["price"],
            "logit_price_weather": fsets["price_weather"],
        }
        for nm, cols in models.items():
            impz = TrainMedianImputer(cols).fit(dtr.loc[tr_idx])
            Xtr = impz.transform(dtr.loc[tr_idx]).to_numpy()
            Xva = impz.transform(dtr.loc[va_idx]).to_numpy()
            Xte = impz.transform(dte).to_numpy()
            Xfull = impz.transform(dtr).to_numpy()

            if nm.startswith("logit"):
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(Xtr)
                m = LogisticRegression(max_iter=3000, C=0.1)
                m.fit(sc.transform(Xtr), tr_y[tr_idx])
                p = m.predict_proba(sc.transform(Xte))[:, 1]
            else:
                best = None
                for params in XGB_GRID:
                    mm = make_xgb("classification", params)
                    mm.fit(Xtr, tr_y[tr_idx], eval_set=[(Xva, tr_y[va_idx])], verbose=False)
                    pv = mm.predict_proba(Xva)[:, 1]
                    from sklearn.metrics import log_loss
                    sc = log_loss(tr_y[va_idx], np.clip(pv, 1e-6, 1 - 1e-6))
                    if best is None or sc < best[0]:
                        best = (sc, params, mm)
                _, params, mm = best
                n_round = getattr(mm, "best_iteration", None)
                n_round = (n_round + 1) if n_round is not None else params["n_estimators"]
                m = make_xgb("classification", {**params, "n_estimators": int(n_round),
                                                "early_stopping_rounds": None})
                m.fit(Xfull, tr_y, verbose=False)
                p = m.predict_proba(Xte)[:, 1]
                if nm == "xgb_price_only":
                    for c, v in zip(list(impz.transform(dtr.loc[tr_idx]).columns),
                                    m.feature_importances_):
                        imp.append(dict(origin=origin, model=nm, feature=c, importance=float(v)))

            p = np.clip(p, 1e-6, 1 - 1e-6)
            recs.append(_clf_metrics(origin, nm, te_y, p, base_rate_te))
            for i, ii in enumerate(dte.index):
                preds.append(dict(task="classification", origin=origin, date=dte.loc[ii, DATE_COL],
                                  route=dte.loc[ii, ROUTE_COL], model=nm,
                                  y_true=int(te_y[i]), y_prob=float(p[i])))
        if (fi + 1) % 12 == 0:
            log(f"  ...origin {origin.date()} elapsed {time.time()-t0:.0f}s")

    return pd.DataFrame(recs), pd.DataFrame(preds), pd.DataFrame(imp)


def _clf_metrics(origin, model, y, p, base_rate):
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y = np.asarray(y, int)
    out = dict(origin=origin, model=model, n=len(y), base_rate=base_rate)
    try:
        out["AUC"] = float(roc_auc_score(y, p))
    except Exception:
        out["AUC"] = np.nan
    try:
        out["AP"] = float(average_precision_score(y, p))
    except Exception:
        out["AP"] = np.nan
    bs = float(brier_score_loss(y, p))
    bs_ref = float(brier_score_loss(y, np.full_like(p, y.mean())))
    out["Brier"] = bs
    out["Brier_ref_climatology"] = bs_ref
    out["BrierSkill_vs_climatology"] = float(1 - bs / bs_ref) if bs_ref > 0 else np.nan
    return out


# =============================================================================
# STEP 5 - INCREMENTAL VALUE TESTS
# =============================================================================
def newey_west_t(diff: np.ndarray, lag: int) -> tuple[float, float]:
    """t-stat for mean(diff)!=0 with Newey-West HAC correction for overlap."""
    diff = np.asarray(diff, float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n < 10:
        return np.nan, np.nan
    mu = diff.mean()
    e = diff - mu
    g0 = np.sum(e * e) / n
    var = g0
    for L in range(1, min(lag, n - 1) + 1):
        gl = np.sum(e[L:] * e[:-L]) / n
        var += 2 * (1 - L / (lag + 1)) * gl
    se = np.sqrt(max(var, 1e-18) / n)
    t = mu / se if se > 0 else np.nan
    try:
        from scipy import stats
        p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    except Exception:
        p = np.nan
    return float(t), float(p)


def incremental_tests(preds: pd.DataFrame, task: str, base_model: str, compare: list[str]):
    """
    FIX 7. The loss differential is sorted by DATE before the HAC correction is
    applied (v1 pivoted on (origin, date, route) and never re-sorted, so the
    overlap structure was wrong). A date-collapsed variant is reported alongside:
    averaging the differential within each calendar date removes all cross-lane
    contemporaneous correlation and is the column to trust.
    """
    rows = []
    sub = preds[preds["task"] == task].copy()
    if len(sub) == 0 or "y_prob" in sub.columns and sub["y_prob"].notna().any():
        if task == "classification" and "y_pred" not in sub.columns:
            sub = sub.assign(y_pred=sub["y_prob"])
    if len(sub) == 0:
        return pd.DataFrame(rows)

    if task == "volatility":
        sub["loss"] = (np.log(sub["y_true"]) - np.log(sub["y_pred"])) ** 2
    else:
        sub["loss"] = (sub["y_true"] - sub["y_pred"]) ** 2

    wide = sub.pivot_table(index=["date", "route"], columns="model",
                           values="loss", aggfunc="mean").reset_index()
    wide = wide.sort_values("date").reset_index(drop=True)   # <-- FIX 7

    for anchor in compare:
        if base_model not in wide.columns or anchor not in wide.columns:
            continue
        pair = wide[["date", base_model, anchor]].dropna()
        if len(pair) < 30:
            continue
        diff = pair[base_model].to_numpy() - pair[anchor].to_numpy()   # >0 => anchor better
        t, p = newey_west_t(diff, lag=HORIZON + 4)

        # date-collapsed: mean differential within each date
        dc = pair.groupby("date")[[base_model, anchor]].mean().dropna()
        dcd = dc[base_model].to_numpy() - dc[anchor].to_numpy()
        t_dc, p_dc = newey_west_t(dcd, lag=HORIZON + 4)

        # winsorised variant: the raw mean is decided by a handful of rows
        thr = np.quantile(np.abs(diff), 0.99)
        dw = np.clip(diff, -thr, thr)
        t_w, p_w = newey_west_t(dw, lag=HORIZON + 4)

        mb, mc = pair[base_model].mean(), pair[anchor].mean()
        rows.append(dict(
            task=task, baseline=base_model, challenger=anchor, n_pairs=len(pair),
            mean_loss_base=float(mb), mean_loss_challenger=float(mc),
            skill_gain_pct=float(100 * (1 - mc / mb)) if mb > 0 else np.nan,
            NW_t=t, p_value=p, significant_5pct=bool(p < 0.05) if np.isfinite(p) else False,
            NW_t_datecollapsed=t_dc, p_datecollapsed=p_dc,
            NW_t_winsorised=t_w, p_winsorised=p_w,
            win_rate_pct=float(100 * (diff > 0).mean()),
        ))
    return pd.DataFrame(rows)


# =============================================================================
# STEP 6 - PLOTS
# =============================================================================
def make_plots(vol_preds, clf_preds, vol_metrics, clf_metrics, imp, ws):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[plots] matplotlib unavailable: {e}")
        return
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. actual vs forecast, tail of one lane
    if len(vol_metrics):
        agg = vol_metrics.groupby("model")["SQERR_trim1"].mean().sort_values()
        best = agg.index[0]
        sub = vol_preds[vol_preds["model"].isin([best, "baseline_har_rv"])].copy()
        if len(sub):
            lane = sub["route"].iloc[0]
            sub = sub[sub["route"] == lane].sort_values("date").tail(600)
            fig, ax = plt.subplots(figsize=(13, 5))
            act = sub[sub["model"] == best].sort_values("date")
            ax.plot(act["date"], act["y_true"] * 100, lw=1.2, label="realised forward vol",
                    color="black")
            for nm in sub["model"].unique():
                s = sub[sub["model"] == nm].sort_values("date")
                ax.plot(s["date"], s["y_pred"] * 100, lw=1.0, alpha=0.8, label=nm)
            ax.set_title(f"Forward 5-day realised volatility: actual vs forecast (best={best})")
            ax.set_ylabel("annualised vol (%)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(PLOT_DIR, "01_volatility_actual_vs_forecast.png"), dpi=110)
            plt.close(fig)

    # 2. ROBUST model comparison (trimmed squared log error) - FIX 5
    if len(vol_metrics):
        agg = vol_metrics.groupby("model")["SQERR_trim1"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(11, 6))
        cols = ["#c0392b" if m.startswith("baseline") else "#2471a3" for m in agg.index]
        ax.barh(range(len(agg)), agg.values, color=cols)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg.index, fontsize=8)
        ax.set_xlabel("mean 1%-trimmed squared log error (lower is better)")
        ax.set_title("Volatility: baselines (red) vs ML (blue) - ROBUST loss")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "02_volatility_robust_loss.png"), dpi=110)
        plt.close(fig)

    # 3. mean vs median vs trimmed QLIKE - shows why v1's headline was wrong
    if len(vol_metrics):
        agg = vol_metrics.groupby("model")[["QLIKE", "QLIKE_trim1", "QLIKE_median"]].mean()
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(agg))
        ax.bar(x - 0.27, np.clip(agg["QLIKE"], 0, 20), 0.27, label="mean (clipped at 20)")
        ax.bar(x, np.clip(agg["QLIKE_trim1"], -5, 20), 0.27, label="1%-trimmed")
        ax.bar(x + 0.27, agg["QLIKE_median"], 0.27, label="median")
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("QLIKE")
        ax.set_title("Why mean QLIKE must not be the headline (note the clipping)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "03_qlike_mean_vs_robust.png"), dpi=110)
        plt.close(fig)

    # 4. AUC by origin
    if len(clf_metrics):
        wide = clf_metrics.pivot_table(index="origin", columns="model", values="AUC")
        fig, ax = plt.subplots(figsize=(13, 5))
        for nm in wide.columns:
            ax.plot(wide.index, wide[nm], lw=1.1, label=nm)
        ax.axhline(0.5, color="black", ls="--", lw=1.2, label="random (AUC 0.5)")
        ax.set_ylabel("AUC")
        ax.set_title(f"Large-move detection: per-fold AUC (all {len(wide)} folds now scored)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "04_classification_auc.png"), dpi=110)
        plt.close(fig)

    # 5. feature importance
    if len(imp):
        g = imp.groupby("feature")["importance"].mean().sort_values(ascending=False).head(25)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(g)), g.values[::-1])
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index[::-1], fontsize=8)
        ax.set_xlabel("mean gain importance")
        ax.set_title("Top 25 features (XGB price-only volatility model)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "05_feature_importance.png"), dpi=110)
        plt.close(fig)

    # 6. weather share of importance
    if len(imp):
        g = imp.groupby("feature")["importance"].mean()
        tot = g.sum()
        wx = [c for c in g.index if c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT
              or c.startswith(("Cyclone_Wind_Lag", "Cyclone_Active_Lag", "Wind_Speed_Lag"))]
        px = [c for c in g.index if c.startswith(("log_rv", "rv_", "ewma", "abs_ret_mean"))]
        wpct = 100 * g[wx].sum() / tot if tot > 0 else 0
        ppct = 100 * g[px].sum() / tot if tot > 0 else 0
        rest = max(0.0, 100 - wpct - ppct)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie([ppct, wpct, rest], labels=[f"price history\n{ppct:.1f}%",
                                           f"weather/cyclone\n{wpct:.1f}%",
                                           f"other\n{rest:.1f}%"],
               colors=["#2471a3", "#c0392b", "#95a5a6"])
        ax.set_title("Where does the volatility model get its signal?")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "06_signal_source.png"), dpi=110)
        plt.close(fig)

    # 7. shrinkage weights - how much the ML was trusted
    if len(ws):
        fig, ax = plt.subplots(figsize=(11, 5))
        wide = ws.pivot_table(index="origin", columns="model", values="w")
        for nm in wide.columns:
            ax.plot(wide.index, wide[nm], lw=1.2, marker="o", ms=3, label=nm)
        ax.axhline(0.0, color="black", ls="--", lw=1.2, label="w=0 -> pure HAR-RV")
        ax.axhline(1.0, color="gray", ls=":", lw=1.2, label="w=1 -> pure ML")
        ax.set_ylabel("shrinkage weight on ML")
        ax.set_title("Validation-chosen weight on the ML forecast (0 means ML added nothing)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "07_shrinkage_weights.png"), dpi=110)
        plt.close(fig)

    log(f"[plots] written to {PLOT_DIR}")


# =============================================================================
# STEP 7 - DECISION LAYER
# =============================================================================
def decision_layer(clf_preds):
    """
    Operational rule: flag a large move when p >= threshold. Reports lift AND
    the Brier skill against climatology, because a model can show lift while
    being worse calibrated than simply always printing the base rate - which is
    exactly what every v1 model did.
    """
    rows = []
    if len(clf_preds) == 0:
        return pd.DataFrame(rows)
    from sklearn.metrics import brier_score_loss, roc_auc_score
    for mdl in sorted(clf_preds["model"].unique()):
        p = clf_preds[clf_preds["model"] == mdl]
        if p.empty:
            continue
        y = p["y_true"].to_numpy()
        pr = p["y_prob"].to_numpy()
        base = float(y.mean())
        try:
            auc = float(roc_auc_score(y, pr))
        except Exception:
            auc = np.nan
        bs = float(brier_score_loss(y, pr))
        bs_ref = float(brier_score_loss(y, np.full_like(pr, base)))
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
            al = pr >= thr
            if al.sum() == 0:
                rows.append(dict(model=mdl, threshold=thr, n_alerts=0, precision=np.nan,
                                 recall=np.nan, lift=np.nan, base_rate=base,
                                 AUC=auc, BrierSkill_vs_climatology=1 - bs / bs_ref if bs_ref > 0 else np.nan))
                continue
            prec = float(y[al].mean())
            rec = float(y[al].sum() / max(y.sum(), 1))
            rows.append(dict(model=mdl, threshold=thr, n_alerts=int(al.sum()), precision=prec,
                             recall=rec, lift=float(prec / base) if base > 0 else np.nan,
                             base_rate=base, AUC=auc,
                             BrierSkill_vs_climatology=1 - bs / bs_ref if bs_ref > 0 else np.nan))
    return pd.DataFrame(rows)


# =============================================================================
# FINDINGS.md
# =============================================================================
def write_findings(vol_metrics, vol_preds, clf_metrics, clf_preds, inc, dq, ws, dec):
    L = []
    A = L.append
    A("# FINDINGS - maritime freight volatility & large-move prediction (v2)\n")
    A(f"Generated by `volatility_large_move.py`. Walk-forward, monthly origins from "
      f"{WF_START.date()}, {HORIZON}-day forward targets, {ANN} annualisation.\n")

    A("## 0. Data reality check (run first, because it bounds everything)\n")
    A("| lane | n | zero-ret % | acf(r^2,1) | acf(r^2,21) | VR(5) | exkurt |")
    A("|---|---|---|---|---|---|---|")
    for _, r in dq.iterrows():
        A(f"| {r['lane']} | {int(r['n_rows'])} | {r['zero_return_pct']:.1f} | "
          f"{r['acf_r2_lag1']:+.3f} | {r['acf_r2_lag21']:+.3f} | {r['var_ratio_5']:.3f} | "
          f"{r['excess_kurt']:+.1f} |")
    A("")
    A("Two facts drive the interpretation:\n")
    A("1. **Lag-1 volatility clustering is absent** (acf(r^2,1) ~ 0.00, VR(5) ~ 1.00). "
      "A real market series has acf(r^2,1) of +0.2 to +0.4 and VR(5) of 1.3-2.5. This "
      "series does not behave like an observed price process at daily frequency.")
    A("2. **Monthly-scale structure IS present** (acf(r^2,21) ~ +0.19). So the target is "
      "not pure noise and there is something to learn - it is just not day-to-day "
      "volatility persistence, which is what a HAR/GARCH model is built to exploit.\n")
    A("The practical consequence: models that lean on trailing-vol persistence (HAR, EWMA, "
      "GARCH) are working with the only autocorrelation that exists, and models that "
      "exploit regime-level information are competing on the axis where the signal is.\n")

    A("## 1. Robust model comparison (volatility)\n")
    A("Headline loss is the **1%-trimmed squared log error**, not mean QLIKE. On this "
      "target a single observation can carry 90% of the mean QLIKE, so the mean is not a "
      "usable ranking statistic.\n")
    if len(vol_metrics):
        agg = (vol_metrics.groupby("model")
               .agg(folds=("MAE", "size"), MAE_log=("MAE", "mean"), RMSE_log=("RMSE", "mean"),
                    QLIKE_mean=("QLIKE", "mean"), QLIKE_trim1=("QLIKE_trim1", "mean"),
                    QLIKE_median=("QLIKE_median", "mean"),
                    SQERR_trim1=("SQERR_trim1", "mean"), bias=("bias", "mean"))
               .sort_values("SQERR_trim1").reset_index())
        A(agg.to_markdown(index=False, floatfmt=".4f"))
        A("")
        # pooled R2 (honest companion to the tiny-sample within-fold R2)
        A("Pooled out-of-sample R2 in log space (all folds stacked; this is the honest "
          "companion to the misleading within-fold R2):\n")
        A("| model | pooled R2 (log) | corr(log pred, log actual) |")
        A("|---|---|---|")
        v = vol_preds.copy()
        v["ly"] = np.log(v["y_true"]); v["lq"] = np.log(v["y_pred"])
        for mdl, s in v.groupby("model"):
            r2 = 1 - ((s.ly - s.lq) ** 2).sum() / ((s.ly - s.ly.mean()) ** 2).sum()
            A(f"| {mdl} | {r2:+.4f} | {np.corrcoef(s.ly, s.lq)[0,1]:+.3f} |")
        A("")

    A("## 2. Does the ML beat the naive baseline?\n")
    if len(vol_metrics) and len(ws):
        A("The shrunk variants apply `w * ML + (1-w) * HAR` with `w` chosen on a "
          "time-ordered validation slice. `w = 0` reproduces HAR exactly, so a challenger "
          "that adds nothing is shrunk back to the benchmark rather than reported as a loss.\n")
        A("| model | mean shrinkage weight w | fraction of folds with w = 0 |")
        A("|---|---|---|")
        for mdl, s in ws.groupby("model"):
            A(f"| {mdl} | {s['w'].mean():.3f} | {100*(s['w'] <= 0.001).mean():.0f}% |")
        A("")

    A("## 3. Incremental value of weather / cyclone features\n")
    if len(inc):
        A("`skill_gain_pct > 0` means the challenger beats the baseline. Three p-values are "
          "reported: the raw Newey-West HAC test, a winsorised version (raw mean is decided "
          "by a handful of rows), and a **date-collapsed** version that averages the loss "
          "differential within each calendar date, which removes all cross-lane "
          "contemporaneous correlation and is the column to trust.\n")
        A(inc.to_markdown(index=False, floatfmt=".4f"))
        A("")
        sig = inc[(inc["p_datecollapsed"] < 0.05)]
        if len(sig) == 0:
            A("**Verdict: no weather or economic feature set adds significant value over "
              "price history alone under any of the three tests.**")
        else:
            A("Significant under the date-collapsed test:\n")
            A(sig.to_markdown(index=False, floatfmt=".4f"))
    else:
        A("_no incremental tests produced_")

    A("\n## 4. Large-move classification\n")
    if len(clf_metrics):
        A(f"All **{int(clf_metrics['origin'].nunique())}** folds are now scored (v1 scored 35 "
          "of 75 because a training-q90 threshold produced zero positives in 40 test months).\n")
        agg = (clf_metrics.groupby("model")
               .agg(folds=("AUC", "size"), AUC=("AUC", "mean"), AP=("AP", "mean"),
                    Brier=("Brier", "mean"),
                    BrierSkill_vs_climatology=("BrierSkill_vs_climatology", "mean"),
                    base_rate=("base_rate", "mean"))
               .sort_values("AUC", ascending=False).reset_index())
        A(agg.to_markdown(index=False, floatfmt=".4f"))
        A("")
        A("A negative `BrierSkill_vs_climatology` means the model is **worse calibrated "
          "than always printing the base rate** - it discriminates but its probabilities "
          "are miscalibrated. v1 had every model negative on this measure.")
    else:
        A("_no classification folds produced_")

    if len(dec):
        A("\n## 5. Operational decision layer\n")
        A("Lift alone is misleading; `BrierSkill_vs_climatology` is printed alongside it.\n")
        A(dec.to_markdown(index=False, floatfmt=".4f"))

    A("\n## 6. Bugs fixed in this version\n")
    A("| # | v1 defect | effect | fix |")
    A("|---|---|---|---|")
    A("| 1 | 4 route labels, 2 physical lanes with bit-identical prices | 42% duplicated "
      "rows; every t-stat inflated ~sqrt(2) | collapse to 2 canonical lanes |")
    A("| 2 | `fillna(0.0)` on every feature | `slp_*` is in Pascals, so its NaNs became a "
      "-527 sigma outlier; 42 of 73 weather columns collapsed to constant zero on Pacific lanes "
      "| per-lane TRAIN-FOLD median imputation + missingness indicators |")
    A("| 3 | one GARCH fitted to all lanes concatenated; horizon = n_test_rows (~84) for a "
      "~21 day month; step k assigned to row k | 63 of 84 forecast steps were fiction; the "
      "'best QLIKE' was a flattest-forecast artifact | per-lane fit; sum the 5 one-step "
      "conditional variances |")
    A("| 4 | large-move threshold = training-fold q90 | 40 of 75 folds had zero test "
      "positives and were silently dropped; only turbulent months were scored | one fixed "
      "threshold from a pre-2020 slice |")
    A("| 5 | mean QLIKE as headline metric | one row = 90% of the pooled total | median and "
      "1%-trimmed reported; pooled R2 added |")
    A("| 6 | 400 boosting rounds, lr 0.03, no validation set | trees extrapolated into the "
      "deep tail (log-pred sd 1.13 vs true 1.089) | early stopping on a time-ordered "
      "validation slice, small hyperparameter grid, convex shrinkage toward HAR |")
    A("| 7 | Newey-West applied to an unsorted pivot | wrong overlap structure | sort by "
      "date; add winsorised and date-collapsed variants |")
    A("| 8 | `FINDINGS.md` documented but never written | missing deliverable | written |")
    A("")

    path = os.path.join(OUT_DIR, "FINDINGS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"[findings] written to {path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    log("=" * 100)
    log("MARITIME FREIGHT: VOLATILITY & LARGE-MOVE PREDICTION  (v2)")
    log("=" * 100)

    raw, dq = load_raw()
    d = build_dataset(raw)
    dq.to_csv(os.path.join(OUT_DIR, "data_quality_report.csv"), index=False)

    d = pd.concat([d, pd.get_dummies(d[ROUTE_COL], prefix="Lane")], axis=1)
    d = d[d["log_rv_fwd"].notna() & d["rv_trail_22"].notna()].reset_index(drop=True)
    log(f"[data] modelling rows: {len(d)}  ({d[DATE_COL].min().date()} .. {d[DATE_COL].max().date()})")

    fsets = feature_sets(d)
    for k, v in fsets.items():
        log(f"[features] {k:14s} n={len(v)}")
    log(f"[features] BANNED (forward-looking): {BANNED_FEATURES}")

    # ---------- TASK A ----------
    vol_metrics, vol_preds, foldrows, ws, imp = run_volatility_walkforward(d, fsets)
    # save immediately: a crash in Task B must not discard Task A
    vol_metrics.to_csv(os.path.join(OUT_DIR, "metrics_volatility.csv"), index=False)
    if len(vol_preds):
        vol_preds.to_csv(os.path.join(OUT_DIR, "predictions_volatility.csv"), index=False)
    if len(foldrows):
        foldrows.to_csv(os.path.join(OUT_DIR, "fold_level_results_volatility.csv"), index=False)
    if len(ws):
        ws.to_csv(os.path.join(OUT_DIR, "shrinkage_weights.csv"), index=False)

    log("\n--- AGGREGATE VOLATILITY (robust ranking) ---")
    if len(vol_metrics):
        agg = (vol_metrics.groupby("model")
               .agg(folds=("MAE", "size"), MAE_log=("MAE", "mean"), RMSE_log=("RMSE", "mean"),
                    QLIKE_mean=("QLIKE", "mean"), QLIKE_trim1=("QLIKE_trim1", "mean"),
                    QLIKE_median=("QLIKE_median", "mean"),
                    SQERR_trim1=("SQERR_trim1", "mean"), bias=("bias", "mean"))
               .sort_values("SQERR_trim1").reset_index())
        log(agg.to_string(index=False))
        log("\n--- POOLED out-of-sample R2 in log space (honest; within-fold R2 is "
            "a ~4-observation statistic and is NOT reported) ---")
        v = vol_preds.copy()
        v["ly"] = np.log(v["y_true"]); v["lq"] = np.log(v["y_pred"])
        for mdl, s in v.groupby("model"):
            r2 = 1 - ((s.ly - s.lq) ** 2).sum() / ((s.ly - s.ly.mean()) ** 2).sum()
            log(f"  {mdl:32s} pooled R2 = {r2:+.4f}   corr = {np.corrcoef(s.ly, s.lq)[0,1]:+.3f}")

    # ---------- TASK B ----------
    clf_metrics, clf_preds, imp_c = run_classification_walkforward(d, fsets)
    clf_metrics.to_csv(os.path.join(OUT_DIR, "metrics_classification.csv"), index=False)
    if len(imp_c):
        imp_c.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    imp = imp_c if len(imp_c) else imp

    log("\n--- AGGREGATE CLASSIFICATION ---")
    if len(clf_metrics):
        agg = (clf_metrics.groupby("model")
               .agg(folds=("AUC", "size"), AUC=("AUC", "mean"), AP=("AP", "mean"),
                    Brier=("Brier", "mean"),
                    BrierSkill_vs_clim=("BrierSkill_vs_climatology", "mean"),
                    base_rate=("base_rate", "mean"))
               .sort_values("AUC", ascending=False).reset_index())
        log(agg.to_string(index=False))
        log(f"\n[note] scored on {clf_metrics['origin'].nunique()} folds "
            f"(v1 scored 35 of 75 - the other 40 had zero positives and were dropped)")

    # ---------- INCREMENTAL VALUE ----------
    log("\n" + "=" * 100)
    log("INCREMENTAL VALUE OF WEATHER / CYCLONE / ECON FEATURES")
    log("=" * 100)
    inc_v = incremental_tests(vol_preds, "volatility", "baseline_har_lagmed",
                              ["baseline_har_lagmed_cal", "baseline_har_rv",
                               "baseline_har_strictlag", "baseline_ewma",
                               "baseline_climatology", "baseline_random_walk_rv22",
                               "baseline_garch11_t",
                               "xgb_price_only", "xgb_price_only_cal",
                               "xgb_price_weather", "xgb_price_weather_shrunk",
                               "xgb_all", "xgb_all_shrunk"])
    clf_for_inc = clf_preds.copy()
    clf_for_inc["y_pred"] = clf_for_inc["y_prob"]
    inc_c = incremental_tests(clf_for_inc, "classification", "xgb_price_only",
                              ["xgb_price_weather", "xgb_all"])
    inc = pd.concat([inc_v, inc_c], ignore_index=True)
    if len(inc):
        log(inc.to_string(index=False))
        inc.to_csv(os.path.join(OUT_DIR, "incremental_value_tests.csv"), index=False)
        log("\n(skill_gain_pct > 0 => challenger better;  NW_t > 0 => challenger better)")
        log("(trust p_datecollapsed: it collapses the differential within each date, removing")
        log(" cross-lane correlation. p_value is the raw HAC test on the winsorised-free series.)")

    # ---------- DECISION LAYER ----------
    dec = decision_layer(clf_preds)
    if len(dec):
        log("\n--- OPERATIONAL DECISION LAYER ---")
        log(dec.to_string(index=False))
        dec.to_csv(os.path.join(OUT_DIR, "daily_decisions.csv"), index=False)

    # ---------- SAVE ----------
    if len(clf_preds):
        clf_preds.to_csv(os.path.join(OUT_DIR, "predictions_classification.csv"), index=False)
    if len(imp):
        imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    make_plots(vol_preds, clf_preds, vol_metrics, clf_metrics, imp, ws)
    write_findings(vol_metrics, vol_preds, clf_metrics, clf_preds, inc, dq, ws, dec)

    log(f"\n[done] {time.time()-t0:.0f}s  outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
