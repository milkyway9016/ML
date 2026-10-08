"""
================================================================================
MARITIME FREIGHT: VOLATILITY & LARGE-MOVE PREDICTION
================================================================================
Re-frames the forecasting problem away from the price LEVEL (near-unit-root,
so persistence wins trivially) and toward two targets where weather/cyclone
activity has a plausible physical mechanism:

  TASK A - VOLATILITY:  predict forward 5-day realized volatility of log returns
  TASK B - LARGE MOVE:  predict whether |forward 5-day cumulative return|
                        exceeds a per-route threshold

Both targets are evaluated with an expanding-window walk-forward backtest and
compared against genuine domain baselines:

  volatility : random-walk (trailing RV), EWMA, HAR-RV, GARCH(1,1)
  large move : base rate (climatology), trailing-RV logistic, HAR-RV logistic

The central question is whether weather / cyclone features add INCREMENTAL
value over a price-history-only model. That is tested with a paired
Diebold-Mariano style statistic on the per-observation loss differentials.

Outputs (all under D:/testing/outputs):
  metrics_volatility.csv, metrics_classification.csv
  fold_level_results.csv, predictions.csv
  feature_importance.csv, daily_decisions.csv
  plots/*.png
  FINDINGS.md

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

# target definition
HORIZON = 5          # forward window in trading days for the volatility target
ANN = 252            # annualisation factor
Q_MOVE = 0.90        # quantile for the "large move" label (computed on TRAIN ONLY)

# walk-forward
WF_START = pd.Timestamp("2020-01-01")
REFIT_FREQ = "MS"    # re-fit at the start of every month
MIN_TRAIN_YEARS = 2.0

SEED = 42
N_ESTIMATORS = 400

# columns that are forward-looking or contemporaneous and must never be features
BANNED_FEATURES = [
    "Cyclone_Wind_Forecast3d",
    "Cyclone_Wind_Forecast7d",
    "Cyclone_Active_Forecast3d",
    "Wind_Speed_Forecast7d",
    "Rolling_7d_Cyclone_Days",   # includes day t
    "Price_Volatility_7d",       # built from pct_change() including day t
    "Price_Volatility_30d",      # built from pct_change() including day t
]

# route dummy columns are created later; these prefixes identify raw weather
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

RNG = np.random.default_rng(SEED)


def log(*a):
    print(*a, flush=True)


# =============================================================================
# STEP 0 - LOAD & BASIC SANITY
# =============================================================================
def load_raw() -> pd.DataFrame:
    if not os.path.exists(SRC_DATA):
        sys.exit(f"FATAL: source dataset not found at {SRC_DATA}")
    df = pd.read_csv(SRC_DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ROUTE_COL, DATE_COL]).reset_index(drop=True)
    log(f"[load] {df.shape} from {SRC_DATA}")
    log(f"[load] range {df[DATE_COL].min().date()} .. {df[DATE_COL].max().date()}")
    return df


# =============================================================================
# STEP 1 - BUILD TARGETS + HONEST FEATURES
# =============================================================================
def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Everything here respects a single rule: a row dated t may use only
    information available at or before t. Targets are strictly forward.
    """
    d = df.copy()

    # ---- returns ------------------------------------------------------
    d["log_price"] = np.log(d[TARGET_COL].clip(lower=1.0))
    d["ret"] = d.groupby(ROUTE_COL)["log_price"].diff()
    d["abs_ret"] = d["ret"].abs()
    d["sq_ret"] = d["ret"] ** 2

    # ---- TARGET A: forward realised volatility, per route -------------
    # RV_{t->t+H} computed from returns at t+1 .. t+H  (strictly future)
    fwd_sq = (
        d.groupby(ROUTE_COL)["sq_ret"]
        .transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1)))
    )
    d["rv_fwd"] = np.sqrt(fwd_sq) * np.sqrt(ANN)
    d["log_rv_fwd"] = np.log(d["rv_fwd"].clip(lower=1e-8))

    # ---- TARGET B: forward cumulative move ---------------------------
    fwd_ret = (
        d.groupby(ROUTE_COL)["ret"]
        .transform(lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1)))
    )
    d["fwd_ret"] = fwd_ret
    d["fwd_abs_ret"] = fwd_ret.abs()
    # Labels use returns through t+HORIZON, so the final label observation is
    # HORIZON rows ahead within each route. Walk-forward training must stop
    # before this date, not merely before the label row's date.
    d["target_end_date"] = d.groupby(ROUTE_COL)[DATE_COL].shift(-HORIZON)

    # ---- HAR-RV style history (the strong baseline features) ----------
    g = d.groupby(ROUTE_COL)["sq_ret"]
    d["rv_d"] = np.sqrt(g.transform(lambda s: s.rolling(1).sum()))
    d["rv_w"] = np.sqrt(g.transform(lambda s: s.rolling(5).sum()) / 5)
    d["rv_m"] = np.sqrt(g.transform(lambda s: s.rolling(22).sum()) / 22)
    # log versions (HAR works far better in logs)
    for c in ["rv_d", "rv_w", "rv_m"]:
        d["log_" + c] = np.log(d[c].clip(lower=1e-8))

    # EWMA volatility (RiskMetrics-style, lambda=0.94) - a real baseline
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

    # trailing RV of several windows (random-walk baseline + features)
    for w in [5, 10, 22, 66]:
        rv = d.groupby(ROUTE_COL)["sq_ret"].transform(
            lambda s, w=w: np.sqrt(s.rolling(w, min_periods=max(3, w // 2)).sum() / w) * np.sqrt(ANN)
        )
        d[f"rv_trail_{w}"] = rv
        d[f"log_rv_trail_{w}"] = np.log(rv.clip(lower=1e-8))

    # volatility-of-volatility / regime features (all trailing)
    d["rv_ratio_5_22"] = d["rv_trail_5"] / d["rv_trail_22"].replace(0, np.nan)
    d["rv_ratio_22_66"] = d["rv_trail_22"] / d["rv_trail_66"].replace(0, np.nan)
    d["ret_skew_22"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: s.rolling(22, min_periods=10).skew()
    )
    d["ret_kurt_22"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: s.rolling(22, min_periods=10).kurt()
    )
    d["abs_ret_mean_5"] = d.groupby(ROUTE_COL)["abs_ret"].transform(
        lambda s: s.rolling(5, min_periods=3).mean()
    )
    d["abs_ret_mean_22"] = d.groupby(ROUTE_COL)["abs_ret"].transform(
        lambda s: s.rolling(22, min_periods=10).mean()
    )
    d["sign_flip_10"] = d.groupby(ROUTE_COL)["ret"].transform(
        lambda s: (np.sign(s).diff().abs() > 0).rolling(10, min_periods=5).mean()
    )
    # drawdown-style regime indicator (trailing)
    d["dd_from_60d_high"] = (
        d[TARGET_COL] / d.groupby(ROUTE_COL)[TARGET_COL].transform(
            lambda s: s.rolling(60, min_periods=20).max()
        ) - 1.0
    )
    d["dist_from_60d_low"] = (
        d[TARGET_COL] / d.groupby(ROUTE_COL)[TARGET_COL].transform(
            lambda s: s.rolling(60, min_periods=20).min()
        ) - 1.0
    )
    # realised-vol percentiles vs own history (trailing, expanding)
    d["rv_pct_rank"] = d.groupby(ROUTE_COL)["rv_trail_22"].transform(
        lambda s: s.expanding(min_periods=60).apply(
            lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else 0.5, raw=True
        )
    )
    return d


def feature_sets(d: pd.DataFrame) -> dict[str, list[str]]:
    """Partition candidate features into price-history-only vs +exogenous."""
    all_cols = [c for c in d.columns]

    price_feats = [c for c in all_cols if c.startswith(("log_rv", "rv_", "ewma", "abs_ret_mean",
                                                       "ret_skew", "ret_kurt", "sign_flip",
                                                       "dd_from", "dist_from"))
                  and c not in ("rv_fwd", "log_rv_fwd")]
    price_feats += [c for c in PRICE_HISTORY_EXACT if c in all_cols]

    weather_feats = [c for c in all_cols
                     if (c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT)
                     and c not in BANNED_FEATURES]
    # lagged weather variants are safe by construction
    weather_feats += [c for c in all_cols
                      if c.startswith(("Cyclone_Wind_Lag", "Cyclone_Active_Lag", "Wind_Speed_Lag"))]

    econ_feats = [c for c in ECON_EXACT if c in all_cols]
    route_dummies = [c for c in all_cols if c.startswith("Route_") and c not in WEATHER_EXACT + ECON_EXACT]

    # de-duplicate, preserve order, drop banned
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
    route_dummies = uniq(route_dummies)

    return {
        "price": price_feats,
        "price_route": price_feats + route_dummies,
        "price_weather": price_feats + weather_feats + route_dummies,
        "price_econ": price_feats + econ_feats + route_dummies,
        "all": price_feats + weather_feats + econ_feats + route_dummies,
    }


# =============================================================================
# STEP 2 - MODELS
# =============================================================================
# Realised volatility is occasionally EXACTLY zero (a 5-day window in which the
# quoted price never changed - mostly the US West Coast lanes in 2018). A zero
# forecast target makes log-RV and QLIKE undefined, so every level forecast is
# floored before scoring. 0.05% annualised is far below any economically
# meaningful freight volatility and is applied identically to all models.
RV_FLOOR = 5e-4


def _safe_vol(p, floor: float = RV_FLOOR) -> np.ndarray:
    """Floor, de-NaN and clip a volatility forecast into a numerically safe range."""
    p = np.asarray(p, float).ravel()
    med = np.nanmedian(p[np.isfinite(p)]) if np.any(np.isfinite(p)) else 1.0
    p = np.where(np.isfinite(p), p, med)
    return np.clip(np.abs(p), floor, 1e4)


def _smear(model, X_train, y_train_log) -> float:
    """
    Duan's smearing factor for the log -> level back-transform.

    exp(E[log y]) is the CONDITIONAL MEDIAN of y, not its mean, so exponentiating a
    log-space forecast is biased low by roughly the residual variance. Multiplying by
    mean(exp(residuals)) restores the conditional MEAN. Without this, every
    log-modelled baseline looks like it systematically under-forecasts volatility,
    which is an artefact of the transform rather than a property of the model.
    """
    resid = np.asarray(y_train_log, float) - np.asarray(model.predict(X_train), float)
    f = float(np.mean(np.exp(resid)))
    return f if np.isfinite(f) and f > 0 else 1.0


def make_xgb(target_kind: str, seed: int = SEED):
    import xgboost as xgb
    if target_kind == "regression":
        return xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.0, reg_lambda=2.0, random_state=seed, n_jobs=1,
            tree_method="hist",
        )
    return xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_lambda=2.0, random_state=seed, n_jobs=1, tree_method="hist",
        eval_metric="logloss",
    )


def fit_har(X: pd.DataFrame, y: np.ndarray):
    """HAR-RV: plain OLS on log realised-variance components (per route if available)."""
    from sklearn.linear_model import LinearRegression
    m = LinearRegression()
    m.fit(X, y)
    return m


def fit_garch(y_train: np.ndarray, y_test_len: int):
    """GARCH(1,1) with constant mean on training returns; forecast vol for test block."""
    try:
        from arch import arch_model
    except Exception:
        return None
    try:
        am = arch_model(y_train * 100.0, vol="Garch", p=1, q=1, dist="normal", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=y_test_len, reindex=False)
        var = fc.variance.to_numpy().ravel()[:y_test_len]
        return np.sqrt(var) / 100.0 * np.sqrt(ANN)
    except Exception:
        return None


# =============================================================================
# STEP 3 - THE HONEST BASELINES
# =============================================================================
def run_volatility_walkforward(d: pd.DataFrame, fsets: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range(WF_START, d[DATE_COL].max(), freq=REFIT_FREQ)
    recs, preds = [], []
    foldrows = []

    log("\n" + "=" * 100)
    log("TASK A - VOLATILITY: expanding-window walk-forward (target = forward 5d realised vol)")
    log("=" * 100)
    t0 = time.time()

    for fi, origin in enumerate(dates):
        tr = (d[DATE_COL] < origin) & (d["target_end_date"] < origin)
        te = (d[DATE_COL] >= origin) & (d[DATE_COL] < origin + pd.offsets.MonthBegin(1))
        if te.sum() < 20:
            continue
        if (d.loc[tr, DATE_COL].max() - d.loc[tr, DATE_COL].min()).days < MIN_TRAIN_YEARS * 365:
            continue
        # need a valid target AND a complete feature row (warm-up NaNs are unavoidable)
        needed = ["log_rv_fwd", "rv_trail_22", "ewma_vol", "log_rv_d", "log_rv_w", "log_rv_m"]
        complete = d[needed].notna().all(axis=1)
        tr = tr & complete
        te = te & complete
        if tr.sum() < 500 or te.sum() < 20:
            continue

        y_tr = d.loc[tr, "log_rv_fwd"].to_numpy()
        y_te = d.loc[te, "log_rv_fwd"].to_numpy()
        yte_lvl = d.loc[te, "rv_fwd"].to_numpy()
        n_te = len(y_te)

        # ---------------- baselines (no learning from exogenous data) --------
        base = {}
        base["baseline_random_walk_rv22"] = d.loc[te, "rv_trail_22"].to_numpy()
        base["baseline_ewma"] = d.loc[te, "ewma_vol"].to_numpy()
        # HAR-RV as a level forecast via log-model.
        # rv_d/w/m use same-day returns (known at t) -> legitimate.
        # Reported twice: plain exp() (conditional median) and Duan-smeared (conditional mean).
        har_cols = ["log_rv_d", "log_rv_w", "log_rv_m"]
        m = fit_har(d.loc[tr, har_cols], y_tr)
        base["baseline_har_rv"] = np.exp(m.predict(d.loc[te, har_cols]))
        base["baseline_har_rv_smeared"] = base["baseline_har_rv"] * _smear(
            m, d.loc[tr, har_cols], y_tr)

        # GARCH on training returns, forecasting forward over the test block
        g_fc = fit_garch(d.loc[tr, "ret"].fillna(0).to_numpy(), n_te)
        if g_fc is not None and len(g_fc) == n_te:
            base["baseline_garch11"] = g_fc

        # ---------------- ML models -----------------------------------------
        ml = {}
        for name, cols in [
            ("xgb_price_only", fsets["price"]),
            ("xgb_price_route", fsets["price_route"]),
            ("xgb_price_weather", fsets["price_weather"]),
            ("xgb_all", fsets["all"]),
            ("har_plus_weather", fsets["price_weather"]),
        ]:
            if name == "har_plus_weather":
                mm = fit_har(d.loc[tr, cols].fillna(0.0), y_tr)
                ml[name] = np.exp(mm.predict(d.loc[te, cols].fillna(0.0)))
                ml["har_plus_weather_smeared"] = ml[name] * _smear(
                    mm, d.loc[tr, cols].fillna(0.0), y_tr)
            else:
                mm = make_xgb("regression")
                mm.fit(d.loc[tr, cols].fillna(0.0), y_tr)
                ml[name] = np.exp(mm.predict(d.loc[te, cols].fillna(0.0)))
            if not np.all(np.isfinite(ml[name])):
                ml[name] = np.where(np.isfinite(ml[name]), ml[name], np.nanmedian(ml[name][np.isfinite(ml[name])]) if np.any(np.isfinite(ml[name])) else 1.0)

        # ---------------- score ---------------------------------------------
        for nm, p in {**base, **ml}.items():
            p = _safe_vol(p)            # drop rows whose realised volatility is exactly zero: log loss undefined
            valid = yte_lvl > RV_FLOOR
            if valid.sum() < 10:
                continue
            a = np.maximum(yte_lvl[valid], RV_FLOOR)
            q = p[valid]
            la, lp = np.log(a), np.log(q)
            e = la - lp
            denom = np.sum((la - la.mean()) ** 2)
            recs.append(dict(
                origin=origin, model=nm, n=int(valid.sum()),
                n_dropped_zero_rv=int((~valid).sum()),
                MAE=float(np.mean(np.abs(e))),
                RMSE=float(np.sqrt(np.mean(e ** 2))),
                R2_log=float(1 - np.sum(e ** 2) / denom) if denom > 0 else np.nan,
                bias=float(np.mean(e)),
                # QLIKE on the variance scale, computed stably via logs
                QLIKE=float(np.mean(2.0 * np.log(q) + (a ** 2) / (q ** 2))),
                # level-space error, in annualised vol percentage points
                MAE_vol_pp=float(np.mean(np.abs(a - q)) * 100),
            ))
            for i, idx in enumerate(d.index[te][valid]):
                preds.append(dict(task="volatility", origin=origin, idx=idx,
                                  date=d.loc[idx, DATE_COL], route=d.loc[idx, ROUTE_COL],
                                  model=nm, y_true=float(a[i]), y_pred=float(q[i])))

        foldrows.append(dict(origin=origin, n_train=int(tr.sum()), n_test=int(te.sum())))
        if (fi + 1) % 12 == 0:
            log(f"  ...fold {fi+1}/{len(dates)} origin {origin.date()} elapsed {time.time()-t0:.0f}s")

    return pd.DataFrame(recs), pd.DataFrame(preds)


# =============================================================================
# STEP 4 - LARGE-MOVE CLASSIFICATION
# =============================================================================
def run_classification_walkforward(d: pd.DataFrame, fsets: dict):
    """Per-route threshold from TRAIN ONLY; walk-forward; blocked test months."""
    recs, preds, imp = [], [], []
    dates = pd.date_range(WF_START, d[DATE_COL].max(), freq=REFIT_FREQ)
    log("\n" + "=" * 100)
    log(f"TASK B - LARGE MOVE: P(|forward {HORIZON}d return| > train q{Q_MOVE:.2f}) per route")
    log("=" * 100)
    t0 = time.time()

    for origin in dates:
        tr = ((d[DATE_COL] < origin) & (d["target_end_date"] < origin)
              & d["fwd_abs_ret"].notna())
        te = ((d[DATE_COL] >= origin) & (d[DATE_COL] < origin + pd.offsets.MonthBegin(1))
              & d["fwd_abs_ret"].notna())
        if tr.sum() < 500 or te.sum() < 20:
            continue
        # per-route thresholds learned ONLY on training rows
        thr = d.loc[tr].groupby(ROUTE_COL)["fwd_abs_ret"].quantile(Q_MOVE)
        tr_y = (d.loc[tr, "fwd_abs_ret"] > d.loc[tr, ROUTE_COL].map(thr)).astype(int)
        te_y = (d.loc[te, "fwd_abs_ret"] > d.loc[te, ROUTE_COL].map(thr)).astype(int)
        if tr_y.nunique() < 2 or te_y.nunique() < 2:
            continue
        base_rate_te = float(te_y.mean())

        models = {
            "xgb_price_only": fsets["price"],
            "xgb_price_weather": fsets["price_weather"],
            "xgb_all": fsets["all"],
            "logit_price_only": fsets["price"],
            "logit_price_weather": fsets["price_weather"],
        }
        for nm, cols in models.items():
            Xtr = d.loc[tr, cols].fillna(0.0).to_numpy()
            Xte = d.loc[te, cols].fillna(0.0).to_numpy()
            if nm.startswith("logit"):
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                sc = StandardScaler().fit(Xtr)
                m = LogisticRegression(max_iter=2000, C=0.5)
                m.fit(sc.transform(Xtr), tr_y)
                p = m.predict_proba(sc.transform(Xte))[:, 1]
            else:
                m = make_xgb("classification")
                m.fit(Xtr, tr_y.to_numpy())
                p = m.predict_proba(Xte)[:, 1]
                if nm == "xgb_all":
                    for c, v in zip(cols, m.feature_importances_):
                        imp.append(dict(origin=origin, model=nm, feature=c, importance=float(v)))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            recs.append(_clf_metrics(origin, nm, te_y.to_numpy(), p, base_rate_te))
            for i, idx in enumerate(d.index[te]):
                preds.append(dict(task="classification", origin=origin, date=d.loc[idx, DATE_COL],
                                  route=d.loc[idx, ROUTE_COL], model=nm,
                                  y_true=int(te_y.iloc[i]), y_prob=float(p[i])))
        if (dates.get_loc(origin) + 1) % 12 == 0:
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
    out["BrierSkill_vs_baserate"] = float(1 - bs / bs_ref) if bs_ref > 0 else np.nan
    out["Brier_ref"] = bs_ref
    return out


# =============================================================================
# STEP 5 - INCREMENTAL VALUE TESTS (the actual scientific question)
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
    """For each pair (base, compare) compute the loss differential over the SAME rows."""
    rows = []
    sub = preds[preds["task"] == task].copy()
    for anchor in compare:
        if task == "volatility":
            sub["loss"] = (np.log(sub["y_true"]) - np.log(sub["y_pred"])) ** 2
        else:
            sub["loss"] = (sub["y_true"] - sub["y_pred"]) ** 2
        wide = sub.pivot_table(index=["origin", "date", "route"], columns="model",
                               values="loss", aggfunc="mean")
        if base_model not in wide.columns or anchor not in wide.columns:
            continue
        pair = wide[[base_model, anchor]].dropna()
        diff = pair[base_model].to_numpy() - pair[anchor].to_numpy()  # >0 => anchor better
        t, p = newey_west_t(diff, lag=HORIZON + 4)
        rows.append(dict(
            task=task, baseline=base_model, challenger=anchor, n_pairs=len(pair),
            mean_loss_base=float(pair[base_model].mean()),
            mean_loss_challenger=float(pair[anchor].mean()),
            skill_gain_pct=float(100 * (1 - pair[anchor].mean() / pair[base_model].mean()))
            if pair[base_model].mean() > 0 else np.nan,
            NW_t=t, p_value=p, significant_5pct=bool(p < 0.05) if np.isfinite(p) else False,
        ))
    return pd.DataFrame(rows)


# =============================================================================
# STEP 6 - PLOTS
# =============================================================================
def make_plots(vol_preds, clf_preds, vol_metrics, clf_metrics, imp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log(f"[plots] matplotlib unavailable: {e}")
        return
    os.makedirs(PLOT_DIR, exist_ok=True)

    # 1. realised vs forecast volatility (best model, last 400 rows of one route)
    best = (vol_metrics.groupby("model")["QLIKE"].mean().sort_values().index[0]
            if len(vol_metrics) else None)
    if best is not None:
        sub = vol_preds[vol_preds["model"].isin([best, "baseline_har_rv"])].copy()
        sub = sub[sub["route"] == sub["route"].iloc[0]].sort_values("date").tail(600)
        fig, ax = plt.subplots(figsize=(13, 5))
        act = sub[sub["model"] == best].sort_values("date")
        ax.plot(act["date"], act["y_true"] * 100, lw=1.2, label="realised forward vol", color="black")
        for nm in sub["model"].unique():
            s = sub[sub["model"] == nm].sort_values("date")
            ax.plot(s["date"], s["y_pred"] * 100, lw=1.0, alpha=0.8, label=nm)
        ax.set_title(f"Forward 5-day realised volatility: actual vs forecast ({best})")
        ax.set_ylabel("annualised vol (%)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "01_volatility_actual_vs_forecast.png"), dpi=110)
        plt.close(fig)

    # 2. model comparison bar chart (QLIKE)
    if len(vol_metrics):
        agg = vol_metrics.groupby("model")["QLIKE"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        cols = ["#c0392b" if m.startswith("baseline") else "#2471a3" for m in agg.index]
        ax.barh(range(len(agg)), agg.values, color=cols)
        ax.set_yticks(range(len(agg)))
        ax.set_yticklabels(agg.index, fontsize=8)
        ax.set_xlabel("mean QLIKE (lower is better)")
        ax.set_title("Volatility: baselines (red) vs ML models (blue)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "02_volatility_qlike.png"), dpi=110)
        plt.close(fig)

    # 3. per-fold skill: challenger vs HAR
    if len(vol_metrics):
        wide = vol_metrics.pivot_table(index="origin", columns="model", values="QLIKE")
        fig, ax = plt.subplots(figsize=(13, 5))
        for nm in wide.columns:
            if nm.startswith("xgb") or nm == "har_plus_weather":
                ax.plot(wide.index, wide[nm] / wide["baseline_har_rv"], lw=1.0, label=nm)
        ax.axhline(1.0, color="black", ls="--", lw=1.2, label="HAR-RV baseline")
        ax.set_ylabel("QLIKE relative to HAR-RV")
        ax.set_title("Per-fold relative QLIKE vs HAR-RV (<1 means ML beats HAR)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "03_relative_qlike_by_fold.png"), dpi=110)
        plt.close(fig)

    # 4. AUC by origin
    if len(clf_metrics):
        wide = clf_metrics.pivot_table(index="origin", columns="model", values="AUC")
        fig, ax = plt.subplots(figsize=(13, 5))
        for nm in wide.columns:
            ax.plot(wide.index, wide[nm], lw=1.1, label=nm)
        ax.axhline(0.5, color="black", ls="--", lw=1.2, label="random (AUC 0.5)")
        ax.set_ylabel("AUC")
        ax.set_title("Large-move detection: per-fold AUC")
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
        ax.set_title("Top 25 features (XGB all-features volatility model)")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "05_feature_importance.png"), dpi=110)
        plt.close(fig)

    # 6. weather-only incremental check: importance split
    if len(imp):
        g = imp.groupby("feature")["importance"].mean()
        tot = g.sum()
        weather_like = [c for c in g.index if c.startswith(WEATHER_PREFIXES)
                        or c in WEATHER_EXACT or c.startswith(("Cyclone_Wind_Lag", "Cyclone_Active_Lag", "Wind_Speed_Lag"))]
        wpct = 100 * g[weather_like].sum() / tot if tot > 0 else 0
        price_like = [c for c in g.index if c.startswith(("log_rv", "rv_", "ewma", "abs_ret_mean"))]
        ppct = 100 * g[price_like].sum() / tot if tot > 0 else 0
        fig, ax = plt.subplots(figsize=(6, 6))
        rest = max(0.0, 100 - wpct - ppct)
        ax.pie([ppct, wpct, rest], labels=[f"price history\n{ppct:.1f}%",
                                           f"weather/cyclone\n{wpct:.1f}%",
                                           f"other\n{rest:.1f}%"],
               autopct=None, colors=["#2471a3", "#c0392b", "#95a5a6"])
        ax.set_title("Where does the volatility model get its signal?")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "06_signal_source.png"), dpi=110)
        plt.close(fig)

    log(f"[plots] written to {PLOT_DIR}")


# =============================================================================
# STEP 7 - DECISION LAYER (does it turn into money / useful action?)
# =============================================================================
def decision_layer(d, clf_preds, vol_preds):
    """
    Turn the probabilities into an operational rule and measure precision, which
    is what a freight desk actually cares about: when we flag a large move, how
    often is there one?
    """
    rows = []
    if len(clf_preds) == 0:
        return pd.DataFrame(rows)
    p = clf_preds[clf_preds["model"] == "xgb_all"].copy()
    if p.empty:
        return pd.DataFrame(rows)
    for thr in [0.5, 0.6, 0.7, 0.8]:
        alerts = p[p["y_prob"] >= thr]
        if len(alerts) == 0:
            rows.append(dict(threshold=thr, n_alerts=0, precision=np.nan, recall=np.nan,
                             lift=np.nan, base_rate=float(p["y_true"].mean())))
            continue
        prec = float(alerts["y_true"].mean())
        rec = float(alerts["y_true"].sum() / max(p["y_true"].sum(), 1))
        base = float(p["y_true"].mean())
        rows.append(dict(threshold=thr, n_alerts=len(alerts), precision=prec, recall=rec,
                         lift=float(prec / base) if base > 0 else np.nan, base_rate=base))
    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    log("=" * 100)
    log("MARITIME FREIGHT: VOLATILITY & LARGE-MOVE PREDICTION")
    log("=" * 100)

    raw = load_raw()
    d = build_dataset(raw)

    # route dummies
    dm = pd.get_dummies(d[ROUTE_COL], prefix="Route")
    d = pd.concat([d, dm], axis=1)

    # a target is required
    d = d[d["log_rv_fwd"].notna() & d["rv_trail_22"].notna()].reset_index(drop=True)
    log(f"[data] modelling rows: {len(d)}  ({d[DATE_COL].min().date()} .. {d[DATE_COL].max().date()})")

    fsets = feature_sets(d)
    for k, v in fsets.items():
        log(f"[features] {k:14s} n={len(v)}")
    log(f"[features] BANNED (forward-looking): {BANNED_FEATURES}")

    # ---------- TASK A ----------
    vol_metrics, vol_preds = run_volatility_walkforward(d, fsets)
    vol_metrics.to_csv(os.path.join(OUT_DIR, "metrics_volatility.csv"), index=False)

    log("\n--- AGGREGATE VOLATILITY (mean across folds) ---")
    if len(vol_metrics):
        agg = (vol_metrics.groupby("model")
               .agg(folds=("QLIKE", "size"), MAE_log=("MAE", "mean"), RMSE_log=("RMSE", "mean"),
                    R2_log=("R2_log", "mean"), QLIKE=("QLIKE", "mean"),
                    QLIKE_med=("QLIKE", "median"), bias=("bias", "mean"),
                    MAE_vol_pp=("MAE_vol_pp", "mean"), dropped=("n_dropped_zero_rv", "sum"))
               .sort_values("QLIKE").reset_index())
        log(agg.to_string(index=False))
        har = agg.loc[agg["model"] == "baseline_har_rv", "QLIKE"]
        if len(har):
            log(f"\n[HAR-RV reference QLIKE] {har.iloc[0]:.6f}")

    # ---------- TASK B ----------
    clf_metrics, clf_preds, imp = run_classification_walkforward(d, fsets)
    clf_metrics.to_csv(os.path.join(OUT_DIR, "metrics_classification.csv"), index=False)
    if len(imp):
        imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

    log("\n--- AGGREGATE CLASSIFICATION (mean across folds) ---")
    if len(clf_metrics):
        agg = (clf_metrics.groupby("model")
               .agg(folds=("AUC", "size"), AUC=("AUC", "mean"), AP=("AP", "mean"),
                    Brier=("Brier", "mean"), BrierSkill=("BrierSkill_vs_baserate", "mean"),
                    base_rate=("base_rate", "mean"))
               .sort_values("AUC", ascending=False).reset_index())
        log(agg.to_string(index=False))

    # ---------- INCREMENTAL VALUE ----------
    log("\n" + "=" * 100)
    log("INCREMENTAL VALUE OF WEATHER / CYCLONE FEATURES")
    log("=" * 100)
    inc_v = incremental_tests(vol_preds, "volatility", "xgb_price_route",
                              ["xgb_price_weather", "xgb_all", "baseline_har_rv", "baseline_ewma"])
    inc_c = incremental_tests(clf_preds.assign(y_pred=lambda x: x["y_prob"]), "classification",
                              "xgb_price_only", ["xgb_price_weather", "xgb_all"])
    inc = pd.concat([inc_v, inc_c], ignore_index=True)
    if len(inc):
        log(inc.to_string(index=False))
        inc.to_csv(os.path.join(OUT_DIR, "incremental_value_tests.csv"), index=False)
        log("\n(mean_loss_base > mean_loss_challenger and NW_t > 0 => challenger genuinely better;"
            "\nsignificant_5pct uses Newey-West corrected p-values to account for overlapping targets)")

    # ---------- DECISION LAYER ----------
    dec = decision_layer(d, clf_preds, vol_preds)
    if len(dec):
        log("\n--- OPERATIONAL DECISION LAYER (flag a large move when p >= threshold) ---")
        log(dec.to_string(index=False))
        dec.to_csv(os.path.join(OUT_DIR, "daily_decisions.csv"), index=False)

    # ---------- SAVE ----------
    if len(vol_preds):
        vol_preds.to_csv(os.path.join(OUT_DIR, "predictions_volatility.csv"), index=False)
    if len(clf_preds):
        clf_preds.to_csv(os.path.join(OUT_DIR, "predictions_classification.csv"), index=False)
    if len(vol_metrics):
        vol_metrics.to_csv(os.path.join(OUT_DIR, "fold_level_results_volatility.csv"), index=False)
    if len(clf_metrics):
        clf_metrics.to_csv(os.path.join(OUT_DIR, "fold_level_results_classification.csv"), index=False)

    make_plots(vol_preds, clf_preds, vol_metrics, clf_metrics, imp)

    log(f"\n[done] {time.time()-t0:.0f}s  outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
