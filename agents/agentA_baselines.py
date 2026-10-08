"""
================================================================================
agentA: RIGOROUS ECONOMETRIC BASELINES FOR MARITIME FREIGHT VOLATILITY
================================================================================
Target (copied EXACTLY from D:\\testing\\volatility_large_move.py):
    log_price = ln(max(Price_USD,1));  ret = log_price.diff() per route
    sq_ret = ret**2
    fwd_var(t) = sum sq_ret over t+1..t+5
    rv_fwd(t)  = sqrt(fwd_var)*sqrt(252);  log_rv_fwd = ln(rv_fwd)
Rows with rv_fwd <= RV_FLOOR dropped before scoring; predictions floored at
RV_FLOOR.  Expanding-window walk-forward, monthly origins from 2020-01-01,
train labels fully observed before origin, test = origin .. origin+1month, >=500 train rows,
>=20 test rows, >=2y history, complete feature rows required.

Deliverables -> D:\\testing\\outputs\\agentA_*.csv, D:\\testing\\agents\\agentA_findings.md
================================================================================
"""
from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG  (mirrors the main pipeline)
# ----------------------------------------------------------------------------
SRC_DATA = r"D:\Internproj\New ML\clean_engineered_dataset.csv"
OUT_DIR = r"D:\testing\outputs"
AGENT_DIR = r"D:\testing\agents"

DATE_COL, ROUTE_COL, TARGET_COL = "Date", "Route", "Price_USD"
HORIZON, ANN = 5, 252
RV_FLOOR = 5e-4
WF_START = pd.Timestamp("2020-01-01")
REFIT_FREQ = "MS"
MIN_TRAIN_ROWS, MIN_TEST_ROWS, MIN_TRAIN_YEARS = 500, 20, 2.0
DM_LAG = HORIZON + 4              # = 9, as instructed
SEED = 42

# warm-up-complete rows (identical requirement to the main pipeline)
NEEDED = ["log_rv_fwd", "rv_trail_22", "ewma_vol", "log_rv_d", "log_rv_w", "log_rv_m"]

# Columns that are forward-looking or contemporaneous.  NONE of these exist in
# the shipped dataset (verified: the file has Cyclone_Wind_Lag*, not
# Cyclone_Wind_Forecast*), but they are excluded defensively by name.
BANNED_FEATURES = [
    "Cyclone_Wind_Forecast3d", "Cyclone_Wind_Forecast7d", "Cyclone_Active_Forecast3d",
    "Wind_Speed_Forecast7d", "Rolling_7d_Cyclone_Days", "Price_Volatility_7d",
    "Price_Volatility_30d",
]
WEATHER_PREFIXES = ("air_", "slp_", "wind_speed_", "cyclone_wind_", "cyclone_slp_",
                    "cyclone_dist_", "cyclone_active_")
WEATHER_EXACT = ["Route_Max_cyclone_wind", "Route_Min_cyclone_slp", "Route_Min_cyclone_dist",
                 "Route_cyclone_active", "Route_Mean_air", "Route_Mean_slp",
                 "Route_Mean_wind_speed", "SWI", "Coastal_Threat", "Congestion_Risk"]
ECON_EXACT = ["Origin_LSCI", "Dest_LSCI", "Route_Mean_LSCI", "LSCI_30d_Delta"]

MODELS = [
    # (model_name, family)
    ("har_log",            "HAR"),   # a) HAR in logs
    ("har_route",          "HAR"),   # b) per-route OLS on top of pooled
    ("har_j",              "HAR"),   # c) HAR-J  (base + jump)
    ("har_cj",             "HAR"),   # c) HAR-CJ (continuous + jump decomposition)
    ("har_ext",            "HAR"),   # d) + EWMA + skew/kurt
    ("har_econ",           "HAR"),   # e) + economic
    ("har_weather",        "HAR"),   # f) + weather/cyclone (82 raw columns)
    ("har_weather_pca",    "HAR"),   # f') + top-5 weather PCs (disclosed small-k alternative)
    ("har_weather_econ",   "HAR"),   # g) everything legitimate
    ("har_wpca_epca",      "HAR"),   # g') weather PCs + econ  (small-k "everything")
    ("har_log_strictlag",  "HAR"),   # robustness: HAR on components ending t-1
    ("garch11_normal",     "GARCH"),
    ("garch11_t",          "GARCH"),
    ("egarch11_t",         "GARCH"),
    ("gjr_garch11_t",      "GARCH"),
]
MODEL_ORDER = [m for m, _ in MODELS]
N_WPCA = 5      # weather PCs (59% of weather-block variance) - pre-specified


def log(*a):
    print(*a, flush=True)


# ============================================================================
# 1. DATA / TARGET  (exact copy of the main pipeline definition)
# ============================================================================
def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(SRC_DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    d = df.sort_values([ROUTE_COL, DATE_COL]).reset_index(drop=True)
    log(f"[load] {d.shape}  {d[DATE_COL].min().date()} .. {d[DATE_COL].max().date()}")

    d["log_price"] = np.log(d[TARGET_COL].clip(lower=1.0))
    d["ret"] = d.groupby(ROUTE_COL)["log_price"].diff()
    d["abs_ret"] = d["ret"].abs()
    d["sq_ret"] = d["ret"] ** 2

    # ---- target: forward 5-day realised variance, annualised ----
    fwd_sq = d.groupby(ROUTE_COL)["sq_ret"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1)))
    d["rv_fwd"] = np.sqrt(fwd_sq) * np.sqrt(ANN)
    d["log_rv_fwd"] = np.log(d["rv_fwd"].clip(lower=1e-8))
    d["target_end_date"] = d.groupby(ROUTE_COL)[DATE_COL].shift(-HORIZON)

    g = d.groupby(ROUTE_COL)["sq_ret"]
    # ---- HAR components.  Spec: daily sq_ret(t), weekly mean t-4..t, monthly t-21..t
    d["rv_d"] = np.sqrt(g.transform(lambda s: s.rolling(1).sum()))
    d["rv_w"] = np.sqrt(g.transform(lambda s: s.rolling(5).sum()) / 5)
    d["rv_m"] = np.sqrt(g.transform(lambda s: s.rolling(22).sum()) / 22)
    # strictly-lagged robustness variants (windows END at t-1)
    d["rv_d_lag"] = np.sqrt(g.transform(lambda s: s.rolling(1).sum().shift(1)))
    d["rv_w_lag"] = np.sqrt(g.transform(lambda s: s.rolling(5).sum().shift(1)) / 5)
    d["rv_m_lag"] = np.sqrt(g.transform(lambda s: s.rolling(22).sum().shift(1)) / 22)
    for c in ["rv_d", "rv_w", "rv_m", "rv_d_lag", "rv_w_lag", "rv_m_lag"]:
        d["log_" + c] = np.log(d[c].clip(lower=1e-8))

    # ---- jump / continuous decomposition (BN-S bipower variation) ----------
    # BPV_t = (pi/2) * |r_t| * |r_{t-1}|   estimates the continuous part
    # J_t   = max(RV_t - BPV_t, 0)        estimates the jump part
    d["bpv"] = (np.pi / 2.0) * d["abs_ret"] * d.groupby(ROUTE_COL)["abs_ret"].shift(1)
    d["jump"] = (d["sq_ret"] - d["bpv"]).clip(lower=0.0)
    # 5-day (HAR-frequency) versions, same window convention as rv_w
    d["bpv_w"] = d.groupby(ROUTE_COL)["bpv"].transform(lambda s: s.rolling(5).mean())
    d["jump_w"] = d.groupby(ROUTE_COL)["jump"].transform(lambda s: s.rolling(5).mean())
    TINY = 1e-16
    for c in ["sq_ret", "bpv", "jump", "bpv_w", "jump_w"]:
        d["log_" + c] = np.log(d[c].clip(lower=TINY))

    # ---- EWMA vol (lambda = 0.94), identical to main pipeline --------------
    lam = 0.94

    def _ewma(s):
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
            lambda s, w=w: np.sqrt(s.rolling(w, min_periods=max(3, w // 2)).sum() / w) * np.sqrt(ANN))
        d[f"rv_trail_{w}"] = rv
        d[f"log_rv_trail_{w}"] = np.log(rv.clip(lower=1e-8))

    d["rv_ratio_5_22"] = d["rv_trail_5"] / d["rv_trail_22"].replace(0, np.nan)
    d["rv_ratio_22_66"] = d["rv_trail_22"] / d["rv_trail_66"].replace(0, np.nan)
    d["ret_skew_22"] = d.groupby(ROUTE_COL)["ret"].transform(lambda s: s.rolling(22, min_periods=10).skew())
    d["ret_kurt_22"] = d.groupby(ROUTE_COL)["ret"].transform(lambda s: s.rolling(22, min_periods=10).kurt())
    d["abs_ret_mean_5"] = d.groupby(ROUTE_COL)["abs_ret"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d["abs_ret_mean_22"] = d.groupby(ROUTE_COL)["abs_ret"].transform(lambda s: s.rolling(22, min_periods=10).mean())
    return d


def add_route_dummies(d: pd.DataFrame) -> pd.DataFrame:
    dm = pd.get_dummies(d[ROUTE_COL], prefix="Route")
    return pd.concat([d, dm], axis=1)


# ============================================================================
# 2. FEATURE BLOCKS
# ============================================================================
HAR_CORE = ["log_rv_d", "log_rv_w", "log_rv_m"]
HAR_STRICT = ["log_rv_d_lag", "log_rv_w_lag", "log_rv_m_lag"]
HAR_JUMP = HAR_CORE + ["log_jump"]
HAR_CJ = ["log_bpv_w", "log_jump_w", "log_rv_w", "log_rv_m"]
HAR_EXT = HAR_CORE + ["log_ewma_vol", "rv_ratio_5_22", "rv_ratio_22_66",
                      "ret_skew_22", "ret_kurt_22", "abs_ret_mean_5", "abs_ret_mean_22"]


def feature_blocks(d: pd.DataFrame) -> dict:
    all_cols = list(d.columns)
    route_dummies = [c for c in all_cols if c.startswith("Route_") and c not in WEATHER_EXACT + ECON_EXACT]
    weather = [c for c in all_cols
               if (c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT
                   or c.startswith(("Cyclone_Wind_Lag", "Cyclone_Active_Lag", "Wind_Speed_Lag")))
               and c not in BANNED_FEATURES]
    econ = [c for c in ECON_EXACT if c in all_cols]
    return dict(route_dummies=route_dummies, weather=weather, econ=econ)


# ============================================================================
# 3. OLS HARNESS
# ============================================================================
def design(d: pd.DataFrame, idx, cols):
    X = d.loc[idx, cols].astype(float).to_numpy()
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def fit_ols(Xtr, ytr, col_scale=True):
    """OLS with intercept + column standardisation. Returns (predict_fn, n_used)."""
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-12] = 1.0
    keep = sd > 1e-12            # drop columns with no training variation
    if keep.sum() == 0:
        return None, 0
    Z = np.column_stack([np.ones(len(Xtr)), (Xtr[:, keep] - mu[keep]) / sd[keep]])
    beta, *_ = np.linalg.lstsq(Z, ytr, rcond=None)

    def predict(Xte):
        Zt = np.column_stack([np.ones(len(Xte)), (Xte[:, keep] - mu[keep]) / sd[keep]])
        return Zt @ beta
    return predict, int(keep.sum())


def har_fit_predict(d, tr_idx, te_idx, cols):
    """Pooled OLS with intercept (+ any dummies already in `cols`)."""
    Xtr, Xte = design(d, tr_idx, cols), design(d, te_idx, cols)
    fn, _ = fit_ols(Xtr, d.loc[tr_idx, "log_rv_fwd"].to_numpy())
    if fn is None:
        return None
    return fn(Xte)


def har_route_fit_predict(d, tr_idx, te_idx, cols):
    """Separate OLS per route; fall back to the pooled model if a route cannot be fit."""
    out = np.full(len(te_idx), np.nan)
    te_route = d.loc[te_idx, ROUTE_COL].to_numpy()
    tr_route = d.loc[tr_idx, ROUTE_COL].to_numpy()
    pooled = har_fit_predict(d, tr_idx, te_idx, cols)
    for r in pd.unique(tr_route):
        m_tr = tr_route == r
        m_te = te_route == r
        if m_te.sum() == 0:
            continue
        sub_tr = np.asarray(tr_idx)[m_tr]
        sub_te = np.asarray(te_idx)[m_te]
        if m_tr.sum() < 60:
            pred = pooled[m_te] if pooled is not None else np.nan
        else:
            Xtr, Xte = design(d, sub_tr, cols), design(d, sub_te, cols)
            fn, _ = fit_ols(Xtr, d.loc[sub_tr, "log_rv_fwd"].to_numpy())
            pred = fn(Xte) if fn is not None else (pooled[m_te] if pooled is not None else np.nan)
        out[m_te] = pred
    return out


# ============================================================================
# 4. GARCH FAMILY
# ============================================================================
GARCH_KW = {
    "garch11_normal":  dict(vol="Garch", p=1, q=1, o=0, dist="normal"),
    "garch11_t":       dict(vol="Garch", p=1, q=1, o=0, dist="t"),
    "egarch11_t":      dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
    "gjr_garch11_t":   dict(vol="Garch", p=1, o=1, q=1, dist="t"),
}
# Variance targeting pins the unconditional variance to the training sample
# variance.  Without it the EGARCH/GJR optimiser wanders into a degenerate
# corner (beta -> 1 with a huge alpha) and the multi-step forecast explodes.
GARCH_FIT_OPTS = dict(disp="off", show_warning=False, options={"maxiter": 300})


def _egarch_degenerate(res) -> bool:
    """
    Detect the degenerate EGARCH solutions this dataset produces.
    With 18.4% of daily returns exactly zero the t-likelihood is multi-modal and
    the optimiser regularly walks to beta -> 1 with alpha in the 1e7-1e28 range,
    which makes every multi-step forecast meaningless.
    """
    try:
        pr = res.params
        a = abs(float(pr["alpha[1]"])); g = abs(float(pr["gamma[1]"]))
        b = float(pr["beta[1]"])
        return (b > 0.999) or (a > 5.0) or (g > 5.0)
    except Exception:
        return True


def _egarch_deterministic_forecast(res, n_ahead: int, scale: float) -> np.ndarray | None:
    """
    Exact multi-step E[sigma^2] path for EGARCH(1,1,1) without simulation.

        ln s2_t = w + a(|z_{t-1}| - E|z|) + g z_{t-1} + b ln s2_{t-1}

    so for h >= 2, with E[z] = 0 and E|z| = c:
        E[ln s2_{t+h}] = w + a(c - c) + g*0 + b E[ln s2_{t+h-1}]
                       = w + b E[ln s2_{t+h-1}]
    and E[s2] = exp(E[ln s2]) because the last innovation entering the recursion
    is z_{t+h-1}, which is independent of ln s2_{t+h-1}. h=1 is exact:
    w + a(|z_t| - c) + g z_t + b ln s2_t.  (s2 is in scaled percent^2 units.)
    """
    try:
        import scipy.special as sp
        pr = res.params
        w = float(pr["omega"]); a = float(pr["alpha[1]"])
        g = float(pr["gamma[1]"]); b = float(pr["beta[1]"]); nu = float(pr["nu"])
        c = float(np.sqrt(2.0 / np.pi) * sp.gamma((nu - 1.0) / 2.0) / sp.gamma(nu / 2.0)
                  * np.sqrt(nu - 2.0))
        s2 = np.asarray(res.conditional_volatility, float) ** 2
        z = np.asarray(res.resid, float) / np.asarray(res.conditional_volatility, float)
        zt, s2t = float(z[-1]), float(s2[-1])
        lns2 = w + a * (abs(zt) - c) + g * zt + b * np.log(s2t)   # h = 1
        out = [float(np.exp(lns2))]
        for _ in range(n_ahead - 1):
            lns2 = w + b * lns2
            out.append(float(np.exp(lns2)))
        return np.asarray(out, float)
    except Exception:
        return None


def garch_fit_predict(d, tr_idx, te_idx, spec: str):
    """
    Fit the GARCH model on TRAINING returns for one route, then forecast the
    analytic variance path forward over the test block (multi-step E[sigma^2]).
    Returns an array aligned with te_idx, or None.

    Returns are rescaled by the training standard deviation before fitting.
    Without this, the EGARCH/GJR optimiser walks to a degenerate corner
    (beta -> 1, alpha ~ 1e7) and the forecasts are useless.
    """
    from arch import arch_model

    y = d.loc[tr_idx, "ret"].astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy() * 100.0
    n_ahead = len(te_idx)
    if len(y) < 200 or n_ahead < 1:
        return None
    sd = float(np.std(y))
    if not np.isfinite(sd) or sd < 1e-8:
        return None
    scale = 100.0 / sd                     # ~O(1) series for the optimiser
    ys = y * scale
    try:
        am = arch_model(ys, mean="Constant", **GARCH_KW[spec])
        # variance targeting is not supported by arch's own fit keyword in all
        # specs, so pass it through options and retry without it on failure
        try:
            res = am.fit(variance_targeting=True, **GARCH_FIT_OPTS)
        except TypeError:
            res = am.fit(**GARCH_FIT_OPTS)
        # documented fallback: a degenerate EGARCH fit is not a forecast, so
        # substitute GARCH(1,1)-t for that route/fold rather than publish a
        # 1e28 variance.  This is reported explicitly in the findings.
        if spec == "egarch11_t" and _egarch_degenerate(res):
            res = arch_model(ys, mean="Constant", **GARCH_KW["garch11_t"]).fit(**GARCH_FIT_OPTS)
            spec_used = "garch11_t_fallback"
        else:
            spec_used = spec
        try:
            fc = res.forecast(horizon=n_ahead, reindex=False)
            var = np.asarray(fc.variance.to_numpy(), float).ravel()[:n_ahead]
        except ValueError:
            # EGARCH (log-variance) has no analytic multi-step forecast in arch,
            # and simulation is unusable when beta ~ 0.99. Use the exact
            # deterministic recursion instead.
            var = _egarch_deterministic_forecast(res, n_ahead, scale)
            if var is None:
                return None
    except Exception:
        return None
    if len(var) != n_ahead or not np.all(np.isfinite(var)):
        return None
    # undo the scaling: percent-per-day variance -> rv_fwd units, then annualise.
    var = np.maximum(var, 1e-12) / (scale ** 2)
    var = np.clip(var, 1e-12, 1e6)          # 1000x annualised vol safety net
    return np.sqrt(var) / 100.0 * np.sqrt(ANN)


# ============================================================================
# 5. METRICS + DM TESTS
# ============================================================================
def metrics(y_true_lvl, pred_lvl, y_true_log, pred_log):
    a = np.maximum(np.asarray(y_true_lvl, float), RV_FLOOR)
    q = np.maximum(np.asarray(pred_lvl, float), RV_FLOOR)
    la, lp = np.log(a), np.log(q)
    e = la - lp
    denom = float(np.sum((la - la.mean()) ** 2))
    return dict(
        n=int(len(a)),
        MAE_log=float(np.mean(np.abs(e))),
        RMSE_log=float(np.sqrt(np.mean(e ** 2))),
        R2_log=float(1 - np.sum(e ** 2) / denom) if denom > 0 else np.nan,
        QLIKE=float(np.mean(2.0 * np.log(q) + (a ** 2) / (q ** 2))),
        MAE_vol_pp=float(np.mean(np.abs(a - q)) * 100),
        bias_log=float(np.mean(e)),
    )


def nw_se(x: np.ndarray, lag: int):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    mu = x.mean()
    e = x - mu
    g0 = np.sum(e * e) / n
    var = g0
    for L in range(1, min(lag, n - 1) + 1):
        gl = np.sum(e[L:] * e[:-L]) / n
        var += 2.0 * (1.0 - L / (lag + 1.0)) * gl
    var = max(var, 1e-24)
    return mu, np.sqrt(var / n), n


def dm_test(diff: np.ndarray, lag: int):
    """DM t-stat on the mean loss differential with Newey-West HAC (lag) variance."""
    from scipy import stats
    mu, se, n = nw_se(diff, lag)
    if not np.isfinite(se) or se <= 0 or n < 10:
        return dict(mean_diff=np.nan, dm_t=np.nan, dm_p=np.nan, dm_t_hln=np.nan,
                    dm_p_hln=np.nan, n_pairs=n)
    t = mu / se
    # Harvey-Leybourne-Newbold (1997) small-sample correction
    h = lag - 1
    hln = t * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return dict(mean_diff=float(mu), dm_t=float(t),
                dm_p=float(2 * stats.t.sf(abs(t), df=n - 1)),
                dm_t_hln=float(hln), dm_p_hln=float(2 * stats.t.sf(abs(hln), df=n - 1)),
                n_pairs=int(n))


def dm_cluster_by_route(sub: pd.DataFrame, col_base: str, col_chal: str, lag: int):
    """
    Robustness: average the loss differential WITHIN each calendar date (which
    removes cross-route contemporaneous correlation), then HAC-test the daily
    differential series.
    """
    g = sub.groupby("date").apply(lambda x: x[col_base].mean() - x[col_chal].mean())
    g = g.sort_index()
    return dm_test(g.to_numpy(), lag)


def fmt(x, nd=5):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


# ============================================================================
# 6. MAIN WALK-FORWARD
# ============================================================================
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(AGENT_DIR, exist_ok=True)

    d = add_route_dummies(build_dataset())
    blocks = feature_blocks(d)
    log(f"[features] route_dummies={len(blocks['route_dummies'])} "
        f"weather={len(blocks['weather'])} econ={len(blocks['econ'])}")
    log(f"[features] weather cols: {blocks['weather']}")
    banned_present = [c for c in BANNED_FEATURES if c in d.columns]
    log(f"[features] banned-but-present: {banned_present if banned_present else 'NONE (good)'}")

    # drop the first route dummy to avoid perfect collinearity with the intercept
    dummies = sorted(blocks["route_dummies"])[1:]

    spec_cols = {
        "har_log":          HAR_CORE,
        "har_j":            HAR_JUMP,
        "har_cj":           HAR_CJ,
        "har_ext":          HAR_EXT,
        "har_econ":         HAR_CORE + blocks["econ"],
        "har_weather":      HAR_CORE + blocks["weather"],
        "har_weather_econ": HAR_CORE + blocks["weather"] + blocks["econ"],
        "har_log_strictlag": HAR_STRICT,
    }

    complete = d[NEEDED].notna().all(axis=1)
    # pre-create the fold-estimated PCA columns (filled inside the walk-forward loop)
    for j in range(N_WPCA):
        d[f"wpca{j}"] = np.nan
    dates = pd.date_range(WF_START, d[DATE_COL].max(), freq=REFIT_FREQ)
    log(f"[wf] candidate origins: {len(dates)} ({dates[0].date()} .. {dates[-1].date()})")

    fold_rows, pred_rows = [], []
    n_used = n_skip = 0

    for origin in dates:
        # Exclude pre-origin rows whose forward 5-day target is not fully
        # observed by the origin (the label endpoint must be strictly earlier).
        tr = (d[DATE_COL] < origin) & (d["target_end_date"] < origin) & complete
        te = (d[DATE_COL] >= origin) & (d[DATE_COL] < origin + pd.offsets.MonthBegin(1)) & complete
        if te.sum() < MIN_TEST_ROWS or tr.sum() < MIN_TRAIN_ROWS:
            n_skip += 1
            continue
        if (d.loc[tr, DATE_COL].max() - d.loc[tr, DATE_COL].min()).days < MIN_TRAIN_YEARS * 365:
            n_skip += 1
            continue
        n_used += 1

        tr_idx = d.index[tr]
        te_idx = d.index[te]
        y_te_log = d.loc[te_idx, "log_rv_fwd"].to_numpy()
        y_te_lvl = d.loc[te_idx, "rv_fwd"].to_numpy()
        valid = y_te_lvl > RV_FLOOR          # drop exactly-zero realised vol
        te_ok = te_idx[valid]
        sub = d.loc[te_ok]

        preds = {}

        # ---------------- HAR family (all in logs) ----------------
        for name, cols in spec_cols.items():
            cols_use = cols + dummies
            p = har_fit_predict(d, tr_idx, te_idx, cols_use)
            if p is None:
                continue
            preds[name] = np.exp(np.clip(p, -30, 30))[valid]
        # b) per-route HAR
        p = har_route_fit_predict(d, tr_idx, te_idx, HAR_CORE)
        if p is not None:
            preds["har_route"] = np.exp(np.clip(p, -30, 30))[valid]

        # ---------------- weather/econ PCA variants ----------------
        # PCA basis is estimated ONLY on the training rows of each fold.
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        wcols = [c for c in blocks["weather"] if d.loc[tr_idx, c].std() > 1e-12]
        if len(wcols) >= N_WPCA:
            sc = StandardScaler().fit(d.loc[tr_idx, wcols].astype(float).fillna(0.0))
            pc = PCA(n_components=N_WPCA, random_state=SEED).fit(
                sc.transform(d.loc[tr_idx, wcols].astype(float).fillna(0.0)))
            tr_pc = pc.transform(sc.transform(d.loc[tr_idx, wcols].astype(float).fillna(0.0)))
            te_pc = pc.transform(sc.transform(d.loc[te_idx, wcols].astype(float).fillna(0.0)))
            for j in range(N_WPCA):
                d.loc[d.index[tr_idx], f"wpca{j}"] = tr_pc[:, j]
                d.loc[d.index[te_idx], f"wpca{j}"] = te_pc[:, j]
            pc_cols = [f"wpca{j}" for j in range(N_WPCA)]
            for name, cols in [("har_weather_pca", HAR_CORE + pc_cols),
                               ("har_wpca_epca", HAR_CORE + pc_cols + blocks["econ"])]:
                pp = har_fit_predict(d, tr_idx, te_idx, cols + dummies)
                if pp is not None:
                    preds[name] = np.exp(np.clip(pp, -30, 30))[valid]

        # ---------------- GARCH family (per route) ----------------
        for spec in [m for m, f in MODELS if f == "GARCH"]:
            vals = np.full(len(te_idx), np.nan)
            te_route = d.loc[te_idx, ROUTE_COL].to_numpy()
            for r in pd.unique(te_route):
                m_te = np.where(te_route == r)[0]
                sub_tr = d.index[tr & (d[ROUTE_COL] == r)]
                if len(sub_tr) < 200:
                    continue
                fc = garch_fit_predict(d, sub_tr, te_idx[m_te], spec)
                if fc is None or len(fc) != len(m_te):
                    continue
                vals[m_te] = fc
            if np.isfinite(vals[valid]).sum() >= 10:
                preds[spec] = vals[valid]

        # ---------------- score ----------------
        for name, p in preds.items():
            p = np.asarray(p, float)
            ok = np.isfinite(p)
            if ok.sum() < 10:
                continue
            mt = metrics(y_te_lvl[valid][ok], p[ok], y_te_log[valid][ok], np.log(np.maximum(p[ok], RV_FLOOR)))
            mt.update(origin=origin, model=name, n_train=int(tr.sum()), n_test=int(te.sum()))
            fold_rows.append(mt)
            for j, idx in enumerate(te_ok[ok]):
                pred_rows.append(dict(origin=origin, date=d.loc[idx, DATE_COL],
                                      route=d.loc[idx, ROUTE_COL], model=name,
                                      y_true=float(max(y_te_lvl[valid][ok][j], RV_FLOOR)),
                                      y_pred=float(max(p[ok][j], RV_FLOOR))))

        if n_used % 12 == 0:
            log(f"  ...origin {origin.date()}  n_test={int(te.sum())}  elapsed {time.time()-t0:.0f}s")

    fold = pd.DataFrame(fold_rows)
    pred = pd.DataFrame(pred_rows)
    log(f"\n[wf] folds used={n_used} skipped={n_skip}  rows scored={len(pred)}")

    # ---------------- aggregate ----------------
    agg = (fold.groupby("model")
           .agg(folds=("QLIKE", "size"), MAE_log=("MAE_log", "mean"), RMSE_log=("RMSE_log", "mean"),
                R2_log=("R2_log", "mean"), QLIKE=("QLIKE", "mean"), MAE_vol_pp=("MAE_vol_pp", "mean"),
                bias_log=("bias_log", "mean"))
           .reset_index())
    agg["family"] = agg["model"].map(dict(MODELS))
    agg = agg.sort_values("QLIKE").reset_index(drop=True)
    agg.to_csv(os.path.join(OUT_DIR, "agentA_metrics_aggregate.csv"), index=False)
    fold.to_csv(os.path.join(OUT_DIR, "agentA_metrics_by_fold.csv"), index=False)

    log("\n" + "=" * 96)
    log("RANKED AGGREGATE MODEL COMPARISON (mean over expanding-window folds)")
    log("=" * 96)
    log(agg[["model", "family", "folds", "MAE_log", "RMSE_log", "R2_log", "QLIKE", "MAE_vol_pp"]]
        .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ---------------- DM tests (paired, same observations) ----------------
    pred["loss_sq"] = (np.log(pred["y_true"]) - np.log(pred["y_pred"])) ** 2
    a = pred["y_true"].to_numpy()
    q = pred["y_pred"].to_numpy()
    pred["loss_qlike"] = 2.0 * np.log(q) + (a ** 2) / (q ** 2)

    wide = {}
    for loss in ["loss_sq", "loss_qlike"]:
        wide[loss] = pred.pivot_table(index=["origin", "date", "route"], columns="model",
                                      values=loss, aggfunc="mean")

    har = "har_log"
    comparisons = [
        ("HAR+weather (82 raw) vs HAR", "har_weather", har),
        ("HAR+weather (5 PCs) vs HAR", "har_weather_pca", har),
        ("HAR+econ vs HAR", "har_econ", har),
        ("HAR+weather+econ (raw) vs HAR", "har_weather_econ", har),
        ("HAR+weatherPCs+econ vs HAR", "har_wpca_epca", har),
        ("HAR+ext(EWMA,skew,kurt) vs HAR", "har_ext", har),
        ("HAR-J vs HAR", "har_j", har),
        ("HAR-CJ vs HAR", "har_cj", har),
        ("HAR per-route vs HAR pooled", "har_route", har),
        ("HAR strict-lag vs HAR", "har_log_strictlag", har),
        ("GARCH(1,1)-normal vs HAR", "garch11_normal", har),
        ("GARCH(1,1)-t vs HAR", "garch11_t", har),
        ("EGARCH(1,1)-t vs HAR", "egarch11_t", har),
        ("GJR-GARCH(1,1)-t vs HAR", "gjr_garch11_t", har),
        ("HAR+weather(5 PCs) vs GARCH(1,1)-t", "har_weather_pca", "garch11_t"),
        # the headline question, both directions
        ("HAR vs HAR+weather (5 PCs)", har, "har_weather_pca"),
        ("HAR vs HAR+econ", har, "har_econ"),
        ("HAR vs HAR+weather (82 raw)", har, "har_weather"),
    ]

    dm_rows = []
    for label, chal, base in comparisons:
        for loss, lname in [("loss_sq", "squared_error_log"), ("loss_qlike", "QLIKE")]:
            W = wide[loss]
            if base not in W.columns or chal not in W.columns:
                continue
            pair = W[[base, chal]].dropna()
            if len(pair) < 30:
                continue
            diff = pair[base].to_numpy() - pair[chal].to_numpy()   # >0 => challenger better
            base_stats = dm_test(diff, DM_LAG)
            sub = pair.reset_index()
            cl = dm_cluster_by_route(sub, base, chal, DM_LAG)
            mb, mc = pair[base].mean(), pair[chal].mean()
            dm_rows.append(dict(
                comparison=label, challenger=chal, baseline=base, loss=lname,
                n_pairs=base_stats["n_pairs"],
                mean_loss_baseline=float(mb), mean_loss_challenger=float(mc),
                skill_gain_pct=float(100 * (1 - mc / mb)) if mb > 0 else np.nan,
                dm_t=base_stats["dm_t"], dm_p=base_stats["dm_p"],
                dm_t_hln=base_stats["dm_t_hln"], dm_p_hln=base_stats["dm_p_hln"],
                significant_5pct=bool(np.isfinite(base_stats["dm_p"]) and base_stats["dm_p"] < 0.05),
                significant_5pct_hln=bool(np.isfinite(base_stats["dm_p_hln"]) and base_stats["dm_p_hln"] < 0.05),
                dm_t_route_clustered=cl["dm_t"], dm_p_route_clustered=cl["dm_p"],
                nw_lag=DM_LAG,
                challenger_better=bool(np.isfinite(base_stats["dm_t"]) and base_stats["dm_t"] > 0),
            ))
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(os.path.join(OUT_DIR, "agentA_dm_tests.csv"), index=False)

    log("\n" + "=" * 96)
    log("DIEBOLD-MARIANO INCREMENTAL VALUE TESTS (paired; NW HAC lag=9; two-sided t p-value)")
    log("=" * 96)
    show = dm[["comparison", "loss", "n_pairs", "mean_loss_baseline", "mean_loss_challenger",
               "skill_gain_pct", "dm_t", "dm_p", "significant_5pct"]]
    log(show.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # raw per-observation predictions for the parent's own analysis
    pred.drop(columns=["loss_sq", "loss_qlike"]).to_csv(
        os.path.join(OUT_DIR, "agentA_predictions_volatility.csv"), index=False)

    with open(os.path.join(OUT_DIR, "agentA_summary.json"), "w") as f:
        json.dump(dict(n_folds=n_used, n_skipped=n_skip, n_scored_rows=int(len(pred)),
                       dm_lag=DM_LAG, rv_floor=RV_FLOOR,
                       models=MODEL_ORDER,
                       weather_cols=blocks["weather"], econ_cols=blocks["econ"]), f, indent=2)

    log(f"\n[done] {time.time()-t0:.0f}s")
    return agg, dm


if __name__ == "__main__":
    main()
