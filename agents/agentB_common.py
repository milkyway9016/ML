"""
agentB_common.py
================
Shared data loading + strictly past-only feature engineering for the
DIRECTION and HORIZON workstream (agent B).

Definitions standardised per the task statement, per Route, after sorting
by (Route, Date):

    log_price = ln(max(Price_USD, 1))
    ret       = log_price.diff()
    sq_ret    = ret**2
    fwd_ret_H(t)   = sum of ret over t+1 .. t+H
    rv_fwd_H(t)    = sqrt(sum of sq_ret over t+1..t+H) * sqrt(252)
    rv_trail22(t)  = sqrt(mean of sq_ret over the 22 days ending at t) * sqrt(252)

EVERY feature used here is computable from information dated <= t.  Weather
and cyclone columns are additionally lagged by one day (shift(1)) so that
nothing contemporaneous with the return at t can leak into the feature set.

Banned source columns (forward-looking / contemporaneous bugs):
    Cyclone_Wind_Forecast3d, Cyclone_Wind_Forecast7d, Cyclone_Active_Forecast3d,
    Wind_Speed_Forecast7d, Rolling_7d_Cyclone_Days, Price_Volatility_7d,
    Price_Volatility_30d
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

SRC_DATA = r"D:\Internproj\New ML\clean_engineered_dataset.csv"
DATE_COL = "Date"
ROUTE_COL = "Route"
PRICE_COL = "Price_USD"
ANN = 252

BANNED_FEATURES = [
    "Cyclone_Wind_Forecast3d",
    "Cyclone_Wind_Forecast7d",
    "Cyclone_Active_Forecast3d",
    "Wind_Speed_Forecast7d",
    "Rolling_7d_Cyclone_Days",
    "Price_Volatility_7d",
    "Price_Volatility_30d",
]

# ---------------------------------------------------------------------------
# Route structure.  Two pairs of routes are exact mirrors of one another
# (identical Date and Price_USD series); verified at runtime in verify_mirrors().
# ---------------------------------------------------------------------------
MIRROR_PAIRS = [
    ("XSICFENE_FarEast_NorthEurope", "XSICNEFE_NorthEurope_FarEast"),
    ("XSICFEUW_FarEast_USWestCoast", "XSICUWFE_USWestCoast_FarEast"),
]
# one representative per mirror pair, used for the deduplicated replication
PRIMARY_ROUTES = [p[0] for p in MIRROR_PAIRS]

WEATHER_PREFIXES = (
    "air_", "slp_", "wind_speed_", "cyclone_wind_", "cyclone_slp_",
    "cyclone_dist_", "cyclone_active_",
)
WEATHER_EXACT = [
    "Route_Max_cyclone_wind", "Route_Min_cyclone_slp", "Route_Min_cyclone_dist",
    "Route_cyclone_active", "Route_Mean_air", "Route_Mean_slp",
    "Route_Mean_wind_speed", "SWI", "Coastal_Threat", "Congestion_Risk",
]
# strictly lagged weather constructs already shipped in the source
LAGGED_WEATHER = [
    "Cyclone_Wind_Lag7d", "Cyclone_Wind_Lag14d", "Cyclone_Wind_Lag30d",
    "Cyclone_Active_Lag7d", "Cyclone_Active_Lag14d", "Cyclone_Active_Lag30d",
    "Wind_Speed_Lag7d", "Wind_Speed_Lag14d", "Wind_Speed_Lag30d",
]
# own-price history constructs already shipped in the source (all lagged)
LAGGED_PRICE = [
    "Price_Lag_7d", "Price_Lag_30d", "Price_Momentum_7d", "Price_Momentum_30d",
    "Price_Zscore_30d",
]
CALENDAR = ["Month", "Week_of_Year"]
HORIZONS = [1, 5, 20]
VOL_K = [1.0, 1.5, 2.0]
FIXED_THR = [0.02, 0.03, 0.05]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(SRC_DATA)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([ROUTE_COL, DATE_COL]).reset_index(drop=True)
    return df


def verify_mirrors(df: pd.DataFrame) -> pd.DataFrame:
    """Byte-level duplicate check on the two suspected mirrored route pairs."""
    rows = []
    for a, b in MIRROR_PAIRS:
        A = df[df[ROUTE_COL] == a].drop(columns=[ROUTE_COL]).sort_values(DATE_COL).reset_index(drop=True)
        B = df[df[ROUTE_COL] == b].drop(columns=[ROUTE_COL]).sort_values(DATE_COL).reset_index(drop=True)
        same_shape = A.shape == B.shape
        date_id = bool(A[DATE_COL].equals(B[DATE_COL])) if same_shape else False
        if same_shape:
            diffs = [c for c in A.columns if not A[c].equals(B[c])]
            maxpd = float(np.nanmax(np.abs(A[PRICE_COL].to_numpy() - B[PRICE_COL].to_numpy())))
        else:
            diffs, maxpd = ["<shape mismatch>"], np.nan
        rows.append(dict(
            route_a=a, route_b=b, n_a=len(A), n_b=len(B),
            dates_identical=date_id,
            n_differing_columns=len(diffs),
            differing_columns=";".join(diffs),
            max_abs_price_diff=maxpd,
            price_and_date_identical=bool(date_id and np.isfinite(maxpd) and maxpd == 0),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _fwd_sum(s: pd.Series, H: int) -> pd.Series:
    """sum of s over t+1..t+H for a within-route series (s must be indexed in
    ascending route-time order)."""
    tmp = s.shift(-1)
    if H > 1:
        tmp = tmp.rolling(H, min_periods=H).sum().shift(-(H - 1))
    return tmp


def _ewma_var(s: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA of squared returns; value at t uses info up to t."""
    x = s.fillna(0.0).to_numpy(dtype=float)
    out = np.empty_like(x)
    v = float(np.mean(x[:20])) if len(x) >= 20 else (float(np.mean(x)) if len(x) else 0.0)
    for i in range(len(x)):
        out[i] = v
        v = lam * v + (1.0 - lam) * x[i]
    return pd.Series(out, index=s.index)


def _max_dd(s: pd.Series, w: int = 60) -> pd.Series:
    x = s.to_numpy(dtype=float)

    def f(a):
        r = a / a[0]
        return float(np.min(r / np.maximum.accumulate(r) - 1.0))

    return pd.Series(x, index=s.index).rolling(w, min_periods=max(10, w // 2)).apply(f, raw=True)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d[PRICE_COL] = d[PRICE_COL].astype(float)

    # ---------------- returns & forward targets (standard definitions) -----
    d["log_price"] = np.log(d[PRICE_COL].clip(lower=1.0))
    g = d.groupby(ROUTE_COL, sort=False)
    d["ret"] = g["log_price"].diff()
    d["sq_ret"] = d["ret"] ** 2
    d["abs_ret"] = d["ret"].abs()
    d["sign_ret"] = np.sign(d["ret"])

    for H in HORIZONS:
        d[f"fwd_ret_{H}"] = g["ret"].transform(lambda s, H=H: _fwd_sum(s, H))
        d[f"rv_fwd_{H}"] = np.sqrt(
            g["sq_ret"].transform(lambda s, H=H: _fwd_sum(s, H))
        ) * np.sqrt(ANN)

    # ---------------- trailing volatility (strictly past) ------------------
    d["rv_trail22"] = np.sqrt(
        g["sq_ret"].transform(lambda s: s.rolling(22, min_periods=22).mean())
    ) * np.sqrt(ANN)

    # returns at lags 1..3 are known at t and are legitimate direction features
    for L in [1, 2, 3, 5]:
        d[f"ret_lag{L}"] = g["ret"].transform(lambda s, L=L: s.shift(L))
    d["ret_sum_3"] = g["ret"].transform(lambda s: s.rolling(3, min_periods=3).sum())
    d["ret_sum_5"] = g["ret"].transform(lambda s: s.rolling(5, min_periods=5).sum())
    d["ret_sum_22"] = g["ret"].transform(lambda s: s.rolling(22, min_periods=22).sum())

    for w in [5, 10, 22, 66]:
        rv = g["sq_ret"].transform(
            lambda s, w=w: np.sqrt(s.rolling(w, min_periods=max(3, w // 2)).mean()) * np.sqrt(ANN)
        )
        d[f"rv_trail_{w}"] = rv
        d[f"log_rv_trail_{w}"] = np.log(rv.clip(lower=1e-8))

    d["ewma_var"] = g["sq_ret"].transform(_ewma_var)
    d["ewma_vol"] = np.sqrt(d["ewma_var"]) * np.sqrt(ANN)
    d["log_ewma_vol"] = np.log(d["ewma_vol"].clip(lower=1e-8))

    d["rv_ratio_5_22"] = d["rv_trail_5"] / d["rv_trail_22"].replace(0, np.nan)
    d["rv_ratio_22_66"] = d["rv_trail_22"] / d["rv_trail_66"].replace(0, np.nan)
    d["log_rv_ratio_5_22"] = np.log(d["rv_ratio_5_22"].clip(lower=1e-8))
    d["log_rv_ratio_22_66"] = np.log(d["rv_ratio_22_66"].clip(lower=1e-8))

    # ---------------- higher moments / persistence -------------------------
    d["ret_skew_22"] = g["ret"].transform(lambda s: s.rolling(22, min_periods=15).skew())
    d["ret_kurt_22"] = g["ret"].transform(lambda s: s.rolling(22, min_periods=15).kurt())
    d["ret_skew_66"] = g["ret"].transform(lambda s: s.rolling(66, min_periods=30).skew())

    d["abs_ret_mean_5"] = g["abs_ret"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    d["abs_ret_mean_22"] = g["abs_ret"].transform(lambda s: s.rolling(22, min_periods=11).mean())
    d["abs_ret_ratio_5_22"] = d["abs_ret_mean_5"] / d["abs_ret_mean_22"].replace(0, np.nan)
    d["neg_ret_frac_22"] = g["ret"].transform(
        lambda s: (s < 0).astype(float).rolling(22, min_periods=11).mean()
    )
    d["sign_flip_10"] = g["ret"].transform(
        lambda s: (s.fillna(0.0).apply(np.sign).diff().abs() > 0).rolling(10, min_periods=5).mean()
    )
    d["sign_flip_22"] = g["ret"].transform(
        lambda s: (s.fillna(0.0).apply(np.sign).diff().abs() > 0).rolling(22, min_periods=11).mean()
    )

    # up / down semi-volatility and a signed vol asymmetry
    d["up_sq"] = np.where(d["ret"] > 0, d["sq_ret"], 0.0)
    d["dn_sq"] = np.where(d["ret"] < 0, d["sq_ret"], 0.0)
    up = d.groupby(ROUTE_COL, sort=False)["up_sq"].transform(
        lambda s: np.sqrt(s.rolling(22, min_periods=22).mean()) * np.sqrt(ANN))
    dn = d.groupby(ROUTE_COL, sort=False)["dn_sq"].transform(
        lambda s: np.sqrt(s.rolling(22, min_periods=22).mean()) * np.sqrt(ANN))
    tot = np.sqrt(up ** 2 + dn ** 2)
    d["up_vol_22"], d["dn_vol_22"] = up, dn
    d["vol_asym_22"] = (dn - up) / tot.replace(0, np.nan)

    # trailing drawdown-style indicators (price levels, strictly past)
    d["dd_from_60d_high"] = d[PRICE_COL] / g[PRICE_COL].transform(
        lambda s: s.rolling(60, min_periods=20).max()) - 1.0
    d["dist_from_60d_low"] = d[PRICE_COL] / g[PRICE_COL].transform(
        lambda s: s.rolling(60, min_periods=20).min()) - 1.0
    d["max_dd_60"] = g[PRICE_COL].transform(_max_dd)
    d["days_neg_60"] = g["ret"].transform(
        lambda s: (s < 0).astype(float).rolling(60, min_periods=20).mean()
    )
    d["ma_ratio_22_66"] = g[PRICE_COL].transform(
        lambda s: s.rolling(22, min_periods=11).mean()
    ) / g[PRICE_COL].transform(lambda s: s.rolling(66, min_periods=33).mean()).replace(0, np.nan)

    # expanding percentile rank of trailing vol vs its own history
    d["rv_pct_rank"] = g["rv_trail_22"].transform(
        lambda s: s.expanding(min_periods=60).apply(
            lambda w: float((w[-1] > w[:-1]).mean()) if len(w) > 1 else 0.5, raw=True)
    )
    d["ret_pct_rank_60"] = g["ret"].transform(
        lambda s: s.rolling(60, min_periods=20).apply(
            lambda w: float((w[-1] > w[:-1]).mean()), raw=True)
    )
    return d


def lag_weather(d: pd.DataFrame) -> pd.DataFrame:
    """Lag every weather / cyclone column by one day.

    The source weather columns are dated at day t, i.e. they are contemporaneous
    with the return r_t.  Predicting returns at t+1..t+H with a variable dated t
    is not strictly wrong, but it is uncomfortably close to nowcasting, so the
    entire weather block is pushed to t-1.  This is a conservative choice: it can
    only weaken the weather signal, never manufacture one.
    """
    wcols = [c for c in d.columns
             if c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT
             or c in LAGGED_WEATHER]
    wcols = [c for c in dict.fromkeys(wcols) if c not in BANNED_FEATURES]
    d = d.copy()
    for c in wcols:
        d[c] = d.groupby(ROUTE_COL, sort=False)[c].shift(1)
    return d


def feature_sets(d: pd.DataFrame) -> dict:
    cal = [c for c in CALENDAR if c in d.columns]
    p = [c for c in d.columns if c.startswith(
        ("ret_lag", "ret_sum_", "rv_trail_", "log_rv_trail_", "ewma_", "log_ewma_",
         "rv_ratio_", "log_rv_ratio_", "ret_skew_", "ret_kurt_", "abs_ret_",
         "neg_ret_frac_", "sign_flip_", "up_vol_", "dn_vol_", "vol_asym_",
         "dd_from_", "dist_from_", "max_dd_", "days_neg_", "ma_ratio_",
         "rv_pct_rank", "ret_pct_rank_"))]
    p += [c for c in LAGGED_PRICE if c in d.columns]
    p = [c for c in dict.fromkeys(p) if c not in BANNED_FEATURES]

    w = [c for c in d.columns
         if c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT or c in LAGGED_WEATHER]
    w = [c for c in dict.fromkeys(w) if c not in BANNED_FEATURES]

    return {
        "price": p + cal,
        "price_weather": p + w + cal,
    }, p, w, cal
