"""
Agent C - shared helpers for the adversarial audit of volatility_large_move.py.
Nothing here imports the pipeline under audit: everything is re-derived from the raw CSV.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SRC = r"D:\Internproj\New ML\clean_engineered_dataset.csv"
OUT = r"D:\testing\outputs"
AGENTS = r"D:\testing\agents"

TARGET_COL = "Price_USD"
DATE_COL = "Date"
ROUTE_COL = "Route"

HORIZON = 5
ANN = 252
WF_START = pd.Timestamp("2020-01-01")
RV_FLOOR = 5e-4

BANNED = [
    "Cyclone_Wind_Forecast3d", "Cyclone_Wind_Forecast7d", "Cyclone_Active_Forecast3d",
    "Wind_Speed_Forecast7d", "Rolling_7d_Cyclone_Days", "Price_Volatility_7d",
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


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Route", "Date"]).reset_index(drop=True)


def independent_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fully independent re-implementation of the target, written from the spec in the
    task brief (not from the pipeline source):
      log_price = ln(max(price,1)); ret = diff(log_price) within route
      rv_fwd(t) = sqrt(sum of sq_ret over t+1..t+5) * sqrt(252)
    Implemented with an explicit forward-shift, no rolling/shift trickery.
    """
    d = df.copy()
    lp = np.log(np.maximum(d["Price_USD"].to_numpy(float), 1.0))
    d["c_log_price"] = lp
    ret = np.full(len(d), np.nan)
    for _, idx in d.groupby("Route").indices.items():
        idx = np.sort(idx)
        r = np.diff(lp[idx])
        ret[idx[1:]] = r
    d["c_ret"] = ret
    sq = ret ** 2  # NaN-safe: only the first row of each route is NaN
    fwd = np.full(len(d), np.nan)
    for _, idx in d.groupby("Route").indices.items():
        idx = np.sort(idx)
        s = sq[idx]
        n = len(s)
        for j in range(n):
            if j + HORIZON <= n - 1:
                fwd[idx[j]] = s[j + 1: j + 1 + HORIZON].sum()
    d["c_rv_fwd"] = np.sqrt(fwd) * np.sqrt(ANN)
    d["c_log_rv_fwd"] = np.log(np.maximum(d["c_rv_fwd"], 1e-8))
    return d


def pipeline_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Transcription of build_dataset()'s target/feature code, for comparison only."""
    d = df.copy()
    d["log_price"] = np.log(d["Price_USD"].clip(lower=1.0))
    d["ret"] = d.groupby("Route")["log_price"].diff()
    d["sq_ret"] = d["ret"] ** 2
    d["rv_fwd"] = np.sqrt(
        d.groupby("Route")["sq_ret"].transform(
            lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1))
        )
    ) * np.sqrt(ANN)
    d["log_rv_fwd"] = np.log(d["rv_fwd"].clip(lower=1e-8))
    d["fwd_ret"] = d.groupby("Route")["ret"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1))
    )
    return d


def price_feature_names(cols) -> list[str]:
    cols = list(cols)
    out = [c for c in cols if c.startswith(("log_rv", "rv_", "ewma", "abs_ret_mean",
                                            "ret_skew", "ret_kurt", "sign_flip",
                                            "dd_from", "dist_from"))
           and c not in ("rv_fwd", "log_rv_fwd")]
    out += [c for c in PRICE_HISTORY_EXACT if c in cols]
    return [c for c in dict.fromkeys(out) if c not in BANNED]


def weather_feature_names(cols) -> list[str]:
    cols = list(cols)
    out = [c for c in cols if (c.startswith(WEATHER_PREFIXES) or c in WEATHER_EXACT)
           and c not in BANNED]
    out += [c for c in cols if c.startswith(("Cyclone_Wind_Lag", "Cyclone_Active_Lag", "Wind_Speed_Lag"))]
    return [c for c in dict.fromkeys(out) if c not in BANNED]


def walkforward_folds(d: pd.DataFrame, extra_needed=("log_rv_fwd", "rv_trail_22", "ewma_vol",
                                                     "log_rv_d", "log_rv_w", "log_rv_m")):
    """Build monthly folds with training labels fully observed before each origin."""
    dates = pd.date_range(WF_START, d["Date"].max(), freq="MS")
    complete = d[list(extra_needed)].notna().all(axis=1)
    target_end_date = d.groupby("Route")["Date"].shift(-HORIZON)
    folds = []
    for origin in dates:
        tr = (d["Date"] < origin) & (target_end_date < origin)
        te = (d["Date"] >= origin) & (d["Date"] < origin + pd.offsets.MonthBegin(1))
        if te.sum() < 20:
            continue
        if (d.loc[tr, "Date"].max() - d.loc[tr, "Date"].min()).days < 2 * 365:
            continue
        tr = tr & complete
        te = te & complete
        if tr.sum() < 500 or te.sum() < 20:
            continue
        folds.append((origin, tr.to_numpy(), te.to_numpy()))
    return folds


# =============================================================================
# Feature construction - faithful transcription of volatility_large_move.py's
# build_dataset() + get_dummies, so every audit script uses the identical design.
# =============================================================================
def build_full(raw: pd.DataFrame) -> pd.DataFrame:
    d = pipeline_derived(raw)
    d["abs_ret"] = d["ret"].abs()

    def _ewma(s: pd.Series) -> pd.Series:
        x = s.fillna(0.0).to_numpy()
        out = np.empty_like(x)
        v = np.nanmean(x[:20]) if len(x) >= 20 else (np.nanmean(x) if len(x) else 0.0)
        for i in range(len(x)):
            out[i] = v
            v = 0.94 * v + 0.06 * x[i]
        return pd.Series(out, index=s.index)

    g = d.groupby(ROUTE_COL)["sq_ret"]
    d["rv_d"] = np.sqrt(g.transform(lambda s: s.rolling(1).sum()))
    d["rv_w"] = np.sqrt(g.transform(lambda s: s.rolling(5).sum()) / 5)
    d["rv_m"] = np.sqrt(g.transform(lambda s: s.rolling(22).sum()) / 22)
    for c in ["rv_d", "rv_w", "rv_m"]:
        d["log_" + c] = np.log(d[c].clip(lower=1e-8))
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
    d["dd_from_60d_high"] = d[TARGET_COL] / d.groupby(ROUTE_COL)[TARGET_COL].transform(
        lambda s: s.rolling(60, min_periods=20).max()) - 1.0
    d["dist_from_60d_low"] = d[TARGET_COL] / d.groupby(ROUTE_COL)[TARGET_COL].transform(
        lambda s: s.rolling(60, min_periods=20).min()) - 1.0
    d["rv_pct_rank"] = d.groupby(ROUTE_COL)["rv_trail_22"].transform(
        lambda s: s.expanding(min_periods=60).apply(
            lambda w: (w[-1] > w[:-1]).mean() if len(w) > 1 else 0.5, raw=True))
    dm = pd.get_dummies(d[ROUTE_COL], prefix="Route")
    d = pd.concat([d, dm], axis=1)
    return d[d["log_rv_fwd"].notna() & d["rv_trail_22"].notna()].reset_index(drop=True)


def feature_groups(d: pd.DataFrame):
    cols = list(d.columns)
    price_feats = price_feature_names(cols)
    weather_feats = weather_feature_names(cols)
    route_dummies = [c for c in cols if c.startswith("Route_")
                     and c not in weather_feats + ECON_EXACT]
    return price_feats, weather_feats, route_dummies
