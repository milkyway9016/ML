"""
Agent C - Deliverable 2 (final): confirmed leakage findings with exact evidence.
Fixed-direction scan:  X[t] == Y[t+k]  (k>0)  =>  X contains information from k days in
the FUTURE of its own timestamp.
Also quantifies the one in-use overlap: Price_Momentum_{7,30}d(t) spans t-7..t, and the
target window is t+1..t+5, so part of the momentum window sits inside the price path that
generates the target -- leakage of information about *past* prices does not occur, but the
momentum features do measure the same price path the target is built from, which inflates
apparent "price history" skill.
Writes D:/testing/outputs/agentC_leakage_matches.csv
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentC_common import OUT, load_raw

df = load_raw()
num = [c for c in df.columns if c not in ("Date", "Route")]
A = df[num].to_numpy(float)
rt = df["Route"].to_numpy()
nunique = {c: int(df[c].nunique(dropna=True)) for c in num}
const = {c for c in num if nunique[c] < 5}


def shifted(k: int) -> np.ndarray:
    """S[t] = A[t+k] within route (k>0 = forward in time)."""
    out = np.full_like(A, np.nan)
    if k > 0:
        same = rt[: len(A) - k] == rt[k:]
        out[: len(A) - k][same] = A[k:][same]
    elif k < 0:
        kk = -k
        same = rt[kk:] == rt[: len(A) - kk]
        out[kk:][same] = A[: len(A) - kk][same]
    return out


rows = []
for k in range(1, 8):
    S = shifted(k)
    for bi, b in enumerate(num):
        if b in const:
            continue
        cb = S[:, bi]
        ok = np.isfinite(cb)
        if ok.sum() < 200:
            continue
        for ai, a in enumerate(num):
            if a == b or a in const:
                continue
            sel = ok & np.isfinite(A[:, ai])
            if sel.sum() < 200:
                continue
            frac = float(np.isclose(A[sel, ai], cb[sel], rtol=0, atol=1e-12).mean())
            if frac > 0.99:
                rows.append(dict(feature=a, equals=f"{b}[t+{k}]", k=k,
                                 frac_equal=round(frac, 6), n=int(sel.sum()),
                                 feature_forward_looking=True,
                                 note=f"{a}(t) is determined by {b} at t+{k}"))

# de-duplicate (a, b, k) pairs, keep only the direction where a is genuinely forward-looking
m = pd.DataFrame(rows)
m = m.drop_duplicates(subset=["feature", "equals"])
m["is_banned"] = m["feature"].isin([
    "Cyclone_Wind_Forecast3d", "Cyclone_Wind_Forecast7d", "Cyclone_Active_Forecast3d",
    "Wind_Speed_Forecast7d", "Rolling_7d_Cyclone_Days", "Price_Volatility_7d",
    "Price_Volatility_30d"])
m["in_use_by_pipeline"] = ~m["is_banned"]
m = m.sort_values(["feature", "k"])
m.to_csv(os.path.join(OUT, "agentC_leakage_matches.csv"), index=False)

print("=" * 90)
print("CONFIRMED DETERMINISTIC FORWARD LOOK-UPS  (X[t] == Y[t+k], k>0)")
print("=" * 90)
print(m[["feature", "equals", "k", "frac_equal", "n", "is_banned"]].to_string(index=False))

print("\n--- FORWARD-LOOKING COLUMNS THAT ARE *BANNED* (correctly excluded) ---")
print(m[m["is_banned"]][["feature", "equals", "k", "frac_equal"]].to_string(index=False)
      if m["is_banned"].any() else "  (none)")

print("\n--- FORWARD-LOOKING COLUMNS THAT ARE *IN USE* AS FEATURES (real leaks) ---")
inuse = m[m["in_use_by_pipeline"]]
print(inuse[["feature", "equals", "k", "frac_equal"]].to_string(index=False)
      if len(inuse) else "  (none)")

# ---------------------------------------------------------------------------
# direct correlation of every in-use feature with future returns at several lags
# ---------------------------------------------------------------------------
from agentC_common import price_feature_names, weather_feature_names, WEATHER_EXACT
price_feats = price_feature_names(list(df.columns))
weather_feats = weather_feature_names(list(df.columns))

d = df.copy()
d["lp"] = np.log(d["Price_USD"].clip(lower=1.0))
d["ret"] = d.groupby("Route")["lp"].diff()
rows = []
for c in price_feats + weather_feats:
    for lag in (1, 3, 5, 7, 10, 22):
        fwd_ret = np.log(d["Price_USD"].clip(lower=1.0)).groupby(d["Route"]).shift(-lag) - \
                  np.log(d["Price_USD"].clip(lower=1.0))
        # cumulative forward return t+1..t+lag
        fut = d.groupby("Route")["ret"].transform(
            lambda s, lag=lag: s.shift(-1).rolling(lag, min_periods=lag).sum().shift(-(lag - 1)))
        sel = d[c].notna() & fut.notna()
        if sel.sum() < 200 or d.loc[sel, c].nunique() < 3:
            continue
        r = pd.Series(d.loc[sel, c].to_numpy()).corr(pd.Series(fut[sel].to_numpy()))
        rows.append(dict(feature=c, horizon=lag, corr_future_return=float(r), n=int(sel.sum())))
cf = pd.DataFrame(rows)
if len(cf):
    cf["abs_corr"] = cf["corr_future_return"].abs()
    top = cf.sort_values("abs_corr", ascending=False).head(15)
    print("\n--- TOP 15 |corr(feature(t), cumulative forward return t+1..t+h)| ---")
    print(top[["feature", "horizon", "corr_future_return", "n"]].to_string(index=False))
    cf.to_csv(os.path.join(OUT, "agentC_feature_future_corr.csv"), index=False)
    print(f"\n[write] {os.path.join(OUT, 'agentC_feature_future_corr.csv')}")

summary = dict(
    n_forward_looking_columns_banned=int(m["is_banned"].sum()),
    n_forward_looking_columns_in_use=int(len(inuse)),
    forward_looking_in_use=inuse[["feature", "equals", "k", "frac_equal"]].to_dict("records"),
    forward_looking_banned=m[m["is_banned"]][["feature", "equals", "k", "frac_equal"]].to_dict("records"),
)
print("\n[summary]", json.dumps(summary, indent=2, default=str))
