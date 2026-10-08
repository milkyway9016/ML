"""
Agent C - Deliverable 3: permutation / null-distribution tests.

Same walk-forward folds, same feature sets and same XGBoost hyper-parameters as
D:/testing/volatility_large_move.py.  Parallelised across FOLDS (joblib/loky).

  model A = price history + route dummies
  model B = price history + weather  + route dummies

Reported skill is the PER-FOLD R2 in log space (mean over folds), which is exactly the
metric the pipeline itself reports as R2_log.  (A pooled R2 across folds is invalid here
because different folds have very different volatility levels.)

NULL A (label permutation): log_rv_fwd shuffled WITHIN route -> zero real signal.
NULL B (weather permutation): real target, weather block shuffled WITHIN route.

Writes D:/testing/outputs/agentC_permutation_null.csv and ..._summary.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentC_common import (OUT, HORIZON, ANN, RV_FLOOR, load_raw, pipeline_derived,
                           price_feature_names, weather_feature_names, walkforward_folds)

SEED = 42
N_EST = 400
N_PERM = 25
N_JOBS = 5
LOGF = open(os.path.join(OUT, "agentC_permutation_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


raw = load_raw()
d = pipeline_derived(raw)
g = d.groupby("Route")["sq_ret"]
d["rv_d"] = np.sqrt(g.transform(lambda s: s.rolling(1).sum()))
d["rv_w"] = np.sqrt(g.transform(lambda s: s.rolling(5).sum()) / 5)
d["rv_m"] = np.sqrt(g.transform(lambda s: s.rolling(22).sum()) / 22)
for c in ["rv_d", "rv_w", "rv_m"]:
    d["log_" + c] = np.log(d[c].clip(lower=1e-8))
for w in [22]:
    rv = d.groupby("Route")["sq_ret"].transform(
        lambda s, w=w: np.sqrt(s.rolling(w, min_periods=max(3, w // 2)).sum() / w)) * np.sqrt(ANN)
    d[f"rv_trail_{w}"] = rv
dm = pd.get_dummies(d["Route"], prefix="Route")
d = pd.concat([d, dm], axis=1)
cols = list(d.columns)
price_feats = price_feature_names(cols)
weather_feats = weather_feature_names(cols)
route_dummies = [c for c in cols if c.startswith("Route_")
                 and c not in weather_feats + ["Route_Mean_LSCI"]]
price_route = price_feats + route_dummies
price_weather = price_feats + weather_feats + route_dummies
d = d[d["log_rv_fwd"].notna() & d["rv_trail_22"].notna()].reset_index(drop=True)
folds = walkforward_folds(d)
log(f"[perm] rows={len(d)} folds={len(folds)} n_price={len(price_feats)} "
    f"n_weather={len(weather_feats)} n_dummies={len(route_dummies)}")
log(f"[perm] n_perm={N_PERM} n_estimators={N_EST} n_jobs={N_JOBS}")

FOLD_DATA = []
for fi, (origin, tr, te) in enumerate(folds):
    FOLD_DATA.append(dict(
        fi=fi, origin=str(origin.date()),
        y=d.loc[te, "log_rv_fwd"].to_numpy(),
        ytr=d.loc[tr, "log_rv_fwd"].to_numpy(),
        rv_lvl=d.loc[te, "rv_fwd"].to_numpy(),
        XA=d.loc[te, price_route].fillna(0.0).to_numpy(np.float32),
        XB=d.loc[te, price_weather].fillna(0.0).to_numpy(np.float32),
        TA=d.loc[tr, price_route].fillna(0.0).to_numpy(np.float32),
        TB=d.loc[tr, price_weather].fillna(0.0).to_numpy(np.float32),
        routes_te=d.loc[te, "Route"].to_numpy(),
        routes_tr=d.loc[tr, "Route"].to_numpy(),
        w_idx=np.array([price_weather.index(c) for c in weather_feats], dtype=int),
    ))


def _perm_within(n, groups, seed):
    rng = np.random.default_rng(seed)
    out = np.arange(n)
    for gr in np.unique(groups):
        idx = np.where(groups == gr)[0]
        out[idx] = idx[rng.permutation(len(idx))]
    return out


def _make_xgb():
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=N_EST, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                            reg_lambda=2.0, random_state=SEED, n_jobs=1, tree_method="hist")


def fit_fold(fd, mode, seed):
    """Returns (fold_index, sseA, sseB, sst) on the fold's own test block."""
    ytr, TA, TB, yte = fd["ytr"], fd["TA"], fd["TB"], fd["y"]
    if mode == "label_perm":
        pl = _perm_within(len(ytr), fd["routes_tr"], seed)
        ytr = ytr[pl]
    elif mode == "weather_perm":
        pw = _perm_within(len(TB), fd["routes_tr"], seed)
        w = fd["w_idx"]
        TB = TB.copy()
        TB[np.ix_(np.arange(len(TB)), w)] = fd["TB"][np.ix_(pw, w)]
    mB = _make_xgb()
    mB.fit(TB, ytr)
    pB = mB.predict(fd["XB"])
    mA = _make_xgb()
    mA.fit(TA, ytr)
    pA = mA.predict(fd["XA"])
    sst = float(np.sum((yte - yte.mean()) ** 2))
    return (fd["fi"], float(np.sum((yte - pA) ** 2)), float(np.sum((yte - pB) ** 2)), sst)


def run(mode, n_rep, seed0):
    rows = []
    for rep in range(n_rep):
        t0 = time.time()
        res = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(fit_fold)(fd, mode, seed0 + 7919 * rep) for fd in FOLD_DATA)
        r2A, r2B, inc = [], [], []
        for fi, sA, sB, sst in res:
            if sst <= 0:
                continue
            r2A.append(1 - sA / sst)
            r2B.append(1 - sB / sst)
            if sA > 0:
                inc.append(100 * (sA - sB) / sA)
        rows.append(dict(rep=rep, mode=mode, n_folds=len(r2A),
                         R2_price_only=float(np.mean(r2A)),
                         R2_price_weather=float(np.mean(r2B)),
                         R2_price_weather_median=float(np.median(r2B)),
                         inc_gain_pct=float(np.mean(inc)),
                         inc_gain_pct_median=float(np.median(inc)),
                         seconds=round(time.time() - t0, 1)))
        log(f"  [{mode}] rep {rep:2d}  R2_A={rows[-1]['R2_price_only']:+.4f}  "
            f"R2_B={rows[-1]['R2_price_weather']:+.4f}  "
            f"inc={rows[-1]['inc_gain_pct']:+.4f}%  ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


t_start = time.time()
log("\n=== REAL ===")
real = run("real", 1, 11); real["test"] = "REAL"
log(f"\n=== NULL A: LABEL PERMUTATION x{N_PERM} ===")
nA = run("label_perm", N_PERM, 1000); nA["test"] = "NULL_label_perm"
log(f"\n=== NULL B: WEATHER PERMUTATION x{N_PERM} ===")
nB = run("weather_perm", N_PERM, 2000); nB["test"] = "NULL_weather_perm"

res = pd.concat([real, nA, nB], ignore_index=True)
res.to_csv(os.path.join(OUT, "agentC_permutation_null.csv"), index=False)
log(f"\n[write] {os.path.join(OUT, 'agentC_permutation_null.csv')}")


def summ(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), mean=float(np.mean(v)),
                sd=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                p05=float(np.percentile(v, 5)), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), p99=float(np.percentile(v, 99)),
                min=float(np.min(v)), max=float(np.max(v)))


summary = {}
for name, sub in (("real", real), ("null_label_perm", nA), ("null_weather_perm", nB)):
    for col in ("R2_price_only", "R2_price_weather", "inc_gain_pct"):
        summary[f"{name}.{col}"] = summ(sub[col])


def pval(null, observed, greater=True):
    null = np.asarray(null, float)
    k = np.sum(null >= observed) if greater else np.sum(null <= observed)
    return float((k + 1) / (len(null) + 1))


real_inc = float(real["inc_gain_pct"].iloc[0])
real_r2 = float(real["R2_price_weather"].iloc[0])
real_r2A = float(real["R2_price_only"].iloc[0])
summary["observed_inc_gain_pct"] = real_inc
summary["observed_R2_price_only"] = real_r2A
summary["observed_R2_price_weather"] = real_r2
summary["p_weather_gain_vs_weather_null"] = pval(nB["inc_gain_pct"], real_inc)
summary["p_weather_gain_vs_label_null"] = pval(nA["inc_gain_pct"], real_inc)
summary["p_R2_B_vs_label_null"] = pval(nA["R2_price_weather"], real_r2)
summary["p_R2_A_vs_label_null"] = pval(nA["R2_price_only"], real_r2A)
summary["weather_null_inc_p95"] = float(np.percentile(nB["inc_gain_pct"], 95))
summary["weather_null_inc_p99"] = float(np.percentile(nB["inc_gain_pct"], 99))
summary["label_null_R2_B_p95"] = float(np.percentile(nA["R2_price_weather"], 95))
summary["label_null_R2_B_p99"] = float(np.percentile(nA["R2_price_weather"], 99))
summary["label_permutation_preserves_no_autocorrelation_note"] = (
    "within-route full shuffle destroys volatility clustering, so this null is the "
    "most conservative possible 'no signal' benchmark: R2 near 0 / negative.")

log("\n" + "=" * 90)
log("SUMMARY")
log("=" * 90)
for k, v in summary.items():
    log(f"  {k}: {v}")

with open(os.path.join(OUT, "agentC_permutation_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)
log(f"\n[done] total {time.time()-t_start:.0f}s")
LOGF.close()
