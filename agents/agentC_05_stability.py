"""
Agent C - Deliverable 4: stability and robustness.

  (a) per-route and per-calendar-year volatility comparison
  (b) expanding window vs rolling 3-year training window
  (c) contribution of the mirrored duplicate routes to the pooled metric
  (d) effect of dropping the two forward-overlapping momentum features
Writes D:/testing/outputs/agentC_stability.csv (+ agentC_stability_incremental.csv)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentC_common import (OUT, ANN, RV_FLOOR, load_raw, build_full, feature_groups,
                           walkforward_folds)

N_EST = 400
SEED = 42
LOGF = open(os.path.join(OUT, "agentC_stability_log.txt"), "w")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOGF.write(s + "\n")
    LOGF.flush()


d = build_full(load_raw())
price_feats, weather_feats, route_dummies = feature_groups(d)
PRICE = price_feats + route_dummies
PW = price_feats + weather_feats + route_dummies
MOM = ["Price_Momentum_7d", "Price_Momentum_30d"]
PRICE_NOMOM = [c for c in PRICE if c not in MOM]
PW_NOMOM = [c for c in PW if c not in MOM]
log(f"[stab] rows={len(d)} price={len(price_feats)} weather={len(weather_feats)} "
    f"dummies={len(route_dummies)}")

folds = walkforward_folds(d)
log(f"[stab] folds={len(folds)}")


def make_xgb():
    import xgboost as xgb
    return xgb.XGBRegressor(n_estimators=N_EST, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                            reg_lambda=2.0, random_state=SEED, n_jobs=1, tree_method="hist")


# ---------------------------------------------------------------------------
# out-of-sample predictions under several specifications
# ---------------------------------------------------------------------------
MODELS = [("xgb_price", PRICE, "xgb"),
          ("xgb_price_weather", PW, "xgb"),
          ("xgb_price_nomom", PRICE_NOMOM, "xgb"),
          ("xgb_price_weather_nomom", PW_NOMOM, "xgb"),
          ("har_like", ["log_rv_d", "log_rv_w", "log_rv_m"], "lin")]

PRED = []
t0 = time.time()
for fi, (origin, tr, te) in enumerate(folds):
    ytr = d.loc[tr, "log_rv_fwd"].to_numpy()
    policies = {"expanding": tr}
    cut = origin - pd.DateOffset(years=3)
    r3 = tr & (d["Date"] >= cut).to_numpy()
    if r3.sum() >= 400:
        policies["rolling3y"] = r3
    # drop the duplicate mirror route (keep only the FE->direction of each lane)
    keep = d["Route"].isin(["XSICFENE_FarEast_NorthEurope", "XSICFEUW_FarEast_USWestCoast"]).to_numpy()
    if (tr & keep).sum() >= 400:
        policies["expanding_dedup"] = tr & keep
    for policy, trp in policies.items():
        for mname, fcols, kind in MODELS:
            if kind == "lin":
                if policy != "expanding":
                    continue
                from sklearn.linear_model import LinearRegression
                mm = LinearRegression().fit(d.loc[trp, fcols], d.loc[trp, "log_rv_fwd"].to_numpy())
            else:
                mm = make_xgb()
                mm.fit(d.loc[trp, fcols].fillna(0.0), d.loc[trp, "log_rv_fwd"].to_numpy())
            p = np.clip(mm.predict(d.loc[te, fcols].fillna(0.0)), np.log(RV_FLOOR), np.log(1e4))
            for i, idx in enumerate(d.index[te]):
                PRED.append(dict(policy=policy, origin=origin, idx=idx,
                                 date=d.loc[idx, "Date"], route=d.loc[idx, "Route"],
                                 model=mname, y=float(d.loc[idx, "log_rv_fwd"]),
                                 p=float(p[i]), n_train=int(trp.sum())))
    if (fi + 1) % 20 == 0:
        log(f"  ...fold {fi+1}/{len(folds)} elapsed {time.time()-t0:.0f}s")

P = pd.DataFrame(PRED)
P["sq_err"] = (P["y"] - P["p"]) ** 2
P["abs_err"] = (P["y"] - P["p"]).abs()
P["year"] = P["date"].dt.year
P["route_pair"] = P["route"].map({
    "XSICFENE_FarEast_NorthEurope": "FE-NEU", "XSICNEFE_NorthEurope_FarEast": "FE-NEU",
    "XSICFEUW_FarEast_USWestCoast": "FE-USW", "XSICUWFE_USWestCoast_FarEast": "FE-USW"})
log(f"[stab] predictions: {len(P)} rows, {P['origin'].nunique()} origins, "
    f"policies={sorted(P['policy'].unique())}")

# ---------------------------------------------------------------------------
rows = []


def add(scope, key, sub):
    if not len(sub):
        return
    for (policy, model), grp in sub.groupby(["policy", "model"]):
        v = grp["sq_err"].to_numpy()
        a = grp["abs_err"].to_numpy()
        rows.append(dict(scope=scope, key=str(key), policy=policy, model=model,
                         n=int(len(v)), mse_log=float(np.mean(v)),
                         mae_log=float(np.mean(a)),
                         rmse_log=float(np.sqrt(np.mean(v)))))


add("ALL", "all", P)
for r, sub in P.groupby("route"):
    add("route", r, sub)
for y, sub in P.groupby("year"):
    add("year", y, sub)
for (y, r), sub in P.groupby(["year", "route"]):
    add("year_route", f"{y}|{r}", sub)
for rp, sub in P.groupby("route_pair"):
    add("route_pair", rp, sub)
for pol, sub in P.groupby("policy"):
    add("policy", pol, sub)
S = pd.DataFrame(rows)

# incremental value of weather, matched rows, per scope
inc_rows = []
scopes = [("ALL", "all", P)]
scopes += [("route", r, s) for r, s in P.groupby("route")]
scopes += [("year", y, s) for y, s in P.groupby("year")]
scopes += [("policy", p, s) for p, s in P.groupby("policy")]
for scope, key, sub in scopes:
    for policy, sub2 in sub.groupby("policy"):
        w = sub2.pivot_table(index=["origin", "date", "route"], columns="model", values="sq_err")
        for base, chal in (("xgb_price", "xgb_price_weather"),
                           ("xgb_price_nomom", "xgb_price_weather_nomom")):
            if base not in w.columns or chal not in w.columns:
                continue
            pair = w[[base, chal]].dropna()
            a = pair[base].to_numpy(); b = pair[chal].to_numpy()
            dd = a - b
            n = len(dd)
            if n < 10:
                continue
            se = dd.std(ddof=1) / np.sqrt(n)
            t = float(dd.mean() / se) if se > 0 else np.nan
            try:
                from scipy import stats
                p = float(2 * (1 - stats.t.cdf(abs(t), df=n - 1)))
            except Exception:
                p = np.nan
            inc_rows.append(dict(scope=scope, key=str(key), policy=policy,
                                 comparison=f"{base} -> {chal}", n=n,
                                 mse_base=float(a.mean()), mse_challenger=float(b.mean()),
                                 weather_gain_pct=float(100 * (1 - b.mean() / a.mean())),
                                 t_stat=t, p_value=p,
                                 significant_5pct=bool(p < 0.05) if np.isfinite(p) else False))
I = pd.DataFrame(inc_rows)

S.to_csv(os.path.join(OUT, "agentC_stability.csv"), index=False)
I.to_csv(os.path.join(OUT, "agentC_stability_incremental.csv"), index=False)
log(f"[write] {os.path.join(OUT, 'agentC_stability.csv')}")
log(f"[write] {os.path.join(OUT, 'agentC_stability_incremental.csv')}")

pd.set_option("display.width", 250)
log("\n--- MEAN LOG-SPACE MSE BY POLICY x MODEL (scope=ALL) ---")
log(S[S.scope == "ALL"][["policy", "model", "n", "mse_log", "mae_log"]].to_string(index=False))

log("\n--- EXPANDING vs ROLLING-3Y vs DEDUP (scope=policy) ---")
log(S[S.scope == "policy"][["key", "model", "n", "mse_log", "mae_log"]].to_string(index=False))

log("\n--- PER-YEAR, expanding, xgb models ---")
piv = S[(S.scope == "year") & (S.policy == "expanding")].pivot_table(
    index="key", columns="model", values="mse_log")
log(piv.to_string())

log("\n--- PER-ROUTE, expanding, xgb models ---")
piv2 = S[(S.scope == "route") & (S.policy == "expanding")].pivot_table(
    index="key", columns="model", values="mse_log")
log(piv2.to_string())

log("\n--- PER-ROUTE-PAIR (independent lanes), expanding ---")
log(S[(S.scope == "route_pair") & (S.policy == "expanding")]
    .pivot_table(index="key", columns="model", values="mse_log").to_string())

log("\n--- WEATHER INCREMENTAL VALUE BY SCOPE ---")
log(I[["scope", "key", "policy", "comparison", "n", "weather_gain_pct", "t_stat",
       "p_value", "significant_5pct"]].to_string(index=False))
log(f"\n[done] {time.time()-t0:.0f}s")
LOGF.close()
