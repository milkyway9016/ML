"""Smoke test: one fold through the whole agentA pipeline."""
import sys, time
import numpy as np, pandas as pd
sys.path.insert(0, r"D:\testing\agents")
import agentA_baselines as A

t0 = time.time()
d = A.add_route_dummies(A.build_dataset())
blocks = A.feature_blocks(d)
print("blocks:", len(blocks["route_dummies"]), len(blocks["weather"]), len(blocks["econ"]))
print("banned present:", [c for c in A.BANNED_FEATURES if c in d.columns])
dummies = sorted(blocks["route_dummies"])[1:]

complete = d[A.NEEDED].notna().all(axis=1)
origin = pd.Timestamp("2020-06-01")
tr = (d["Date"] < origin) & complete
te = (d["Date"] >= origin) & (d["Date"] < origin + pd.offsets.MonthBegin(1)) & complete
tr_idx, te_idx = d.index[tr], d.index[te]
print("n_tr", len(tr_idx), "n_te", len(te_idx), "build", f"{time.time()-t0:.0f}s")

specs = {
    "har_log": A.HAR_CORE, "har_j": A.HAR_JUMP, "har_cj": A.HAR_CJ,
    "har_ext": A.HAR_EXT, "har_econ": A.HAR_CORE + blocks["econ"],
    "har_weather": A.HAR_CORE + blocks["weather"],
    "har_weather_econ": A.HAR_CORE + blocks["weather"] + blocks["econ"],
    "har_log_strictlag": A.HAR_STRICT,
}
y = d.loc[te_idx, "log_rv_fwd"].to_numpy()
for nm, cols in specs.items():
    t1 = time.time()
    p = A.har_fit_predict(d, tr_idx, te_idx, cols + dummies)
    ok = np.isfinite(p).all()
    print(f"{nm:20s} finite={ok} mean={p.mean():.4f} true_mean={y.mean():.4f} {time.time()-t1:.1f}s")
p = A.har_route_fit_predict(d, tr_idx, te_idx, A.HAR_CORE)
print("har_route finite", np.isfinite(p).all(), "mean", np.mean(p))

for spec in A.GARCH_KW:
    t1 = time.time()
    vals = np.full(len(te_idx), np.nan)
    te_route = d.loc[te_idx, A.ROUTE_COL].to_numpy()
    for r in pd.unique(te_route):
        m = np.where(te_route == r)[0]
        sub_tr = d.index[tr & (d[A.ROUTE_COL] == r)]
        fc = A.garch_fit_predict(d, sub_tr, te_idx[m], spec)
        if fc is not None:
            vals[m] = fc
    print(f"{spec:16s} finite_frac={np.isfinite(vals).mean():.2f} mean={np.nanmean(vals)*100:.3f}% "
          f"true_mean={np.nanmean(d.loc[te_idx,'rv_fwd'])*100:.3f}%  {time.time()-t1:.1f}s")
print("total", f"{time.time()-t0:.0f}s")
