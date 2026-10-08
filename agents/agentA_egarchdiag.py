"""Trace EGARCH fit/forecast behaviour across folds."""
import sys
sys.path.insert(0, r"D:\testing\agents")
import numpy as np, pandas as pd
from arch import arch_model
import agentA_baselines as A

d = A.add_route_dummies(A.build_dataset())
complete = d[A.NEEDED].notna().all(axis=1)
origins = pd.date_range("2020-01-01", d["Date"].max(), freq="MS")
bad, rows = [], []
for o in origins:
    tr = (d["Date"] < o) & complete
    te = (d["Date"] >= o) & (d["Date"] < o + pd.offsets.MonthBegin(1)) & complete
    if te.sum() < 20 or tr.sum() < 500:
        continue
    for r in d.loc[te, "Route"].unique():
        sub = d.index[tr & (d["Route"] == r)]
        y = d.loc[sub, "ret"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy() * 100.0
        sd = float(np.std(y)); scale = 100.0 / sd
        try:
            am = arch_model(y * scale, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t")
            try:
                res = am.fit(variance_targeting=True, disp="off", show_warning=False)
            except TypeError:
                res = am.fit(disp="off", show_warning=False)
            p = res.params
            rows.append(dict(origin=o, route=r, ok=True, conv=res.convergence_flag,
                             beta=p.get("beta[1]", np.nan), alpha=p.get("alpha[1]", np.nan),
                             gamma=p.get("gamma[1]", np.nan), omega=p.get("omega", np.nan),
                             llf=res.loglikelihood))
        except Exception as e:
            rows.append(dict(origin=o, route=r, ok=False, conv=-1, err=str(e)[:60]))
            bad.append((o, r, str(e)[:80]))
df = pd.DataFrame(rows)
print(df.groupby("ok").size().to_string())
print("\nconvergence flags:", df[df.ok]["conv"].value_counts().to_dict())
ok = df[df.ok]
print("\nparameter summary (scaled units):")
print(ok[["omega", "alpha", "gamma", "beta"]].describe().to_string(float_format=lambda v: f"{v:.4f}"))
print("\nbeta > 0.999 cases:", int((ok["beta"] > 0.999).sum()), "of", len(ok))
print("beta close to 1 by route:")
print(ok.groupby("route")["beta"].agg(["mean", "max"]).to_string(float_format=lambda v: f"{v:.4f}"))
print("\nfirst 10 failures:")
for b in bad[:10]:
    print("  ", b)
df.to_csv(r"D:\testing\outputs\agentA_egarch_diag.csv", index=False)
