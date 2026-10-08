"""Validate the deterministic EGARCH recursion against arch's analytic h=1 forecast."""
import sys
sys.path.insert(0, r"D:\testing\agents")
import numpy as np, pandas as pd
from arch import arch_model
import agentA_baselines as A

d = A.add_route_dummies(A.build_dataset())
complete = d[A.NEEDED].notna().all(axis=1)
for o in ["2020-01-01", "2020-06-01", "2021-06-01", "2023-01-01", "2025-01-01"]:
    o = pd.Timestamp(o)
    tr = (d["Date"] < o) & complete
    te = (d["Date"] >= o) & (d["Date"] < o + pd.offsets.MonthBegin(1)) & complete
    if te.sum() < 20:
        continue
    for r in d.loc[te, "Route"].unique():
        sub = d.index[tr & (d["Route"] == r)]
        y = d.loc[sub, "ret"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy() * 100.0
        sd = float(np.std(y)); scale = 100.0 / sd
        am = arch_model(y * scale, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t")
        try:
            res = am.fit(variance_targeting=True, disp="off", show_warning=False)
        except TypeError:
            res = am.fit(disp="off", show_warning=False)
        # arch analytic h=1
        try:
            a1 = np.asarray(res.forecast(horizon=1, reindex=False).variance.to_numpy(), float).ravel()[0]
        except Exception as e:
            a1 = np.nan
        det = A._egarch_deterministic_forecast(res, 25, scale)
        print(f"{o.date()} {r[:20]:22s} analytic_h1={a1:.6f} det_h1={det[0]:.6f} "
              f"ratio={det[0]/a1:.6f}  det path[0,1,4,24]={np.round(det[[0,1,4,24]],4)}")
