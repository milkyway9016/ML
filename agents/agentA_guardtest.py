"""Verify the EGARCH stability guard: how often does the fallback trigger, and are forecasts sane?"""
import sys
sys.path.insert(0, r"D:\testing\agents")
import numpy as np, pandas as pd
from arch import arch_model
import agentA_baselines as A

d = A.add_route_dummies(A.build_dataset())
complete = d[A.NEEDED].notna().all(axis=1)
origins = pd.date_range("2020-01-01", d["Date"].max(), freq="MS")
n_tot = n_deg = n_fail = 0
means = []
for o in origins:
    tr = (d["Date"] < o) & complete
    te = (d["Date"] >= o) & (d["Date"] < o + pd.offsets.MonthBegin(1)) & complete
    if te.sum() < 20 or tr.sum() < 500:
        continue
    for r in d.loc[te, "Route"].unique():
        sub = d.index[tr & (d["Route"] == r)]
        n_tot += 1
        y = d.loc[sub, "ret"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy() * 100.0
        sd = float(np.std(y)); scale = 100.0 / sd
        am = arch_model(y * scale, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="t")
        try:
            res = am.fit(variance_targeting=True, disp="off", show_warning=False)
        except TypeError:
            res = am.fit(disp="off", show_warning=False)
        if A._egarch_degenerate(res):
            n_deg += 1
        fc = A.garch_fit_predict(d, sub, d.index[te & (d["Route"] == r)], "egarch11_t")
        if fc is None:
            n_fail += 1
        else:
            means.append(float(np.mean(fc)))
print(f"fits={n_tot} degenerate={n_deg} ({100*n_deg/n_tot:.1f}%) forecast_failures={n_fail}")
m = np.array(means)
print("mean EGARCH forecast (annualised vol) across folds/routes:",
      "min", round(m.min(), 4), "median", round(np.median(m), 4), "max", round(m.max(), 4))
print("share above 5.0 (500% ann.):", round(float((m > 5).mean()), 4))
# compare with garch11_t
gm = []
for o in origins:
    tr = (d["Date"] < o) & complete
    te = (d["Date"] >= o) & (d["Date"] < o + pd.offsets.MonthBegin(1)) & complete
    if te.sum() < 20 or tr.sum() < 500:
        continue
    for r in d.loc[te, "Route"].unique():
        sub = d.index[tr & (d["Route"] == r)]
        fc = A.garch_fit_predict(d, sub, d.index[te & (d["Route"] == r)], "garch11_t")
        if fc is not None:
            gm.append(float(np.mean(fc)))
gm = np.array(gm)
print("mean GARCH(1,1)-t forecast:",
      "min", round(gm.min(), 4), "median", round(np.median(gm), 4), "max", round(gm.max(), 4))
