import sys, time
sys.path.insert(0, r"D:\testing\agents")
import numpy as np, pandas as pd
import agentA_baselines as A

d = A.add_route_dummies(A.build_dataset())
complete = d[A.NEEDED].notna().all(axis=1)
o = pd.Timestamp("2020-06-01")
tr = (d["Date"] < o) & complete
te = (d["Date"] >= o) & (d["Date"] < o + pd.offsets.MonthBegin(1)) & complete
ti, tei = d.index[tr], d.index[te]
tc = d.loc[tei, "Route"].to_numpy()
for r in pd.unique(tc):
    m = np.where(tc == r)[0]
    sub = d.index[tr & (d["Route"] == r)]
    parts = []
    for spec in ["garch11_normal", "garch11_t", "egarch11_t", "gjr_garch11_t"]:
        t0 = time.time()
        fc = A.garch_fit_predict(d, sub, tei[m], spec)
        val = "None" if fc is None else str(round(float(np.mean(fc)), 4))
        parts.append(spec + "=" + val + f"({time.time()-t0:.1f}s)")
    truth = round(float(d.loc[tei[m], "rv_fwd"].mean()), 4)
    print(r[:24].ljust(26), " ".join(parts), " true=", truth)
