"""
Agent C - Deliverable 4b: mirrored duplicate routes + effective sample size.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentC_common import OUT, load_raw, pipeline_derived

df = load_raw()
pairs = [("XSICFENE_FarEast_NorthEurope", "XSICNEFE_NorthEurope_FarEast"),
         ("XSICFEUW_FarEast_USWestCoast", "XSICUWFE_USWestCoast_FarEast")]

report = {"pairs": []}
allcols = [c for c in df.columns if c not in ("Date", "Route")]
for a, b in pairs:
    da = df[df["Route"] == a].set_index("Date").sort_index()
    db = df[df["Route"] == b].set_index("Date").sort_index()
    common = da.index.intersection(db.index)
    info = dict(route_a=a, route_b=b, n_a=len(da), n_b=len(db), n_common_dates=len(common))
    diffs = {}
    for c in allcols:
        va, vb = da.loc[common, c], db.loc[common, c]
        eq = np.isclose(va.to_numpy(float), vb.to_numpy(float), rtol=0, atol=0, equal_nan=True)
        if not eq.all():
            diffs[c] = int((~eq).sum())
    info["columns_identical_on_common_dates"] = len(allcols) - len(diffs)
    info["columns_differing"] = diffs
    report["pairs"].append(info)
    print(f"[dup] {a}\n      {b}")
    print(f"      rows {len(da)} vs {len(db)}, common dates {len(common)}")
    print(f"      columns byte-identical on common dates: {len(allcols) - len(diffs)}/{len(allcols)}")
    print(f"      differing columns: {diffs if diffs else 'NONE - perfect mirror'}")

# ---- effective sample size -------------------------------------------------
d = pipeline_derived(df)
d = d[d["log_rv_fwd"].notna()].reset_index(drop=True)
d["route_pair"] = d["Route"].map({
    "XSICFENE_FarEast_NorthEurope": "FE-NEU", "XSICNEFE_NorthEurope_FarEast": "FE-NEU",
    "XSICFEUW_FarEast_USWestCoast": "FE-USW", "XSICUWFE_USWestCoast_FarEast": "FE-USW"})
rows = []
for rp, g in d.groupby("route_pair"):
    x = g["log_rv_fwd"].dropna().to_numpy()
    x = x - x.mean()
    n = len(x)
    # lag-1 autocorrelation of the target -> effective n
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    n_eff = n * (1 - r1) / (1 + r1) if r1 > -1 else np.nan
    rows.append(dict(route_pair=rp, n_obs=n, n_unique_dates=g["Date"].nunique(),
                     n_duplicated_rows=n - g["Date"].nunique(),
                     lag1_autocorr_log_rv_fwd=round(r1, 4), n_effective=round(n_eff, 1)))
    # overlaps of the target window: each return enters 5 different targets
eff = pd.DataFrame(rows)
print("\n--- EFFECTIVE SAMPLE SIZE ---")
print(eff.to_string(index=False))

tot = dict(
    total_modelling_rows=int(len(d)),
    n_routes=int(d["Route"].nunique()),
    n_independent_directional_lanes=int(d["route_pair"].nunique()),
    n_duplicated_rows=int(len(d) - d["Date"].nunique()),
    duplication_ratio=round(len(d) / d["Date"].nunique(), 4),
    n_unique_route_dates=int(d["Date"].nunique()),
    n_effective_independent_obs_after_lag1=float(eff["n_effective"].sum()),
    target_window_overlap_factor=5,
)
report["effective_sample"] = tot
report["per_pair"] = eff.to_dict("records")
print("\n--- SUMMARY ---")
for k, v in tot.items():
    print(f"  {k}: {v}")

with open(os.path.join(OUT, "agentC_duplicate_routes.json"), "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"\n[write] {os.path.join(OUT, 'agentC_duplicate_routes.json')}")
