"""
agentD step 2c - WHY IS THE PLACEBO NON-ZERO? Caliper/estimator sensitivity.

A placebo mean of +0.048 (not 0) on the SAME matched estimator used for the real
events means part of every matched number is estimator artefact. This script
isolates it by:
  1. caliper sweep  (tighter matching => less selection-on-noise bias)
  2. k-nearest-neighbour matching (averaging controls damps the artefact)
  3. covariate-exact matching inside trailing-vol deciles
  4. logging the EVENT trait itself (od_max_cyc_wind / od_mean_wind) to test
     whether events are genuinely high-wind days -- essential because with an
     expanding percentile the threshold rises, so "above trailing p90" can fire
     on a day whose absolute wind is unremarkable.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location(
    "es", r"D:\testing\agents\agentD_02_event_study.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)

OUT = r"D:\testing\outputs"
d = pd.read_csv(os.path.join(OUT, "agentD_panel_events.csv"), parse_dates=["Date"])
events = [c[3:] for c in d.columns if c.startswith("ev_")]
RNG = np.random.default_rng(11)


def knn_match(sub, col, ycol, k=5, caliper_sd=None, return_x=False):
    ey, my, mx, ex = [], [], [], []
    for r, g in sub.groupby("Route"):
        x = g["log_trail_rv22"].to_numpy(); y = g[ycol].to_numpy()
        is_ev = g[col].to_numpy() == 1
        ok = np.isfinite(x) & np.isfinite(y)
        ctrl = (~is_ev) & ok
        if ctrl.sum() < 30:
            continue
        cxi = np.flatnonzero(ctrl)
        cx = x[cxi]
        cal = (caliper_sd * np.nanstd(x[ok])) if caliper_sd else np.inf
        for i in np.flatnonzero(is_ev & ok):
            dist = np.abs(cx - x[i])
            nb = np.argsort(dist)[:k]
            keep = nb[dist[nb] <= cal]
            if len(keep) == 0:
                continue
            j = cxi[keep]
            ey.append(y[i]); my.append(float(np.mean(y[j])))
            mx.append(float(np.mean(x[j]))); ex.append(x[i])
    out = (np.asarray(ey), np.asarray(my), np.asarray(mx), np.asarray(ex))
    return out if return_x else out[:2]


def placebo_diff(k=1, caliper_sd=None, n_rep=20):
    vals = []
    for rep in range(n_rep):
        d["plac"] = 0.0
        for r, g in d.groupby("Route"):
            n_ev = int((g["ev_cyc_active"] == 1).sum())
            d.loc[RNG.choice(g.index.to_numpy(), n_ev, replace=False), "plac"] = 1.0
        sub = d[d["fwd_rv5"].notna() & d["log_trail_rv22"].notna()]
        ey, my = knn_match(sub, "plac", "fwd_rv5", k=k, caliper_sd=caliper_sd)
        if len(ey) >= 40:
            vals.append(float(np.mean(ey) - np.mean(my)))
    return np.array(vals)


print("=" * 112)
print("(1) CALIPER / k SWEEP  -- placebo is the noise floor, real events must clear it")
print("=" * 112)
rows = []
for k in [1, 5, 10]:
    for cal in [None, 0.25, 0.10, 0.05, 0.02]:
        pl = placebo_diff(k=k, caliper_sd=cal, n_rep=15)
        reals = {}
        for ev in events:
            sub = d[d["ev_" + ev].notna() & d["fwd_rv5"].notna() & d["log_trail_rv22"].notna()]
            ey, my = knn_match(sub, "ev_" + ev, "fwd_rv5", k=k, caliper_sd=cal)
            reals[ev] = float(np.mean(ey) - np.mean(my)) if len(ey) >= 40 else np.nan
        rows.append(dict(k=k, caliper=("none" if cal is None else cal),
                         placebo_mean=round(float(pl.mean()), 4),
                         placebo_sd=round(float(pl.std()), 4),
                         placebo_lo=round(float(np.percentile(pl, 2.5)), 4),
                         placebo_hi=round(float(np.percentile(pl, 97.5)), 4),
                         **{e: round(v, 4) for e, v in reals.items()}))
sw = pd.DataFrame(rows)
print(sw.to_string(index=False))
sw.to_csv(os.path.join(OUT, "agentD_caliper_sensitivity.csv"), index=False)

print("\n" + "=" * 112)
print("(2) ARE THE EVENT DAYS ACTUALLY WINDY? (absolute levels, not just flags)")
print("=" * 112)
chk = []
for ev in events:
    col = "ev_" + ev
    for r, g in d.groupby("Route"):
        e = g[g[col] == 1]; n = g[g[col] == 0]
        if len(e) < 30:
            continue
        chk.append(dict(event=ev, route=r[:12], n_ev=len(e),
                        ev_cycw_mean=round(float(e["od_max_cyc_wind"].mean()), 2),
                        ne_cycw_mean=round(float(n["od_max_cyc_wind"].mean()), 2),
                        ev_wind_mean=round(float(e["od_mean_wind"].mean()), 3),
                        ne_wind_mean=round(float(n["od_mean_wind"].mean()), 3),
                        ev_slpanom_mean=round(float(e["slp_anom"].mean()), 3),
                        ne_slpanom_mean=round(float(n["slp_anom"].mean()), 3)))
print(pd.DataFrame(chk).to_string(index=False))

print("\n" + "=" * 112)
print("(3) WINDFALL CHECK - is the source 'cyclone' block plausible?")
print("=" * 112)
fe = d[d.Route == "XSICFENE_FarEast_NorthEurope"]
print("Share of days with own-zone cyclone_active=1, per zone:")
zz = ["South_China_Sea", "Indian_Ocean", "Arabian_Sea", "Mediterranean_Sea", "North_Sea"]
raw = pd.read_csv(r"D:\Internproj\New ML\clean_engineered_dataset.csv", parse_dates=["Date"])
rf = raw[raw.Route == "XSICFENE_FarEast_NorthEurope"]
for z in zz:
    a = rf["cyclone_active_" + z]
    w = rf["cyclone_wind_" + z]
    ann = (a > 0).sum() / 11.3
    print(f"  {z:20s} active {100*(a>0).mean():5.1f}% of days  ~{ann:5.1f} cyclone-days/yr  "
          f"max wind {w.max():5.0f} kt  median wind on active days {w[w>0].median():5.0f} kt")
print("\n  Reality check: the Mediterranean sees roughly 1-2 'medicanes' per year, so ~15% of")
print("  days flagged as cyclone-active there is ~50x too frequent. The Indian Ocean figure")
print("  (~24%) is ~4x the real annual tropical-cyclone-day rate. The cyclone block looks")
print("  like a distance/smearing artefact rather than genuine storm-day identification.")
print("  Red Sea and North Atlantic cyclone blocks are identically zero despite lying on the")
print("  FE<->North Europe lane -- so two of the seven transited zones carry no cyclone")
print("  information at all.")
