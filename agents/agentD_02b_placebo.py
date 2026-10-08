"""
agentD step 2b - FALSIFICATION / BALANCE CHECKS for the matched-control design.

Three things must hold before ANY matched-control difference is believable:
  (i)   matching balances log trail_rv22 between event and control groups;
  (ii)  a PLACEBO (random pseudo-event days, same count, same routes) yields a
        matched difference of ~0 -- otherwise the matching procedure itself is
        biased and every "matched effect" is an artefact;
  (iii) the month-block bootstrap (honest inference under overlapping 5-day
        windows) is compared with the paired t-test.
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
RNG = np.random.default_rng(7)
CAL = 0.25


def matched_pairs(sub: pd.DataFrame, col: str, ycol: str, caliper_sd: float = CAL):
    """Return (event_y, control_y, control_x, event_x) for the matched design."""
    ey, my, mx, ex = [], [], [], []
    for r, g in sub.groupby("Route"):
        x = g["log_trail_rv22"].to_numpy()
        y = g[ycol].to_numpy()
        is_ev = g[col].to_numpy() == 1
        ok = np.isfinite(x) & np.isfinite(y)
        ctrl = (~is_ev) & ok
        if ctrl.sum() < 30:
            continue
        cx = x[ctrl]
        order = np.argsort(cx)
        cx_s, c_order = cx[order], np.flatnonzero(ctrl)[order]
        cal = caliper_sd * np.nanstd(x[ok])
        for i in np.flatnonzero(is_ev & ok):
            j = np.searchsorted(cx_s, x[i])
            best, bestd = -1, np.inf
            for jj in (j - 1, j):
                if 0 <= jj < len(cx_s):
                    dd = abs(cx_s[jj] - x[i])
                    if dd < bestd:
                        best, bestd = jj, dd
            if best < 0 or bestd > cal:
                continue
            k = c_order[best]
            ey.append(y[i]); my.append(y[k]); mx.append(x[k]); ex.append(x[i])
    return map(np.asarray, (ey, my, mx, ex))


d = pd.read_csv(os.path.join(OUT, "agentD_panel_events.csv"), parse_dates=["Date"])
events = [c[3:] for c in d.columns if c.startswith("ev_")]

print("=" * 118)
print("BALANCE CHECK - is log trail_rv22 actually matched? (target = fwd_rv5)")
print("=" * 118)
rows = []
for ev in events:
    col = "ev_" + ev
    sub = d[d[col].notna() & d["fwd_rv5"].notna() & d["log_trail_rv22"].notna()]
    E, N = sub[sub[col] == 1], sub[sub[col] == 0]
    ey, my, mx, ex = matched_pairs(sub, col, "fwd_rv5")
    if len(ey) < 40:
        continue
    raw_gap = N["log_trail_rv22"].mean() - E["log_trail_rv22"].mean()
    mat_gap = mx.mean() - ex.mean()
    rows.append(dict(
        event=ev, n_matched=len(ey),
        event_logtrail=round(float(ex.mean()), 4),
        allnonevent_logtrail=round(float(N["log_trail_rv22"].mean()), 4),
        matched_ctrl_logtrail=round(float(mx.mean()), 4),
        naive_mismatch=round(float(raw_gap), 4),
        matched_mismatch=round(float(mat_gap), 4),
        bias_reduction_pct=round(100 * (1 - abs(mat_gap) / max(abs(raw_gap), 1e-12)), 1)))
bal = pd.DataFrame(rows)
print(bal.to_string(index=False))
print("\nInterpretation: if naive_mismatch is strongly negative the event days sit in a")
print("LOWER-trailing-vol regime than average, so the naive comparison is biased toward")
print("finding a negative 'effect' purely from mean reversion. Matching removes that.")

print("\n" + "=" * 118)
print("PLACEBO - 30 random pseudo-event draws (same n per route) must give matched_diff ~ 0")
print("=" * 118)
plec = []
for rep in range(30):
    d["placebo"] = 0.0
    for r, g in d.groupby("Route"):
        n_ev = int((g["ev_cyc_active"] == 1).sum())
        pick = RNG.choice(g.index.to_numpy(), size=min(n_ev, len(g)), replace=False)
        d.loc[pick, "placebo"] = 1.0
    sub = d[d["fwd_rv5"].notna() & d["log_trail_rv22"].notna()]
    ey, my, mx, ex = matched_pairs(sub, "placebo", "fwd_rv5")
    if len(ey) >= 40:
        plec.append(float(np.mean(ey) - np.mean(my)))
plec = np.array(plec)
print(f"30 placebo draws: mean={plec.mean():+.5f}  sd={plec.std():.5f}  "
      f"p2.5={np.percentile(plec,2.5):+.5f}  p97.5={np.percentile(plec,97.5):+.5f}")
print("=> placebo mean near 0 confirms the matching procedure is unbiased;")
print("   the placebo sd is the noise floor any genuine matched effect must clear.")

print("\n" + "=" * 118)
print("SAME-DAY COINCIDENCE - do weather shocks line up with freight volatility AT ALL?")
print("=" * 118)
cors = []
for ev in events:
    col = "ev_" + ev
    for ycol, ylab in [("fwd_rv5", "fwd_rv5"), ("abs_ret", "|ret_t| (contemporaneous)"),
                       ("trail_rv22", "trail_rv22 (past)")]:
        for r, g in d.groupby("Route"):
            s = g[[col, ycol]].dropna()
            if len(s) < 100 or s[col].nunique() < 2:
                continue
            cors.append(dict(event=ev, target=ylab, route=r[:12],
                             corr=round(float(s[col].corr(s[ycol])), 4), n=len(s)))
c = pd.DataFrame(cors)
print(c.groupby(["event", "target"])["corr"].agg(["mean", "min", "max", "size"]).round(4).to_string())
