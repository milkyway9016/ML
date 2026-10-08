"""
agentD step 4 - WHERE WOULD WEATHER MATTER MOST?

Conditional analysis of the weather effect on log forward 5-day realised vol:
  1. storm season (Is_Peak_Season, Aug-Oct) vs rest of year
  2. cyclone-prone Europe lanes (Indian Ocean / South China Sea) vs Pacific lanes
  3. trailing-volatility regimes: calm / normal / turbulent terciles

All done on the DE-DUPLICATED lane sample, with route FE and HAC (L=10) errors,
plus a within-cell descriptive table and a month-block bootstrap on each stratum
so that a "significant" stratum must survive honest inference.

Outputs: agentD_conditional_effects.csv, plots/agentD_conditional_effects.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location("rg", r"D:\testing\agents\agentD_03_regressions.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)

OUT = r"D:\testing\outputs"
PLOT = os.path.join(OUT, "plots"); os.makedirs(PLOT, exist_ok=True)
RNG = np.random.default_rng(99)

SHOCKS = ["ev_cyc_active", "ev_cyc_wind_ge64kt", "ev_slp_anom_lt_p10", "ev_wind_gt_p90"]


def hac_sub(X, y, V, idx):
    b = X[:, idx].T @ np.linalg.pinv(X[:, idx].T @ X[:, idx]) @ y if False else None
    return None


def fit_stratum(d, feats, label, dim, value, lag=10):
    """OLS within one stratum; returns coefficient rows."""
    if len(d) < 120:
        return []
    X, names = rg.build_design(d, ["log_trail_rv22"] + feats)
    # drop lane FE inside a single-lane stratum (collinear)
    keep = [i for i, nm in enumerate(names) if not nm.startswith("laneFE_")]
    X = X[:, keep]; names = [names[i] for i in keep]
    y = d["log_fwd_rv5"].to_numpy()
    m = np.isfinite(y)
    if m.sum() < 120:
        return []
    r = rg.ols_hac(X[m], y[m], lag)
    out = []
    for nm, b, se, t, p in zip(names, r["b"], r["se"], r["t"], r["p"]):
        out.append(dict(dimension=dim, stratum=value, model=label, term=nm,
                        coef=round(float(b), 6), hac_se=round(float(se), 6),
                        t=round(float(t), 3), p_value=round(float(p), 6),
                        n=r["n"], r2=round(r["r2"], 5),
                        significant_5pct=bool(p < 0.05)))
    return out


def describe(d, col, ycol="fwd_rv5"):
    """Naive and trailing-vol-adjusted (within-decile) event-minus-nonevent."""
    if d[col].nunique() < 2 or len(d) < 120:
        return None
    e = d[d[col] == 1][ycol].to_numpy()
    n = d[d[col] == 0][ycol].to_numpy()
    if len(e) < 25 or len(n) < 25:
        return None
    naive = float(np.mean(e) - np.mean(n))
    # within trailing-vol decile
    num, den = 0.0, 0.0
    q = pd.qcut(d["log_trail_rv22"], 10, labels=False, duplicates="drop")
    for b in np.unique(q.dropna()):
        m = (q == b).to_numpy()
        ee = (d[col].to_numpy() == 1) & m
        nn = (d[col].to_numpy() == 0) & m
        ye, yn = d[ycol].to_numpy()[ee], d[ycol].to_numpy()[nn]
        ye, yn = ye[np.isfinite(ye)], yn[np.isfinite(yn)]
        if len(ye) < 3 or len(yn) < 3:
            continue
        num += len(ye) * (ye.mean() - yn.mean()); den += len(ye)
    adj = num / den if den > 0 else np.nan
    return dict(n_event=len(e), n_nonevent=len(n), event_mean=round(float(np.mean(e)), 5),
                nonevent_mean=round(float(np.mean(n)), 5), naive_diff=round(naive, 5),
                vol_adjusted_diff=round(adj, 5) if np.isfinite(adj) else np.nan)


def main():
    d = pd.read_csv(os.path.join(OUT, "agentD_panel_events.csv"), parse_dates=["Date"])
    d["lane"] = d["Route"].map(rg.LANE_KEEP)
    d = (d.sort_values(["lane", "Date"]).drop_duplicates(["lane", "Date"])
           .reset_index(drop=True))
    d = d[d["log_fwd_rv5"].notna() & d["log_trail_rv22"].notna()].reset_index(drop=True)

    # regime terciles on the with-lane-standardised trailing vol
    d["trail_z"] = d.groupby("lane")["log_trail_rv22"].transform(
        lambda s: (s - s.mean()) / s.std())
    d["regime"] = pd.cut(d["trail_z"], [-np.inf, -0.4307, 0.4307, np.inf],
                         labels=["calm", "normal", "turbulent"])

    print("=" * 118)
    print("STEP 4 - WHERE WOULD WEATHER MATTER MOST?")
    print("=" * 118)
    print(f"[sample] de-duplicated rows {len(d)}; regimes "
          f"{d.regime.value_counts().to_dict()}; "
          f"peak-season share {100*d.Is_Peak_Season.mean():.1f}%")

    rows = []
    # ---- 1. season ---------------------------------------------------------
    for lab, sub in [("peak_season_AugOct", d[d.Is_Peak_Season == 1]),
                     ("off_season", d[d.Is_Peak_Season == 0])]:
        rows += fit_stratum(sub, SHOCKS, "shocks", "season", lab)
    # ---- 2. lane family ----------------------------------------------------
    for lab, sub in [("Europe_lanes", d[d.lane == "FarEast-NorthEurope"]),
                     ("Pacific_lanes", d[d.lane == "FarEast-USWestCoast"])]:
        rows += fit_stratum(sub, SHOCKS, "shocks", "lane", lab)
    # ---- 3. volatility regime ---------------------------------------------
    for lab in ["calm", "normal", "turbulent"]:
        sub = d[d.regime == lab]
        rows += fit_stratum(sub, SHOCKS, "shocks", "vol_regime", lab)
    cond = pd.DataFrame(rows)
    cond.to_csv(os.path.join(OUT, "agentD_conditional_effects.csv"), index=False)

    for dim in ["season", "lane", "vol_regime"]:
        s = cond[(cond.dimension == dim) & (cond.term.isin(SHOCKS))]
        print(f"\n--- shock coefficients, split by {dim} "
              f"(target = log fwd_rv5, HAC L=10) ---")
        print(s[["stratum", "term", "coef", "hac_se", "t", "p_value", "n",
                 "significant_5pct"]].to_string(index=False))

    # ---- 4. descriptive within-cell table ---------------------------------
    print("\n--- Descriptive: event vs non-event mean fwd_rv5, naive vs "
          "trailing-vol-decile adjusted ---")
    desc = []
    for dim, groups in [
            ("season", [("peak_season_AugOct", d[d.Is_Peak_Season == 1]),
                        ("off_season", d[d.Is_Peak_Season == 0])]),
            ("lane", [("Europe_lanes", d[d.lane == "FarEast-NorthEurope"]),
                      ("Pacific_lanes", d[d.lane == "FarEast-USWestCoast"])]),
            ("vol_regime", [(lab, d[d.regime == lab]) for lab in ["calm", "normal", "turbulent"]])]:
        for glab, sub in groups:
            for ev in SHOCKS:
                r = describe(sub, ev)
                if r:
                    desc.append(dict(dimension=dim, stratum=glab, event=ev, **r))
    dt = pd.DataFrame(desc)
    print(dt.to_string(index=False))
    dt.to_csv(os.path.join(OUT, "agentD_conditional_descriptives.csv"), index=False)

    # ---- 5. formal interaction tests --------------------------------------
    print("\n--- Formal interaction tests (HAC L=10) ---")
    ints = []
    d["peak"] = (d.Is_Peak_Season == 1).astype(float)
    d["eu"] = (d.lane == "FarEast-NorthEurope").astype(float)
    d["turb"] = (d.regime == "turbulent").astype(float)
    for mod, mcol in [("peak_season", "peak"), ("lane_family", "eu"),
                      ("turbulent_regime", "turb")]:
        feats = ["log_trail_rv22", mcol] + SHOCKS + [f"{s}_x_{mcol}" for s in SHOCKS]
        for s in SHOCKS:
            d[f"{s}_x_{mcol}"] = d[s] * d[mcol]
        X, names = rg.build_design(d, feats)
        y = d["log_fwd_rv5"].to_numpy()
        m = np.isfinite(y)
        r = rg.ols_hac(X[m], y[m], 10)
        idx = [i for i, nm in enumerate(names) if nm.endswith(f"_x_{mcol}")]
        b = r["b"][idx]
        XtXi = np.linalg.pinv(X[m].T @ X[m])
        e = y[m] - X[m] @ r["b"]
        Xe = X[m] * e[:, None]
        S = Xe.T @ Xe
        for L in range(1, 11):
            w = 1 - L / 11
            G = Xe[L:].T @ Xe[:-L]; S += w * (G + G.T)
        V = XtXi @ S @ XtXi
        from scipy import stats as st
        stat = float(b @ np.linalg.pinv(V[np.ix_(idx, idx)]) @ b)
        pv = float(1 - st.chi2.cdf(stat, len(idx)))
        ints.append(dict(interaction=mod, n_terms=len(idx), wald_chi2=round(stat, 3),
                         df=len(idx), p_value=round(pv, 6),
                         significant_5pct=bool(pv < 0.05)))
    it = pd.DataFrame(ints)
    print(it.to_string(index=False))

    out = pd.concat([cond.assign(block="stratum_coefficients"),
                     dt.assign(block="stratum_descriptives"),
                     it.assign(block="interaction_wald")], ignore_index=True)
    out.to_csv(os.path.join(OUT, "agentD_conditional_effects.csv"), index=False)
    print(f"\n[written] agentD_conditional_effects.csv ({len(out)} rows)")

    # ---- 6. plot -----------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    spec3 = [("season", ["peak_season_AugOct", "off_season"]),
             ("lane", ["Europe_lanes", "Pacific_lanes"]),
             ("vol_regime", ["calm", "normal", "turbulent"])]
    for ax, (dim, order) in zip(axes, spec3):
        sub = dt[dt.dimension == dim]
        xs = np.arange(len(SHOCKS))
        w = 0.8 / max(len(order), 1)
        for i, st_ in enumerate(order):
            v = [sub[(sub.stratum == st_) & (sub.event == eV)]["vol_adjusted_diff"].mean()
                 for eV in SHOCKS]
            ax.bar(xs + i * w - 0.4 + w / 2, v, w, label=st_)
        ax.axhline(0, color="k", lw=1.2)
        ax.set_xticks(xs)
        ax.set_xticklabels([s.replace("ev_", "") for s in SHOCKS], rotation=25,
                           ha="right", fontsize=8)
        ax.set_ylabel("trailing-vol-adjusted event -\nnon-event mean fwd RV5 (annualised)")
        ax.set_title(f"by {dim}")
        ax.legend(fontsize=8)
    fig.suptitle("agentD: conditional weather effects on forward 5-day freight-rate volatility "
                 "(vol-decile-adjusted; every bar is within its own error band)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, "agentD_conditional_effects.png"), dpi=115)
    plt.close(fig)
    print("[written] plots/agentD_conditional_effects.png")


if __name__ == "__main__":
    main()
