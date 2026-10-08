"""
agentD step 3 - DOES WEATHER ADD ANYTHING BEYOND VOLATILITY PERSISTENCE?

Plain OLS on logs:
    log fwd_rv5(t) = a + b1 log trail_rv22(t) [+ route FE]
    log fwd_rv5(t) = a + b1 log trail_rv22(t) + b2' weather(t) [+ route FE]
HAC (Newey-West, L=10) standard errors, because the 5-day target overlaps.

HONESTY NOTE ON THE SAMPLE
The source file contains each lane TWICE: FE->NE and NE->FE have byte-identical
Price_USD series (verified elsewhere in this project), likewise FE->USWC and
USWC->FE. That is 4 rows per lane-date, not 4 independent routes, and it inflates
any naive n by ~2x while halving HAC standard errors. Headline numbers are
therefore reported on the DE-DUPLICATED sample (one physical lane per pair).

Outputs: agentD_regressions.csv
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = r"D:\testing\outputs"

# keep one direction per physical lane
LANE_KEEP = {"XSICFENE_FarEast_NorthEurope": "FarEast-NorthEurope",
             "XSICFEUW_FarEast_USWestCoast": "FarEast-USWestCoast",
             "XSICNEFE_NorthEurope_FarEast": "FarEast-NorthEurope",
             "XSICUWFE_USWestCoast_FarEast": "FarEast-USWestCoast"}

SHOCK_FEATS = ["ev_cyc_active", "ev_cyc_wind_gt_p90", "ev_cyc_wind_ge64kt",
               "ev_slp_anom_lt_p10", "ev_wind_gt_p90", "ev_cyc_and_wind"]
CONT_FEATS = ["od_max_cyc_wind", "cyc_wind_anom", "slp_anom", "wind_anom"]


def ols_hac(X: np.ndarray, y: np.ndarray, lag: int = 10):
    """OLS with Newey-West HAC covariance."""
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ y)
    e = y - X @ b
    Xe = X * e[:, None]
    S = Xe.T @ Xe
    for L in range(1, min(lag, n - 1) + 1):
        w = 1 - L / (lag + 1)
        G = Xe[L:].T @ Xe[:-L]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    se = np.sqrt(np.maximum(np.diag(V), 1e-30))
    t = b / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    resid_var = float(e @ e / (n - k))
    ybar = y.mean()
    tss = float(((y - ybar) ** 2).sum())
    return dict(b=b, se=se, t=t, p=p, r2=float(1 - (e @ e) / tss),
                adj_r2=float(1 - (1 - (e @ e) / tss) * (n - 1) / (n - k)),
                rmse=float(np.sqrt(resid_var)), n=n, k=k)


def build_design(d: pd.DataFrame, feats: list[str]):
    cols = [np.ones(len(d))]
    names = ["const"]
    for c in feats:
        v = d[c].to_numpy(dtype=float)
        m = np.isfinite(v)
        v = np.where(m, v, np.nanmedian(v[m]) if m.any() else 0.0)
        cols.append(v)
        names.append(c)
        if (~m).any() and c in CONT_FEATS:
            cols.append((~m).astype(float))
            names.append(c + "_isna")
    # route FE (one dummy dropped)
    lanes = sorted(d["lane"].unique())
    for L in lanes[1:]:
        cols.append((d["lane"].to_numpy() == L).astype(float))
        names.append(f"laneFE_{L}")
    return np.column_stack(cols), names


def run_model(d, feats, label, lag=10):
    X, names = build_design(d, feats)
    y = d["log_fwd_rv5"].to_numpy()
    m = np.isfinite(y)
    r = ols_hac(X[m], y[m], lag)
    return r, names, label


def main():
    d = pd.read_csv(os.path.join(OUT, "agentD_panel_events.csv"), parse_dates=["Date"])
    d["lane"] = d["Route"].map(LANE_KEEP)
    dup = len(d)
    d = (d.sort_values(["lane", "Date"])
           .drop_duplicates(subset=["lane", "Date"], keep="first")
           .reset_index(drop=True))
    print("=" * 118)
    print("STEP 3 - DOES WEATHER ADD ANYTHING BEYOND VOLATILITY PERSISTENCE?")
    print("=" * 118)
    print(f"[sample] source rows {dup} -> de-duplicated lane-dates {len(d)} "
          f"(identical opposite-direction prices removed)")
    print(f"[sample] lanes: {d.groupby('lane').size().to_dict()}")

    d = d[d["log_fwd_rv5"].notna() & d["log_trail_rv22"].notna()].reset_index(drop=True)
    print(f"[sample] estimable rows {len(d)}  "
          f"({d.Date.min().date()} .. {d.Date.max().date()})")

    models = [
        ("M0_trail_only", []),
        ("M1_trail+shocks", SHOCK_FEATS),
        ("M2_trail+continuous", CONT_FEATS),
        ("M3_trail+shocks+continuous", SHOCK_FEATS + CONT_FEATS),
        ("M4_trail+cyc_shocks_only", ["ev_cyc_active", "ev_cyc_wind_gt_p90", "ev_cyc_wind_ge64kt"]),
        ("M5_trail+slp_wind_shocks_only", ["ev_slp_anom_lt_p10", "ev_wind_gt_p90"]),
    ]
    fits, rows = {}, []
    for lab, feats in models:
        r, names, _ = run_model(d, ["log_trail_rv22"] + feats, lab, lag=10)
        fits[lab] = (r, names)
        for nm, b, se, t, p in zip(names, r["b"], r["se"], r["t"], r["p"]):
            rows.append(dict(model=lab, term=nm, coef=round(float(b), 6),
                             hac_se=round(float(se), 6), t=round(float(t), 3),
                             p_value=round(float(p), 6), n=r["n"],
                             r2=round(r["r2"], 6), adj_r2=round(r["adj_r2"], 6),
                             significant_5pct=bool(p < 0.05)))
    res = pd.DataFrame(rows)

    base_r2 = fits["M0_trail_only"][0]["r2"]
    print("\n--- Model fit (target = log forward 5-day realised volatility) ---")
    fit_tab = []
    for lab, _ in models:
        r = fits[lab][0]
        fit_tab.append(dict(model=lab, n=r["n"], k=r["k"], R2=round(r["r2"], 5),
                            adj_R2=round(r["adj_r2"], 5),
                            incremental_R2_vs_M0=round(r["r2"] - base_r2, 6),
                            pct_incremental=round(100 * (r["r2"] - base_r2) / base_r2, 3)
                            if base_r2 > 0 else np.nan))
    fitt = pd.DataFrame(fit_tab)
    print(fitt.to_string(index=False))

    # incremental F-test (HAC Wald) for each nesting vs M0
    print("\n--- Wald test: do the added weather terms jointly differ from zero? (HAC) ---")
    wald = []
    r0, n0 = fits["M0_trail_only"]
    for lab, feats in models[1:]:
        r1, n1 = fits[lab]
        idx = [i for i, nm in enumerate(n1) if nm in feats]
        b = r1["b"][idx]
        X, _ = build_design(d[d.log_fwd_rv5.notna()], ["log_trail_rv22"] + feats)
        y = d.loc[d.log_fwd_rv5.notna(), "log_fwd_rv5"].to_numpy()
        m = np.isfinite(y)
        X, y = X[m], y[m]
        XtXi = np.linalg.pinv(X.T @ X)
        e = y - X @ r1["b"]
        Xe = X * e[:, None]
        S = Xe.T @ Xe
        for L in range(1, 11):
            w = 1 - L / 11
            G = Xe[L:].T @ Xe[:-L]
            S += w * (G + G.T)
        V = XtXi @ S @ XtXi
        Vsub = V[np.ix_(idx, idx)]
        stat = float(b @ np.linalg.pinv(Vsub) @ b)
        q = len(idx)
        pv = float(1 - stats.chi2.cdf(stat, q))
        wald.append(dict(model=lab, n_terms=q, wald_chi2=round(stat, 3), df=q,
                         p_value=round(pv, 6), significant_5pct=bool(pv < 0.05),
                         incremental_R2=round(r1["r2"] - r0["r2"], 6)))
    wt = pd.DataFrame(wald)
    print(wt.to_string(index=False))

    # ---------------- WITHIN-LANE R2 (the fair persistence benchmark) -------
    # Lane fixed effects absorb the huge level difference between the two lanes,
    # so pooled R2 understates persistence. Report the within-lane R2 too.
    print("\n--- WITHIN-LANE R2 (route-mean removed) - the fair persistence benchmark ---")
    win_rows = []
    for lab, feats in [("M0_trail_only", []), ("M1_trail+shocks", SHOCK_FEATS),
                       ("M3_trail+shocks+continuous", SHOCK_FEATS + CONT_FEATS)]:
        num, den, n_tot = 0.0, 0.0, 0
        for lane, g in d.groupby("lane"):
            y = g["log_fwd_rv5"].to_numpy()
            m = np.isfinite(y)
            g2 = g[m]
            y = y[m]
            X, _ = build_design(g2, ["log_trail_rv22"] + feats)
            X = X[:, [i for i, nm in enumerate(_)
                      if not nm.startswith("laneFE_")]]
            b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
            e = y - X @ b
            num += float(e @ e)
            den += float(((y - y.mean()) ** 2).sum())
            n_tot += len(y)
        win_rows.append(dict(model=lab, n=n_tot, within_lane_R2=round(1 - num / den, 5)))
    wr = pd.DataFrame(win_rows)
    print(wr.to_string(index=False))
    wr.to_csv(os.path.join(OUT, "agentD_regression_within_lane_r2.csv"), index=False)
    base_w = wr.loc[wr.model == "M0_trail_only", "within_lane_R2"].iloc[0]
    print(f"  => weather raises within-lane R2 from {base_w:.4f} to "
          f"{wr.loc[wr.model=='M3_trail+shocks+continuous','within_lane_R2'].iloc[0]:.4f}")

    # ---------------- full (duplicated) sample robustness -------------------
    d2 = pd.read_csv(os.path.join(OUT, "agentD_panel_events.csv"), parse_dates=["Date"])
    d2["lane"] = d2["Route"]
    d2 = d2[d2["log_fwd_rv5"].notna() & d2["log_trail_rv22"].notna()].reset_index(drop=True)
    rob = []
    for lab, feats in [("M0_trail_only", []), ("M1_trail+shocks", SHOCK_FEATS),
                       ("M3_trail+shocks+continuous", SHOCK_FEATS + CONT_FEATS)]:
        r, names, _ = run_model(d2, ["log_trail_rv22"] + feats, lab, lag=10)
        rob.append(dict(sample="duplicated-4-rows", model=lab, n=r["n"],
                        R2=round(r["r2"], 5), adj_R2=round(r["adj_r2"], 5)))
    for lab, feats in [("M0_trail_only", []), ("M1_trail+shocks", SHOCK_FEATS),
                       ("M3_trail+shocks+continuous", SHOCK_FEATS + CONT_FEATS)]:
        r = fits[lab][0]
        rob.append(dict(sample="dedup-2-lanes", model=lab, n=r["n"],
                        R2=round(r["r2"], 5), adj_R2=round(r["adj_r2"], 5)))
    rt = pd.DataFrame(rob)
    print("\n--- Sample-definition robustness (R2 with vs without the duplicated rows) ---")
    print(rt.to_string(index=False))

    # ---------------- key coefficient read-out ------------------------------
    print("\n--- Key weather coefficients in M3 (physical direction) ---")
    m3 = res[res.model == "M3_trail+shocks+continuous"]
    print(m3[["term", "coef", "hac_se", "t", "p_value", "significant_5pct"]]
          .to_string(index=False))

    out = pd.concat([res.assign(block="regression_coefficients"),
                     fitt.assign(block="model_fit").rename(columns={"model": "model"}),
                     wt.assign(block="wald_incremental")], ignore_index=True)
    out.to_csv(os.path.join(OUT, "agentD_regressions.csv"), index=False)
    rt.to_csv(os.path.join(OUT, "agentD_regression_sample_robustness.csv"), index=False)
    print(f"\n[written] agentD_regressions.csv ({len(out)} rows)")

    # persistence-only benchmark for context
    print("\n--- Context: how much does trailing volatility alone explain? ---")
    print(f"  R2 of log_fwd_rv5 on log_trail_rv22 alone = {base_r2:.4f}")
    print(f"  slope on log_trail_rv22 = {fits['M0_trail_only'][0]['b'][1]:.4f} "
          f"(HAC t = {fits['M0_trail_only'][0]['t'][1]:.2f})")
    print("  A slope far below 1 means forward volatility mean-reverts strongly "
          "relative to the trailing 22-day level.")


if __name__ == "__main__":
    main()
