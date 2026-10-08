"""
agentD step 2 - EVENT STUDY (core deliverable)

Weather-shock events are defined ONLY from past/contemporaneous information
(trailing percentiles computed on an expanding window shifted by one day).

For each event definition we measure the average behaviour of FUTURE volatility
aligned on the event day (t-5 .. t+15), and contrast:
  (a) NAIVE   : event days vs all non-event days
  (b) MATCHED : event days vs non-event days matched on log trail_rv22
                (volatility clusters, so (a) is contaminated by persistence)
Inference: Welch t-test, month-block bootstrap CI, and Newey-West HAC t on the
event dummy (overlapping 5-day target).

Outputs: agentD_event_study.csv, agentD_event_windows.csv,
         plots/agentD_event_study.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = r"D:\testing\outputs"
PLOT = os.path.join(OUT, "plots")
os.makedirs(PLOT, exist_ok=True)

RNG = np.random.default_rng(20240517)
MIN_OBS = 90          # trailing window for percentile thresholds
N_BOOT = 2000
LAG_NW = 20           # ~ 4 x overlapping horizon


# ---------------------------------------------------------------------------
def trailing_pct_rank(s: pd.Series, p: float, min_obs: int = MIN_OBS) -> pd.Series:
    """Expanding trailing p-quantile, shifted by 1 day => strictly past info."""
    q = s.shift(1).expanding(min_periods=min_obs).quantile(p)
    return q


def make_events(d: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    d = d.sort_values(["Route", "Date"]).reset_index(drop=True)
    ev = {}

    # --- 1. any cyclone active in an own zone of the route ------------------
    ev["cyc_active"] = d["od_cyc_active"].astype(float)

    # --- 2. cyclone wind anomaly above own trailing 90th percentile ---------
    thr_w = d.groupby("Route", group_keys=False)["od_max_cyc_wind"].apply(
        lambda s: trailing_pct_rank(s, 0.90))
    ev["cyc_wind_gt_p90"] = (d["od_max_cyc_wind"] > thr_w).astype(float)

    # --- 3. severe cyclone: own-zone max wind >= 64 kt (Cat-1 threshold) ----
    ev["cyc_wind_ge64kt"] = (d["od_max_cyc_wind"] >= 64).astype(float)

    # --- 4. deep low: own-zone mean SLP anomaly below own trailing 10th pct -
    thr_s = d.groupby("Route", group_keys=False)["slp_anom"].apply(
        lambda s: trailing_pct_rank(s, 0.10))
    ev["slp_anom_lt_p10"] = (d["slp_anom"] < thr_s).astype(float)

    # --- 5. wind shock: own-zone mean wind above own trailing 90th pct ------
    thr_ws = d.groupby("Route", group_keys=False)["od_mean_wind"].apply(
        lambda s: trailing_pct_rank(s, 0.90))
    ev["wind_gt_p90"] = (d["od_mean_wind"] > thr_ws).astype(float)

    # --- 6. compound: cyclone active AND wind above trailing 90th pct -------
    ev["cyc_and_wind"] = ((ev["cyc_active"] > 0) & (ev["wind_gt_p90"] > 0)).astype(float)

    # a warm-up row with no trailing threshold must not silently count as "event"
    guard = {"cyc_wind_gt_p90": thr_w, "slp_anom_lt_p10": thr_s, "wind_gt_p90": thr_ws}
    src = {"cyc_active": d["od_cyc_active"], "cyc_wind_gt_p90": d["od_max_cyc_wind"],
           "cyc_wind_ge64kt": d["od_max_cyc_wind"], "slp_anom_lt_p10": d["slp_anom"],
           "wind_gt_p90": d["od_mean_wind"], "cyc_and_wind": pd.Series(1.0, index=d.index)}
    for k, v in ev.items():
        s = pd.Series(np.asarray(v, float), index=d.index)
        if k in guard:
            s = s.where(guard[k].notna() & src[k].notna())   # NaN during warm-up
        else:
            s = s.where(src[k].notna())
        d["ev_" + k] = s
    return d, list(ev.keys())


# ---------------------------------------------------------------------------
def matched_control(d: pd.DataFrame, evcol: str, ycol: str,
                    caliper_sd: float = 0.25):
    """
    For every event row, pick the non-event row of the SAME route whose
    log trail_rv22 is closest and lies inside a caliper. Matching is done
    WITHIN the event's own volatility regime, so persistence cannot explain
    any surviving difference. Returns (event_y, matched_y, n_matched).
    """
    ey, my = [], []
    for r, g in d.groupby("Route"):
        x = g["log_trail_rv22"].to_numpy()
        is_ev = g[evcol].to_numpy() == 1
        ok = np.isfinite(x)
        ctrl = (~is_ev) & ok & g[ycol].notna().to_numpy()
        if ctrl.sum() < 30:
            continue
        cx = x[ctrl]
        order = np.argsort(cx)
        cx_s, c_order = cx[order], np.flatnonzero(ctrl)[order]
        cal = caliper_sd * np.nanstd(x[ok])
        for i in np.flatnonzero(is_ev & ok):
            if not np.isfinite(g[ycol].to_numpy()[i]):
                continue
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
            ey.append(g[ycol].to_numpy()[i])
            my.append(g[ycol].to_numpy()[k])
    return np.array(ey), np.array(my), len(ey)


def newey_west_t(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float]:
    """HAC (Newey-West) t-stat for the slope in y = a + b x."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x = y[m], x[m]
    n = len(y)
    if n < 20:
        return np.nan, np.nan
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, min(lag, n - 1) + 1):
        w = 1 - L / (lag + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(max(V[1, 1], 1e-30))
    t = b[1] / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def block_bootstrap(a: np.ndarray, b: np.ndarray, dates: np.ndarray,
                    stat: str = "diff", n_boot: int = N_BOOT):
    """
    Month-block bootstrap over the pooled (event, control) pairs:
    resample calendar months with replacement, recompute the statistic.
    This handles serial dependence from overlapping 5-day forward windows.
    """
    if len(a) == 0:
        return np.nan, np.nan, np.nan
    periods = pd.PeriodIndex(pd.to_datetime(dates), freq="M").astype(str)
    up = np.unique(periods)
    idx_by_p = {p: np.flatnonzero(periods == p) for p in up}
    def f(ia, ib):
        if stat == "diff":
            return np.mean(a[ia]) - np.mean(b[ib])
        if stat == "ratio":
            return np.mean(a[ia]) / max(np.mean(b[ib]), 1e-12)
        return np.median(a[ia]) - np.median(b[ib])
    obs = f(np.arange(len(a)), np.arange(len(b)))
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = RNG.integers(0, len(up), len(up))
        ia, ib = [], []
        for k in pick:
            ia.append(idx_by_p[up[k]])
            ib.append(idx_by_p[up[k]])
        ia = np.concatenate(ia) if ia else np.array([], int)
        ib = np.concatenate(ib) if ib else np.array([], int)
        ia = ia[ia < len(a)]
        ib = ib[ib < len(b)]
        draws[i] = f(ia, ib) if len(ia) and len(ib) else np.nan
    draws = draws[np.isfinite(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


# ---------------------------------------------------------------------------
def event_study(d: pd.DataFrame, events: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, win_rows = [], []

    for ev in events:
        col = "ev_" + ev
        for ycol, ylab in [("fwd_rv5", "fwd_rv5"), ("fwd_abs_mean5", "fwd_abs_ret")]:
            sub = d[d[col].notna() & d[ycol].notna() & d["log_trail_rv22"].notna()]
            E = sub[sub[col] == 1]
            N = sub[sub[col] == 0]
            if len(E) < 40 or len(N) < 40:
                continue

            # ---- (a) NAIVE -------------------------------------------------
            a, b = E[ycol].to_numpy(), N[ycol].to_numpy()
            t_naive, p_naive = stats.ttest_ind(a, b, equal_var=False)
            naive = _naive_boot(a, b, E["Date"].to_numpy(), N["Date"].to_numpy(), ycol)

            # ---- (b) MATCHED on log trailing vol ---------------------------
            ey, my, nm = matched_control(sub, col, ycol)
            if nm >= 40:
                mt, mp = stats.ttest_rel(ey, my)
                mdiff = float(np.mean(ey) - np.mean(my))
                mlo, mhi = _paired_boot(ey, my, sub, col, ycol)
            else:
                mt = mp = mdiff = mlo = mhi = np.nan

            rows.append(dict(
                event=ev, target=ylab,
                n_event=len(E), n_nonevent=len(N), n_matched=nm,
                event_mean=round(float(a.mean()), 5),
                nonevent_mean=round(float(b.mean()), 5),
                naive_diff=round(naive[0], 5),
                naive_ci_lo=round(naive[1], 5), naive_ci_hi=round(naive[2], 5),
                naive_t=round(float(t_naive), 3), naive_p=round(float(p_naive), 5),
                matched_diff=round(mdiff, 5),
                matched_ci_lo=round(mlo, 5), matched_ci_hi=round(mhi, 5),
                matched_t=round(float(mt), 3) if np.isfinite(mt) else np.nan,
                matched_p=round(float(mp), 5) if np.isfinite(mp) else np.nan,
                pct_of_naive_surviving_matching=(
                    round(100 * mdiff / naive[0], 1)
                    if np.isfinite(mdiff) and abs(naive[0]) > 1e-12 else np.nan),
                nw_t=round(newey_west_t(sub[ycol].to_numpy(),
                                        sub[col].to_numpy().astype(float), LAG_NW)[0], 3),
                nw_p=round(newey_west_t(sub[ycol].to_numpy(),
                                        sub[col].to_numpy().astype(float), LAG_NW)[1], 5),
            ))

        # ---- aligned event window t-5 .. t+15 ------------------------------
        sub = d[d[col].notna()].copy()
        sub["is_ev"] = sub[col] == 1
        for off in range(-5, 16):
            sh = sub.groupby("Route")[["fwd_rv5", "fwd_abs_mean5", "is_ev"]].shift(-off)
            tmp = pd.DataFrame({
                "ev": sh["is_ev"], "rv": sh["fwd_rv5"], "ar": sh["fwd_abs_mean5"]})
            for lab, c in [("fwd_rv5", "rv"), ("fwd_abs_ret", "ar")]:
                g = tmp[tmp["ev"].notna()].groupby("ev")[c]
                if 1.0 in g.groups and 0.0 in g.groups:
                    mean_ev = g.mean().get(1.0, np.nan)
                    mean_ne = g.mean().get(0.0, np.nan)
                    cnt = g.count().get(1.0, 0)
                    win_rows.append(dict(event=ev, target=lab, offset=off,
                                         n_event=cnt, mean_event=mean_ev,
                                         mean_nonevent=mean_ne,
                                         diff=mean_ev - mean_ne))
    return pd.DataFrame(rows), pd.DataFrame(win_rows)


def _naive_boot(a, b, da, db, ycol):
    periods_a = pd.PeriodIndex(pd.to_datetime(da), freq="M").astype(str)
    periods_b = pd.PeriodIndex(pd.to_datetime(db), freq="M").astype(str)
    up = np.unique(np.concatenate([periods_a, periods_b]))
    ia_by_p = {p: np.flatnonzero(periods_a == p) for p in up}
    ib_by_p = {p: np.flatnonzero(periods_b == p) for p in up}
    obs = float(np.mean(a) - np.mean(b))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.integers(0, len(up), len(up))
        ia = np.concatenate([ia_by_p[up[k]] for k in pick])
        ib = np.concatenate([ib_by_p[up[k]] for k in pick])
        draws[i] = (np.mean(a[ia]) - np.mean(b[ib])) if len(ia) and len(ib) else np.nan
    draws = draws[np.isfinite(draws)]
    return obs, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _paired_boot(ey, my, sub, col, ycol):
    """Bootstrap the MEAN PAIRED difference by resampling calendar months."""
    # rebuild the event/control date pairs to allow month-block resampling
    dates = []
    for r, g in sub.groupby("Route"):
        x = g["log_trail_rv22"].to_numpy()
        is_ev = g[col].to_numpy() == 1
        ok = np.isfinite(x) & g[ycol].notna().to_numpy()
        ctrl = (~is_ev) & ok
        if ctrl.sum() < 30:
            continue
        cx = x[ctrl]
        order = np.argsort(cx)
        cx_s, c_order = cx[order], np.flatnonzero(ctrl)[order]
        cal = 0.25 * np.nanstd(x[ok])
        dts = g["Date"].to_numpy()
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
            dates.append(dts[i])
    dates = pd.to_datetime(dates)
    if len(dates) != len(ey):
        return np.nan, np.nan
    d_ = ey - my
    periods = pd.PeriodIndex(dates, freq="M").astype(str)
    up = np.unique(periods)
    idx_by_p = {p: np.flatnonzero(periods == p) for p in up}
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = RNG.integers(0, len(up), len(up))
        ii = np.concatenate([idx_by_p[up[k]] for k in pick])
        draws[i] = np.mean(d_[ii]) if len(ii) else np.nan
    draws = draws[np.isfinite(draws)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# ---------------------------------------------------------------------------
def make_plot(win: pd.DataFrame, ev_tab: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    key = ["cyc_active", "cyc_wind_ge64kt", "slp_anom_lt_p10", "wind_gt_p90"]
    key = [k for k in key if k in set(win["event"])]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (lab, tlab) in zip(axes, [("Forward RV5", "fwd_rv5"),
                                      ("Forward |ret| (5d mean)", "fwd_abs_ret")]):
        for i, ev in enumerate(key):
            w = win[(win.event == ev) & (win.target == tlab)].sort_values("offset")
            if w.empty:
                continue
            ax.plot(w["offset"], w["mean_event"], "-o", ms=4, lw=1.8,
                    color=colors[i], label=f"{ev} (n={int(w['n_event'].max())})")
            ax.plot(w["offset"], w["mean_nonevent"], "--", lw=1.2,
                    color=colors[i], alpha=0.55)
        ax.axvline(0, color="k", ls=":", lw=1)
        ax.set_xlabel("trading days relative to event (0 = event day)")
        ax.set_ylabel("annualised vol" if tlab == "fwd_rv5" else "mean |log return|")
        ax.set_title(f"Event study: {lab}\nsolid = event days, dashed = non-event days")
        ax.legend(fontsize=8)
    fig.suptitle("agentD event study - forward 5-day freight-rate volatility aligned on "
                 "weather-shock days", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, "agentD_event_study.png"), dpi=115)
    plt.close(fig)


def main():
    d = pd.read_csv(os.path.join(OUT, "agentD_panel.csv"), parse_dates=["Date"])
    d, events = make_events(d)
    d.to_csv(os.path.join(OUT, "agentD_panel_events.csv"), index=False)
    print(f"[events] {len(events)} definitions constructed from past-only trailing percentiles")

    tab, win = event_study(d, events)
    tab.to_csv(os.path.join(OUT, "agentD_event_study.csv"), index=False)
    win.to_csv(os.path.join(OUT, "agentD_event_windows.csv"), index=False)

    pd.set_option("display.width", 300)
    print("\n" + "=" * 130)
    print("EVENT STUDY - forward volatility on weather-shock days")
    print("=" * 130)
    show = tab.copy()
    print(show.to_string(index=False))

    make_plot(win, tab)
    print(f"\n[written] agentD_event_study.csv, agentD_event_windows.csv, "
          f"plots/agentD_event_study.png")


if __name__ == "__main__":
    main()
