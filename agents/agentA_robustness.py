"""
Robustness suite on the saved agentA predictions.

Three things the headline (mean-over-folds) metrics hide:
  1. the mean per-fold QLIKE is dominated by a handful of observations;
  2. R2 computed within a single test month is a tiny-sample statistic;
  3. the two 'reversed' routes are byte-identical price series, so every
     cross-sectional statistic double-counts.
This script reports pooled metrics, trimmed metrics, and DM tests after
collapsing the duplicate route pairs.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = r"D:\testing\outputs"
DM_LAG = 9
MODELS = ["har_log", "har_route", "har_j", "har_cj", "har_ext", "har_econ", "har_weather",
          "har_weather_pca", "har_weather_econ", "har_wpca_epca", "har_log_strictlag",
          "garch11_normal", "garch11_t", "egarch11_t", "gjr_garch11_t"]
HAR = "har_log"

p = pd.read_csv(os.path.join(OUT, "agentA_predictions_volatility.csv"))
p["date"] = pd.to_datetime(p["date"])
p["la"] = np.log(p["y_true"]); p["lp"] = np.log(p["y_pred"])
p["loss_sq"] = (p["la"] - p["lp"]) ** 2
p["loss_qlike"] = 2 * np.log(p["y_pred"]) + p["y_true"] ** 2 / p["y_pred"] ** 2


def nw_t(x, lag=DM_LAG):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n
    mu = x.mean(); e = x - mu
    g0 = (e * e).sum() / n
    v = g0
    for L in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - L / (lag + 1)) * (e[L:] * e[:-L]).sum() / n
    se = np.sqrt(max(v, 1e-24) / n)
    if se <= 0:
        return np.nan, np.nan, n
    t = mu / se
    h = lag - 1
    hln = t * np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    return t, 2 * stats.t.sf(abs(hln), df=n - 1), n


# --------------------------------------------------------------------------
# 1. POOLED metrics (all scored rows together, no per-fold averaging)
# --------------------------------------------------------------------------
rows = []
for m, g in p.groupby("model"):
    la, lp, a, q = g["la"].to_numpy(), g["lp"].to_numpy(), g["y_true"].to_numpy(), g["y_pred"].to_numpy()
    ls = (la - lp) ** 2
    lq = 2 * np.log(q) + a ** 2 / q ** 2
    keep = np.ones(len(g), bool)
    # trimmed: drop the top 1% of observations by total squared-error loss
    thr = np.quantile(ls, 0.99)
    k2 = ls <= thr
    rows.append(dict(
        model=m, n=len(g), n_dates=g["date"].nunique(),
        MAE_log=np.abs(la - lp).mean(), RMSE_log=np.sqrt(ls.mean()),
        R2_log=1 - ls.sum() / ((la - la.mean()) ** 2).sum(),
        QLIKE=lq.mean(),
        MAE_vol_pp=np.abs(a - q).mean() * 100,
        QLIKE_trim1=lq[k2].mean(), MAE_log_trim1=np.abs(la - lp)[k2].mean(),
        R2_log_trim1=1 - ls[k2].sum() / ((la - la.mean()) ** 2).sum(),
        QLIKE_median_obs=float(np.median(lq)),
        share_loss_top1pct=float(ls[np.argsort(-ls)[:max(1, len(g)//100)]].sum() / ls.sum()),
    ))
pool = pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)
pool.to_csv(os.path.join(OUT, "agentA_metrics_pooled.csv"), index=False)
print("=" * 110)
print("POOLED metrics over all scored rows (compare with the mean-over-folds file)")
print("=" * 110)
print(pool.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

# --------------------------------------------------------------------------
# 2. DM tests under three refinements
# --------------------------------------------------------------------------
# (a) collapse duplicate route pairs: mean loss per (origin, date, direction) then
#     average the two identical directions -> one observation per date per lane.
def collapse(g):
    return g.groupby(["origin", "date"]).mean(numeric_only=True)


pairs = [
    ("HAR+weather (82 raw) vs HAR", "har_weather"),
    ("HAR+weather (5 PCs) vs HAR", "har_weather_pca"),
    ("HAR+econ vs HAR", "har_econ"),
    ("HAR+ext vs HAR", "har_ext"),
    ("HAR-J vs HAR", "har_j"),
    ("HAR per-route vs HAR", "har_route"),
    ("HAR strict-lag vs HAR", "har_log_strictlag"),
    ("GARCH(1,1)-t vs HAR", "garch11_t"),
    ("EGARCH(1,1)-t vs HAR", "egarch11_t"),
]

res = []
for label, chal in pairs:
    for lossname in ["loss_sq", "loss_qlike"]:
        W = p.pivot_table(index=["origin", "date", "route"], columns="model", values=lossname, aggfunc="mean")
        if chal not in W.columns or HAR not in W.columns:
            continue
        pair = W[[HAR, chal]].dropna().reset_index()
        d_all = pair[HAR].to_numpy() - pair[chal].to_numpy()
        t_all, p_all, n_all = nw_t(d_all)

        # (a) duplicate-collapsed, daily series
        pc = pair.copy()
        pc["d"] = pc[HAR] - pc[chal]
        day = pc.groupby("date")["d"].mean().sort_index()
        t_day, p_day, n_day = nw_t(day.to_numpy())

        # (b) 1%-trimmed on the paired squared-error loss
        pc["ls"] = pair[HAR].to_numpy() + pair[chal].to_numpy()
        k = pc["ls"] <= np.quantile(pc["ls"], 0.99)
        t_tr, p_tr, n_tr = nw_t(pc.loc[k, "d"].to_numpy())

        # (c) sign test on the daily collapsed differential
        nz = day[day != 0]
        n_pos = int((nz > 0).sum())
        sign_p = float(stats.binomtest(n_pos, len(nz), 0.5).pvalue) if len(nz) > 0 else np.nan

        res.append(dict(comparison=label, challenger=chal, loss=lossname,
                        n_pairs=n_all, n_dates=n_day,
                        skill_gain_pct=100 * (1 - pair[chal].mean() / pair[HAR].mean()),
                        dm_t_all=t_all, dm_p_all=p_all,
                        dm_t_dailycollapsed=t_day, dm_p_dailycollapsed=p_day,
                        dm_t_trim1=t_tr, dm_p_trim1=p_tr,
                        sign_frac_challenger_better=float(n_pos / len(nz)) if len(nz) else np.nan,
                        sign_test_p=sign_p))
rb = pd.DataFrame(res)
rb.to_csv(os.path.join(OUT, "agentA_dm_robustness.csv"), index=False)
print("\n" + "=" * 110)
print("DM ROBUSTNESS: all-pairs / duplicate-collapsed-to-daily / 1%-trimmed  (+ sign test)")
print("=" * 110)
print(rb[rb["loss"] == "loss_sq"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print()
print(rb[rb["loss"] == "loss_qlike"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --------------------------------------------------------------------------
# 3. how concentrated is the loss?  (diagnostic table)
# --------------------------------------------------------------------------
print("\n" + "=" * 110)
print("LOSS CONCENTRATION: share of total squared-error loss from the top 1% of observations")
print("=" * 110)
for m in [HAR, "har_weather_pca", "har_weather", "garch11_t"]:
    g = p[p["model"] == m]
    ls = g["loss_sq"].to_numpy()
    for frac in [0.001, 0.01, 0.05]:
        k = max(1, int(frac * len(ls)))
        top = np.sort(ls)[::-1][:k].sum() / ls.sum()
        print(f"  {m:18s} top {100*frac:5.1f}% of obs -> {100*top:6.2f}% of total loss")
    print()
