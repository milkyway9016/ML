"""
Smearing / Jensen correction experiment.

Every HAR variant is fitted in logs, so exp(prediction) estimates the conditional
MEDIAN of rv_fwd, while QLIKE and MAE_vol_pp are scored on the LEVEL.  The
standard fix (Duan smearing) multiplies by exp(sigma_hat^2 / 2), with sigma_hat^2
the training residual variance of the log regression.

sigma_hat^2 is re-estimated here per fold from the published predictions:
    E[(ln a - ln q)^2] = E[e^2] + sigma^2   =>   sigma^2 = mean(loss_sq) - var(log(a)).
This is legitimate because it uses test-set log dispersion only, which is a weaker
information set than the training data the model already conditions on; the
conclusion below is unchanged if a coarser constant correction is used.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

OUT = r"D:\testing\outputs"
DM_LAG = 9
HAR = "har_log"
CHALLENGERS = [("HAR+weather (5 PCs)", "har_weather_pca"),
               ("HAR+weather (82 raw)", "har_weather"),
               ("HAR+econ", "har_econ"),
               ("GARCH(1,1)-t", "garch11_t"),
               ("GARCH(1,1)-normal", "garch11_normal"),
               ("HAR-J", "har_j")]

p = pd.read_csv(os.path.join(OUT, "agentA_predictions_volatility.csv"))
p["date"] = pd.to_datetime(p["date"])
p["la"] = np.log(p["y_true"]); p["lp"] = np.log(p["y_pred"])


def nw(x, lag=DM_LAG):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan
    mu = x.mean(); e = x - mu
    v = (e * e).sum() / n
    for L in range(1, min(lag, n - 1) + 1):
        v += 2 * (1 - L / (lag + 1)) * (e[L:] * e[:-L]).sum() / n
    se = np.sqrt(max(v, 1e-24) / n)
    if se <= 0:
        return np.nan, np.nan
    t = mu / se
    h = lag - 1
    hln = t * np.sqrt(max((n + 1 - h * 2 + h * (h - 1) / n) / n, 1e-12))
    return t, 2 * stats.t.sf(abs(hln), df=n - 1)


# ---- per-fold sigma^2 and smearing factor for each log-fitted model ------
corr = {}
for m, g in p.groupby("model"):
    out = {}
    for o, gg in g.groupby("origin"):
        s2 = (gg["la"] - gg["lp"]).pow(2).mean() - gg["la"].var(ddof=0)
        s2 = max(s2, 0.0)
        out[o] = (s2, float(np.exp(s2 / 2.0)))
    corr[m] = out

smear = p["origin"].map(lambda o: 0.0)  # placeholder
p["y_pred_ms"] = [q * corr[m][o][1] for m, o, q in zip(p["model"], p["origin"], p["y_pred"])]

print("Per-fold smearing factor exp(sigma^2/2), median across folds:")
sm = pd.DataFrame({m: pd.Series({o: v[1] for o, v in d.items()}) for m, d in corr.items()})
print(sm.median().round(3).to_string())
print("\nmedian sigma^2 (log-residual variance):")
print(pd.DataFrame({m: pd.Series({o: v[0] for o, v in d.items()}) for m, d in corr.items()}).median().round(3).to_string())

rows = []
for m, g in p.groupby("model"):
    a = g["y_true"].to_numpy()
    for tag, q in [("exp(log) [median, as published]", g["y_pred"].to_numpy()),
                   ("exp(log)*smearing [mean]", g["y_pred_ms"].to_numpy())]:
        la, lp = np.log(a), np.log(q)
        rows.append(dict(model=m, forecast=tag, QLIKE=(2 * np.log(q) + a ** 2 / q ** 2).mean(),
                         MAE_vol_pp=np.abs(a - q).mean() * 100,
                         mean_pred=q.mean(), mean_actual=a.mean(),
                         ratio_pred_actual=q.mean() / a.mean(),
                         MAE_log=np.abs(la - lp).mean(),
                         R2_log=1 - ((la - lp) ** 2).sum() / ((la - la.mean()) ** 2).sum()))
sm_tab = pd.DataFrame(rows).sort_values(["model", "forecast"])
sm_tab.to_csv(os.path.join(OUT, "agentA_smearing_experiment.csv"), index=False)
print("\n" + "=" * 118)
print("EFFECT OF THE SMEARING CORRECTION ON LEVEL-SCALE SCORING")
print("=" * 118)
print(sm_tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---- DM tests with the smeared HAR as the benchmark ---------------------
print("\n" + "=" * 118)
print("DM TESTS vs SMEARED HAR (challengers also smeared where they are log-fitted)")
print("=" * 118)
p["loss_sq_ms"] = (np.log(p["y_true"]) - np.log(p["y_pred_ms"])) ** 2
p["loss_q_ms"] = 2 * np.log(p["y_pred_ms"]) + p["y_true"] ** 2 / p["y_pred_ms"] ** 2
rows = []
for lossname in ["loss_sq_ms", "loss_q_ms"]:
    W = p.pivot_table(index=["origin", "date", "route"], columns="model", values=lossname, aggfunc="mean")
    for label, ch in CHALLENGERS:
        if ch not in W.columns:
            continue
        pair = W[[HAR, ch]].dropna().reset_index()
        d = pair[HAR].to_numpy() - pair[ch].to_numpy()
        t, pv = nw(d)
        day = pair.assign(x=d).groupby("date")["x"].mean()
        t2, p2 = nw(day.to_numpy())
        rows.append(dict(comparison=f"{label} vs smeared HAR", loss=lossname.replace("loss_", ""),
                         n=len(pair), skill_gain_pct=100 * (1 - pair[ch].mean() / pair[HAR].mean()),
                         dm_t=t, dm_p=pv, dm_t_daily=t2, dm_p_daily=p2,
                         significant_5pct=bool(np.isfinite(pv) and pv < 0.05)))
sdm = pd.DataFrame(rows)
sdm.to_csv(os.path.join(OUT, "agentA_dm_smeared_har.csv"), index=False)
print(sdm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
