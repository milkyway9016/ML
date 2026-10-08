"""Deep diagnostic of the evaluation: why negative R2 / huge QLIKE?"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r"D:\testing\agents")
pd.set_option("display.width", 200)

p = pd.read_csv(r"D:\testing\outputs\agentA_predictions_volatility.csv")
fold = pd.read_csv(r"D:\testing\outputs\agentA_metrics_by_fold.csv")
p["date"] = pd.to_datetime(p["date"])
p["la"] = np.log(p["y_true"]); p["lp"] = np.log(p["y_pred"])
p["e"] = p["la"] - p["lp"]
p["qlike_obs"] = 2 * np.log(p["y_pred"]) + p["y_true"] ** 2 / p["y_pred"] ** 2

print("=== POOLED (over all scored rows) vs MEAN-OF-FOLD metrics ===")
rows = []
for m, g in p.groupby("model"):
    la, lp = g["la"].to_numpy(), g["lp"].to_numpy()
    sst = ((la - la.mean()) ** 2).sum()
    r2_pool = 1 - ((la - lp) ** 2).sum() / sst
    r2_vs_zero = 1 - ((la - lp) ** 2).sum() / (la ** 2).sum()
    fm = fold[fold["model"] == m]
    rows.append(dict(model=m, R2_pooled=r2_pool, R2_foldmean=fm["R2_log"].mean(),
                     corr=float(np.corrcoef(la, lp)[0, 1]),
                     n_folds_neg_r2=int((fm["R2_log"] < 0).sum()), n_folds=len(fm),
                     mean_act=g["y_true"].mean(), mean_pred=g["y_pred"].mean(),
                     med_pred=g["y_pred"].median(),
                     QLIKE_pooled=g["qlike_obs"].mean(),
                     QLIKE_medfold=fm["QLIKE"].median(),
                     QLIKE_foldmean=fm["QLIKE"].mean(),
                     QLIKE_maxfold=fm["QLIKE"].max(),
                     MAE_log_pooled=np.abs(g["e"]).mean()))
diag = pd.DataFrame(rows).sort_values("QLIKE_pooled")
print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\n=== where do the extreme QLIKE values come from? (har_log & har_weather & har_econ) ===")
for m in ["har_log", "har_weather", "har_econ", "har_weather_econ", "har_route", "garch11_t"]:
    g = p[p["model"] == m].copy()
    g["contrib"] = g["qlike_obs"]
    top = g.nlargest(3, "contrib")[["origin", "date", "route", "y_true", "y_pred", "contrib"]]
    tot = g["contrib"].sum()
    print(f"\n-- {m}: pooled mean QLIKE {g['contrib'].mean():.3f}; top-3 obs contribute "
          f"{100*g.nlargest(10,'contrib')['contrib'].sum()/tot:.1f}% of the total loss (top 10 obs)")
    print(top.to_string(index=False))
    print(f"   y_pred quantiles: {np.round(np.percentile(g['y_pred'],[0,1,5,50,95,99,100]),5)}")
    print(f"   y_true quantiles: {np.round(np.percentile(g['y_true'],[0,1,5,50,95,99,100]),5)}")

print("\n=== is the target predictable at all? autocorrelation / AR(1) sanity ===")
import agentA_baselines as A
d = A.build_dataset()
complete = d[A.NEEDED].notna().all(axis=1)
sub = d[complete & (d["Date"] >= pd.Timestamp("2020-01-01"))]
x = sub["log_rv_fwd"].to_numpy(); l1 = sub["log_rv_w"].to_numpy(); l2 = sub["log_rv_m"].to_numpy()
print("corr(log_rv_fwd, log_rv_w) =", round(np.corrcoef(x, l1)[0, 1], 4))
print("corr(log_rv_fwd, log_rv_m) =", round(np.corrcoef(x, l2)[0, 1], 4))
print("std(log_rv_fwd) overall =", round(x.std(), 4), " std(log_rv_w) =", round(l1.std(), 4))

print("\n=== per-fold target level: does the test month sit inside the train range? ===")
rows = []
for o, g in p[p["model"] == "har_log"].groupby("origin"):
    tr = d[complete & (d["Date"] < o)]
    rows.append(dict(origin=o, mean_tr_log=tr["log_rv_fwd"].mean(), mean_te_log=g["la"].mean(),
                     sd_te=g["la"].std(), n=len(g)))
r = pd.DataFrame(rows)
r["gap"] = r["mean_te_log"] - r["mean_tr_log"]
print(r.describe().to_string(float_format=lambda v: f"{v:.3f}"))
print("\nlargest train/test level gaps:")
print(r.reindex(r["gap"].abs().sort_values(ascending=False).index).head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n=== within-month: is there enough variation to beat the fold mean? ===")
v = p[p["model"] == "har_log"].groupby("origin")["la"].std()
print("sd of log_rv_fwd WITHIN test month: mean", round(v.mean(), 4), "median", round(v.median(), 4))
print("sd of log_rv_fwd ACROSS all rows  :", round(p[p['model']=='har_log']['la'].std(), 4))
