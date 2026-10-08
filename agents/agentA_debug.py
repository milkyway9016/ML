"""Debug: EGARCH failure + are exogenous columns route-specific?"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r"D:\testing\agents")
import agentA_baselines as A
from arch import arch_model

d = A.add_route_dummies(A.build_dataset())
complete = d[A.NEEDED].notna().all(axis=1)
origin = pd.Timestamp("2020-06-01")
tr = (d["Date"] < origin) & complete
r = "XSICFENE_FarEast_NorthEurope"
sub_tr = d.index[tr & (d["Route"] == r)]
y = d.loc[sub_tr, "ret"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy() * 100.0
print("y", len(y), y.std())

for spec, kw in A.GARCH_KW.items():
    try:
        am = arch_model(y, mean="Constant", **kw)
        res = am.fit(disp="off", show_warning=False, options={"maxiter": 150})
        fc = res.forecast(horizon=30, reindex=False)
        var = np.asarray(fc.variance.to_numpy(), float)
        print(f"{spec:16s} OK nobs={res.nobs} var_shape={var.shape} "
              f"finite={np.isfinite(var).all()} min={np.nanmin(var):.4g} mean={np.nanmean(var):.4g}")
    except Exception as e:
        print(f"{spec:16s} FAIL {type(e).__name__}: {str(e)[:200]}")

# --- are exogenous vars route-specific or duplicated across routes? ---
print("\n--- distinct values per date, by column (4 routes present per date) ---")
for c in ["Origin_LSCI", "Dest_LSCI", "Route_Mean_LSCI", "LSCI_30d_Delta", "Route_Mean_air",
          "SWI", "Coastal_Threat", "Congestion_Risk", "Route_Max_cyclone_wind",
          "air_South_China_Sea", "cyclone_active_South_China_Sea", "Month"]:
    g = d.groupby("Date")[c].nunique()
    print(f"  {c:32s} nunique/date mean={g.mean():.2f} max={g.max()}")

print("\n--- R^2 of weather block for log_rv_fwd (in-sample, pooled) ---")
import itertools
blocks = A.feature_blocks(d)
trio = d[A.HAR_CORE + blocks["weather"] + ["Route"]].dropna()
X = pd.get_dummies(trio[blocks["weather"]], drop_first=True).astype(float)
X = np.column_stack([np.ones(len(X)), X])
y2 = trio["log_rv_fwd"].to_numpy()
b, *_ = np.linalg.lstsq(X, y2, rcond=None)
r2 = 1 - ((y2 - X @ b) ** 2).sum() / ((y2 - y2.mean()) ** 2).sum()
print(f"weather-only OLS in-sample R2 = {r2:.4f}  (n={len(y2)}, k={X.shape[1]})")

# PCA of weather block
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
W = d[blocks["weather"]].astype(float).fillna(0.0)
Z = StandardScaler().fit_transform(W)
p = PCA(n_components=10, random_state=0).fit(Z)
print("weather PCA explained var:", np.round(p.explained_variance_ratio_, 4), "cum", round(p.explained_variance_ratio_.cumsum()[-1], 4))
