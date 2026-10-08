"""Data-quality / feature-structure diagnostics for the report."""
import numpy as np, pandas as pd, sys
sys.path.insert(0, r"D:\testing\agents")
import agentA_baselines as A
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d = A.add_route_dummies(A.build_dataset())
blocks = A.feature_blocks(d)
print("blocks: route_dummies", len(blocks["route_dummies"]), "weather", len(blocks["weather"]),
      "econ", len(blocks["econ"]))

# ---- 1. how route-specific are the exogenous columns? -------------------
print("\n--- between-route variation share (1 - mean within-date variance/total variance) ---")
rows = []
for c in blocks["weather"] + blocks["econ"]:
    x = d[c].astype(float)
    tot = x.var()
    within = d.groupby("Date")[c].transform(lambda s: s - s.mean()).var()
    icc = 1 - within / tot if tot > 0 else np.nan
    nun = d.groupby("Date")[c].nunique().mean()
    rows.append(dict(col=c, between_route_share=icc, mean_nunique_per_date=nun,
                     nunique_all=d[c].nunique()))
icc = pd.DataFrame(rows).sort_values("between_route_share")
print(icc.head(8).to_string(index=False))
print("...")
print(icc.tail(5).to_string(index=False))
print(f"\ncols with between_route_share < 0.01 (i.e. effectively global/duplicated): "
      f"{int((icc['between_route_share'] < 0.01).sum())} / {len(icc)}")
print(f"cols constant across all rows: {int((icc['nunique_all'] <= 1).sum())}")

# ---- 2. weather-only explanatory power for log_rv_fwd -------------------
print("\n--- in-sample R2 for log_rv_fwd ---")
base = d.dropna(subset=["log_rv_fwd"] + A.NEEDED).copy()
W = base[blocks["weather"]].astype(float).fillna(0.0)
W = W.loc[:, W.std() > 0]
y = base["log_rv_fwd"].to_numpy()
ec = base[blocks["econ"]].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
for nm, X in [
    (f"weather only ({W.shape[1]} non-const)", W.to_numpy()),
    ("econ only (4 raw)", ec.to_numpy()),
    ("HAR core only", base[A.HAR_CORE].astype(float).replace([np.inf, -np.inf], np.nan).fillna(-36.8).to_numpy()),
    ("HAR + weather", np.column_stack([base[A.HAR_CORE].astype(float).replace([np.inf, -np.inf], np.nan).fillna(-36.8).to_numpy(), W.to_numpy()])),
]:
    Xs = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    r2 = 1 - ((y - Xs @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    print(f"  {nm:30s} R2={r2:.4f} adjR2={adj:.4f}  k={X.shape[1]} n={len(y)}")

# ---- 3. PCA of weather block -------------------------------------------
Z = StandardScaler().fit_transform(W)
pca = PCA(n_components=12, random_state=0).fit(Z)
print("\nweather PCA explained-var ratio:", np.round(pca.explained_variance_ratio_, 4))
print("cumulative:", np.round(pca.explained_variance_ratio_.cumsum(), 4))

# ---- 4. zero-return / zero-RV structure ---------------------------------
print("\n--- zero-return structure ---")
print(f"share of exactly-zero daily returns: {(d['sq_ret'] == 0).mean():.4f}")
print(d.assign(zero=(d['sq_ret'] == 0)).groupby("Route")["zero"].mean().round(4).to_string())
print(f"share of zero daily returns by year (2018/2019):")
print(d.assign(zero=(d['sq_ret'] == 0), yr=d['Date'].dt.year).groupby("yr")["zero"].mean().round(4).to_string())
