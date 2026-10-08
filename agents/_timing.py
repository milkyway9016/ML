import time, numpy as np, pandas as pd, xgboost as xgb
rng = np.random.default_rng(0)
X = rng.normal(size=(8400, 40)); y = (rng.random(8400) < 0.4).astype(int)
t0 = time.time()
for _ in range(3):
    m = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                          reg_lambda=2.0, random_state=42, n_jobs=1, tree_method="hist",
                          eval_metric="logloss")
    m.fit(X, y)
print("xgb 250 trees x3 fits: %.2fs -> %.2fs/fit" % (time.time()-t0, (time.time()-t0)/3))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit_transform(X)
t0 = time.time()
for _ in range(3):
    LogisticRegression(max_iter=2000, C=0.5).fit(Xs, y)
print("logit x3: %.2fs" % (time.time()-t0))
import os
print("cpus", os.cpu_count())
