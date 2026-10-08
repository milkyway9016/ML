"""Augment the aggregate metrics table with pooled / trimmed diagnostics."""
import os
import numpy as np
import pandas as pd

OUT = r"D:\testing\outputs"
agg = pd.read_csv(os.path.join(OUT, "agentA_metrics_aggregate.csv"))
pool = pd.read_csv(os.path.join(OUT, "agentA_metrics_pooled.csv"))

cols = ["model", "n", "R2_log", "QLIKE", "QLIKE_trim1", "MAE_vol_pp", "MAE_log_trim1",
        "QLIKE_median_obs"]
pl = pool[cols].rename(columns={
    "n": "n_rows_pooled", "R2_log": "R2_log_pooled", "QLIKE": "QLIKE_pooled",
    "QLIKE_trim1": "QLIKE_pooled_trim1pct", "MAE_vol_pp": "MAE_vol_pp_pooled",
    "MAE_log_trim1": "MAE_log_pooled_trim1pct", "QLIKE_median_obs": "QLIKE_median_per_obs"})
out = agg.merge(pl, on="model", how="left")
out = out[["model", "family", "folds", "MAE_log", "RMSE_log", "R2_log", "QLIKE", "MAE_vol_pp",
           "n_rows_pooled", "R2_log_pooled", "QLIKE_pooled", "QLIKE_pooled_trim1pct",
           "MAE_vol_pp_pooled", "MAE_log_pooled_trim1pct", "QLIKE_median_per_obs"]]
out.to_csv(os.path.join(OUT, "agentA_metrics_aggregate.csv"), index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
