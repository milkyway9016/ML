"""Diagnostic: verify target definition + data pitfalls before running the full study."""
import numpy as np, pandas as pd

SRC = r"D:\Internproj\New ML\clean_engineered_dataset.csv"
HORIZON, ANN = 5, 252

d = pd.read_csv(SRC)
d["Date"] = pd.to_datetime(d["Date"])
d = d.sort_values(["Route", "Date"]).reset_index(drop=True)

d["log_price"] = np.log(d["Price_USD"].clip(lower=1.0))
d["ret"] = d.groupby("Route")["log_price"].diff()
d["sq_ret"] = d["ret"] ** 2

fwd_sq = d.groupby("Route")["sq_ret"].transform(
    lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1)))
d["rv_fwd"] = np.sqrt(fwd_sq) * np.sqrt(ANN)
d["log_rv_fwd"] = np.log(d["rv_fwd"].clip(lower=1e-8))

print("rows", len(d), "rv_fwd NaN", d["rv_fwd"].isna().sum())
print("rv_fwd <= 5e-4 :", (d["rv_fwd"] <= 5e-4).sum())
print(d.loc[d["rv_fwd"] <= 5e-4, "Route"].value_counts())
print(d.loc[d["rv_fwd"] <= 5e-4, "Date"].dt.year.value_counts())
print("frac price unchanged:", (d["Price_USD"].diff().abs() < 1e-9).mean())
print("exact-zero sq_ret share:", (d["sq_ret"] == 0).mean())
print("\nrv_fwd quantiles:", np.nanpercentile(d["rv_fwd"], [1, 5, 25, 50, 75, 95, 99]).round(4))

# are exogenous columns route-specific or global on a date?
for c in ["Origin_LSCI", "Dest_LSCI", "Route_Mean_LSCI", "LSCI_30d_Delta",
          "Route_Mean_air", "SWI", "Coastal_Threat", "Congestion_Risk", "Route_Max_cyclone_wind"]:
    nun = d.groupby("Date")[c].nunique()
    print(f"{c:26s} median nunique per date = {nun.median():.1f}")

# near-duplicate / constant columns
nun = d.nunique()
print("\nconstant cols:", [c for c in d.columns if nun[c] <= 1])
print("\nultra-low-variance cols (nunique<=3):", [c for c in d.columns if nun[c] <= 3])

# leakage check: do 'lagged' weather cols actually correlate with same-day raw weather?
for a, b in [("Cyclone_Wind_Lag7d", "Route_Max_cyclone_wind"), ("Wind_Speed_Lag7d", "Route_Mean_wind_speed")]:
    print(f"corr({a},{b}) = {d[[a,b]].corr().iloc[0,1]:.3f}")

# date coverage: which origins pass the filters
WF_START = pd.Timestamp("2020-01-01")
origins = pd.date_range(WF_START, d["Date"].max(), freq="MS")
ok = []
for o in origins:
    tr = d["Date"] < o
    te = (d["Date"] >= o) & (d["Date"] < o + pd.offsets.MonthBegin(1))
    span_ok = (d.loc[tr, "Date"].max() - d.loc[tr, "Date"].min()).days >= 2 * 365
    ok.append((o, int(tr.sum()), int(te.sum()), span_ok))
ok = pd.DataFrame(ok, columns=["origin", "n_tr", "n_te", "span_ok"])
print("\nfirst 4 origins:\n", ok.head(4).to_string(index=False))
print("origins with n_te>=20:", int((ok.n_te >= 20).sum()), "/", len(ok))
print("origins usable:", int(((ok.n_te >= 20) & (ok.n_tr >= 500) & ok.span_ok).sum()))
print("last origin:", ok.origin.max().date())
