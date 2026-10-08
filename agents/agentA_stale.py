"""Quantify stale-quote (step-function) structure and its effect on the target."""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)

d = pd.read_csv(r"D:\Internproj\New ML\clean_engineered_dataset.csv")
d["Date"] = pd.to_datetime(d["Date"])
d = d.sort_values(["Route", "Date"]).reset_index(drop=True)

# ---- 1. are the two directions literally duplicated prices? -------------
piv = d.pivot_table(index="Date", columns="Route", values="Price_USD")
print("=== correlation matrix of price levels across routes ===")
print(piv.corr().round(4).to_string())
print("\n=== fraction of dates on which the two directions have IDENTICAL price ===")
for a, b in [("XSICFENE_FarEast_NorthEurope", "XSICNEFE_NorthEurope_FarEast"),
             ("XSICFEUW_FarEast_USWestCoast", "XSICUWFE_USWestCoast_FarEast")]:
    m = piv[[a, b]].dropna()
    print(f"  {a[:12]} vs {b[:12]}: identical on {100*(m[a]==m[b]).mean():.2f}% of dates  "
          f"(corr {m[a].corr(m[b]):.4f}, mean |log ratio| {np.log(m[a]/m[b]).abs().mean():.5f})")

# ---- 2. stale-quote / stall-age structure -------------------------------
d["ret"] = d.groupby("Route")["Price_USD"].transform(lambda s: np.log(s.clip(lower=1)).diff())
def stall_age(s):
    out = np.zeros(len(s), dtype=int)
    c = 0
    for i in range(1, len(s)):
        c = c + 1 if s.iloc[i] == s.iloc[i - 1] else 0
        out[i] = c
    return pd.Series(out, index=s.index)
d["stall"] = d.groupby("Route")["Price_USD"].transform(stall_age)

fwd = d.groupby("Route")["ret"].transform(
    lambda s: (s ** 2).shift(-1).rolling(5, min_periods=5).sum().shift(-4))
d["rv_fwd"] = np.sqrt(fwd) * np.sqrt(252)
d["log_rv_fwd"] = np.log(d["rv_fwd"].clip(lower=1e-8))

print("\n=== mean forward 5d RV by CURRENT stall age (days since last price change) ===")
g = d.dropna(subset=["rv_fwd"]).groupby("stall")["rv_fwd"].agg(["mean", "median", "size"])
print(g.head(20).to_string(float_format=lambda v: f"{v:.4f}"))
print("\ncorr(stall, log_rv_fwd) =", round(d[["stall", "log_rv_fwd"]].dropna().corr().iloc[0, 1], 4))
print("corr(log_rv_w, log_rv_fwd) =",
      round(np.log(np.sqrt(d.groupby('Route')['ret'].transform(lambda s:(s**2).rolling(5).mean())).clip(lower=1e-8))
            .corr(d["log_rv_fwd"]), 4))
print("corr(stall, log_rv_w) =",
      round(d[["stall"]].join(np.log(np.sqrt(d.groupby('Route')['ret'].transform(lambda s:(s**2).rolling(5).mean())).clip(lower=1e-8)).rename("x")).dropna().corr().iloc[0, 1], 4))

# ---- 3. the 'quiet then jump' mechanism ---------------------------------
print("\n=== P(large move in next 5d | stall age) ===")
d["big_fwd"] = (d["rv_fwd"] > 1.0).astype(float)
d.loc[d["rv_fwd"].isna(), "big_fwd"] = np.nan
print(d.dropna(subset=["big_fwd"]).groupby("stall")["big_fwd"].agg(["mean", "size"]).head(18)
      .to_string(float_format=lambda v: f"{v:.4f}"))

# ---- 4. how much of "volatility" is just quote-refresh timing? -----------
print("\n=== share of dates that are quote refreshes (|ret|>0) by year ===")
d["yr"] = d["Date"].dt.year
print(d.groupby("yr")["ret"].apply(lambda s: (s.abs() > 0).mean()).round(4).to_string())

# ---- 5. does stall age beat HAR features? ------------------------------
print("\n=== simple pooled OLS on log_rv_fwd, full sample (in-sample R2) ===")
base = d.dropna(subset=["log_rv_fwd"]).copy()
base["log_rv_d"] = np.log(base["ret"].abs().clip(lower=1e-8))
base["rv_w"] = np.sqrt(base.groupby("Route")["ret"].transform(lambda s: (s ** 2).rolling(5).mean()))
base["rv_m"] = np.sqrt(base.groupby("Route")["ret"].transform(lambda s: (s ** 2).rolling(22).mean()))
base["log_rv_w"] = np.log(base["rv_w"].clip(lower=1e-8))
base["log_rv_m"] = np.log(base["rv_m"].clip(lower=1e-8))
# stall age in the PAST 5 days (strictly lagged version of the mechanism)
base["stall_lag"] = base.groupby("Route")["stall"].shift(1)
for nm, cols in [("HAR core", ["log_rv_d", "log_rv_w", "log_rv_m"]),
                 ("HAR + stall_lag", ["log_rv_d", "log_rv_w", "log_rv_m", "stall_lag"]),
                 ("stall_lag only", ["stall_lag"]),
                 ("stall_lag + log_rv_m", ["stall_lag", "log_rv_m"])]:
    X = base[cols].astype(float).fillna(0.0).to_numpy()
    X = np.column_stack([np.ones(len(X)), X])
    y = base["log_rv_fwd"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"  {nm:22s} R2={r2:.4f}")
