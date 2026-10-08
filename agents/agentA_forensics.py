"""Raw-data forensics: the extreme-volatility events and stale-price runs."""
import numpy as np, pandas as pd
pd.set_option("display.width", 220)

d = pd.read_csv(r"D:\Internproj\New ML\clean_engineered_dataset.csv")
d["Date"] = pd.to_datetime(d["Date"])
d = d.sort_values(["Route", "Date"]).reset_index(drop=True)
d["ret"] = d.groupby("Route")["Price_USD"].transform(lambda s: np.log(s.clip(lower=1)).diff())

print("=== largest absolute daily log returns ===")
print(d.reindex(d["ret"].abs().sort_values(ascending=False).index)
      .head(12)[["Date", "Route", "Price_USD", "ret"]].to_string(index=False))

print("\n=== raw price path, FarEast->USWestCoast, 2020-07 .. 2020-09 ===")
s = d[(d["Route"] == "XSICFEUW_FarEast_USWestCoast") &
      (d["Date"] >= "2020-07-01") & (d["Date"] <= "2020-09-15")][["Date", "Price_USD", "ret"]]
print(s.to_string(index=False))
print("\n=== how many DISTINCT prices in that window? ===", s["Price_USD"].nunique(), "of", len(s))

print("\n=== stale-price runs: consecutive identical Price_USD per route ===")
def maxrun(v):
    mx = cur = 1
    for i in range(1, len(v)):
        cur = cur + 1 if v[i] == v[i - 1] else 1
        mx = max(mx, cur)
    return mx
runs = d.groupby("Route")["Price_USD"].apply(lambda s: maxrun(s.to_numpy()))
print(runs.to_string())
print("\n=== overall: distribution of daily |ret| ===")
print(d["ret"].abs().describe(percentiles=[.5, .9, .99, .999]).to_string())

print("\n=== how much of the 5-day forward RV variance comes from the top 1% of days? ===")
H = 5
d["sq"] = d["ret"] ** 2
top = d["sq"].nlargest(int(0.01 * len(d))).sum()
print(f"top 1% of sq_ret accounts for {100*top/d['sq'].sum():.1f}% of total sum of squared returns")

print("\n=== target tail: rv_fwd > 3 (300% annualised) ===")
fwd = d.groupby("Route")["sq"].transform(lambda s: s.shift(-1).rolling(H, min_periods=H).sum().shift(-(H-1)))
d["rv_fwd"] = np.sqrt(fwd) * np.sqrt(252)
big = d[d["rv_fwd"] > 3]
print(f"count = {len(big)} rows ({100*len(big)/len(d):.2f}%)")
print(big.groupby([big["Date"].dt.to_period("M"), "Route"]).size().to_string())

print("\n=== 2021-02: the fold with the -1.84 train/test level gap ===")
s = d[(d["Route"] == "XSICFEUW_FarEast_USWestCoast") &
      (d["Date"] >= "2021-01-15") & (d["Date"] <= "2021-03-05")][["Date", "Price_USD", "ret"]]
print(s.to_string(index=False))
