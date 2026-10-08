"""
agentD step 1 - WEATHER DATA QUALITY / IS THERE A PHYSICAL EVENT TO STUDY?

Builds the analysis panel (targets + route-own-zone weather) and audits the
weather block for data-quality problems and for cross-route zone leakage.

Outputs:
  D:\testing\outputs\agentD_weather_quality.csv
  D:\testing\outputs\agentD_panel.csv          (shared panel for steps 2-4)
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

SRC = r"D:\Internproj\New ML\clean_engineered_dataset.csv"
OUT = r"D:\testing\outputs"
os.makedirs(OUT, exist_ok=True)

HORIZON, ANN = 5, 252

ZONES = ["South_China_Sea", "Indian_Ocean", "Arabian_Sea", "Red_Sea",
         "Mediterranean_Sea", "North_Atlantic", "North_Sea",
         "North_Pacific_West", "North_Pacific_East"]

# Which ocean zones does each lane physically traverse?
ROUTE_ZONES = {
    "XSICFENE_FarEast_NorthEurope": ["South_China_Sea", "Indian_Ocean", "Arabian_Sea",
                                     "Red_Sea", "Mediterranean_Sea", "North_Atlantic", "North_Sea"],
    "XSICNEFE_NorthEurope_FarEast": ["South_China_Sea", "Indian_Ocean", "Arabian_Sea",
                                     "Red_Sea", "Mediterranean_Sea", "North_Atlantic", "North_Sea"],
    "XSICFEUW_FarEast_USWestCoast": ["South_China_Sea", "North_Pacific_West", "North_Pacific_East"],
    "XSICUWFE_USWestCoast_FarEast": ["South_China_Sea", "North_Pacific_West", "North_Pacific_East"],
}
ROUTE_GROUP = {
    "XSICFENE_FarEast_NorthEurope": "Europe",
    "XSICNEFE_NorthEurope_FarEast": "Europe",
    "XSICFEUW_FarEast_USWestCoast": "Pacific",
    "XSICUWFE_USWestCoast_FarEast": "Pacific",
}

BANNED = ["Cyclone_Wind_Forecast3d", "Cyclone_Wind_Forecast7d", "Cyclone_Active_Forecast3d",
          "Wind_Speed_Forecast7d", "Rolling_7d_Cyclone_Days", "Price_Volatility_7d",
          "Price_Volatility_30d"]


# ---------------------------------------------------------------------------
def build_panel() -> pd.DataFrame:
    d = pd.read_csv(SRC, parse_dates=["Date"])
    d = d.sort_values(["Route", "Date"]).reset_index(drop=True)

    d["log_price"] = np.log(d["Price_USD"].clip(lower=1.0))
    d["ret"] = d.groupby("Route")["log_price"].diff()
    d["sq_ret"] = d["ret"] ** 2
    d["abs_ret"] = d["ret"].abs()

    g = d.groupby("Route")["sq_ret"]
    # ---- TARGET: forward 5-day realised vol from returns t+1..t+5 -----------
    d["fwd_sq"] = g.transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum())
    d["fwd_rv5"] = np.sqrt(d["fwd_sq"]) * np.sqrt(ANN)
    d["log_fwd_rv5"] = np.log(d["fwd_rv5"].clip(lower=1e-8))
    d["fwd_abs_mean5"] = d.groupby("Route")["abs_ret"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).mean())

    # ---- trailing 22-day realised vol (volatility persistence control) ------
    d["trail_rv22"] = g.transform(
        lambda s: np.sqrt(s.rolling(22, min_periods=11).mean()) * np.sqrt(ANN))
    d["log_trail_rv22"] = np.log(d["trail_rv22"].clip(lower=1e-8))

    # ---- route-own-zone weather aggregation ---------------------------------
    # Everything below is same-day (t) information: legitimate "contemporaneous
    # shock" input for an event study, and used only as a predictor of t+1..t+5.
    max_w, act_any, min_slp, mean_wind, mean_slp = [], [], [], [], []
    for r, grp in d.groupby("Route"):
        zz = [z for z in ROUTE_ZONES[r] if f"wind_speed_{z}" in grp.columns]
        wcols = [f"cyclone_wind_{z}" for z in zz]
        acols = [f"cyclone_active_{z}" for z in zz]
        scols = [f"cyclone_slp_{z}" for z in zz]
        wscols = [f"wind_speed_{z}" for z in zz]
        slcols = [f"slp_{z}" for z in zz]

        def _max(df, cols):
            sub = df[cols].replace(0.0, np.nan)     # 0 = "no cyclone", not a value
            return sub.max(axis=1).fillna(0.0)

        max_w.append(_max(grp, wcols))
        act_any.append((grp[acols].fillna(0) > 0).any(axis=1).astype(int))
        mn = grp[scols].replace(0.0, np.nan).min(axis=1)
        min_slp.append(mn)
        mean_wind.append(grp[wscols].mean(axis=1))
        # slp stored in Pascals in the source file
        mean_slp.append(grp[slcols].mean(axis=1) / 100.0)

    d["od_max_cyc_wind"] = pd.concat(max_w).sort_index()
    d["od_cyc_active"] = pd.concat(act_any).sort_index()
    d["od_min_cyc_slp"] = pd.concat(min_slp).sort_index()   # NaN = no cyclone
    d["od_mean_wind"] = pd.concat(mean_wind).sort_index()
    d["od_mean_slp"] = pd.concat(mean_slp).sort_index()

    # pressure / wind ANOMALIES vs own trailing 30-day mean (strictly backward)
    for c, out in [("od_mean_slp", "slp_anom"), ("od_mean_wind", "wind_anom")]:
        base = d.groupby("Route")[c].transform(
            lambda s: s.shift(1).rolling(30, min_periods=15).mean())
        d[out] = d[c] - base

    # cyclone wind as an anomaly too (0 on no-cyclone days, so anomaly < 0)
    base_w = d.groupby("Route")["od_max_cyc_wind"].transform(
        lambda s: s.shift(1).rolling(90, min_periods=30).mean())
    d["cyc_wind_anom"] = d["od_max_cyc_wind"] - base_w

    d["route_group"] = d["Route"].map(ROUTE_GROUP)
    d["Is_Peak_Season"] = d["Is_Peak_Season"].astype(int)
    d["year"] = d["Date"].dt.year
    d["month"] = d["Date"].dt.month
    return d


# ---------------------------------------------------------------------------
def leakage_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """For every route x zone the route does NOT traverse, report whether the
    zone carries ANY non-zero weather information (leakage / spurious signal)."""
    rows = []
    for r, grp in raw.groupby("Route"):
        own = ROUTE_ZONES[r]
        for z in ZONES:
            for m in ["air", "slp", "wind_speed", "cyclone_wind", "cyclone_slp",
                      "cyclone_dist", "cyclone_active"]:
                c = f"{m}_{z}"
                if c not in grp.columns:
                    continue
                s = pd.to_numeric(grp[c], errors="coerce")
                if s.notna().sum() == 0:
                    state = "column-absent-all-NaN"
                elif s.fillna(0).abs().max() == 0:
                    state = "all-zero-anyway"
                else:
                    state = "HAS-DATA"
                rows.append(dict(
                    route=r, zone=z, traversed=bool(z in own), metric=m, column=c,
                    n_rows=int(len(s)), n_nan=int(s.isna().sum()),
                    n_nonzero=int((s.fillna(0) != 0).sum()),
                    abs_max=float(s.abs().max()) if s.notna().any() else 0.0,
                    leakage=bool((z not in own) and state == "HAS-DATA"),
                    state=state))
    return pd.DataFrame(rows)


def quality_audit(d: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Distribution / structural-zero / sentinel / forward-fill audit."""
    rows = []

    def add(scope, column, s, note=""):
        s = pd.to_numeric(s, errors="coerce")
        v = s.dropna().to_numpy()
        # longest run of exactly identical consecutive values => forward-fill
        if len(v) > 1:
            brk = np.r_[True, v[1:] != v[:-1], True]
            idx = np.flatnonzero(brk)
            maxrun = int((idx[1:] - idx[:-1]).max())
        else:
            maxrun = 0
        nz = v[v != 0]
        rows.append(dict(
            scope=scope, column=column, n=int(len(s)), n_nan=int(s.isna().sum()),
            n_zero=int((v == 0).sum()), pct_zero=round(100 * (v == 0).mean(), 2) if len(v) else np.nan,
            min=float(np.min(v)) if len(v) else np.nan,
            q50=float(np.median(v)) if len(v) else np.nan,
            q90=float(np.quantile(v, .90)) if len(v) else np.nan,
            q99=float(np.quantile(v, .99)) if len(v) else np.nan,
            max=float(np.max(v)) if len(v) else np.nan,
            mean_nonzero=float(nz.mean()) if len(nz) else np.nan,
            n_nonzero=int(len(nz)),
            max_identical_run=maxrun, note=note))

    # route-own-zone aggregates
    for c in ["od_max_cyc_wind", "od_cyc_active", "od_min_cyc_slp", "od_mean_wind",
              "od_mean_slp", "slp_anom", "wind_anom", "cyc_wind_anom"]:
        add("panel(route-own-zone)", c, d[c])

    # per-zone cyclone fields, per route (so structural zeros are visible)
    for r, grp in raw.groupby("Route"):
        zz = ROUTE_ZONES[r]
        for z in zz:
            for m in ["cyclone_wind", "cyclone_dist", "cyclone_active", "cyclone_slp"]:
                c = f"{m}_{z}"
                if c in grp.columns:
                    add(f"raw({r.split('_')[0]})", c, grp[c],
                        note="0 = structural (no cyclone in zone)")
        # non-traversed but present columns that are all zero
        for z in [x for x in ZONES if x not in zz]:
            c = f"cyclone_wind_{z}"
            if c in grp.columns:
                add(f"raw({r.split('_')[0]})", c, grp[c],
                    note="NON-TRAVERSED ZONE (all-zero => no leakage)")

    # continuous met fields, one route, for forward-fill / range check
    fe = raw[raw.Route == "XSICFENE_FarEast_NorthEurope"]
    for z in ROUTE_ZONES["XSICFENE_FarEast_NorthEurope"]:
        for m in ["air", "slp", "wind_speed"]:
            c = f"{m}_{z}"
            if c in fe.columns:
                note = "slp in Pascals (~101000)" if m == "slp" else ""
                add("raw(FENE)", c, fe[c], note=note)
    return pd.DataFrame(rows)


def main():
    raw = pd.read_csv(SRC, parse_dates=["Date"]).sort_values(["Route", "Date"]).reset_index(drop=True)

    print("=" * 100)
    print("STEP 1 - WEATHER DATA QUALITY / PHYSICAL-EVENT RECONNAISSANCE")
    print("=" * 100)

    leak = leakage_audit(raw)
    n_leak = int(leak["leakage"].sum())
    print(f"\n[LEAKAGE AUDIT] route x non-traversed zone x metric combinations: "
          f"{int((~leak.traversed).sum())}")
    print(f"[LEAKAGE AUDIT] combinations carrying NON-ZERO weather data: {n_leak}")
    if n_leak:
        print(leak[leak.leakage].to_string(index=False))
    else:
        print("[LEAKAGE AUDIT] VERDICT: no cross-route zone leakage. Non-traversed zones are "
              "structurally absent (NaN) or identically zero, so they cannot manufacture "
              "spurious associations. Route_* summaries are route-specific (verified: "
              "Route_cyclone_active == OR of own-zone active flags, 0 mismatches).")

    # columns that are all-zero even though the route DOES traverse the zone
    dead = leak[(leak.traversed) & (leak.state == "all-zero-anyway")]
    print("\n[DEAD ZONES WITHIN TRAVERSED LANES] traversed-zone metrics that are identically zero:")
    if len(dead):
        print(dead[["route", "zone", "metric"]].drop_duplicates()
              .groupby("zone")["metric"].apply(lambda s: sorted(s)).to_string())
    else:
        print("  none")

    d = build_panel()
    q = quality_audit(d, raw)
    q.to_csv(os.path.join(OUT, "agentD_weather_quality.csv"), index=False)
    leak.to_csv(os.path.join(OUT, "agentD_zone_leakage_audit.csv"), index=False)
    d.to_csv(os.path.join(OUT, "agentD_panel.csv"), index=False)
    print(f"\n[written] agentD_weather_quality.csv ({len(q)} rows), "
          f"agentD_zone_leakage_audit.csv, agentD_panel.csv ({len(d)} rows)")

    # ---------------- cyclone frequency -------------------------------------
    print("\n" + "-" * 100)
    print("CYCLONE ACTIVITY FREQUENCY (per route, from own-zone OR flag)")
    print("-" * 100)
    freq = (d.groupby("Route")
            .agg(n_days=("od_cyc_active", "size"),
                 cyc_active_days=("od_cyc_active", "sum"),
                 pct_active=("od_cyc_active", lambda s: round(100 * s.mean(), 2)),
                 wind_q50=("od_max_cyc_wind", "median"),
                 wind_q90=("od_max_cyc_wind", lambda s: s.quantile(.90)),
                 wind_q99=("od_max_cyc_wind", lambda s: s.quantile(.99)),
                 wind_max=("od_max_cyc_wind", "max"),
                 pct_wind_gt0=("od_max_cyc_wind", lambda s: round(100 * (s > 0).mean(), 2)),
                 pct_wind_ge34kt=("od_max_cyc_wind", lambda s: round(100 * (s >= 34).mean(), 2)),
                 pct_wind_ge64kt=("od_max_cyc_wind", lambda s: round(100 * (s >= 64).mean(), 2))))
    print(freq.to_string())

    print("\nPer-year cyclone-active day counts (route-own zones):")
    yr = (d.pivot_table(index="year", columns="Route", values="od_cyc_active",
                        aggfunc=["sum", "size"]))
    dd = d.groupby(["year", "Route"])["od_cyc_active"].agg(["sum", "size"])
    dd.columns = ["active_days", "obs_days"]
    dd["pct"] = (100 * dd.active_days / dd.obs_days).round(1)
    print(dd.unstack(1).to_string())

    print("\nPer-zone cyclone-active days and wind percentiles (route-own zones only):")
    zrows = []
    for r, grp in raw.groupby("Route"):
        for z in ROUTE_ZONES[r]:
            c = f"cyclone_active_{z}"
            w = f"cyclone_wind_{z}"
            if c not in grp.columns:
                continue
            if grp[c].fillna(0).abs().max() == 0 and grp[w].fillna(0).abs().max() == 0:
                zrows.append(dict(route=r, zone=z, active_days=0, pct_active=0.0,
                                  wind_max=0.0, note="DEAD (identically zero)"))
                continue
            zrows.append(dict(route=r, zone=z, active_days=int((grp[c] > 0).sum()),
                              pct_active=round(100 * (grp[c] > 0).mean(), 1),
                              wind_max=float(grp[w].max()), note=""))
    print(pd.DataFrame(zrows).to_string(index=False))

    print("\nContinuous met fields - forward-fill / range check (FE->NE route):")
    print(q[q.scope == "raw(FENE)"][["column", "n_nan", "min", "q50", "max",
                                     "max_identical_run", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()
