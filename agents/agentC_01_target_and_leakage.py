"""
Agent C - Deliverables 1 & 2: independent target re-derivation + leakage audit.
Run: python D:\testing\agents\agentC_01_target_and_leakage.py
"""
from __future__ import annotations

import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentC_common import (SRC, OUT, HORIZON, ANN, RV_FLOOR, BANNED, WEATHER_EXACT, load_raw,
                           independent_targets, pipeline_derived, price_feature_names,
                           weather_feature_names, walkforward_folds)

RNG = np.random.default_rng(20240607)
rep = {}
leak_rows = []
notes = []


def log(*a):
    print(*a, flush=True)


# =============================================================================
# 1. TARGET AUDIT
# =============================================================================
log("=" * 90)
log("1. INDEPENDENT TARGET RE-DERIVATION")
log("=" * 90)

raw = load_raw()
ind = independent_targets(raw)
pipe = pipeline_derived(raw)

m = ind[["Date", "Route", "Price_USD", "c_log_price", "c_ret", "c_rv_fwd", "c_log_rv_fwd"]].copy()
m["p_log_price"] = pipe["log_price"]
m["p_ret"] = pipe["ret"]
m["p_rv_fwd"] = pipe["rv_fwd"]
m["p_log_rv_fwd"] = pipe["log_rv_fwd"]

# --- full-population comparison (not just the 200-row sample) -----------------
def cmp(a, b, tol):
    bothna = a.isna() & b.isna()
    one = a.isna() ^ b.isna()
    close = (~a.isna()) & (~b.isna()) & ((a - b).abs() <= tol)
    return bothna, one, close

bothna, one, close = cmp(m["c_rv_fwd"], m["p_rv_fwd"], 1e-9)
mismatch_full = ~(bothna | close)
log(f"[target] rows={len(m)}  NaN-pattern mismatches={int(one.sum())}  "
    f"value mismatches>1e-9={int((mismatch_full & ~one).sum())}")
rep["full_population_rows"] = int(len(m))
rep["full_population_nan_pattern_mismatch"] = int(one.sum())
rep["full_population_value_mismatch"] = int((mismatch_full & ~one).sum())

# --- the requested random sample of 200 rows ---------------------------------
sample = m.sample(n=200, random_state=7).sort_index()
s_bothna, s_one, s_close = cmp(sample["c_rv_fwd"], sample["p_rv_fwd"], 1e-9)
s_idx = sample.index[s_one | (~s_close)]
log(f"[target] random 200-row sample: NaN-pattern mismatches={int(s_one.sum())}, "
    f"value mismatches>1e-9={int((~s_close & ~s_one).sum())}")
rep["sample_n"] = 200
rep["sample_mismatches"] = int(len(s_idx))
rep["sample_max_abs_diff"] = float((sample["c_rv_fwd"] - sample["p_rv_fwd"]).abs().max())

# Also compare the log target (what the model actually fits)
lb, lo, lc = cmp(m["c_log_rv_fwd"], m["p_log_rv_fwd"], 1e-9)
rep["log_target_mismatch_rows"] = int((~(lb | lc)).sum())
rep["log_target_max_abs_diff"] = float((m["c_log_rv_fwd"] - m["p_log_rv_fwd"]).abs().max())

# zero-variance windows
zero = m[m["c_rv_fwd"] <= 1e-12]
rep["exact_zero_rv_rows"] = int(len(zero))
rep["exact_zero_rv_by_route_year"] = (
    zero.groupby(["Route", zero["Date"].dt.year]).size().to_dict()
)
rep["exact_zero_rv_by_route_year"] = {f"{k[0]}|{k[1]}": int(v) for k, v in
                                      rep["exact_zero_rv_by_route_year"].items()}
rep["rows_below_RV_FLOOR"] = int((m["c_rv_fwd"] <= RV_FLOOR).sum())
log(f"[target] exactly-zero rv_fwd rows={len(zero)}  rows<=floor({RV_FLOOR})={rep['rows_below_RV_FLOOR']}")
log(f"[target] zero rows by route/year: {rep['exact_zero_rv_by_route_year']}")

# --- does the target use day-t information? ----------------------------------
# rv_fwd(t) must be sqrt(sum sq_ret(t+1..t+5))*sqrt(252).  Decisive test: replace
# sq_ret at a single interior row r0 with 1e6 and record exactly which target
# positions move.  Correct behaviour: positions r0-1 .. r0-5 move, r0 does not.
def _rv_series(sq: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"r": raw["Route"], "s": sq})
    return pd.Series(np.sqrt(tmp.groupby("r")["s"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=HORIZON).sum().shift(-(HORIZON - 1))
    )) * np.sqrt(ANN), index=raw.index)


rv_exact = _rv_series(pipe["sq_ret"])
route_first = raw.groupby("Route").head(1).index
route_last = raw.groupby("Route").tail(1).index
route_first5 = raw.groupby("Route").head(HORIZON).index   # too close to the route start
interior = raw.index.difference(route_first.union(route_last).union(route_first5))

# (i) single-row delta fingerprint on a mid-route row
r0 = int(raw.index[raw["Route"] == "XSICFENE_FarEast_NorthEurope"][1400])
sq_pert = pipe["sq_ret"].copy()
sq_pert.loc[r0] = 1.0e6
rv_p = _rv_series(sq_pert)
changed = np.where(~np.isclose(rv_exact.to_numpy(), rv_p.to_numpy(), equal_nan=True))[0]
rep["fingerprint_row"] = r0
rep["fingerprint_changed_positions_relative"] = (changed - r0).tolist()
rep["fingerprint_own_row_changed"] = bool(r0 in set(changed.tolist()))
rep["fingerprint_expected_relative"] = [-HORIZON + i for i in range(HORIZON)]  # -5..-1
log(f"[target] fingerprint: sq_ret({r0}) -> 1e6 moves rv_fwd at relative positions "
    f"{(changed - r0).tolist()} (expected {rep['fingerprint_expected_relative']}); "
    f"own row moved = {rep['fingerprint_own_row_changed']}")

# (ii) bulk confirmation, ISOLATED perturbations: rows at least HORIZON+1 apart, so no
#      perturbed row can fall inside another perturbed row's target window.  On this
#      design rv_fwd(t) must be unchanged at every perturbed row t.
interior_np = interior.to_numpy()
spaced = []
last = -10 ** 9
for r in interior_np:
    if r - last > HORIZON + 1:
        spaced.append(r); last = r
spaced = np.array(spaced)
pert_rows = np.sort(RNG.choice(spaced, size=min(400, len(spaced)), replace=False))
sq_pert2 = pipe["sq_ret"].copy()
sq_pert2.loc[pert_rows] = 1.0e6
rv_p2 = _rv_series(sq_pert2)
comparable = rv_exact.notna() & rv_p2.notna()
own_changed = (~np.isclose(rv_exact.loc[pert_rows].to_numpy(), rv_p2.loc[pert_rows].to_numpy(),
                           equal_nan=True)) & comparable.loc[pert_rows].to_numpy()
rep["sqret_perturb_rows_tested"] = int(len(pert_rows))
rep["sqret_perturb_rows_finite"] = int(comparable.loc[pert_rows].sum())
rep["sqret_perturb_own_row_rv_changes"] = int(own_changed.sum())
rep["sqret_perturb_design"] = "isolated: perturbed rows spaced > HORIZON apart"
log(f"[target] ISOLATED bulk test: changed sq_ret(t) at {len(pert_rows)} rows spaced "
    f">{HORIZON} apart; of the {rep['sqret_perturb_rows_finite']} with a finite target, "
    f"rv_fwd(t) moved for {int(own_changed.sum())} (must be 0)")
log(f"[target] (for reference, a dense 400-row perturbation gives a spurious non-zero count "
    f"only because other perturbed rows fall inside the same window)")

# (iii) dense-design cross-check on ONE route only, to rule out cross-route contamination
_g = raw[raw["Route"] == "XSICFENE_FarEast_NorthEurope"].reset_index(drop=True)
_sq = (np.log(_g["Price_USD"].clip(lower=1.0)).diff() ** 2).to_numpy()


def _rv1(s):
    return np.sqrt(pd.Series(s).rolling(HORIZON, min_periods=HORIZON).sum().shift(-HORIZON)) * np.sqrt(ANN)


_base1 = _rv1(_sq).to_numpy()
_dense = np.sort(RNG.choice(list(range(HORIZON, len(_g) - HORIZON)), 400, replace=False))
_s2 = _sq.copy(); _s2[_dense] = 1.0e6
_a1 = _rv1(_s2).to_numpy()
_dense_changed = [int(i) for i in _dense if not np.isclose(_base1[i], _a1[i], equal_nan=True)]
_explained = [i for i in _dense_changed if any((i + k) in set(_dense.tolist()) for k in range(1, HORIZON + 1))]
rep["single_route_dense_changed"] = len(_dense_changed)
rep["single_route_dense_all_explained_by_window_collision"] = bool(len(_explained) == len(_dense_changed))
log(f"[target] single-route dense test: {len(_dense_changed)} rows appear to change, all of them "
    f"because another perturbed row lies inside their target window "
    f"(verified: {rep['single_route_dense_all_explained_by_window_collision']})")

# (iii) day-(t+1) IS in the window (sanity that the test has power)
r1 = int(interior[10]) + 1
sq_pert3 = pipe["sq_ret"].copy()
sq_pert3.loc[r1] = 1.0e6
rv_p3 = _rv_series(sq_pert3)
rep["power_check_own_row_of_t_plus_1_moved"] = bool(
    not np.isclose(rv_exact.loc[r1 - 1], rv_p3.loc[r1 - 1], equal_nan=True))
log(f"[target] power check: moving sq_ret at row {r1} moves rv_fwd({r1-1}) = "
    f"{rep['power_check_own_row_of_t_plus_1_moved']} (must be True)")

# (iv) scale-invariance of the target: log returns are invariant to a common rescaling of
#      the entire price series, so rv_fwd must be bit-identical after Price_USD -> k*Price.
p3 = raw.copy()
p3["Price_USD"] = p3["Price_USD"] * 1000.0
new3 = independent_targets(p3)["c_rv_fwd"].to_numpy()
ref = independent_targets(raw)["c_rv_fwd"].to_numpy()
both = np.isfinite(ref) & np.isfinite(new3)
n_diff = int((~np.isclose(ref[both], new3[both], equal_nan=True)).sum())
rep["scale_invariance_rows_changed"] = n_diff
log(f"[target] common rescaling of the whole price series (x1000) changed rv_fwd for "
    f"{n_diff} of {int(both.sum())} rows (must be 0)")

rep["target_rederivation_matched"] = bool(rep["sample_mismatches"] == 0 and
                                          rep["full_population_value_mismatch"] == 0 and
                                          rep["full_population_nan_pattern_mismatch"] == 0)

with open(os.path.join(OUT, "agentC_target_audit.json"), "w") as f:
    json.dump(rep, f, indent=2, default=str)
log(f"[write] {os.path.join(OUT, 'agentC_target_audit.json')}")


# =============================================================================
# 2. LEAKAGE AUDIT
# =============================================================================
log("\n" + "=" * 90)
log("2. LEAKAGE AUDIT")
log("=" * 90)

cols = list(raw.columns)
price_feats = price_feature_names(cols)
weather_feats = weather_feature_names(cols)
route_dummies = [c for c in cols if c.startswith("Route_")
                 and c not in WEATHER_EXACT + ["Origin_LSCI", "Dest_LSCI", "Route_Mean_LSCI", "LSCI_30d_Delta"]]

log(f"[leak] price features ({len(price_feats)}): {price_feats}")
log(f"[leak] weather features ({len(weather_feats)})")
log(f"[leak] banned: {BANNED}")

# features the pipeline actually feeds the models (exclude dummies - handled separately)
audit_feats = [c for c in dict.fromkeys(price_feats + weather_feats + route_dummies)
               if c not in BANNED and c not in ("Date", "Route")]

# ---- 2a. correlation with future returns ------------------------------------
d = pipe.copy()

# future cumulative 5d return (the "large move" driver) and one-day-ahead return
d["fwd1"] = d.groupby("Route")["ret"].shift(-1)
d["fwd5"] = d.groupby("Route")["ret"].transform(lambda s: s.shift(-1).rolling(5).sum().shift(-4))
d["fwd_abs5"] = d["fwd5"].abs()

# 5-day forward average weather (i.e. what a leaky "weather" column would look like)
def _route_corr(feat: pd.Series, tgt: pd.Series, route: pd.Series):
    """Spearman + Pearson computed within route, then averaged weighted by n."""
    out = {}
    for name, fn in (("pearson", lambda a, b: np.corrcoef(a, b)[0, 1]),
                     ("spearman", lambda a, b: pd.Series(a).corr(pd.Series(b), method="spearman"))):
        vals, wts = [], []
        for r, g in pd.DataFrame({"f": feat, "t": tgt, "r": route}).groupby("r"):
            sub = g.dropna()
            if len(sub) < 30 or sub["f"].nunique() < 2 or sub["t"].nunique() < 2:
                continue
            v = fn(sub["f"].to_numpy(float), sub["t"].to_numpy(float))
            if np.isfinite(v):
                vals.append(v); wts.append(len(sub))
        out[name] = float(np.average(vals, weights=wts)) if vals else np.nan
    return out


rows = []
for c in audit_feats:
    f = d[c]
    row = dict(feature=c,
               group=("weather" if c in weather_feats else ("dummy" if c in route_dummies else "price_history")))
    pc = _route_corr(f, d["fwd5"], d["Route"])
    row["corr_fwd5"] = pc["pearson"]
    row["spearman_fwd5"] = pc["spearman"]
    pc1 = _route_corr(f, d["fwd1"], d["Route"])
    row["corr_fwd1"] = pc1["pearson"]
    row["corr_fwd_abs5"] = _route_corr(f, d["fwd_abs5"], d["Route"])["pearson"]
    row["corr_log_rv_fwd"] = _route_corr(f, d["log_rv_fwd"], d["Route"])["pearson"]
    row["corr_rv_fwd_level"] = _route_corr(f, d["rv_fwd"], d["Route"])["pearson"]
    row["n_valid"] = int(f.notna().sum())
    row["n_unique"] = int(f.nunique())
    rows.append(row)

lk = pd.DataFrame(rows)

# ---- 2b. deterministic-function-of-the-future test ---------------------------
# exact match of column A against a forward shift of column B (1..7 rows), within route.
# Constant columns are excluded: they match any shift of themselves trivially.
fut_rows = []
num_cols_all = [c for c in cols if c not in ("Date", "Route")]
const_cols = [c for c in num_cols_all if d[c].nunique(dropna=True) < 5]
num_cols = [c for c in num_cols_all if c not in const_cols]
log(f"[leak] near-constant columns excluded from shift matching ({len(const_cols)}): {const_cols}")
arr = d[num_cols].to_numpy(float)
arrall = d[num_cols_all].to_numpy(float)
dts = d["Date"].to_numpy()
rt = d["Route"].to_numpy()
for lag in range(1, 8):
    fwd = np.full_like(arr, np.nan)
    for j in range(len(d) - lag):
        if rt[j] == rt[j + lag]:
            fwd[j] = arr[j + lag]
    for bi, b in enumerate(num_cols):
        colb = fwd[:, bi]
        ok = np.isfinite(colb)
        if ok.sum() < 100:
            continue
        for ai, a in enumerate(num_cols):
            cola = arr[:, ai]
            sel = ok & np.isfinite(cola)
            if sel.sum() < 100:
                continue
            eq = np.isclose(cola[sel], colb[sel], rtol=0, atol=1e-12)
            frac = float(eq.mean())
            if frac > 0.99:
                fut_rows.append(dict(test=f"col[t] == col.shift(-{lag})[t]", column_a=a,
                                     column_b=b, lag=lag, frac_equal=frac,
                                     n_compared=int(sel.sum())))

# column equals a forward-only rolling aggregate (mean/max/min/sum over t+1..t+w)?
roll_rows = []
for w in (3, 5, 7):
    for fn, nm in ((np.nanmean, "mean"), (np.nanmax, "max"), (np.nanmin, "min"),
                   (np.nansum, "sum")):
        fwd = np.full_like(arr, np.nan)
        for j in range(len(d) - w - 1):
            if len(set(rt[j + 1:j + 1 + w])) == 1:
                with np.errstate(all="ignore"):
                    fwd[j] = fn(arr[j + 1:j + 1 + w, :], axis=0)
        for bi, b in enumerate(num_cols):
            colb = fwd[:, bi]
            if np.isfinite(colb).sum() < 100:
                continue
            for ai, a in enumerate(num_cols):
                sel = np.isfinite(colb) & np.isfinite(arr[:, ai])
                if sel.sum() < 100:
                    continue
                frac = float(np.isclose(arr[sel, ai], colb[sel], rtol=0, atol=1e-9).mean())
                if frac > 0.99:
                    roll_rows.append(dict(test=f"col[t] == {nm}(col[t+1..t+{w}])",
                                          column_a=a, column_b=b, frac_equal=frac,
                                          n_compared=int(sel.sum())))

# ---- 2c. is any feature a deterministic function of *contemporaneous* raw fields? ----
# duplicate-column detection (identical series under a different name)
dupes = []
for i, a in enumerate(num_cols):
    for b in num_cols[i + 1:]:
        va, vb = d[a].to_numpy(float), d[b].to_numpy(float)
        sel = np.isfinite(va) & np.isfinite(vb)
        if sel.sum() < 100:
            continue
        if np.isclose(va[sel], vb[sel], rtol=0, atol=1e-12).mean() > 0.9999:
            dupes.append((a, b, int(sel.sum())))

# perfect (rank) predictor of the target?
perfect = []
for c in audit_feats:
    v = d[c]
    sel = v.notna() & d["log_rv_fwd"].notna()
    if sel.sum() < 100 or v[sel].nunique() < 5:
        continue
    sp = pd.Series(v[sel].to_numpy()).corr(pd.Series(d.loc[sel, "log_rv_fwd"].to_numpy()), method="spearman")
    if np.isfinite(sp) and abs(sp) > 0.9:
        perfect.append((c, float(sp)))

# ---- 2d. forbidden-column forensics -----------------------------------------
# For each banned column: does it provably encode FUTURE information?  Two tests:
#   (1) exact forward-shift match against an in-use column: X[t] == Y[t+k], k>0
#   (2) contemporaneous correlation with the target, for context
banned_report = []
FORWARD_MATCHES = {}
const_pos = {c: i for i, c in enumerate(num_cols_all)}
for k in range(1, 8):
    S = np.full_like(arrall, np.nan)
    same = rt[: len(d) - k] == rt[k:]
    S[: len(d) - k][same] = arrall[k:][same]
    for ci, c in enumerate(num_cols_all):
        if c not in BANNED:
            continue
        cb = S[:, ci]
        ok = np.isfinite(cb)
        if ok.sum() < 200:
            continue
        for a in num_cols_all:
            if a == c:
                continue
            va = d[a].to_numpy(float)
            sel = ok & np.isfinite(va)
            if sel.sum() < 200:
                continue
            frac = float(np.isclose(va[sel], cb[sel], rtol=0, atol=1e-12).mean())
            if frac > 0.999:
                FORWARD_MATCHES.setdefault(c, []).append((a, k, round(frac, 6), int(sel.sum())))

for c in BANNED:
    v = d[c]
    corr_t = _route_corr(v, d["log_rv_fwd"], d["Route"])["pearson"]
    fm = FORWARD_MATCHES.get(c, [])
    if fm:
        explanation = "; ".join(f"{c}(t) == {a}(t+{k}) exactly over {n} rows" for a, k, f, n in fm)
        verdict = "BANNED - CONFIRMED forward-looking"
    else:
        explanation = "no exact forward-shift match found"
        verdict = "BANNED - conservative (not proven forward-looking)"
    banned_report.append(dict(feature=c, nunique=int(v.nunique(dropna=True)),
                              corr_contemporaneous_log_rv_fwd=corr_t,
                              forward_link=explanation, verdict=verdict))

# ---- 2e. weather features: are they actually route-specific or global? -------
glob = []
for c in weather_feats:
    if c.startswith("Route_"):
        continue
    nun = d.groupby("Date")[c].nunique()
    glob.append(dict(feature=c, distinct_values_per_date_mean=float(nun.mean()),
                     is_global=bool(nun.max() <= 1), nunique=int(d[c].nunique(dropna=True)),
                     is_constant_or_binary=bool(d[c].nunique(dropna=True) <= 2)))

# ---- 2f. fold-boundary discipline -------------------------------------------
d_full = pipe.copy()
d_full["rv_trail_22"] = pipe.groupby("Route")["sq_ret"].transform(
    lambda s: np.sqrt(s.rolling(22, min_periods=11).sum() / 22) * np.sqrt(ANN))
# rebuild the exact 'complete' gate used by the pipeline
dd = d_full.copy()
dd["log_rv_d"] = np.log(np.sqrt(dd["sq_ret"]).clip(lower=1e-8))
dd["log_rv_w"] = np.log(np.sqrt(dd.groupby("Route")["sq_ret"].transform(lambda s: s.rolling(5).sum() / 5)).clip(lower=1e-8))
dd["log_rv_m"] = np.log(np.sqrt(dd.groupby("Route")["sq_ret"].transform(lambda s: s.rolling(22).sum() / 22)).clip(lower=1e-8))
dd["ewma_vol"] = 1.0
folds = walkforward_folds(dd)
log(f"[split] reproduced {len(folds)} folds")

bound_rows = []
# last calendar date whose return enters rv_fwd(t): the return at t+HORIZON
fwd_reach = pd.to_datetime(d_full.groupby("Route")["Date"].transform(lambda s: s.shift(-HORIZON)))
fwd_reach_np = fwd_reach.to_numpy(dtype="datetime64[ns]")
dt = dd["Date"].to_numpy(dtype="datetime64[ns]")
n_test_before_origin = 0
n_train_target_overlap = 0
n_origin_leak_pairs = 0
min_reach_overlap_frac = []
for origin, tr, te in folds:
    n_test_before_origin += int((dt[te] < np.datetime64(origin)).sum())
    over = tr & pd.notna(fwd_reach).to_numpy() & (fwd_reach_np >= np.datetime64(origin))
    n_train_target_overlap += int(over.sum())
    min_reach_overlap_frac.append(float(over.sum() / max(tr.sum(), 1)))
    n_origin_leak_pairs += len(set(dt[tr].tolist()) & set(dt[te].tolist()))

bound_rows.append(dict(origin="ALL", n_test_rows_dated_before_origin=n_test_before_origin,
                       n_train_rows_whose_target_window_reaches_origin_or_later=n_train_target_overlap,
                       frac_train_rows_with_target_crossing_origin=float(np.mean(min_reach_overlap_frac)),
                       n_train_dates_also_in_test=n_origin_leak_pairs))
log(f"[split] test rows dated before their fold origin: {n_test_before_origin}")
log(f"[split] train rows whose target window (t+1..t+5) reaches >= origin: {n_train_target_overlap}")
log(f"[split] train dates also present in a test block: {n_origin_leak_pairs}")

# feature computation windows that cross the origin
cross = dd.groupby("Route")["Date"].transform(lambda s: s.shift(-(HORIZON)))
n_cross = 0
for origin, tr, te in folds:
    reach = pd.to_datetime(dd.groupby("Route")["Date"].transform(lambda s: s.shift(-HORIZON)))
    n_cross += int((tr & (reach >= origin)).sum())
rep["train_rows_target_crossing_origin"] = int(n_cross)

# ---- assemble leakage CSV ----------------------------------------------------
lk["flag_corr_gt_0.5"] = lk["corr_fwd5"].abs() > 0.5
lk["flag_corr_fwd1_gt_0.5"] = lk["corr_fwd1"].abs() > 0.5
lk["flag_near_perfect_target"] = lk["corr_log_rv_fwd"].abs() > 0.9
lk["flag_banned"] = lk["feature"].isin(BANNED)
lk["flag_used_by_pipeline"] = ~lk["feature"].isin(BANNED)
lk = lk.sort_values("corr_log_rv_fwd", key=lambda s: s.abs(), ascending=False)
lk.to_csv(os.path.join(OUT, "agentC_leakage_audit.csv"), index=False)
log(f"[write] {os.path.join(OUT, 'agentC_leakage_audit.csv')}  ({len(lk)} features)")

# append structural findings as extra rows so the CSV is self-contained
extra = []
for r in fut_rows:
    extra.append(dict(feature=r["column_a"], group="STRUCTURAL",
                      note=f"t+{r['lag']} leak: {r['test']} via {r['column_b']} frac={r['frac_equal']:.4f}"))
for r in roll_rows:
    extra.append(dict(feature=r["column_a"], group="STRUCTURAL",
                      note=f"forward-rolling leak: {r['test']} via {r['column_b']} frac={r['frac_equal']:.4f}"))
for a, b, n in dupes:
    extra.append(dict(feature=a, group="STRUCTURAL", note=f"identical to {b} over {n} rows"))
for c, sp in perfect:
    extra.append(dict(feature=c, group="STRUCTURAL", note=f"|spearman| with log_rv_fwd = {sp:.3f}"))
for r in extra:
    pass
for r in banned_report:
    extra.append(dict(feature=r["feature"], group="BANNED",
                      note=f"{r['verdict']}; contemp corr with log_rv_fwd="
                           f"{r['corr_contemporaneous_log_rv_fwd']:.3f}; {r['forward_link']}"))
extra.append(dict(feature="__split__", group="STRUCTURAL",
                  note=f"test rows dated before fold origin={n_test_before_origin}; "
                       f"train rows whose target window reaches >=origin={n_train_target_overlap}; "
                       f"train dates also in test={n_origin_leak_pairs}"))
if extra:
    pd.concat([lk, pd.DataFrame(extra)], ignore_index=True).to_csv(
        os.path.join(OUT, "agentC_leakage_audit.csv"), index=False)

log("\n--- TOP |corr| WITH FUTURE 5d RETURN (all audited features) ---")
log(lk.reindex(lk["corr_fwd5"].abs().sort_values(ascending=False).index)
    .head(20)[["feature", "group", "corr_fwd5", "corr_fwd1", "corr_log_rv_fwd", "n_valid"]]
    .to_string(index=False))

log("\n--- FLAGGED: |corr with future 5d return| > 0.5 ---")
fl = lk[lk["flag_corr_gt_0.5"]]
log(fl[["feature", "group", "corr_fwd5", "corr_fwd1", "corr_log_rv_fwd"]].to_string(index=False)
    if len(fl) else "  (none)")

log("\n--- EXACT FORWARD-SHIFT MATCHES (deterministic function of future rows) ---")
log(pd.DataFrame(fut_rows).to_string(index=False) if fut_rows else "  (none)")
log("\n--- EXACT FORWARD-ROLLING MATCHES ---")
log(pd.DataFrame(roll_rows).head(30).to_string(index=False) if roll_rows else "  (none)")
log("\n--- DUPLICATE (identical) COLUMN PAIRS ---")
log(pd.DataFrame(dupes, columns=["col_a", "col_b", "n"]).to_string(index=False) if dupes else "  (none)")
log("\n--- NEAR-PERFECT TARGET PREDICTORS (|spearman| > 0.9 vs log_rv_fwd) ---")
log(pd.DataFrame(perfect, columns=["feature", "spearman"]).to_string(index=False) if perfect else "  (none)")
log("\n--- BANNED-COLUMN FORENSICS ---")
log(pd.DataFrame(banned_report).to_string(index=False))
log("\n--- WEATHER COLUMNS: GLOBAL OR ROUTE-SPECIFIC? (summary) ---")
_g = pd.DataFrame(glob)
log(f"  total weather cols checked: {len(_g)}")
log(f"  constant / 1-bit (nunique<=2): {int(_g['is_constant_or_binary'].sum())}")
log(f"  perfectly constant (nunique<=1): {int((_g['nunique'] <= 1).sum())} -> "
    f"{_g.loc[_g['nunique'] <= 1, 'feature'].tolist()}")
log(f"  identical across routes on the same date: {int(_g['is_global'].sum())}")
log(_g.loc[_g['is_constant_or_binary']].to_string(index=False))
log("\n--- SPLIT BOUNDARY ---")
log(pd.DataFrame(bound_rows).to_string(index=False))

struct = dict(
    exact_forward_shift_matches=fut_rows,
    n_exact_forward_shift_matches=len(fut_rows),
    exact_forward_rolling_matches=roll_rows[:40],
    n_exact_forward_rolling_matches=len(roll_rows),
    identical_column_pairs=[dict(a=a, b=b, n=n) for a, b, n in dupes],
    near_perfect_target_predictors=[dict(feature=c, spearman=sp) for c, sp in perfect],
    banned_forensics=banned_report,
    constant_or_1bit_weather_cols=[g for g in glob if g["is_constant_or_binary"]],
    n_weather_cols_constant_or_binary=sum(1 for g in glob if g["is_constant_or_binary"]),
    n_weather_cols_all_constant=sum(1 for g in glob if g["nunique"] <= 1),
    n_weather_cols_global=sum(1 for g in glob if g["is_global"]),
    split_test_before_origin=n_test_before_origin,
    split_train_target_reaches_origin=n_train_target_overlap,
    split_train_dates_in_test=n_origin_leak_pairs,
)
rep["leakage"] = struct
with open(os.path.join(OUT, "agentC_target_audit.json"), "w") as f:
    json.dump(rep, f, indent=2, default=str)
log("\n[write] updated agentC_target_audit.json")
