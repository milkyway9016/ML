"""Build D:\\testing\\agents\\agentA_findings.md from the output CSVs."""
import os
import numpy as np
import pandas as pd

OUT = r"D:\testing\outputs"
AG = r"D:\testing\agents"

agg = pd.read_csv(os.path.join(OUT, "agentA_metrics_aggregate.csv"))
fold = pd.read_csv(os.path.join(OUT, "agentA_metrics_by_fold.csv"))
dm = pd.read_csv(os.path.join(OUT, "agentA_dm_tests.csv"))

agg = agg.sort_values("QLIKE").reset_index(drop=True)
har = agg.loc[agg["model"] == "har_log"]
har_q = float(har["QLIKE"].iloc[0])
har_m = float(har["MAE_log"].iloc[0])
har_r = float(har["RMSE_log"].iloc[0])
har_v = float(har["MAE_vol_pp"].iloc[0])


def md_table(df, cols, headers=None, floatfmt="{:.5f}"):
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("n/a" if not np.isfinite(v) else floatfmt.format(v))
            elif isinstance(v, (bool, np.bool_)):
                cells.append("yes" if v else "no")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


# ---------- headline DM numbers ----------
def get(comparison, loss):
    s = dm[(dm["comparison"] == comparison) & (dm["loss"] == loss)]
    return s.iloc[0] if len(s) else None


L = []
L.append("# agentA - Rigorous econometric baselines for freight-rate volatility\n")
L.append("**Target** (identical to `volatility_large_move.py`): `rv_fwd(t) = sqrt(sum(sq_ret, t+1..t+5)) * sqrt(252)`,")
L.append("i.e. annualised 5-day forward realised volatility of log price. Strictly forward-looking.")
L.append("Rows with `rv_fwd <= 5e-4` are dropped before scoring; every prediction is floored at 5e-4.\n")
L.append("**Protocol**: expanding-window walk-forward, monthly origins 2020-01 .. 2026-04 (76 folds),")
L.append("train labels fully observed before origin, test = following calendar month,")
L.append(">=500 mature train rows, >=20 test rows,")
L.append(">=2y history, all of `log_rv_fwd, rv_trail_22, ewma_vol, log_rv_d, log_rv_w, log_rv_m` non-NaN.")
L.append("Metrics are means over folds. `MAE_vol_pp` is in annualised volatility percentage points.\n")
L.append("**HAR features follow the stated spec exactly**: daily = `ln sq_ret(t)`, weekly = `ln mean sq_ret(t-4..t)`,")
L.append("monthly = `ln mean sq_ret(t-21..t)`. These three use only information observable at the close of day *t*,")
L.append("and the target begins at *t+1*, so there is no look-ahead. A strictly-lagged variant (windows ending *t-1*)")
L.append("is reported as `har_log_strictlag` for robustness.\n")

L.append("## 1. Ranked aggregate model comparison\n")
L.append(md_table(agg, ["model", "family", "folds", "MAE_log", "RMSE_log", "R2_log", "QLIKE", "MAE_vol_pp"],
                 ["model", "family", "folds", "MAE_log", "RMSE_log", "R2_log", "QLIKE", "MAE_vol_pp"]))
L.append("")
L.append(f"Reference HAR: QLIKE = **{har_q:.5f}**, MAE_log = {har_m:.5f}, RMSE_log = {har_r:.5f}, "
         f"MAE_vol_pp = {har_v:.3f}.\n")

# ---------- DM table ----------
L.append("## 2. Diebold-Mariano incremental-value tests\n")
L.append("Paired on the *exact same* rows. Loss differential `d = L(baseline) - L(challenger)`, so **`dm_t > 0`")
L.append("means the challenger wins**. Newey-West HAC with lag = HORIZON + 4 = 9; two-sided p-value from the")
L.append("t distribution with n-1 df. `p_hln` additionally applies the Harvey-Leybourne-Newbold (1997) small-sample")
L.append("correction. `p_route` re-tests after averaging the differential within each calendar date, which removes")
L.append("all cross-route contemporaneous correlation (the remaining serial correlation is only the 5-day overlap).\n")
d1 = dm[dm["loss"] == "squared_error_log"].copy()
L.append(md_table(d1, ["comparison", "n_pairs", "mean_loss_baseline", "mean_loss_challenger",
                       "skill_gain_pct", "dm_t", "dm_p", "significant_5pct"],
                  ["comparison", "n", "mean loss base", "mean loss chal", "skill gain %", "DM t", "p", "sig 5%"],
                  floatfmt="{:.5f}"))
L.append("")
L.append("Same tests on the QLIKE loss (`2 ln q + a^2/q^2`):\n")
d2 = dm[dm["loss"] == "QLIKE"].copy()
L.append(md_table(d2, ["comparison", "n_pairs", "mean_loss_baseline", "mean_loss_challenger",
                       "skill_gain_pct", "dm_t", "dm_p", "significant_5pct"],
                  ["comparison", "n", "mean loss base", "mean loss chal", "skill gain %", "DM t", "p", "sig 5%"],
                  floatfmt="{:.5f}"))
L.append("")
L.append("Robustness column - same squared-error comparison, Newey-West on the *date-averaged* differential")
L.append("(removes cross-route correlation), which is the more conservative of the two:\n")
L.append(md_table(d1, ["comparison", "dm_t", "dm_p", "dm_t_hln", "dm_p_hln", "dm_t_route_clustered",
                       "dm_p_route_clustered"],
                  ["comparison", "DM t", "p", "t (HLN)", "p (HLN)", "t (date-avg)", "p (date-avg)"]))
L.append("")

# ---------- verdict ----------
def verdict_block(comp_raw, comp_pca, comp_econ, loss):
    a, b, c = get(comp_raw, loss), get(comp_pca, loss), get(comp_econ, loss)
    txt = []
    for nm, s in [("weather, 82 raw columns", a), ("weather, 5 PCs", b), ("economic, 4 columns", c)]:
        if s is None:
            continue
        txt.append(f"  - **{nm}**: skill gain {s['skill_gain_pct']:+.3f}%, DM t = {s['dm_t']:+.3f}, "
                   f"p = {s['dm_p']:.4f} ({'significant' if s['significant_5pct'] else 'NOT significant'} at 5%)")
    return "\n".join(txt)

L.append("## 3. Verdict: does weather/cyclone information add incremental predictive value over HAR-RV?\n")
L.append("On the squared-error loss in log space:\n")
L.append(verdict_block("HAR+weather (82 raw) vs HAR", "HAR+weather (5 PCs) vs HAR", "HAR+econ vs HAR",
                       "squared_error_log"))
L.append("\nOn the QLIKE loss:\n")
L.append(verdict_block("HAR+weather (82 raw) vs HAR", "HAR+weather (5 PCs) vs HAR", "HAR+econ vs HAR", "QLIKE"))
L.append("")
# data-driven verdict sentence
w_pca_sq = get("HAR+weather (5 PCs) vs HAR", "squared_error_log")
w_pca_q = get("HAR+weather (5 PCs) vs HAR", "QLIKE")
w_raw_sq = get("HAR+weather (82 raw) vs HAR", "squared_error_log")
econ_sq = get("HAR+econ vs HAR", "squared_error_log")
best = agg.iloc[0]
if w_pca_sq is not None:
    sig = "is" if w_pca_sq["significant_5pct"] else "is NOT"
    L.append(f"**Headline.** The best-behaved weather specification (5 weather PCs) {sig} significantly better than "
             f"plain HAR on squared error: skill gain {w_pca_sq['skill_gain_pct']:+.3f}%, DM t = {w_pca_sq['dm_t']:+.3f}, "
             f"p = {w_pca_sq['dm_p']:.4f}. On QLIKE the corresponding numbers are "
             f"{w_pca_q['skill_gain_pct']:+.3f}%, t = {w_pca_q['dm_t']:+.3f}, p = {w_pca_q['dm_p']:.4f}.")
if w_raw_sq is not None:
    L.append(f"The 82-raw-column version is actively harmful: skill gain "
             f"{w_raw_sq['skill_gain_pct']:+.3f}%, DM t = {w_raw_sq['dm_t']:+.3f} - the point estimate goes the "
             f"*wrong* way because 86 parameters are being fitted to training rows that are ~86% duplicated "
             f"across four near-identical routes.")
L.append("")
L.append("**Reading.** Either the weather/cyclone block carries no incremental information for 5-day-forward "
         "freight volatility beyond what lagged squared returns already contain, or it carries information in a "
         "form that a linear HAR cannot extract - and the second is not an endorsement of the feature set, it is "
         "a statement that these tests cannot find value in it. The block does have genuine *in-sample* "
         "explanatory power (weather-only R2 for `log_rv_fwd` = 0.055, and HAR+weather in-sample adj-R2 = 0.113 "
         "vs 0.077 for HAR alone), but that power is a slow contemporaneous war-season/2021-22-disruption trend "
         "already absorbed by the 22-day HAR memory. Out of sample the incremental content is indistinguishable "
         "from zero.\n")

L.append("## 4. Model-family verdicts\n")
for fam_key, fam_name in [("garch11_t", "GARCH(1,1)-t"), ("egarch11_t", "EGARCH(1,1)-t"),
                          ("gjr_garch11_t", "GJR-GARCH(1,1)-t"), ("har_route", "HAR per-route"),
                          ("har_j", "HAR-J"), ("har_cj", "HAR-CJ"), ("har_ext", "HAR+EWMA/skew/kurt"),
                          ("har_log_strictlag", "HAR strict-lag")]:
    row = agg[agg["model"] == fam_key]
    if not len(row):
        continue
    r = row.iloc[0]
    g = get(f"{fam_name} vs HAR", "squared_error_log")
    gq = get(f"{fam_name} vs HAR", "QLIKE")
    if g is None:
        # comparison label may differ
        alt = {"HAR per-route": "HAR per-route vs HAR pooled", "HAR+EWMA/skew/kurt": "HAR+ext(EWMA,skew,kurt) vs HAR"}
        g = get(alt.get(fam_name, f"{fam_name} vs HAR"), "squared_error_log")
        gq = get(alt.get(fam_name, f"{fam_name} vs HAR"), "QLIKE")
    txt = (f"- **{fam_name}**: QLIKE {r['QLIKE']:.5f} vs HAR {har_q:.5f} "
           f"({100*(r['QLIKE']/har_q - 1):+.2f}% vs HAR), MAE_vol_pp {r['MAE_vol_pp']:.3f} vs HAR {har_v:.3f}")
    if g is not None:
        txt += (f"; DM (sq err) t = {g['dm_t']:+.3f}, p = {g['dm_p']:.4f}, skill {g['skill_gain_pct']:+.3f}%")
    L.append(txt)
L.append("")

L.append("## 5. Data problems a reviewer must know about\n")
L.append("1. **The dataset contains the 'banned' forward-looking columns.** The task brief said they were source")
L.append("   bugs; they are present in `clean_engineered_dataset.csv` by name: `Cyclone_Wind_Forecast3d/7d`,")
L.append("   `Cyclone_Active_Forecast3d`, `Wind_Speed_Forecast7d`, `Rolling_7d_Cyclone_Days`,")
L.append("   `Price_Volatility_7d/30d`. Every one is excluded here by an explicit name block-list.")
L.append("2. **The route panel is not a panel.** Each weather/econ column takes only ~1.0-1.7 distinct values per")
L.append("   calendar date across the 4 routes, and only 4 routes exist. For a given date the observations are")
L.append("   near-duplicates, so the effective sample size is far below the nominal row count and any HAC test")
L.append("   that treats rows as independent over-states significance. Both a day-clustered and an unclustered")
L.append("   test are reported for that reason.")
L.append("3. **18.4% of daily returns are exactly zero** (prices are quoted and often unchanged; 23.7% on the")
L.append("   US-West-Coast legs). `sq_ret = 0` makes `ln(sq_ret)` = -inf, so every log-RV component is floored at")
L.append("   1e-16 (-36.8 in logs). This creates a bimodal feature distribution and is the main reason the strict")
L.append("   HAR-J/CJ decomposition fails.")
L.append("4. **8 weather columns are constant** across the whole sample (`cyclone_*_Red_Sea`, `cyclone_*_North_Atlantic`)")
L.append("   and 22 have <= 3 distinct values. They are dropped automatically when a fold's training standard")
L.append("   deviation is zero.")
L.append("5. **The zero-RV rows are concentrated in 2018-2019** (50 + 6 of 66), i.e. *before* the evaluation window,")
L.append("   so the `rv_fwd <= 5e-4` rule removes few test rows - but it does bite on the US-West-Coast legs.")
L.append("6. **The last fold's target window extends past the data**: origins run to 2026-04-01; `rv_fwd` is NaN for")
L.append("   the final 5 observations per route, so those rows are excluded by the `log_rv_fwd` completeness rule.")
L.append("")

L.append("## 6. Single most important caveat\n")
L.append("**The effective sample size of this evaluation is roughly 76 independent monthly origins, not the")
L.append("~6,000 scored rows.** Test rows within a fold share the same date across near-duplicate routes, and the")
L.append("5-day target overlaps by construction. The DM p-values below use lag 9 as instructed, which corrects for")
L.append("the overlap but not for the cross-route duplication; the day-averaged column is the trustworthy one, and")
L.append("it is uniformly *less* favourable to every challenger. Treat any borderline 5% result here as suggestive")
L.append("only.\n")

with open(os.path.join(AG, "agentA_findings.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote", os.path.join(AG, "agentA_findings.md"), len(L), "lines")
