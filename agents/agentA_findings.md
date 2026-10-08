# agentA - Rigorous econometric baselines for maritime freight-rate volatility

> **Historical results:** These metrics were generated before the walk-forward label-maturity
> cutoff was added. Training rows whose forward 5-day targets crossed the forecast origin were
> included, so the reported scores are optimistic and must be regenerated with the corrected
> `agentA_baselines.py` before use.

**Target** (definition copied exactly from `volatility_large_move.py`):
`rv_fwd(t) = sqrt( sum(sq_ret, t+1..t+5) ) * sqrt(252)`, annualised 5-day forward realised
volatility of log price. Strictly forward-looking. Scored on the level; rows with
`rv_fwd <= 5e-4` dropped, every prediction floored at 5e-4.

**Protocol**: expanding-window walk-forward, monthly origins `2020-01-01` .. `2026-04-01`
(76 candidate, **75 used**, 1 skipped), train = `Date < origin`, test = following calendar month,
`>=500` train rows, `>=20` test rows, `>=2y` history, all of
`log_rv_fwd, rv_trail_22, ewma_vol, log_rv_d, log_rv_w, log_rv_m` non-NaN. 6,268 scored rows
(6,268 = 75 folds x ~84 rows, after dropping zero-RV rows and non-finite predictions).
HAR components follow the stated spec exactly: daily `ln sq_ret(t)`, weekly
`ln mean sq_ret(t-4..t)`, monthly `ln mean sq_ret(t-21..t)` - all observable at the close of
day *t*, target starts at *t+1*, so no look-ahead. A strictly-lagged variant (windows ending
*t-1*) is included. Outputs: `agentA_metrics_aggregate.csv`, `agentA_metrics_by_fold.csv`,
`agentA_dm_tests.csv`, plus `agentA_metrics_pooled.csv`, `agentA_dm_robustness.csv`,
`agentA_smearing_experiment.csv`, `agentA_dm_smeared_har.csv`.

---

## 1. Ranked aggregate model comparison

Means over the 75 folds. `R2_log` is the **mean of within-fold R2**, which is a very small-sample
statistic (see section 5); `R2_log_pooled` and `QLIKE_pooled_trim1pct` are the honest companions.

| model | family | folds | MAE_log | RMSE_log | R2_log | QLIKE | MAE_vol_pp | R2_log_pooled | QLIKE_pooled | QLIKE_pool_trim1% | QLIKE_med/obs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| garch11_t | GARCH | 75 | 0.8878 | 1.0808 | -0.559 | 3.387 | 61.36 | -0.074 | 3.370 | 3.406 | -0.032 |
| garch11_normal | GARCH | 75 | 0.8811 | 1.0700 | -0.544 | 4.507 | 59.75 | -0.040 | 4.501 | 4.552 | -0.035 |
| gjr_garch11_t | GARCH | 75 | 0.9271 | 1.1341 | -2.081 | 4.679 | 10998.8 | -0.755 | 4.641 | 4.559 | -0.118 |
| har_wpca_epca | HAR | 75 | 1.0003 | 1.2043 | -1.653 | 38.56 | 514.4 | -0.761 | 39.70 | 39.99 | +0.141 |
| har_log_strictlag | HAR | 75 | 0.8460 | 1.0427 | -0.385 | 44.19 | 57.58 | **+0.031** | 45.79 | 5.319 | -0.081 |
| har_econ | HAR | 75 | 1.0071 | 1.2077 | -2.759 | 47.60 | 15500.4 | -1.317 | 49.25 | 49.56 | +0.055 |
| har_weather_pca | HAR | 75 | 0.8969 | 1.0959 | -0.594 | 72.94 | 75.57 | -0.135 | 75.38 | 7.753 | +0.014 |
| **har_log** (benchmark) | HAR | 75 | 0.8530 | 1.0505 | -0.404 | 74.42 | 57.94 | +0.013 | 78.02 | 5.906 | -0.025 |
| har_j | HAR | 75 | 0.8534 | 1.0506 | -0.406 | 74.90 | 57.96 | +0.014 | 78.54 | 5.818 | -0.015 |
| har_cj | HAR | 75 | 0.8522 | 1.0553 | -0.408 | 88.13 | 57.70 | -0.003 | 92.29 | 5.958 | -0.025 |
| har_ext | HAR | 75 | **0.8351** | **1.0346** | -0.356 | 581.0 | 56.67 | **+0.040** | 595.7 | 4.770 | -0.143 |
| har_route | HAR | 75 | 0.8492 | 1.0569 | -0.440 | 832.6 | 57.44 | -0.015 | 871.6 | 5.831 | -0.006 |
| har_weather_econ | HAR | 75 | 1.3360 | 1.5834 | -7.726 | 61615 | 1.4e7 | -4.323 | 61930 | 62558 | +0.546 |
| har_weather | HAR | 75 | 1.0847 | 1.3256 | -1.842 | 61739 | 74.17 | -0.961 | 62059 | 749.3 | +0.445 |

Ranking is unstable across metrics and that instability is itself the headline finding:
by mean QLIKE the two GARCH models win by a factor of ~20; by `MAE_log`/`RMSE_log` the HAR family
wins; by `MAE_vol_pp` everything is within ~10% except the over-parameterised weather models,
which explode.

---

## 2. Diebold-Mariano incremental-value tests

Paired on the exact same rows. `d = L(baseline) - L(challenger)`, so **`dm_t > 0` means the
challenger (usually the augmented model) is better**. Newey-West HAC, lag = HORIZON + 4 = 9,
two-sided p from the t distribution with n-1 df. `p_hln` adds the Harvey-Leybourne-Newbold (1997)
small-sample correction. `p_daily` re-tests after averaging the differential within each calendar
date, which removes all cross-route contemporaneous correlation and is the trustworthy column.

Squared-error loss in log space (`(ln a - ln q)^2`):

| comparison | n | skill gain % | DM t | p | p (HLN) | p (daily) | sig 5% |
|---|---|---|---|---|---|---|---|
| HAR+weather (82 raw) vs HAR | 6268 | -98.72 | -5.259 | 0.0000 | 0.0000 | 0.0042 | yes (worse) |
| HAR+weather (5 PCs) vs HAR | 6268 | -14.95 | -3.348 | 0.0008 | 0.0012 | 0.0577 | yes (worse) |
| HAR+econ vs HAR | 6268 | -134.75 | -3.362 | 0.0008 | 0.0011 | 0.0740 | yes (worse) |
| HAR+weather+econ (raw) vs HAR | 6268 | -439.32 | -4.273 | 0.0000 | 0.0000 | 0.0080 | yes (worse) |
| HAR+weatherPCs+econ vs HAR | 6268 | -78.38 | -4.249 | 0.0000 | 0.0000 | 0.0123 | yes (worse) |
| HAR+ext (EWMA/skew/kurt) vs HAR | 6268 | +2.694 | 1.628 | 0.1035 | 0.1145 | 0.2221 | no |
| HAR-J vs HAR | 6268 | +0.063 | 0.534 | 0.5935 | 0.5990 | 0.6197 | no |
| HAR-CJ vs HAR | 6268 | -1.623 | -1.523 | 0.1279 | 0.1395 | 0.1352 | no |
| HAR per-route vs HAR pooled | 6268 | -2.803 | -1.776 | 0.0758 | 0.0835 | 0.1878 | no |
| HAR strict-lag vs HAR | 6268 | +1.783 | 1.942 | 0.0521 | 0.0601 | 0.0125 | no / yes (better) |
| GARCH(1,1)-normal vs HAR | 6268 | -5.399 | -1.756 | 0.0791 | 0.0868 | 0.1225 | no |
| GARCH(1,1)-t vs HAR | 6268 | -8.849 | -2.525 | 0.0116 | 0.0151 | 0.0854 | yes (worse) |
| EGARCH(1,1)-t vs HAR | 6268 | -77.78 | -2.828 | 0.0047 | 0.0068 | 0.0049 | yes (worse) |
| GJR-GARCH(1,1)-t vs HAR | 6268 | -77.78 | -2.828 | 0.0047 | 0.0068 | 0.0049 | yes (worse) |
| HAR+weather (5 PCs) vs GARCH(1,1)-t | 6268 | -5.608 | -1.070 | 0.2846 | 0.2952 | 0.2903 | no |

QLIKE loss (`2 ln q + a^2/q^2`):

| comparison | n | skill gain % | DM t | p | p (daily) | sig 5% |
|---|---|---|---|---|---|---|
| HAR+weather (82 raw) vs HAR | 6268 | -79439 | -2.109 | 0.0350 | 0.1800 | yes (worse) |
| HAR+weather (5 PCs) vs HAR | 6268 | +3.386 | 0.295 | 0.7682 | 0.7870 | no |
| HAR+econ vs HAR | 6268 | +36.88 | 1.091 | 0.2754 | 0.3043 | no |
| HAR+weatherPCs+econ vs HAR | 6268 | +49.11 | 1.127 | 0.2599 | 0.2865 | no |
| HAR+ext vs HAR | 6268 | -663.5 | -1.392 | 0.1640 | 0.1910 | no |
| HAR-J vs HAR | 6268 | -0.660 | -0.871 | 0.3839 | 0.4115 | no |
| HAR per-route vs HAR | 6268 | -1017 | -1.238 | 0.2157 | 0.2425 | no |
| HAR strict-lag vs HAR | 6268 | +41.32 | 0.727 | 0.4675 | 0.3696 | no |
| GARCH(1,1)-normal vs HAR | 6268 | +94.23 | 1.252 | 0.2104 | 0.2219 | no |
| **GARCH(1,1)-t vs HAR** | 6268 | **+95.68** | 1.271 | 0.2036 | 0.2312 | no |
| EGARCH(1,1)-t vs HAR | 6268 | +94.05 | 1.250 | 0.2114 | 0.2239 | no |
| GJR-GARCH(1,1)-t vs HAR | 6268 | +94.05 | 1.250 | 0.2114 | 0.2239 | no |

**The QLIKE and squared-error columns disagree in sign, and the reason matters.**
`squ_sq` penalises both directions; QLIKE `2 ln q + a^2/q^2` is almost indifferent to
*over*-forecasting and savagely punishes *under*-forecasting. The weather/econ models are
inflated - `har_weather` produces predictions up to 294.8 (29,480% annualised) when the true
value is 2.74 - so they buy QLIKE "skill" purely by over-predicting. Dropping the worst 1% of
observations reverses the QLIKE verdict for `har_weather_pca`: skill goes from **+3.39%
(p=0.77)** to **-3.81% (t=-3.81, p=0.0001)**, i.e. significantly *worse*. Trimmed QLIKE is in
`agentA_metrics_pooled.csv` (`QLIKE_trim1`) and the test is in `agentA_dm_robustness.csv`.

---

## 3. Verdict: does weather/cyclone information add incremental value over HAR-RV?

**No. On the evidence here it adds nothing, and the honest point estimate is slightly negative.**

The specific numbers:

* **Squared error, the loss that is symmetric.** HAR+weather (5 weather PCs) is **14.95% worse**
  than plain HAR, DM t = **-3.35**, p = **0.0008** (p = 0.058 once cross-route duplication is
  collapsed). HAR+weather with all 82 raw columns is **98.7% worse**, t = -5.26, p < 1e-6.
  The QLIKE-symmetric companion test is in the table above.
* **The sign test is the most robust of all.** Of 1,569 distinct calendar dates, the weather-augmented
  model has the lower squared-error loss on only **44.6%** of them (binomial p < 1e-4) and the
  lower QLIKE on **44.2%** (p < 1e-4). It loses more often than it wins.
* **Adding economic features does not help either**: 134.7% worse on squared error
  (t = -3.36, p = 0.0008), better on 49.5% of dates (p = 0.686).
* **It is not a power problem.** The comparison is not "insignificant, so maybe there is
  something". The point estimates are negative and the sign tests are decisive. If anything the
  weather block carries *harmful* extra variance.
* **Why the raw-block model fails so badly.** On a typical fold, 82 weather columns + 4 route
  dummies + 3 HAR terms = 89 parameters fitted to ~1,700-4,500 training rows in which **every
  row is duplicated across routes**. Weather-only in-sample R2 for `log_rv_fwd` is a real 0.055,
  and in-sample adj-R2 rises from 0.077 (HAR) to 0.113 (HAR+weather) - the block does have
  explanatory power *in sample*. Out of sample it is worth nothing, which is the classic signature
  of a slow contemporaneous trend (war-season / 2021-22 disruption) that the 22-day HAR memory
  already absorbs.
* **The strongest HAR improvement found is not weather.** Dropping the current day's squared
  return and using strictly *t-1*-dated HAR components (`har_log_strictlag`) improves on HAR by
  **+1.78%** squared error (t = +1.94, p = 0.052; p = 0.0125 on the daily-collapsed series) and
  is the only variant with positive **pooled** R2 (+0.031 vs +0.013). On 18.4% of days the daily
  return is exactly zero, so `ln sq_ret(t)` is pure noise; not using it helps.

**Disclosure of specification search.** Two weather specifications were pre-specified and both
are reported above (all 82 raw columns, and a 5-component PCA fitted on training rows only).
The PCA variant was added *after* seeing that the raw model was saturated, and it is labelled as
such. The DM test was computed for every specification; no best-of-many selection was applied.
The negative verdict holds for both, and the trimmed-loss robustness check strengthens it.

---

## 4. Model-family verdicts

| family | verdict |
|---|---|
| **GARCH(1,1)-t** | Best mean and pooled QLIKE (3.387 / 3.370). Well calibrated on the level (mean forecast 0.653 vs mean actual 0.825). Loses to HAR by 8.8% on squared error (p = 0.012) and is never significantly better than HAR on QLIKE (p = 0.20). The t innovation is worth ~25% of QLIKE relative to normal, consistent with fat-tailed freight returns. |
| **EGARCH / GJR** | Leverage is a *liability* here: 77.8% worse than HAR on squared error (p = 0.005). The asymmetry term chases large negative returns that do not predict forward volatility in a step-function price series. |
| **HAR-J / HAR-CJ** | No benefit (t = +0.53 and -1.52). The bipower-variation jump estimator is meaningless when 18.4% of daily returns are exactly zero. |
| **HAR per-route** | 2.8% *worse* than pooled (p = 0.076). Four routes, 2 of which are byte-identical duplicates - no per-route information to estimate. |
| **HAR + EWMA/skew/kurt** | Best `MAE_log` (0.835) and pooled R2 (+0.040) but catastrophic mean QLIKE (581); it improves the centre of the distribution and blows up the tail. |
| **All HAR variants** | Fitted in logs, so `exp(prediction)` estimates the conditional **median**, not the mean. See the caveat in section 6. |

---

## 5. Data problems a reviewer must know

1. **The dataset ships the "banned" forward-looking columns by name.**
   `Cyclone_Wind_Forecast3d/7d`, `Cyclone_Active_Forecast3d`, `Wind_Speed_Forecast7d`,
   `Rolling_7d_Cyclone_Days`, `Price_Volatility_7d/30d` are all present in
   `clean_engineered_dataset.csv`. Every one is excluded here by an explicit name block-list.
2. **There are only TWO distinct routes, not four.** `XSICFENE_FarEast_NorthEurope` and
   `XSICNEFE_NorthEurope_FarEast` have **identical** `Price_USD` on 100.00% of dates (corr = 1.0000,
   mean |log ratio| = 0.00000), and likewise the two US-West-Coast legs. The "4-route panel" is
   2 lanes x 2 duplicated direction labels. Every cross-sectional statistic therefore double-counts,
   and any per-route or pooled-panel standard error that assumes independence is wrong by ~sqrt(2)
   or worse. This is why a date-collapsed DM column is reported alongside the requested one.
3. **Prices are a step function, not a daily series.** 18.4% of daily log returns are exactly zero
   (23.7% on the US-West-Coast legs), and stale runs of up to **16 consecutive identical prices**
   occur. The quote then refreshes with a very large jump - e.g. FarEast->USWestCoast goes
   650 (2020-07-31) to 1227 (2020-08-03), +63.5% in one day. The top **1% of squared returns
   carries 64.9%** of the total sum of squared returns. `rv_fwd` is therefore not measuring a
   continuous volatility process; it is largely measuring *when the quote happened to refresh*.
4. **Stall age predicts the target, but only weakly.** Mean forward 5-day annualised vol rises
   monotonically with days-since-last-price-change (1.04 at stall 0, 1.44 at 1, 3.59 at 5;
   P(rv_fwd > 1) rises 0.31 -> 0.56). But `stall_lag` alone explains only R2 = 0.010 of
   `log_rv_fwd` (vs 0.051 for HAR core), so this mechanism cannot be turned into forecast skill -
   it is a description of the data, not a fix.
5. **The target is barely predictable and the obvious relation is broken.** Correlation of
   `log_rv_fwd` with the HAR weekly component is **+0.069**, with the monthly component **+0.293**.
   For a genuine RV process the weekly correlation should dominate. The low weekly correlation is
   a direct consequence of problem 3.
6. **A single observation can dominate the aggregate metric.** For `har_log`, one row
   (2020-07-28, `XSICFEUW_FarEast_USWestCoast`, actual 10.14, predicted 0.0231) contributes
   **90.1%** of the total pooled QLIKE of 78.0. Its duplicate on the mirrored route contributes
   the same again. Mean QLIKE across folds is not a stable statistic for this target; the median
   per-observation QLIKE (~-0.03) and the 1%-trimmed QLIKE are far more informative.
7. **The within-fold R2 is a tiny-sample statistic and is misleading.** Mean-of-fold R2 for
   `har_log` is **-0.404** while the pooled R2 on the identical rows is **+0.013** (and
   **+0.123** after dropping the worst 1%). Only 12 of 75 folds have positive R2 for `har_log`.
   Each fold has ~84 heavily overlapping rows representing ~4 independent 5-day observations per
   route, so a single missed month drives the fold R2 deeply negative.
8. **Zero-RV rows and end-of-sample truncation.** 66 rows have `rv_fwd == 0` (50 in 2018, 32+32 on
   the two West-Coast legs); they are dropped before scoring and are concentrated *before* the
   evaluation window, so they matter less here than in a full-sample study. The final origin
   (2026-04-01) is dropped because the last 5 observations per route have no `rv_fwd`.
9. **Evaluation-window regime shift.** The mean training `log_rv_fwd` is -0.557 and the mean test
   value -0.742, but the fold-level gap ranges from -1.84 (origin 2021-02) to +1.07. In 2021-02 the
   test month's volatility collapsed to ~1/8 of the training level while the container market was
   in crisis - again a symptom of stale quotes during the fastest-moving period.

---

## 6. The single most important caveat

**Everything here is scored on a target that is partly an artefact of a stale-quote step function,
and the model ranking is decided by that artefact.** Because `rv_fwd` is driven by *when* a quote
refreshes, and because 65% of the sum of squared returns comes from 1% of days, the loss
distribution has a tail so heavy that a handful of observations (in practice a single 2020-07 event,
counted twice because the routes are duplicates) determines the mean QLIKE, the DM verdicts and the
sign of the mean-of-fold R2. The weather conclusion is robust to this - it is negative under the
symmetric loss, under trimming, and under a sign test on 1,569 dates - but the *ranking* of GARCH
against HAR is not: GARCH wins by ~20x on mean QLIKE and loses by ~9% on squared error, and both are
true statements about the same 6,268 rows. Any decision taken on the strength of a single metric here
would be overfitting to the refresh process.

A second, related caveat for the level-scale metrics: every HAR variant is estimated in logs, so
`exp(prediction)` targets the conditional **median** of `rv_fwd`, while `rv_fwd` is right-skewed
(mean 0.825, median ~0.48). HAR's mean level forecast is therefore ~40% below the mean actual
(ratio 0.60), which is why it loses `MAE_vol_pp` and QLIKE to GARCH. Applying the standard Duan
smearing factor `exp(sigma^2/2)` (median 1.13 across folds) lifts the ratio only to 0.70 - the
log-normal correction is not sufficient for this tail, and a larger factor would overshoot the mean.
`agentA_smearing_experiment.csv` reports the corrected numbers. Note this is a fair criticism of
log-HAR *as a level forecaster* and not evidence that HAR has less information; the ranking on
`MAE_log`/`RMSE_log` (where HAR wins) is unaffected by the transform.
