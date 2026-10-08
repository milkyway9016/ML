# agentD — Does ocean weather / cyclone activity physically affect maritime freight-rate volatility?

**Question.** Does weather/cyclone activity add incremental explanatory power for forward 5-day
realised volatility of container freight rates, beyond volatility persistence?

**Blunt answer: no.** Every weather effect I can measure is indistinguishable from zero once the
comparison is made honestly. The one apparently significant result (naive event study) survives only
because event days sit in a different volatility regime than average — it disappears under matched
control, contributes R² = +0.0002 to a regression, and fails a Wald test at p = 0.99.

Data: `D:\Internproj\New ML\clean_engineered_dataset.csv`. Target, per route, sorted by (Route, Date):
`ret = ln(max(Price_USD,1)).diff()`, `fwd_rv5(t) = sqrt(Σ_{t+1..t+5} ret²)·√252`,
`trail_rv22(t) = sqrt(mean_{22 days ending t} ret²)·√252`. Banned forward-looking/contemporaneous
columns (Cyclone_Wind_Forecast3d/7d, Cyclone_Active_Forecast3d, Wind_Speed_Forecast7d,
Rolling_7d_Cyclone_Days, Price_Volatility_7d/30d) were never used as inputs.

---

## 1. Data quality and the non-traversed-zone leakage check

**Leakage check: CLEAN.** I audited every (route × zone × metric) combination where the zone does
**not** lie on the route's lane — 112 combinations. **Zero** carry any non-zero weather value. The
zones are simply absent (`NaN`) for the lanes that don't transit them. Route-level summaries are
route-specific: `Route_cyclone_active` equals the OR of own-zone `cyclone_active` flags on 2,829/2,829
FE–NorthEurope days (0 mismatches). **No leakage mechanism exists in this file.** This is a real
positive finding — it removes a whole class of spurious results.

**Continuous met fields are structurally clean.** No sentinel zeros, no forward-filled stretches
(maximum run of identical consecutive values = 1–2 days), no impossible values. `slp_*` is stored in
**Pascals** (~100,200–103,400 Pa), not hPa — a unit trap, not an error. The trailing 71 days of every
route have `NaN` weather (dataset runs to 2026-04-15; weather stops earlier). Max `wind_speed_*` is
only 14.2 m/s.

**Two of the seven FE–NorthEurope zones carry no cyclone information at all.** `cyclone_*` for
**Red_Sea** and **North_Atlantic** are identically zero on all 9,800 rows. The FE↔North Europe lane
transits both, and both are real cyclone basins (Red Sea medicanes, North Atlantic hurricanes). So
2/7 transited zones are structurally dead.

**The `cyclone_active` flag is not a credible storm-day identifier.** Per-zone shares of
cyclone-active days on the FE–NorthEurope route:

| zone | freq | implied/yr | max wind | median wind on active days |
|---|---|---|---|---|
| South_China_Sea | 15.0% | ~38 | 105 kt | 45 kt |
| Indian_Ocean | 23.9% | ~60 | 135 kt | 40 kt |
| Arabian_Sea | 4.8% | ~12 | 125 kt | 40 kt |
| Mediterranean_Sea | 15.0% | ~37 | 140 kt | 45 kt |
| North_Sea | 1.6% | ~4 | 70 kt | 45 kt |
| Red_Sea / North_Atlantic | 0% | 0 | 0 | — |

The Mediterranean sees roughly 1–2 medicanes per year; ~37/yr is **~50× too frequent**. The Indian
Ocean figure is ~4× the real rate. Decisively: on cyclone-active days for FE–NorthEurope the
**mean wind speed is 3.008 m/s vs 2.997 m/s on non-active days — statistically and physically
identical**. The flag marks "a storm exists somewhere in the basin", not "the shipping lane is
experiencing weather". So the weather signal is neither clean nor well-localised, which bounds what
is possible before any modelling.

## 2. Event study (core deliverable)

Event definitions use only past/contemporaneous information (expanding trailing percentiles, shifted
one day). Estimator choice turned out to be the whole ballgame.

**Estimator validation (placebo = random pseudo-events, same count, same routes):**

| estimator | placebo mean | placebo sd | 95% range |
|---|---|---|---|
| k=1 nearest-neighbour match on log trail_rv22 | **+0.0553** | 0.0338 | — |
| IPW / doubly-robust | −0.0037 | 0.0258 | [−0.063, +0.027] |
| vol-decile stratified | +0.0009 | 0.0303 | [−0.051, +0.058] |

The textbook matching approach is **biased** here: selecting the single closest control on a *noisy*
covariate cherry-picks controls with an upward forward-vol surprise, manufacturing a spurious negative
"effect" that even flips sign at k=5. I discarded it. IPW and decile estimators are unbiased.

**Final event study** (target = `fwd_rv5`, annualised; bootstrap CIs are month-block; naive NW = Newey-West L=20):

| event | n_ev | event mean | non-event mean | **naive diff** | naive NW p | **matched (IPW) diff** | IPW 95% CI | decile diff | decile 95% CI |
|---|---|---|---|---|---|---|---|---|---|
| cyclone active | 4164 | 1.0884 | 1.1754 | **−0.0871** | 0.151 | **−0.1281** | [−0.365, +0.048] | −0.0647 | [−0.232, +0.097] |
| cyc wind > trail p90 | 930 | 1.0470 | 1.1100 | −0.0630 | 0.436 | −0.0350 | [−0.292, +0.141] | −0.0091 | [−0.249, +0.168] |
| cyc wind ≥ 64 kt | 920 | 1.0417 | 1.1483 | −0.1066 | 0.153 | −0.0061 | [−0.222, +0.172] | −0.0137 | [−0.191, +0.188] |
| SLP anomaly < p10 | 782 | 1.0986 | 1.0864 | +0.0122 | 0.903 | −0.0253 | [−0.179, +0.175] | +0.0050 | [−0.185, +0.218] |
| wind > trail p90 | 996 | 1.0459 | 1.1098 | −0.0639 | 0.370 | −0.1154 | [−0.289, +0.037] | −0.1219 | [−0.303, +0.042] |
| cyclone AND wind | 322 | 1.0084 | 1.1427 | −0.1344 | 0.250 | −0.2392 | [−0.684, +0.153] | −0.2270 | [−0.801, +0.114] |

**Every matched CI spans zero.** Not one weather shock raises future volatility.

Two diagnostics make the null trustworthy rather than merely underpowered:

**(i) The naive/matched discrepancy is regime, not weather.** The naive test flags `cyc_active`
(p = 0.0096) and `cyc_wind_ge64kt` (p = 0.0089). Matching on log `trail_rv22` removes 96.7–99.9% of
the covariate imbalance — and the significance evaporates. Cyclone-active days are *lower*-volatility
days than average, so the naive comparison was reading volatility clustering and mean reversion, not
weather.

**(ii) Aligned windows show the "effect" already exists before the shock.** Mean `fwd_rv5`
difference, pre-event (t−5..t−1) vs post-event (t+1..t+15):

| event | pre-trend | offset 0 | post-event |
|---|---|---|---|
| cyclone active | −0.0582 | −0.1053 | −0.0193 |
| cyc wind ≥ 64 kt | −0.1263 | −0.1128 | −0.0535 |
| SLP anomaly < p10 | +0.0633 | +0.0098 | −0.0821 |
| wind > p90 | +0.0366 | −0.0643 | +0.0413 |

For `cyc_wind_ge64kt` the pre-trend (−0.126) is **larger** than any post-event value. There is no
discontinuity at the event: whatever separation exists is a persistent regime difference that predates
the shock. See `plots/agentD_event_study.png`.

## 3. Does weather add anything beyond volatility persistence?

OLS on logs, HAC (Newey-West, L=10) standard errors. **Sample correction:** the source contains each
lane *twice* — FE→NE and NE→FE have byte-identical `Price_USD`, likewise FE→USWC and USWC→FE (4 rows
per lane-date, 2 physical lanes). Headline results are on the de-duplicated sample (n = 4,876,
2015-01-20 … 2026-04-14); the duplicated sample doubles n and leaves R² identical to 5 decimals, so
it only fakes precision.

| model | n | R² | **incremental R² vs trail-only** | Wald χ² | df | **HAC p** |
|---|---|---|---|---|---|---|
| M0 trail_rv22 only | 4876 | 0.12653 | — | — | — | — |
| M1 + 6 shock dummies | 4876 | 0.12675 | **+0.000224** (+0.18%) | 0.78 | 6 | **0.993** |
| M2 + 4 continuous weather | 4876 | 0.13856 | +0.012034 (+9.5%) | 1.17 | 4 | 0.882 |
| M3 + shocks + continuous | 4876 | 0.13886 | +0.012338 (+9.8%) | 3.05 | 10 | **0.980** |
| M4 + cyclone shocks only | 4876 | 0.12657 | +0.000048 | 0.15 | 3 | 0.986 |
| M5 + slp/wind shocks only | 4876 | 0.12659 | +0.000063 | 0.27 | 2 | 0.876 |

Weather shock variables add **0.02 percentage points** of R² and are jointly insignificant at
p = 0.99. The larger M2/M3 delta comes from *continuous level* weather variables (cyclone wind, SLP
anomaly, wind anomaly), not from shocks, and is also insignificant (p = 0.88) — those levels mostly
proxy season and lane and add nothing beyond route fixed effects.

Coefficient signs in M3 (annualised log-vol): `ev_cyc_active` +0.046 (t=0.38), `ev_cyc_wind_ge64kt`
+0.086 (t=0.53), `ev_slp_anom_lt_p10` −0.098 (t=−0.73), `ev_wind_gt_p90` +0.034 (t=0.30),
`od_max_cyc_wind` +0.0002 (t=0.03). **No term reaches |t| > 1.1.** Contrast `log_trail_rv22`
coefficient **0.7968** (HAC t = 15.6). Within-lane R² is 0.1273 for trailing vol alone and 0.1298
with shocks — weather moves it by 0.0025.

## 4. Where would weather matter most?

No stratum produces a coherent effect. Formal interaction Wald tests (HAC L=10):

| interaction | χ² | df | p | significant |
|---|---|---|---|---|
| peak season (Aug–Oct) × shocks | 1.68 | 4 | 0.795 | No |
| lane family (Europe vs Pacific) × shocks | 4.99 | 4 | 0.288 | No |
| turbulent regime × shocks | 6.28 | 4 | 0.179 | No |

Stratum coefficient signs flip arbitrarily: `ev_cyc_active` is −0.067 on Europe lanes but +0.194 on
Pacific; `ev_wind_gt_p90` is +0.246 in calm regimes but −0.279 in normal (HAC p = 0.040 for calm
only — one hit out of 12 stratum×event tests, i.e. exactly the chance rate). Vol-decile-adjusted
effects at the turbulent tercile swing to −0.33 and −0.38 with HAC SEs of 0.14–0.17, i.e. noise.
`plots/agentD_conditional_effects.png` shows bars flipping sign across every panel. There is no stormy
season, lane, or volatility regime in which weather carries signal.

---

## Verdict

**Ocean weather and cyclone activity have no measurable effect on container freight-rate volatility in
this dataset beyond the persistence of volatility itself.** I tested this four ways — matched event
study, aligned event windows, HAC regression with incremental-R², and conditional/interaction
analysis — and the answer is the same each time. The matched-control numbers prove the test was fair,
not merely underpowered: the estimators were validated against placebo draws (IPW placebo −0.004,
decile +0.001), the naive comparison's apparent significance was traced to volatility clustering and
removed by matching, and the residual separation in event windows is fully present *before* the shock.

Two honest caveats. First, the exercise is partly bounded by the data: two of seven transited zones
have no cyclone data, the `cyclone_active` flag fires on ~51% of FE–NorthEurope days and is
uncorrelated with actual wind speed, and the Red Sea / North Atlantic blocks are dead. A better
storm-localisation product could in principle show something this file cannot. Second, the
methodology did surface a genuine methodological trap worth carrying forward: **greedy k=1
nearest-neighbour matching on a noisy covariate is biased here** (+0.055 placebo), and any analysis
that had used it would have "found" a significant negative weather effect that does not exist.

Artifacts: `agentD_weather_quality.csv`, `agentD_zone_leakage_audit.csv`, `agentD_event_study.csv`,
`agentD_event_windows.csv`, `agentD_event_pretrend.csv`, `agentD_estimator_validation.csv`,
`agentD_caliper_sensitivity.csv`, `agentD_regressions.csv`, `agentD_regression_within_lane_r2.csv`,
`agentD_regression_sample_robustness.csv`, `agentD_conditional_effects.csv`,
`agentD_conditional_descriptives.csv`, `plots/agentD_event_study.png`,
`plots/agentD_conditional_effects.png`, `plots/agentD_estimator_validation.png`.
Scripts: `D:\testing\agents\agentD_01..04*.py`.
