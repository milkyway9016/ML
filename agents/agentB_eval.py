"""
agentB_eval.py
==============
Direction + large-move + alerting walk-forward backtest for the
DIRECTION / HORIZON workstream.

One fold per calendar month, origin = first day of the month, starting
2020-01-01. Train on pre-origin rows whose horizon-specific label endpoint is
also before the origin (expanding window), test on the following calendar
month. Requires >= 500 train rows, >= 20 test rows and >= 2 years of history.

Targets
-------
direction   : sign(fwd_ret_H) for H in {1, 5, 20}
large move  : |fwd_ret_H| > threshold, thresholds learned on TRAINING data only:
                vol-scaled : k * rv_trail22 * sqrt(H/252),  k in {1.0, 1.5, 2.0}
                fixed      : 2%, 3%, 5%
                (plus the existing pipeline's per-route train q90, for reference)

Models: logistic regression (standardised, L2) and XGBoost, on
        PRICE  vs  PRICE+WEATHER feature sets.  n_jobs=1, tree_method='hist'.

Variants
--------
  both     : all four routes (matches the existing D:\\testing pipeline)
  primary2 : one representative per mirrored route pair, i.e. two genuinely
             independent lanes.  Reported because the mirror pairs are exact
             duplicates and inflate any pooled statistic.

Usage:  python agentB_eval.py --variant both|primary2 --out <path.csv> [--horizons 1,5,20]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agentB_common import (  # noqa: E402
    HORIZONS, VOL_K, FIXED_THR, PRIMARY_ROUTES, ROUTE_COL, DATE_COL,
    build_features, feature_sets, lag_weather, load_raw, log, verify_mirrors,
)

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss  # noqa: E402

SEED = 42
WF_START = pd.Timestamp("2020-01-01")
MIN_TRAIN_YEARS = 2.0
MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 20
TOP_FRAC = 0.10
PEAK_MONTHS = (8, 9, 10)
XGB_TREES = 250


def make_xgb():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=XGB_TREES, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        reg_lambda=2.0, random_state=SEED, n_jobs=1, tree_method="hist",
        eval_metric="logloss", verbosity=0,
    )


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def _clf_metrics(y, p, train_rate):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    out = dict(n=len(y), pos_rate=float(y.mean()) if len(y) else np.nan)
    if len(y) == 0 or y.min() == y.max():
        return {**out, "AUC": np.nan, "AP": np.nan, "Brier": np.nan,
                "Brier_ref": np.nan, "BrierSkill": np.nan,
                "Acc": np.nan, "MajBase": np.nan, "BalAcc": np.nan}
    out["AUC"] = float(roc_auc_score(y, p))
    out["AP"] = float(average_precision_score(y, p))
    bs = float(brier_score_loss(y, p))
    if np.isfinite(train_rate):
        # Brier skill against ALWAYS predicting the TRAINING base rate
        ref = float(brier_score_loss(y, np.full(len(y), float(train_rate))))
        out.update(Brier=bs, Brier_ref=ref,
                   BrierSkill=float(1.0 - bs / ref) if ref > 0 else np.nan)
    else:
        # no meaningful training rate (pooled seasonal sub-sample): fall back to
        # the sub-sample event rate purely as a reference point
        ref = float(y.mean())
        ref = max(min(ref, 1 - 1e-9), 1e-9)
        out.update(Brier=bs, Brier_ref=float(brier_score_loss(y, np.full(len(y), ref))),
                   BrierSkill=np.nan)
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())
    maj = float(max(y.mean(), 1.0 - y.mean()))
    tpr = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    tnr = float(1.0 - pred[y == 0].mean()) if (y == 0).any() else np.nan
    out.update(Acc=acc, MajBase=maj, BalAcc=float(np.nanmean([tpr, tnr])))
    return out


def _top_decile(y, p, frac=TOP_FRAC):
    """Precision / recall / lift in the top `frac` of predicted probability."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    if n == 0 or y.sum() == 0:
        return dict(n_alert=k, precision=np.nan, recall=np.nan, lift=np.nan)
    order = np.argsort(-p, kind="mergesort")[:k]
    prec = float(y[order].mean())
    base = float(y.mean())
    return dict(n_alert=k,
                precision=prec,
                recall=float(y[order].sum() / y.sum()),
                lift=float(prec / base) if base > 0 else np.nan,
                base_rate=base)


def _pooled_metrics(y, p, train_rate):
    """Metrics computed on the pooled out-of-sample record (single ranking).

    For direction, `train_rate` is the mean training base rate; because the
    pooled sample spans many folds, a fixed-probability reference is not
    meaningful there, so BrierSkill is left NaN unless a rate is supplied.
    """
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    m = _clf_metrics(y, p, train_rate)
    k = max(1, int(np.ceil(TOP_FRAC * len(y))))
    if len(y) and y.sum() > 0:
        order = np.argsort(-p, kind="mergesort")[:k]
        base = float(y.mean())
        m["precision_at_10pct"] = float(y[order].mean())
        m["recall_at_10pct"] = float(y[order].sum() / y.sum())
        m["lift_at_10pct"] = float(m["precision_at_10pct"] / base) if base > 0 else np.nan
        m["base_rate"] = base
    return m


def newey_west_t(diff: np.ndarray, lag: int):
    """Two-sided t-test on mean(diff) with Newey-West HAC standard errors."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    mu = d.mean()
    e = d - mu
    var = float(np.sum(e * e) / n)
    L = min(lag, n - 1)
    for l in range(1, L + 1):
        gl = float(np.sum(e[l:] * e[:-l]) / n)
        var += 2.0 * (1.0 - l / (L + 1.0)) * gl
    var = max(var, 1e-18)
    se = np.sqrt(var / n)
    if se <= 0:
        return np.nan, np.nan
    t = mu / se
    try:
        from scipy import stats
        p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
    except Exception:
        p = np.nan
    return float(t), float(p)


# ---------------------------------------------------------------------------
def build_folds(d: pd.DataFrame):
    origins = pd.date_range(WF_START, d[DATE_COL].max(), freq="MS")
    folds = []
    for origin in origins:
        end = origin + pd.offsets.MonthBegin(1)
        tr = (d[DATE_COL] < origin).to_numpy()
        te = ((d[DATE_COL] >= origin) & (d[DATE_COL] < end)).to_numpy()
        if not tr.any() or not te.any():
            continue
        span = (d.loc[tr, DATE_COL].max() - d.loc[tr, DATE_COL].min()).days
        if span < MIN_TRAIN_YEARS * 365:
            continue
        if tr.sum() < MIN_TRAIN_ROWS or te.sum() < MIN_TEST_ROWS:
            continue
        folds.append(dict(origin=origin, tr=tr, te=te, n_train=int(tr.sum()),
                          n_test=int(te.sum())))
    return folds


def label_defs(d: pd.DataFrame, tr: pd.Series, te: pd.Series, H: int):
    """Yield (def_name, y_train, y_test) for every large-move definition."""
    fr = d[f"fwd_ret_{H}"].to_numpy()
    rv = d["rv_trail22"].to_numpy()
    scope = tr | te
    ok = scope & np.isfinite(fr) & np.isfinite(rv)
    for k in VOL_K:
        thr = k * rv * np.sqrt(H / 252.0)
        yield (f"vol_k{k:.1f}", H, (np.abs(fr) > thr), ok)
    for t in FIXED_THR:
        yield (f"fixed_{int(t*100)}pct", H, (np.abs(fr) > t), ok)
    # per-route train-only q90 (matches the pre-existing pipeline)
    tmp = pd.DataFrame({"r": d[ROUTE_COL].to_numpy(), "a": np.abs(fr)})
    q = tmp.loc[tr.to_numpy()].groupby("r")["a"].quantile(0.90)
    thr_r = tmp["r"].map(q).to_numpy()
    yield ("route_q90", H, (np.abs(fr) > thr_r), ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="both", choices=["both", "primary2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--horizons", default="1,5,20")
    ap.add_argument("--fast", action="store_true", help="xgb_price_weather only for definitions")
    args = ap.parse_args()

    horizons = [int(x) for x in args.horizons.split(",") if x]
    t0 = time.time()

    raw = load_raw()
    mir = verify_mirrors(raw)
    log("[mirror check]")
    log(mir.to_string(index=False))

    if args.variant == "primary2":
        raw = raw[raw[ROUTE_COL].isin(PRIMARY_ROUTES)].reset_index(drop=True)
    log(f"[variant] {args.variant}: {raw.shape}, routes={sorted(raw[ROUTE_COL].unique())}")

    d = build_features(raw)
    d = lag_weather(d)
    fs, price_cols, wcols, cal = feature_sets(d)
    log(f"[features] price={len(fs['price'])} price_weather={len(fs['price_weather'])} "
        f"(weather block={len(wcols)}, calendar={len(cal)})")
    log(f"[features] price_weather n = {len(fs['price_weather'])}")

    for c in ["fwd_ret_1", "fwd_ret_5", "fwd_ret_20", "rv_trail22"]:
        if c not in d.columns:
            raise SystemExit(f"missing {c}")
    d = d[np.isfinite(d["rv_trail22"].to_numpy())].reset_index(drop=True)
    log(f"[rows] modelling rows = {len(d)}")

    # matrices
    Xp = d[fs["price"]].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    Xw = d[fs["price_weather"]].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    # for the logistic model, NaN->0 then standardised; XGB uses the same 0-fill
    # (xgboost hist handles NaN natively but 0-fill keeps the two variants identical)
    sets = {"price": Xp, "price_weather": Xw}

    folds = build_folds(d)
    log(f"[folds] {len(folds)} usable folds, first={folds[0]['origin'].date()} "
        f"last={folds[-1]['origin'].date()}")

    dcols = ["origin", "variant", "kind", "H", "definition", "model", "features",
             "n_train", "n_test", "train_rate", "test_rate"]
    dir_rows, mv_rows, al_rows = [], [], []
    oos = []          # pooled out-of-sample predictions
    skipped = []
    # Each forward-return label ends H route observations after its row. Keep
    # this per horizon because the experiment evaluates 1-, 5-, and 20-day
    # labels in the same monthly folds.
    target_end_by_h = {
        H: d.groupby(ROUTE_COL)[DATE_COL].shift(-H).to_numpy()
        for H in horizons
    }

    for fi, f in enumerate(folds):
        origin, tr, te = f["origin"], f["tr"], f["te"]
        peak_all = np.asarray(pd.DatetimeIndex(d[DATE_COL]).month.isin(PEAK_MONTHS))
        date_all = d[DATE_COL].to_numpy()

        model_preds = {}   # (kind,H,def) -> {(features,model): (ytr_rate, yte, pte)}

        # ---------------- direction --------------------------------------
        for H in horizons:
            y = (d[f"fwd_ret_{H}"].to_numpy() > 0).astype(int)
            ok = np.isfinite(d[f"fwd_ret_{H}"].to_numpy())
            mature = target_end_by_h[H] < origin
            trm, tem = tr & mature & ok, te & ok
            if trm.sum() < MIN_TRAIN_ROWS or tem.sum() < MIN_TEST_ROWS:
                continue
            if (d.loc[trm, DATE_COL].max() - d.loc[trm, DATE_COL].min()).days < MIN_TRAIN_YEARS * 365:
                continue
            ytr, yte = y[trm], y[tem]
            if ytr.min() == ytr.max() or yte.min() == yte.max():
                skipped.append(dict(origin=origin, kind="direction", H=H, reason="constant label"))
                continue
            train_rate = float(ytr.mean())
            Ztr_h, Zte_h = {}, {}
            for nm, M in sets.items():
                scaler = StandardScaler().fit(M[trm])
                Ztr_h[nm] = scaler.transform(M[trm])
                Zte_h[nm] = scaler.transform(M[tem])
            store = {}
            for fe in ["price", "price_weather"]:
                Xa, Xb = Ztr_h[fe], Zte_h[fe]
                lg = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED)
                lg.fit(Xa, ytr)
                pl = lg.predict_proba(Xb)[:, 1]
                xg = make_xgb()
                xg.fit(sets[fe][trm], ytr)
                px = xg.predict_proba(sets[fe][tem])[:, 1]
                store[(fe, "logit")] = pl
                store[(fe, "xgb")] = px
                for alg, p in [("logit", pl), ("xgb", px)]:
                    m = _clf_metrics(yte, p, train_rate)
                    dir_rows.append(dict(origin=origin, variant=args.variant, kind="direction",
                                         H=H, definition="sign", model=alg, features=fe,
                                         n_train=int(trm.sum()), n_test=int(tem.sum()),
                                         train_rate=train_rate, test_rate=float(yte.mean()), **m))
                    for i, gidx in enumerate(np.where(tem)[0]):
                        oos.append(dict(origin=origin, variant=args.variant, kind="direction", H=H,
                                        definition="sign", model=alg, features=fe,
                                        date=date_all[gidx], route=d[ROUTE_COL].iloc[gidx],
                                        peak=bool(peak_all[gidx]), y=yte[i], p=float(p[i])))
            model_preds[("direction", H, "sign")] = (train_rate, yte, tem, store)

        # ---------------- large moves ------------------------------------
        defs = []
        for H in horizons:
            mature = target_end_by_h[H] < origin
            tr_h = tr & mature
            for nm, HH, lab, okm in label_defs(d, pd.Series(tr_h), pd.Series(te), H):
                defs.append((nm, HH, lab, okm))
        for nm, H, lab, okm in defs:
            mature = target_end_by_h[H] < origin
            trm, tem = tr & mature & okm, te & okm
            if trm.sum() < MIN_TRAIN_ROWS or tem.sum() < MIN_TEST_ROWS:
                continue
            if (d.loc[trm, DATE_COL].max() - d.loc[trm, DATE_COL].min()).days < MIN_TRAIN_YEARS * 365:
                continue
            ytr, yte = lab[trm].astype(int), lab[tem].astype(int)
            if ytr.min() == ytr.max():
                skipped.append(dict(origin=origin, kind="largemove", H=H, definition=nm,
                                    reason="constant train label"))
                continue
            if yte.sum() == 0 or yte.sum() == len(yte):
                skipped.append(dict(origin=origin, kind="largemove", H=H, definition=nm,
                                    reason="degenerate test label"))
                continue
            train_rate = float(ytr.mean())
            Ztr_h, Zte_h = {}, {}
            for feature_name, M in sets.items():
                scaler = StandardScaler().fit(M[trm])
                Ztr_h[feature_name] = scaler.transform(M[trm])
                Zte_h[feature_name] = scaler.transform(M[tem])
            store = {}
            feats = ["price", "price_weather"] if not args.fast else ["price_weather"]
            for fe in feats:
                Xa, Xb = Ztr_h[fe], Zte_h[fe]
                lg = LogisticRegression(max_iter=3000, C=0.5, random_state=SEED)
                lg.fit(Xa, ytr)
                pl = lg.predict_proba(Xb)[:, 1]
                xg = make_xgb()
                xg.fit(sets[fe][trm], ytr)
                px = xg.predict_proba(sets[fe][tem])[:, 1]
                store[(fe, "logit")] = pl
                store[(fe, "xgb")] = px
                for alg, p in [("logit", pl), ("xgb", px)]:
                    m = _clf_metrics(yte, p, train_rate)
                    a = _top_decile(yte, p)
                    mv_rows.append(dict(origin=origin, variant=args.variant, kind="largemove",
                                        H=H, definition=nm, model=alg, features=fe,
                                        n_train=int(trm.sum()), n_test=int(tem.sum()),
                                        train_rate=train_rate, test_rate=float(yte.mean()),
                                        **m))
                    for season, mask in [("all", np.ones(len(yte), bool)),
                                         ("peak", peak_all[tem]),
                                         ("off", ~peak_all[tem])]:
                        if mask.sum() < 10:
                            continue
                        aa = _top_decile(yte[mask], p[mask])
                        al_rows.append(dict(origin=origin, variant=args.variant, kind="largemove",
                                            H=H, definition=nm, model=alg, features=fe,
                                            season=season, n_test=int(mask.sum()),
                                            test_rate=float(yte[mask].mean()), **aa))
                    for i, gidx in enumerate(np.where(tem)[0]):
                        oos.append(dict(origin=origin, variant=args.variant, kind="largemove", H=H,
                                        definition=nm, model=alg, features=fe,
                                        date=date_all[gidx], route=d[ROUTE_COL].iloc[gidx],
                                        peak=bool(peak_all[gidx]), y=int(yte[i]), p=float(p[i])))
            model_preds[("largemove", H, nm)] = (train_rate, yte, tem, store)

        # alerting for the direction task too (top-decile budget on sign calls)
        for H in horizons:
            key = ("direction", H, "sign")
            if key not in model_preds:
                continue
            train_rate, yte, tem, store = model_preds[key]
            for (fe, alg), p in store.items():
                for season, mask in [("all", np.ones(len(yte), bool)),
                                     ("peak", peak_all[tem]),
                                     ("off", ~peak_all[tem])]:
                    if mask.sum() < 10:
                        continue
                    aa = _top_decile(yte[mask], p[mask])
                    al_rows.append(dict(origin=origin, variant=args.variant, kind="direction",
                                        H=H, definition="sign", model=alg, features=fe,
                                        season=season, n_test=int(mask.sum()),
                                        test_rate=float(yte[mask].mean()), **aa))

        if (fi + 1) % 5 == 0:
            log(f"  fold {fi+1}/{len(folds)} origin {origin.date()} "
                f"n_tr={f['n_train']} n_te={f['n_test']} elapsed {time.time()-t0:.0f}s")

    dir_df = pd.DataFrame(dir_rows)
    mv_df = pd.DataFrame(mv_rows)
    al_df = pd.DataFrame(al_rows)
    oos_df = pd.DataFrame(oos)

    # ---------------- pooled metrics ------------------------------------
    pooled = []
    if len(oos_df):
        for keys, grp in oos_df.groupby(["kind", "H", "definition", "model", "features", "variant"]):
            tr_rate = np.nan
            if keys[0] == "direction":
                sub = dir_df[(dir_df.H == keys[1]) & (dir_df.model == keys[3])
                             & (dir_df.features == keys[4])]
            else:
                sub = mv_df[(mv_df.H == keys[1]) & (mv_df.definition == keys[2])
                            & (mv_df.model == keys[3]) & (mv_df.features == keys[4])]
            if len(sub):
                tr_rate = float(sub["train_rate"].mean())
            m = _pooled_metrics(grp["y"].to_numpy(), grp["p"].to_numpy(), tr_rate)
            pooled.append(dict(kind=keys[0], H=keys[1], definition=keys[2], model=keys[3],
                               features=keys[4], variant=keys[5], **m))
        # seasonal pooled
        for peak_val in [True, False]:
            sub = oos_df[oos_df.peak == peak_val]
            for keys, grp in sub.groupby(["kind", "H", "definition", "model", "features", "variant"]):
                m = _pooled_metrics(grp["y"].to_numpy(), grp["p"].to_numpy(), np.nan)
                pooled.append(dict(kind=keys[0], H=keys[1], definition=keys[2], model=keys[3],
                                   features=keys[4], variant=keys[5],
                                   season="peak" if peak_val else "off", **m))
    pooled_df = pd.DataFrame(pooled)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dir_df.to_csv(out, index=False)
    mv_df.to_csv(out.replace(".csv", "_largemove.csv"), index=False)
    al_df.to_csv(out.replace(".csv", "_alerting.csv"), index=False)
    oos_df.to_csv(out.replace(".csv", "_oos_predictions.csv"), index=False)
    pooled_df.to_csv(out.replace(".csv", "_pooled.csv"), index=False)
    mir.to_csv(out.replace(".csv", "_mirror_check.csv"), index=False)
    if skipped:
        pd.DataFrame(skipped).to_csv(out.replace(".csv", "_skipped.csv"), index=False)

    log(f"\n[done] variant={args.variant} {time.time()-t0:.0f}s")
    log(f"  direction rows   : {len(dir_df)}")
    log(f"  large-move rows  : {len(mv_df)}")
    log(f"  alerting rows    : {len(al_df)}")
    log(f"  pooled rows      : {len(pooled_df)}")
    if len(dir_df):
        g = (dir_df.groupby(["H", "features", "model"])
             .agg(folds=("AUC", "size"), Acc=("Acc", "mean"), Maj=("MajBase", "mean"),
                  BalAcc=("BalAcc", "mean"), AUC=("AUC", "mean"),
                  BrierSkill=("BrierSkill", "mean"),
                  pos_BS=("BrierSkill", lambda s: float((s > 0).mean())))
             .reset_index())
        log("\nDIRECTION (fold-mean):")
        log(g.to_string(index=False))
    if len(pooled_df):
        log("\nPOOLED per-obs metrics (head):")
        log(pooled_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
