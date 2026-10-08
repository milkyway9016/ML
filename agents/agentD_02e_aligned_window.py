"""
agentD step 2e - ALIGNED EVENT WINDOW (t-5 .. t+15).

FIX: the first attempt shifted the event flag and compared it with the
UNSHIFTED target, which is why every window came out NaN. The correct
construction, given fwd_rv5(t) is the volatility of t+1..t+5, is:

  row t, offset k :  event status = I(event at t)
                     outcome      = fwd_rv5 at date t+k   (i.e. the forward
                                    volatility whose window starts at t+k)

so offset 0 reads "forward volatility starting the day after the shock", and
offsets -5..-1 are a clean PRE-TREND check (outcomes realised before/around the
shock). Windows are built on the de-duplicated lane sample.

Outputs: agentD_event_windows.csv (overwrites), plots/agentD_event_study.png
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import importlib.util
spec = importlib.util.spec_from_file_location("rg", r"D:\testing\agents\agentD_03_regressions.py")
rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)

OUT = r"D:\testing\outputs"
PLOT = os.path.join(OUT, "plots"); os.makedirs(PLOT, exist_ok=True)

KEY = ["cyc_active", "cyc_wind_ge64kt", "slp_anom_lt_p10", "wind_gt_p90"]


def main():
    d = pd.read_csv(os.path.join(OUT, "agentD_panel_events.csv"), parse_dates=["Date"])
    d["lane"] = d["Route"].map(rg.LANE_KEEP)
    d = (d.sort_values(["lane", "Date"]).drop_duplicates(["lane", "Date"])
           .reset_index(drop=True))

    rows = []
    for ev in KEY:
        col = "ev_" + ev
        for lane, g in d.groupby("lane"):
            g = g.sort_values("Date")
            # future outcomes indexed by the DATE each forward window starts on
            fut = g.set_index("Date")[["fwd_rv5", "fwd_abs_mean5"]]
            for k in range(-5, 16):
                target_dates = g["Date"] + pd.tseries.offsets.BDay(k)
                mapped = fut.reindex(pd.DatetimeIndex(target_dates))
                tmp = pd.DataFrame({
                    "ev": g[col].to_numpy(),
                    "rv": mapped["fwd_rv5"].to_numpy(),
                    "ar": mapped["fwd_abs_mean5"].to_numpy()})
                for lab, c in [("fwd_rv5", "rv"), ("fwd_abs_ret", "ar")]:
                    s = tmp.dropna(subset=["ev", c])
                    e = s[s.ev == 1][c]
                    n = s[s.ev == 0][c]
                    if len(e) < 10 or len(n) < 10:
                        continue
                    rows.append(dict(event=ev, lane=lane, target=lab, offset=k,
                                     n_event=len(e), mean_event=float(e.mean()),
                                     mean_nonevent=float(n.mean()),
                                     diff=float(e.mean() - n.mean())))
    win = pd.DataFrame(rows)
    agg = (win.groupby(["event", "target", "offset"])
           .apply(lambda x: pd.Series({
                   "n_event": int(x["n_event"].sum()),
                   "mean_event": np.average(x["mean_event"], weights=x["n_event"]),
                   "mean_nonevent": np.average(x["mean_nonevent"], weights=x["n_event"]),
                   "diff": np.average(x["diff"], weights=x["n_event"])}))
           .reset_index())
    agg.to_csv(os.path.join(OUT, "agentD_event_windows.csv"), index=False)

    pd.set_option("display.width", 200)
    print("=" * 110)
    print("ALIGNED EVENT WINDOW - mean forward RV5 by offset relative to event day")
    print("=" * 110)
    for ev in KEY:
        s = agg[(agg.event == ev) & (agg.target == "fwd_rv5")].sort_values("offset")
        print(f"\n--- {ev}  (n_event at offset 0 = "
              f"{int(s[s.offset==0].n_event.iloc[0]) if len(s[s.offset==0]) else 0})")
        print(s[["offset", "mean_event", "mean_nonevent", "diff"]].round(4).to_string(index=False))

    pretrend = []
    for ev in KEY:
        s = agg[(agg.event == ev) & (agg.target == "fwd_rv5")]
        pre = s[(s.offset >= -5) & (s.offset <= -1)]["diff"].mean()
        post = s[(s.offset >= 1) & (s.offset <= 15)]["diff"].mean()
        zero = s[s.offset == 0]["diff"].mean()
        pretrend.append(dict(event=ev, pre_trend_mean_diff=round(pre, 5),
                             offset0_diff=round(zero, 5), post_mean_diff=round(post, 5)))
    pt = pd.DataFrame(pretrend)
    print("\n--- Pre-trend vs post-event contrast ---")
    print(pt.to_string(index=False))
    pt.to_csv(os.path.join(OUT, "agentD_event_pretrend.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8))
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (lab, tlab) in zip(axes, [("Forward 5-day realised volatility", "fwd_rv5"),
                                      ("Forward 5-day mean |log return|", "fwd_abs_ret")]):
        for i, ev in enumerate(KEY):
            w = agg[(agg.event == ev) & (agg.target == tlab)].sort_values("offset")
            if w.empty:
                continue
            ax.plot(w["offset"], w["mean_event"], "-o", ms=4.5, lw=2.0, color=cmap[i],
                    label=f"{ev}  (n={int(w[w.offset==0].n_event.iloc[0])})")
            ax.plot(w["offset"], w["mean_nonevent"], "--", lw=1.3, color=cmap[i], alpha=0.5)
        ax.axvline(0, color="k", ls=":", lw=1.2)
        ax.set_xlabel("trading days relative to the weather shock (0 = shock day)")
        ax.set_ylabel("annualised volatility" if tlab == "fwd_rv5" else "mean |log return|")
        ax.set_title(f"{lab}\nsolid = event days, dashed = non-event days")
        ax.legend(fontsize=8)
    fig.suptitle("agentD: event study - forward freight-rate volatility around weather shocks. "
                 "Any vertical scatter is already present BEFORE the shock and does not "
                 "discontinuously rise through it", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, "agentD_event_study.png"), dpi=118)
    plt.close(fig)
    print("\n[written] agentD_event_windows.csv, agentD_event_pretrend.csv, "
          "plots/agentD_event_study.png")


if __name__ == "__main__":
    main()
