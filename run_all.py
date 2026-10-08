"""
================================================================================
run_all.py - one-command reproduction of the whole D:\\testing project
================================================================================
Runs, in dependency order:

  1. volatility_large_move.py   the main pipeline (targets, baselines, both tasks)
  2. agents/agentA_*.py         HAR-RV / GARCH family + Diebold-Mariano tests
  3. agents/agentB_*.py         direction, multi-horizon large moves, alerting
  4. agents/agentC_*.py         target audit, leakage audit, permutation nulls
  5. agents/agentD_*.py         weather data quality + event study

Each script is discovered by glob and run with the same interpreter. Failures in
one agent do not abort the rest; a summary is printed at the end.

Usage:
    python D:\\testing\\run_all.py            # everything
    python D:\\testing\\run_all.py --main     # only the main pipeline
    python D:\\testing\\run_all.py --agents   # only the agents
================================================================================
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

PROJECT = os.path.dirname(os.path.abspath(__file__))
AGENTS_DIR = os.path.join(PROJECT, "agents")
OUT_DIR = os.path.join(PROJECT, "outputs")


def _scripts(pattern_root: str) -> list[str]:
    found = sorted(glob.glob(os.path.join(pattern_root, "*.py")))
    return [f for f in found if not os.path.basename(f).startswith("_")]


def _run(path: str, label: str) -> dict:
    print("\n" + "=" * 100)
    print(f"RUNNING {label}: {os.path.basename(path)}")
    print("=" * 100, flush=True)
    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        proc = subprocess.run([sys.executable, path], cwd=PROJECT, env=env)
        rc = proc.returncode
    except Exception as exc:  # pragma: no cover
        print(f"  FAILED to launch: {exc}")
        rc = -1
    dt = time.time() - t0
    status = "OK" if rc == 0 else f"FAILED (rc={rc})"
    print(f"  -> {status} in {dt:.0f}s", flush=True)
    return {"script": os.path.basename(path), "label": label, "rc": rc,
            "seconds": round(dt, 1), "status": status}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", action="store_true", help="run only the main pipeline")
    ap.add_argument("--agents", action="store_true", help="run only the agent scripts")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

    results: list[dict] = []
    t_start = time.time()

    run_main = (not args.agents) or args.main
    run_agents = (not args.main) or args.agents

    if run_main:
        main_script = os.path.join(PROJECT, "volatility_large_move.py")
        if os.path.exists(main_script):
            results.append(_run(main_script, "MAIN PIPELINE"))
        else:
            print(f"!! main pipeline not found at {main_script}")

    if run_agents:
        for path in _scripts(AGENTS_DIR):
            results.append(_run(path, "AGENT"))

    print("\n" + "=" * 100)
    print("RUN SUMMARY")
    print("=" * 100)
    print(f"{'script':52s} {'status':16s} {'seconds':>9s}")
    for r in results:
        print(f"{r['script']:52s} {r['status']:16s} {r['seconds']:9.1f}")
    failed = [r for r in results if r["rc"] != 0]
    print(f"\n{len(results)} scripts, {len(failed)} failed, "
          f"total {time.time() - t_start:.0f}s")
    print(f"Outputs: {OUT_DIR}")

    if failed:
        print("\nFailed scripts:")
        for r in failed:
            print(f"  {r['script']} (rc={r['rc']})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
