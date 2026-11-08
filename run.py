"""Reproduce the paper's main results.

    python run.py offline              # Figure 3 (a, c): offline loss by budget, paired differences
    python run.py online               # Figure 3 (b, d), Figure 4 and Table 1: the online stream
    python run.py all                  # both

Add --workload taskmaster3 or --workload socrateach to run one workload. Results are written to
results/<workload>/; `python plot.py` draws the figures from them.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data import WORKLOADS, load
from models import CostModel, PairFeatures, RiskModel

RESULTS = Path(__file__).resolve().parent / "results"
A = "CUCB-SC-MT"


def setup(w, log):
    t0 = time.time()
    cost = CostModel(w)
    log(f"  cost model: bandwidth ×{cost.scale:g}, ridge {cost.ridge:g}, β_c = {cost.beta:.3f}")
    risk = RiskModel(w, PairFeatures(w))
    log(f"  risk model: bandwidth ×{risk.scale:g}, ridge {risk.ridge:g}, β_r = {risk.beta:.3f} "
        f"({len(w.risk_pairs):,} labeled pairs, {time.time() - t0:.0f}s)")
    return cost, risk


def paired_differences(table, losses, boot, method=A):
    """Δ = method − baseline (pp of the no-cache cost) at matched budgets, and with no budget against
    every unbounded baseline and against the method's own k = 1000."""
    rows = []
    budgets = [b for b in table.budget.unique() if b != "all"]
    pairs = [((method, b), (m, b)) for b in budgets for m in table.method.unique() if m != method]
    pairs += [((method, "all"), (m, "all")) for m in table.method.unique() if m != method]
    pairs += [((method, "all"), (method, max(budgets, key=int)))]
    for a, b in pairs:
        if a in losses and b in losses:
            d, lo, hi = boot(losses[a] - losses[b])
            rows.append({"budget": a[1], "versus": b[0], "versus_budget": b[1], "delta": d, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def run_offline(name, log=print):
    import offline
    w = load(name)
    log(f"\n== {w.label}: offline ({len(w.cal):,} calibration, {len(w.val):,} validation, {len(w.test):,} test states)")
    cost, risk = setup(w, log)
    table, losses = offline.run(w, cost, risk, log=log)
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "offline_loss.csv", index=False)
    np.savez_compressed(out / "offline_test_losses.npz",
                        **{f"{m}|{b}": v for (m, b), v in losses.items()}, test_states=w.state_ids[w.test])
    paired = paired_differences(table, losses, offline.Bootstrap(w))
    paired.to_csv(out / "offline_paired_differences.csv", index=False)
    log(f"  wrote {out / 'offline_loss.csv'}")


def run_online(name, log=print):
    import online
    w = load(name)
    log(f"\n== {w.label}: online ({len(np.unique(w.conv_ids)):,} conversations, {w.n:,} requests)")
    cost, risk = setup(w, log)
    out = RESULTS / name
    if not (out / "offline_loss.csv").exists():
        raise SystemExit(f"{out / 'offline_loss.csv'} is missing: the online methods use the offline selections, "
                         f"so run `python run.py offline --workload {name}` first.")
    online.run(w, cost, risk, out, log=log)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("part", choices=["offline", "online", "all"])
    p.add_argument("--workload", choices=list(WORKLOADS), action="append",
                   help="workload(s) to run (default: both)")
    args = p.parse_args()
    for name in args.workload or list(WORKLOADS):
        if args.part in ("offline", "all"):
            run_offline(name)
        if args.part in ("online", "all"):
            run_online(name)
