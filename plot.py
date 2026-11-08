"""Figures 3 and 4 from the results in results/<workload>/ (run `python run.py all` first).

    python plot.py      # writes results/figure3.pdf and results/figure4.pdf
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS = Path(__file__).resolve().parent / "results"
WORKLOADS = {"taskmaster3": "Taskmaster-3", "socrateach": "SocraTeach"}
METHODS = ["CUCB-SC-MT", "ContextCache", "ElastiCache-MT", "vCache", "Joint-state threshold", "Single-turn"]
COLORS = {"CUCB-SC-MT": "#2a78d6", "CLCB-LS-MT": "#2a78d6", "ContextCache": "#1baf7a", "ElastiCache-MT": "#eb6834",
          "vCache": "#e87ba4", "Joint-state threshold": "#008300", "Single-turn": "#eda100"}
LABELS = {"CUCB-SC-MT": "CUCB-SC-MT / CLCB-LS-MT", "Single-turn": "Single-turn (CUCB-SC-Cont)"}
BUDGETS = ["100", "200", "300", "500", "1000", "all"]
STRONGEST_UNBOUNDED = {"taskmaster3": "ElastiCache-MT", "socrateach": "ContextCache"}   # dotted in Figure 4

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def bars(ax, table, title):
    table = table.assign(budget=table.budget.astype(str))
    width = 0.8 / len(METHODS)
    top = 110.0
    for i, m in enumerate(METHODS):
        name = "CLCB-LS-MT" if m == "CUCB-SC-MT" and "CLCB-LS-MT" in set(table.method) else m
        t = table[table.method == name].set_index("budget").reindex(BUDGETS)
        x = np.arange(len(BUDGETS)) - 0.4 + width * (i + 0.5)
        y = t["loss"].to_numpy(float)
        err = np.vstack([y - t["loss_lo"].to_numpy(float), t["loss_hi"].to_numpy(float) - y])
        ax.bar(x, y - 50, bottom=50, width=width * 0.92, color=COLORS[m], zorder=2)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor="#222222", elinewidth=0.8, capsize=1.5, zorder=3)
        for xi, yi, hi in zip(x, y, t["loss_hi"].to_numpy(float)):
            if np.isfinite(yi):
                ax.text(xi, max(yi, hi) + 1, f"{yi:.1f}", rotation=90, ha="center", va="bottom", fontsize=6.5)
                top = max(top, hi + 10)
    ax.axhline(100, color="#555555", ls="--", lw=0.9, zorder=1)
    ax.set_xticks(np.arange(len(BUDGETS)), [b if b != "all" else "No\nbudget" for b in BUDGETS])
    ax.set_ylim(50, top)
    ax.set_ylabel("Loss (% of no-cache cost)")
    ax.set_xlabel("Cache budget k")
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    ax.set_title(title, y=-0.32, fontsize=10)


def figure3():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    letters = iter("abcd")
    for row, (name, label) in enumerate(WORKLOADS.items()):
        for col, part in enumerate(["offline", "online"]):
            ax = axes[row, col]
            path = RESULTS / name / f"{part}_loss.csv"
            if path.exists():
                bars(ax, pd.read_csv(path, dtype={"budget": str}), f"({next(letters)}) {label} {part}")
            else:
                next(letters)
                ax.set_axis_off()
    handles = [Patch(color=COLORS[m], label=LABELS.get(m, m)) for m in METHODS]
    handles.append(Line2D([], [], color="#555555", ls="--", label="No cache"))
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3)
    fig.savefig(RESULTS / "figure3.pdf", bbox_inches="tight")
    plt.close(fig)


def running(loss, cost, t0):
    r = 100 * np.cumsum(loss) / np.cumsum(cost)
    t = np.arange(1, len(loss) + 1)
    return t[t0:], r[t0:]


def figure4():
    online = ["CLCB-LS-MT", "ContextCache", "ElastiCache-MT", "vCache", "Joint-state threshold", "Single-turn"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    letters = iter("abcd")
    for row, (name, label) in enumerate(WORKLOADS.items()):
        path = RESULTS / name / "online_losses.npz"
        for col, k in enumerate(["200", "1000"]):
            ax = axes[row, col]
            title = f"({next(letters)}) {label} online, k = {k}"
            if not path.exists():
                ax.set_axis_off()
                continue
            z = np.load(path)
            cost = z["cost"]
            t0 = int(min(max(50, len(cost) // 100), len(cost) - 1))
            for m in online:
                ax.plot(*running(z[f"{m}|{k}"], cost, t0), color=COLORS[m], lw=1.6)
            for m in ["CLCB-LS-MT", STRONGEST_UNBOUNDED[name]]:
                ax.plot(*running(z[f"{m}|all"], cost, t0), color=COLORS[m], lw=1.6, ls=":")
            ax.axhline(100, color="#555555", ls="--", lw=0.9)
            ax.set_xlabel("Requests served")
            ax.set_ylabel("Running loss\n(% of no-cache cost so far)")
            ax.grid(color="#e6e6e6")
            ax.set_title(title, y=-0.3, fontsize=10)
    handles = [Line2D([], [], color=COLORS[m], lw=2, label=LABELS.get(m, m) if m != "Single-turn" else "Single-turn")
               for m in online]
    handles += [Line2D([], [], color=COLORS[m], lw=2, ls=":", label=f"{m}, no budget")
                for m in ["CLCB-LS-MT", "ElastiCache-MT", "ContextCache"]]
    handles.append(Line2D([], [], color="#555555", ls="--", label="No cache"))
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=3)
    fig.savefig(RESULTS / "figure4.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure3()
    figure4()
    print(f"Wrote {RESULTS / 'figure3.pdf'} and {RESULTS / 'figure4.pdf'}")
