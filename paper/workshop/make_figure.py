"""Figure 1 of the workshop paper: which friction dominates, by cost level.

Reads ``paper/findings_sweep.json`` (written by
``python -m experiments.findings misspecification_sweep --json ...``) and plots, for each
of the 12 contracts, the ratio

    |effect of a +10% volatility error on CVaR-95| / |effect of proportional cost c|

at c = 1, 5, 25, 50 bp. Above 1 the volatility error hurts more; below 1 the cost does.
No number is computed here that is not a ratio of two entries of that file.

Run from the repository root:

    python paper/workshop/make_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "findings_sweep.json"
TARGET = HERE / "fig_friction_ratio.pdf"

COSTS = ("0.0001", "0.0005", "0.0025", "0.005")
COST_BP = (1, 5, 25, 50)

# Categorical slots 1 and 2 of the reference palette; validated on a white surface.
STYLE = {
    "weekly": {"color": "#2a78d6", "marker": "o", "label": "weekly rebalancing"},
    "daily": {"color": "#eb6834", "marker": "^", "label": "daily rebalancing"},
}
INK, MUTED, HAIRLINE = "#0b0b0b", "#52514e", "#c3c2b7"


def main() -> None:
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))["misspecification_sweep"]["rows"]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.edgecolor": HAIRLINE,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,  # embed TrueType, not Type 3
    })
    fig, ax = plt.subplots(figsize=(4.6, 1.9))

    for row in rows:
        vol = abs(row["vol_effect"]["1.1"][0])
        ratios = [vol / abs(row["cost_effect"][c][0]) for c in COSTS]
        s = STYLE[row["rebalancing"]]
        ax.plot(COST_BP, ratios, color=s["color"], lw=1.4, alpha=0.9,
                marker=s["marker"], ms=4.5, mec="white", mew=0.8, zorder=3)

    ax.axhline(1.0, color=MUTED, lw=0.8, zorder=2)
    ax.text(0.9, 1.1, "equal effect", color=MUTED, ha="left", va="bottom", fontsize=7.5)
    ax.text(0.9, 0.9, "cost hurts more below", color=MUTED, ha="left", va="top", fontsize=7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(COST_BP, [f"{c} bp" for c in COST_BP])
    ax.minorticks_off()
    ax.set_yticks([0.3, 1, 3, 10, 30], ["0.3", "1", "3", "10", "30"])
    ax.set_xlim(0.8, 60)
    ax.set_ylim(0.2, 50)
    ax.set_xlabel("proportional transaction cost $c$")
    ax.set_ylabel("|vol-error effect|\n/ |cost effect|", linespacing=1.1)
    ax.grid(True, which="major", axis="y", color="#e1e0d9", lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Line2D([], [], color=s["color"], marker=s["marker"], lw=1.4, ms=4.5,
                   mec="white", mew=0.8, label=s["label"])
        for s in STYLE.values()
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=7.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(TARGET)
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
