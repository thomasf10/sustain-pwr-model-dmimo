"""Small plotting helpers for the GUI.

Sweep figures reuse the manuscript figures in :mod:`fr3_power.plotting` so the
GUI and the CLI scripts render identically. The single-operating-point breakdown
is specific to the interactive view and lives here.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from fr3_power import PowerBreakdown


def breakdown_bar(b: PowerBreakdown):
    """Stacked bar of one operating point: digital / analog / PA.

    Each bar stacks the load-independent part (opaque) over the load-dependent
    part (transparent), matching the convention of the manuscript figures.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    comps = [("Digital", "tab:green", b.digital),
             ("Analog", "tab:orange", b.analog),
             ("PA", "tab:red", b.pa)]
    for i, (_name, color, split) in enumerate(comps):
        ax.bar(i, split.load_ind, color=mcolors.to_rgba(color, 0.9),
               edgecolor="black")
        ax.bar(i, split.load_dep, bottom=split.load_ind,
               color=mcolors.to_rgba(color, 0.4), edgecolor="black")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([c[0] for c in comps])
    ax.set_ylabel("Power consumption [W]")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig
