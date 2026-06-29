"""Stacked-bar figures for power consumption and ergodic rates.

The functions take already-computed result arrays and produce the figures of
the manuscript. LaTeX rendering is optional (``use_tex``); when off, math-text
is used so the scripts run without a LaTeX installation.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .frame_average import LoadSplit


def setup_style(use_tex: bool = False, font_size: int = 15) -> None:
    """Apply the common Matplotlib style used by all figures."""
    plt.rcParams.update({
        "font.size": font_size,
        "text.usetex": use_tex,
        "font.family": "serif",
        "xtick.direction": "in",
        "ytick.direction": "in",
    })


def _bars_stacked(ax, x, splits: Sequence[LoadSplit], color, hatch=None):
    """Draw load-independent (opaque) over load-dependent (transparent) bars."""
    load_ind = [s.load_ind for s in splits]
    load_dep = [s.load_dep for s in splits]
    c_ind = mcolors.to_rgba(color, alpha=0.9)
    c_dep = mcolors.to_rgba(color, alpha=0.4)
    p_ind = ax.bar(x, load_ind, edgecolor="black", color=c_ind, hatch=hatch)
    p_dep = ax.bar(x, load_dep, bottom=load_ind, edgecolor="black",
                   color=c_dep, hatch=hatch)
    return p_ind, p_dep


def plot_digital_analog(sweep_values, digital: Sequence[LoadSplit],
                        analog: Sequence[LoadSplit], *, title="", ylim=(0, 720),
                        xlabel=r"Number of RF chains, $M_\mathrm{RF}$"):
    """Fig 2a: digital and analog consumption, load-ind/dep stacked, side by side."""
    fig, ax = plt.subplots()
    for i in range(len(sweep_values)):
        x = [i * 3 - 0.4, i * 3 + 0.4]
        di, dd = digital[i], analog[i]
        c_ind = [mcolors.to_rgba("tab:green", alpha=0.9),
                 mcolors.to_rgba("tab:orange", alpha=0.9)]
        c_dep = [mcolors.to_rgba("tab:green", alpha=0.4),
                 mcolors.to_rgba("tab:orange", alpha=0.4)]
        p1 = ax.bar(x, [di.load_ind, dd.load_ind], edgecolor="black",
                    color=c_ind, hatch=[None, "///"])
        p2 = ax.bar(x, [di.load_dep, dd.load_dep], bottom=[di.load_ind, dd.load_ind],
                    edgecolor="black", color=c_dep, hatch=[None, "///"])
    ax.set_ylabel(r"Power consumption, $P_\mathrm{cons}$ [W]")
    ax.set_ylim(ylim)
    ax.set_xticks([i * 3 for i in range(len(sweep_values))], sweep_values)
    ax.set_xlabel(xlabel)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend((p1[0], p2[0], p1[1], p2[1]),
              ("Digital (load-indep.)", "Digital (load-dep.)",
               "Analog (load-indep.)", "Analog (load-dep.)"),
              loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_pa(sweep_values, pa: Sequence[LoadSplit], *, title="", ylim=(0, 720),
            xlabel=r"Number of antennas, $M_\mathrm{ant}$"):
    """Fig 2b: power-amplifier consumption, load-ind/dep stacked."""
    fig, ax = plt.subplots()
    x = [i * 3 for i in range(len(sweep_values))]
    p1, p2 = _bars_stacked(ax, x, pa, "tab:red")
    ax.set_ylabel(r"Power consumption, $P_\mathrm{cons}$ [W]")
    ax.set_ylim(ylim)
    ax.set_xticks(x, sweep_values)
    ax.set_xlabel(xlabel)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend((p1[0], p2[0]),
              ("Power amplifier (load-indep.)", "Power amplifier (load-dep.)"),
              loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_rates(sweep_values, R_DL, R_UL, *, title="",
               xlabel=r"Number of RF chains, $M_\mathrm{RF}$"):
    """Fig 2c: ergodic DL/UL sum rates [Gbit/s]."""
    fig, ax = plt.subplots()
    for i in range(len(sweep_values)):
        x = [i * 3 - 0.4, i * 3 + 0.4]
        y = [R_UL[i] * 1e-9, R_DL[i] * 1e-9]
        colors = [mcolors.to_rgba("tab:cyan", alpha=0.7),
                  mcolors.to_rgba("tab:blue", alpha=0.7)]
        p1 = ax.bar(x, y, edgecolor="black", color=colors, hatch=[None, "///"])
    ax.set_ylabel(r"Ergodic sum rate, $R_i$ [Gbit/s]")
    ax.set_xticks([i * 3 for i in range(len(sweep_values))], sweep_values)
    ax.set_xlabel(xlabel)
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend((p1[0], p1[1]),
              (r"Uplink, $i=\mathrm{UL}$", r"Downlink, $i=\mathrm{DL}$"),
              loc="upper left")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax
