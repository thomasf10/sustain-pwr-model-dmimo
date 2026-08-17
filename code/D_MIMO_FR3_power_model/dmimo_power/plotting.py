"""Figures comparing the three MIMO deployments.

Reuses the manuscript's Matplotlib style from
:func:`fr3_power.plotting.setup_style` so these figures sit next to the
co-located ones without restyling.

Both palettes below are the Okabe-Ito colourblind-safe set and were checked with
the six-check validator (lightness band, chroma floor, CVD separation of every
adjacent pair, normal-vision floor, contrast against the surface) rather than
picked by eye. The default Matplotlib green/orange pair fails outright: under
protanopia its separation is dE 0.7, i.e. indistinguishable. Series identity is
additionally carried by marker shape, so it never rests on colour alone.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from fr3_power.plotting import setup_style  # noqa: F401  (re-exported)

from .scenarios import Deployment

#: Fixed hue per deployment, assigned by identity and never cycled. The order
#: below is the order the validator was run on.
DEPLOYMENT_COLOR = {
    Deployment.COLOCATED: "#0072B2",      # blue
    Deployment.CENTRALIZED_S1: "#CC79A7",  # reddish purple
    Deployment.CENTRALIZED_S2: "#D55E00",  # vermillion
    Deployment.DISTRIBUTED_S3: "#009E73",  # bluish green
}

#: Marker shape per deployment: the secondary encoding of identity. The two
#: centralized splits deliver the same rate and differ only in the fronthaul, so
#: they take the two filled quadrilaterals.
DEPLOYMENT_MARKER = {
    Deployment.COLOCATED: "o",
    Deployment.CENTRALIZED_S1: "s",
    Deployment.CENTRALIZED_S2: "D",
    Deployment.DISTRIBUTED_S3: "^",
}

#: Power blocks of the breakdown figure, in stacking order.
BLOCK_COLOR = {
    "AP digital": "#0072B2",
    "AP analog": "#D55E00",
    "AP PA": "#009E73",
    "fronthaul": "#56B4E9",
    "CPU": "#CC79A7",
}


def budget_label(result) -> str:
    """Point label for a transmit-budget sweep."""
    return f"{result.P_budget:g} W"


def ap_count_label(result) -> str:
    """Point label for a sweep over the number of APs."""
    return f"L={result.L}"


def save_network_figures(scenarios, deployments, out_dir, *, show=False,
                         annotate=False):
    """Save an overview of every distinct network geometry being compared.

    Draws the APs, the users, the central unit and the fronthaul links with
    :func:`mimo_helpers.plot_network`, for the *first* Monte Carlo drop of each
    topology: seeding with ``cfg.seed`` reproduces exactly the layout the rate
    simulation starts from, so the picture matches the numbers rather than being
    an unrelated illustration. Because the seed is shared, the users are in the
    same places in every panel, and only the AP layout changes between them.

    Geometries are deduplicated across all the scenarios given. Deployments
    differing only in the functional split share a layout, and the co-located
    baseline is the same single site at every point of an AP-count sweep, so
    each picture is drawn exactly once.

    Args:
        scenarios: One :class:`~dmimo_power.scenarios.Scenario`, or an iterable
            of them (e.g. one per AP count of a sweep).
        deployments: Iterable of :class:`Deployment` to cover.
        out_dir: Directory to write ``network_L{L}x{M}.png`` into.
        show: Open a window per figure.
        annotate: Label each AP and UE with its index.

    Returns:
        List of the written paths, in drawing order.
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    import mimo_helpers as mh
    from .scenarios import Scenario, topologies

    if isinstance(scenarios, Scenario):
        scenarios = [scenarios]

    geometries = {}
    for scenario in scenarios:
        geometries.update(topologies(scenario, deployments))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for (L, M, _placement), cfg in sorted(geometries.items()):
        # Same seed as the Monte Carlo, so this is drop 0 of that very run.
        ap_pos, ue_pos = mh.draw_positions(cfg, np.random.default_rng(cfg.seed))
        path = out_dir / f"network_L{L}x{M}.png"
        ax = mh.plot_network(cfg, ap_pos, ue_pos, annotate=annotate, show=False)
        kind = "co-located" if L == 1 else "distributed"
        ax.set_title(f"{kind}: $L={L}$ AP{'s' if L > 1 else ''} "
                     rf"$\times$ $M={M}$ antennas, $K={cfg.K}$ users")
        ax.figure.savefig(path, dpi=150, bbox_inches="tight")
        written.append(path)
        if show:
            plt.show()
        else:
            plt.close(ax.figure)
    return written


def plot_rate_vs_power(results_by_deployment, *, title="", point_label=budget_label,
                       ax=None):
    """Delivered sum rate against network power, one connected series per deployment.

    The same axes serve both sweeps, which is what makes them comparable: only
    the parameter moving along each curve changes. A series sitting up and to
    the left dominates one sitting down and to the right. Points are connected
    because consecutive markers are the same deployment at neighbouring sweep
    values, not independent samples.

    Args:
        results_by_deployment: Mapping ``Deployment -> sequence of
            OperatingResult``, ordered along the sweep.
        title: Axes title.
        point_label: Callable turning a result into an endpoint label, or
            ``None`` to omit labels. Only the first and last marker of each
            series are labelled; a number on every point would bury the curves.
        ax: Optional axes to draw into.

    Returns:
        ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.4, 5.2))
    else:
        fig = ax.figure

    for deployment, results in results_by_deployment.items():
        if not results:
            continue
        power = np.array([r.power for r in results])
        rate = np.array([r.R_total * 1e-9 for r in results])
        color = DEPLOYMENT_COLOR[deployment]
        marker = DEPLOYMENT_MARKER[deployment]

        if len(results) == 1:
            # A single reference point (e.g. co-located in an AP-count sweep):
            # an open marker, so it does not read as a truncated curve.
            ax.plot(power, rate, marker, color=color, ms=11, mfc="white",
                    mew=2.2, label=deployment.label, zorder=4)
        else:
            ax.plot(power, rate, "-", color=color, lw=2, zorder=2)
            ax.plot(power, rate, marker, color=color, ms=8, mec="white",
                    mew=1.2, label=deployment.label, zorder=3)

        if point_label is not None:
            indices = (0,) if len(results) == 1 else (0, -1)
            for idx, dy in zip(indices, ((6, -13), (6, 5))):
                ax.annotate(point_label(results[idx]), (power[idx], rate[idx]),
                            textcoords="offset points", xytext=dy,
                            fontsize=9, color="0.35")

    ax.set_xlabel(r"Network power consumption, $P_\mathrm{net}$ [W]")
    ax.set_ylabel(r"Delivered sum rate, $R_\mathrm{DL}+R_\mathrm{UL}$ [Gbit/s]")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="best", framealpha=0.92, fontsize=10)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_power_breakdown(results: Sequence, *, title="", ax=None):
    """Stacked power breakdown of one operating point per deployment.

    Shows where the consumption actually goes, which is what explains the
    positions in :func:`plot_rate_vs_power`. Segments carry a thin surface gap
    so adjacent fills never touch, and a legend covering every block is always
    present: in-segment labels are dropped when a segment is too short to hold
    text without colliding with its neighbour, so identity must not depend on
    them. Label ink is chosen for contrast against its own fill rather than
    taking the series colour.

    Args:
        results: One ``OperatingResult`` per deployment, at a common budget.
        title: Axes title.
        ax: Optional axes to draw into.

    Returns:
        ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.6, 5.4))
    else:
        fig = ax.figure

    labels = [r.deployment.label.replace(", ", ",\n") for r in results]
    x = np.arange(len(results))
    # Headroom for the total annotation and the legend above the tallest bar.
    y_max = max(r.power for r in results) * 1.30
    name_floor = 0.10 * y_max   # tall enough for "name\nvalue"
    value_floor = 0.045 * y_max  # tall enough for the value alone

    for i, r in enumerate(results):
        bottom = 0.0
        for name, value in _blocks_of(r).items():
            if value <= 0:
                continue
            ax.bar(x[i], value, bottom=bottom, width=0.6,
                   color=BLOCK_COLOR[name], edgecolor="white", linewidth=2)
            text = (f"{name}\n{value:.0f} W" if value >= name_floor
                    else (f"{value:.0f} W" if value >= value_floor else None))
            if text:
                ax.text(x[i], bottom + value / 2, text, ha="center", va="center",
                        fontsize=8.5, color=_ink_for(BLOCK_COLOR[name]))
            bottom += value
        ax.text(x[i], bottom + 0.015 * y_max, f"{r.power:.0f} W", ha="center",
                va="bottom", fontsize=11, color="0.2")

    # One proxy handle per block, independent of which bars actually show it:
    # the co-located site has no fronthaul and no CPU, and the legend must still
    # say what those colours mean.
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
               for c in BLOCK_COLOR.values()]
    ax.legend(handles, list(BLOCK_COLOR), loc="upper left", ncol=3,
              fontsize=9.5, framealpha=0.92)

    ax.set_xticks(x, labels)
    ax.set_ylim(0, y_max)
    ax.set_ylabel(r"Power consumption, $P_\mathrm{net}$ [W]")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def _ink_for(fill: str) -> str:
    """Readable text colour on a given fill: white on the dark steps, ink on light."""
    return "white" if fill in ("#0072B2", "#D55E00", "#009E73") else "#1a1a1a"


def _blocks_of(result) -> dict:
    """Per-block consumption [W] of one result, co-located or distributed."""
    b = result.breakdown
    if hasattr(b, "fronthaul"):        # NetworkBreakdown
        return {"AP digital": b.ap_digital.total, "AP analog": b.ap_analog.total,
                "AP PA": b.ap_pa.total, "fronthaul": b.fronthaul.total,
                "CPU": b.cpu.total}
    # Co-located PowerBreakdown: no fronthaul and no central unit at all.
    return {"AP digital": b.digital.total, "AP analog": b.analog.total,
            "AP PA": b.pa.total, "fronthaul": 0.0, "CPU": 0.0}
