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
    "AP sync": "#E69F00",
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


def plot_power_vs_aps(results_by_deployment, *, title="", ax=None,
                      colocated_result=None):
    """Network power against the number of APs, at a fixed transmit budget.

    The second evaluation. The same $L M$ antennas are redistributed over a
    growing number of APs, so the x axis is how finely the array is chopped and
    the y axis is what that chopping costs. The co-located baseline is the
    $L = 1$ point: it is a different model rather than the limit of the
    distributed one, since a one-AP network would still pay a fronthaul link
    and a central unit, so it is drawn once as an open marker that the three
    split curves start from.

    Args:
        results_by_deployment: Mapping ``Deployment -> sequence of
            OperatingResult`` ordered by AP count, for the distributed splits.
        title: Axes title.
        ax: Optional axes to draw into.
        colocated_result: The ``L = 1`` reference point, drawn as an open marker.

    Returns:
        ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
    else:
        fig = ax.figure

    for deployment, results in results_by_deployment.items():
        if not results:
            continue
        L = np.array([r.L for r in results])
        power = np.array([r.power for r in results])
        color = DEPLOYMENT_COLOR[deployment]
        ax.plot(L, power, "-", color=color, lw=2, zorder=2)
        ax.plot(L, power, DEPLOYMENT_MARKER[deployment], color=color, ms=7,
                mec="white", mew=1.1, label=deployment.label, zorder=3)

    if colocated_result is not None:
        ax.plot([1], [colocated_result.power], DEPLOYMENT_MARKER[Deployment.COLOCATED],
                color=DEPLOYMENT_COLOR[Deployment.COLOCATED], ms=11, mfc="white",
                mew=2.2, label=Deployment.COLOCATED.label, zorder=4)

    # Base-2 log axis, since the sweep halves the array at each step, but with
    # the AP counts themselves as tick labels rather than powers of two.
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"Number of APs, $L$   (antennas per AP $M = 128/L$)")
    ax.set_ylabel(r"Network power consumption, $P_\mathrm{net}$ [W]")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="best", framealpha=0.92, fontsize=10)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_ee_vs_aps(results_by_deployment, *, title="", ax=None,
                   colocated_result=None, Pi_FH=None):
    """Energy efficiency against the number of APs, at a fixed transmit budget.

    The fourth evaluation, and the one that decides whether distributing pays:
    the power of :func:`plot_power_vs_aps` cannot answer that on its own,
    because the delivered rate is not constant along the sweep.

    ``Pi_FH`` draws the ceiling of Remark rem:fh_ceiling, ``1 / (Pi_FH L)``. It
    is a *downlink* bound on the data-sharing splits: keeping only the
    duplicated payload ``L R_DL`` in eq. pnet gives ``R_DL / P_net <= 1 /
    (Pi_FH L)`` whatever the bandwidth, the transmit power or the precoder. It
    is drawn as a reference line rather than as a series, and it does not bound
    S1, which forwards samples rather than payload and so duplicates nothing.

    Args:
        results_by_deployment: Mapping ``Deployment -> sequence of
            OperatingResult`` ordered by AP count.
        title: Axes title.
        ax: Optional axes to draw into.
        colocated_result: The single-site reference, drawn as an open marker.
        Pi_FH: Fronthaul traffic coefficient [W per bit/s]; ``None`` omits the
            ceiling.

    Returns:
        ``(fig, ax)``.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
    else:
        fig = ax.figure

    if Pi_FH:
        L_line = np.array([1.0, 200.0])
        ax.plot(L_line, 1.0 / (Pi_FH * L_line) * 1e-6, "--", color="0.45", lw=1.4,
                zorder=1, label=r"ceiling $1/(\Pi_\mathrm{FH} L)$")

    for deployment, results in results_by_deployment.items():
        if not results:
            continue
        L = np.array([r.L for r in results])
        ee = np.array([r.energy_efficiency for r in results]) * 1e-6
        color = DEPLOYMENT_COLOR[deployment]
        ax.plot(L, ee, "-", color=color, lw=2, zorder=2)
        ax.plot(L, ee, DEPLOYMENT_MARKER[deployment], color=color, ms=7,
                mec="white", mew=1.1, label=deployment.label, zorder=3)

    if colocated_result is not None:
        ax.plot([1], [colocated_result.energy_efficiency * 1e-6],
                DEPLOYMENT_MARKER[Deployment.COLOCATED],
                color=DEPLOYMENT_COLOR[Deployment.COLOCATED], ms=11, mfc="white",
                mew=2.2, label=Deployment.COLOCATED.label, zorder=4)

    from matplotlib.ticker import FixedLocator, ScalarFormatter

    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(0.85, 165)
    # Log y: the ceiling is a straight line on it, and the splits span more than
    # a decade over the sweep.
    ax.set_yscale("log")
    ax.set_xlabel(r"Number of APs, $L$   (antennas per AP $M = 128/L$)")
    ax.set_ylabel(r"Energy efficiency, $EE_\mathrm{net}$ [Mbit/J]")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", framealpha=0.92, fontsize=9.5)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_breakdown_vs_aps(results_by_deployment, *, title="", axes=None,
                          colocated_result=None):
    """Stacked per-block consumption over the AP-count sweep, one panel per split.

    The third evaluation, which turns the totals of :func:`plot_power_vs_aps`
    into the mechanism behind them. One panel per split so the stacks stay
    readable; the block order and colours are those of
    :func:`plot_power_breakdown`, so the two figures can be read together.

    Args:
        results_by_deployment: Mapping ``Deployment -> sequence of
            OperatingResult`` ordered by AP count.
        title: Figure title.
        axes: Optional sequence of axes, one per deployment.
        colocated_result: If given, drawn as the leftmost bar of every panel,
            the single-site deployment the sweep starts from.

    Returns:
        ``(fig, axes)``.
    """
    deployments = [d for d, r in results_by_deployment.items() if r]
    if axes is None:
        # Independent y scales, matching pgf.breakdown_vs_aps. The totals span a
        # factor five across the splits, so a shared scale flattens the cheapest
        # panel until its block structure is unreadable, which is the one thing
        # this figure has to show. Cross-split totals belong to
        # plot_power_vs_aps, which puts them all on one axis.
        fig, axes = plt.subplots(1, len(deployments),
                                 figsize=(4.5 * len(deployments), 4.4), sharey=False)
        axes = [axes] if len(deployments) == 1 else list(axes)
    else:
        axes = list(axes)
        fig = axes[0].figure

    for ax, deployment in zip(axes, deployments):
        results = list(results_by_deployment[deployment])
        labels, points = [], []
        if colocated_result is not None:
            labels.append("1*")
            points.append(colocated_result)
        labels += [str(r.L) for r in results]
        points += results

        x = np.arange(len(points))
        for i, r in enumerate(points):
            bottom = 0.0
            for name, value in _blocks_of(r).items():
                if value <= 0:
                    continue
                ax.bar(x[i], value, bottom=bottom, width=0.72,
                       color=BLOCK_COLOR[name], edgecolor="white", linewidth=0.6)
                bottom += value
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_xlabel(r"Number of APs, $L$")
        ax.set_title(deployment.label)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(r"Power consumption, $P_\mathrm{net}$ [W]")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
               for c in BLOCK_COLOR.values()]
    fig.legend(handles, list(BLOCK_COLOR), loc="lower center",
               ncol=len(BLOCK_COLOR), fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return fig, axes


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
                "AP PA": b.ap_pa.total, "AP sync": b.ap_sync.total,
                "fronthaul": b.fronthaul.total, "CPU": b.cpu.total}
    # Co-located PowerBreakdown: no synchronization, no fronthaul, no CPU.
    return {"AP digital": b.digital.total, "AP analog": b.analog.total,
            "AP PA": b.pa.total, "AP sync": 0.0, "fronthaul": 0.0, "CPU": 0.0}
