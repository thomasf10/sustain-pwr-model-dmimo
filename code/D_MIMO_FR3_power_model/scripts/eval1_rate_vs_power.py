"""Evaluation 1 of the manuscript: delivered rate against consumed power.

The locus traced by sweeping the total transmit budget ``P_TX``, reported for
the co-located baseline and the three functional splits, at ``L = 32, 64, 128``
APs, i.e. ``M = 4, 2, 1`` antennas each. Every point holds the total antenna
count at ``L*M = 128``, so distributing never adds hardware, it only spreads it.

Two choices the manuscript leaves open are settled here and stated in the
caption rather than left implicit:

* **The amplifiers are re-sized at every point** (``PASizing.PER_AP_BUDGET``,
  ``P_max = P_TX / (LM)``). Each point is therefore a deployment dimensioned for
  the budget it is given, not one fixed deployment being dimmed. The alternative
  (fix the rating and vary the load ``xbar``) traces a differently shaped curve
  and answers a different question; it is not what this script plots.
* **The co-located point is the baseline model, not the ``L = 1`` limit of the
  distributed one.** A distributed network with one AP would still pay a
  fronthaul link and a central unit; the baseline pays neither, which is why it
  is drawn as an open marker rather than as the left end of a curve.

Both directions carry data, so the axis is the sum rate ``R_DL + R_UL``.

    python scripts/eval1_rate_vs_power.py
    python scripts/eval1_rate_vs_power.py --realizations 40 --Q 128
    python scripts/eval1_rate_vs_power.py --no-cache      # recompute the rates

Writes ``figures/eval1_rate_vs_power.{png,tex}``, copies the pgfplots source
into the manuscript's ``figs/`` directory, and records the run in a manifest.
Rates are cached in ``data/rates_dmimo.json`` keyed on the rate configuration,
so a re-run only recomputes what actually changed.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmimo_power import manifest, pgf, plotting, unsourced_parameters  # noqa: E402
from dmimo_power.scenarios import Deployment, Scenario, evaluate  # noqa: E402

#: Total transmit budgets [W]. Centred on the nominal P_TX = L*M*P_max = 12.8 W
#: of the parameter table, spanning a factor 32 around it so the static floor is
#: visible at the low end and the amplifier term dominates at the high end.
DEFAULT_BUDGETS = (1.6, 3.2, 6.4, 12.8, 25.6, 51.2)

#: AP counts, one panel each, at the fixed total of 128 antennas.
DEFAULT_AP_COUNTS = (32, 64, 128)

OUT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = OUT_DIR / "data" / "runs"
PAPER_FIG_DIR = OUT_DIR.parent.parent / "sustain-dmimo-pwr-model-paper" / "figs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS),
                    help="total transmit power budgets [W]")
    ap.add_argument("--ap-counts", type=int, nargs="+", default=list(DEFAULT_AP_COUNTS),
                    help="AP counts, one panel each (must divide the antenna total)")
    ap.add_argument("--antennas", type=int, default=128,
                    help="total antenna count L*M, held fixed everywhere")
    ap.add_argument("--K", type=int, default=20, help="users")
    ap.add_argument("--Q", type=int, default=64,
                    help="subcarriers evaluated per drop (a Monte Carlo subset "
                         "of the 3333 the band carries)")
    ap.add_argument("--realizations", type=int, default=20,
                    help="Monte Carlo drops per point")
    ap.add_argument("--linear-x", action="store_true",
                    help="linear power axis; the default is logarithmic, since "
                         "the four deployments span a factor 37 in P_net within "
                         "one panel and a linear axis buries the cheap ones")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the cached rates and recompute")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening a window")
    return ap.parse_args()


def panel_title(L: int, M: int) -> str:
    return f"$L={L}$, $M={M}$"


def run(args, base):
    """Evaluate every deployment at every budget, for each AP count."""
    use_cache = not args.no_cache

    panels, all_results = {}, []
    for L in args.ap_counts:
        scenario = base.with_aps(L)
        title = panel_title(L, scenario.M)
        print(f"\n=== {title} ({scenario.M_tot} antennas total) ===")
        columns = {}
        for deployment in Deployment:
            print(f"{deployment.label}:")
            series = [evaluate(scenario, deployment, budget, use_cache=use_cache)
                      for budget in args.budgets]
            columns[deployment] = series
            all_results += series
        panels[title] = columns
    return panels, all_results


def print_table(panels) -> None:
    header = (f"{'topology':>12}  {'deployment':<26}{'P_TX':>7}{'R_DL':>9}{'R_UL':>9}"
              f"{'R_tot':>9}{'P_net':>9}{'EE':>9}")
    print(header)
    print(f"{'':>12}  {'':<26}{'[W]':>7}{'[Gb/s]':>9}{'[Gb/s]':>9}{'[Gb/s]':>9}"
          f"{'[W]':>9}{'[Mb/J]':>9}")
    print("-" * len(header))
    for title, columns in panels.items():
        for deployment, series in columns.items():
            for r in series:
                print(f"{title:>12}  {deployment.label:<26}{r.P_budget:>7.1f}"
                      f"{r.rates.R_DL*1e-9:>9.2f}{r.rates.R_UL*1e-9:>9.2f}"
                      f"{r.R_total*1e-9:>9.2f}{r.power:>9.1f}"
                      f"{r.energy_efficiency*1e-6:>9.1f}")
            print()


def save_png(panels, args) -> Path:
    """One row of axes, one per AP count, on shared limits so they compare.

    Per-point budget labels are omitted: four series of six points each would
    put 48 numbers on one row of axes. The sweep runs from the smallest budget
    at the bottom of a curve to the largest at the top, which the title states,
    and the numbers themselves are in the manifest.
    """
    from matplotlib.ticker import FixedLocator, ScalarFormatter

    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4),
                             sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    for ax, (title, columns) in zip(axes, panels.items()):
        plotting.plot_rate_vs_power(columns, title=title, ax=ax, point_label=None)
        if not args.linear_x:
            # Matplotlib's default log ticks collide on this range; place them by
            # hand on the 1-2-5 decade steps and label them as plain numbers.
            ax.set_xscale("log")
            ticks = [100, 200, 500, 1000, 2000, 5000]
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_minor_locator(FixedLocator([]))
            ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel("")
        ax.get_legend().remove()
    axes[0].set_ylabel(r"Delivered sum rate, $R_\mathrm{DL}+R_\mathrm{UL}$ [Gbit/s]")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{args.antennas} antennas, $K={args.K}$ users; "
                 rf"$P_\mathrm{{TX}}={args.budgets[0]:g}\dots{args.budgets[-1]:g}$ W "
                 "increasing upward along each curve, amplifiers re-sized at "
                 "every point", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))

    path = FIG_DIR / "eval1_rate_vs_power.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def main() -> None:
    args = parse_args()
    for L in args.ap_counts:
        if args.antennas % L:
            raise SystemExit(f"L={L} does not divide the antenna total {args.antennas}")

    # The topology of the first panel, used for the parameter record; the
    # per-point parameters vary with L and are rebuilt inside evaluate().
    base = Scenario(L=args.ap_counts[0], M=args.antennas // args.ap_counts[0],
                    K=args.K, Q=args.Q, n_realizations=args.realizations)

    print(f"Evaluation 1: rate against power, {args.antennas} antennas, "
          f"K={args.K} users, Q={args.Q} subcarriers, "
          f"{args.realizations} drops/point")
    print(f"budgets [W]: {', '.join(f'{b:g}' for b in args.budgets)}")
    print()
    print(unsourced_parameters(
        base.power_params(Deployment.CENTRALIZED_S1, 12.8)))

    panels, all_results = run(args, base)
    print()
    print_table(panels)

    plotting.setup_style(use_tex=False, font_size=12)
    png = save_png(panels, args)
    tex = pgf.rate_vs_power_panels(panels, FIG_DIR / "eval1_rate_vs_power.tex",
                                   xmode="normal" if args.linear_x else "log")

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(tex, PAPER_FIG_DIR / tex.name)
    shutil.copy(png, PAPER_FIG_DIR / png.name)

    doc = manifest.build(
        name="eval1_rate_vs_power",
        description=("Evaluation 1: delivered rate against network power over a "
                     "transmit-budget sweep, per AP count, with the amplifiers "
                     "re-sized at every point"),
        scenario=base,
        sweep={"budget": {"variable": "total transmit power", "unit": "W",
                          "values": args.budgets,
                          "fixed_total_antennas": args.antennas,
                          "ap_counts": args.ap_counts,
                          "pa_sizing": "re-sized at every point (per-AP budget)"}},
        params_by_deployment={d: base.power_params(d, 12.8) for d in Deployment},
        results=all_results,
        figures=[png.relative_to(OUT_DIR), tex.relative_to(OUT_DIR)],
    )
    path = manifest.write(doc, RUN_DIR)

    print("wrote:")
    for p in (png, tex):
        print(f"  {p.relative_to(OUT_DIR)}")
    print(f"  {(PAPER_FIG_DIR / tex.name)}")
    print(f"  {path.relative_to(OUT_DIR)}   (run parameters)")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
