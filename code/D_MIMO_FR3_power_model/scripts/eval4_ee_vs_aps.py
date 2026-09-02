"""Evaluation 4 of the manuscript: energy efficiency over the AP-count sweep.

The same sweep as ``eval23_ap_sweep.py``, read through eq. ee_net instead of
through ``P_net``. It is a separate evaluation rather than a third panel of that
script because it answers a different question: ``P_net`` alone cannot say
whether distributing pays, since the delivered rate is not constant along the
sweep, so the power figure and this one have to be read together.

The ceiling ``1 / (Pi_FH L)`` of Remark rem:fh_ceiling is drawn alongside. It
bounds the *data-sharing* splits, S2 and S3, whose fronthaul carries the payload
duplicated to every AP; it does not bound S1, which forwards samples and
duplicates nothing, and S1 is expected to sit above it at large ``L``.

Rates and powers come from the same cache as evaluations 1 to 3, keyed on the
rate configuration, so running this after ``eval23_ap_sweep.py`` at the same
budget recomputes nothing and the two figures are guaranteed to describe one
campaign rather than two.

    python scripts/eval4_ee_vs_aps.py
    python scripts/eval4_ee_vs_aps.py --realizations 100 --Q 128
    python scripts/eval4_ee_vs_aps.py --budget 25.6

Writes ``figures/eval4_ee_vs_aps.{png,tex}``, copies the pgfplots source into
the manuscript's ``figs/``, and records the run in a manifest.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmimo_power import manifest, pgf, plotting, unsourced_parameters  # noqa: E402
from dmimo_power.scenarios import Deployment, Scenario, evaluate  # noqa: E402

#: AP counts of the sweep, at the fixed total of 128 antennas. The same list as
#: eval23_ap_sweep.py, so the two figures cover identical points.
DEFAULT_AP_COUNTS = (2, 4, 8, 16, 32, 64, 128)

OUT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = OUT_DIR / "data" / "runs"
PAPER_FIG_DIR = OUT_DIR.parent.parent / "sustain-dmimo-pwr-model-paper" / "figs"

SPLITS = (Deployment.CENTRALIZED_S1, Deployment.CENTRALIZED_S2,
          Deployment.DISTRIBUTED_S3)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ap-counts", type=int, nargs="+", default=list(DEFAULT_AP_COUNTS),
                    help="AP counts to sweep (each must divide the antenna total)")
    ap.add_argument("--antennas", type=int, default=128,
                    help="total antenna count L*M, held fixed across the sweep")
    ap.add_argument("--budget", type=float, default=12.8,
                    help="total transmit power held fixed across the sweep [W]")
    ap.add_argument("--K", type=int, default=20, help="users")
    ap.add_argument("--Q", type=int, default=64,
                    help="subcarriers evaluated per drop (a Monte Carlo subset)")
    ap.add_argument("--realizations", type=int, default=20,
                    help="Monte Carlo drops per point")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the cached rates and recompute")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening a window")
    return ap.parse_args()


def run(args):
    """Evaluate every split at every AP count, plus the co-located baseline.

    One progress bar over the whole campaign, in units of evaluated points. The
    rate model is cached, so points already computed by another evaluation
    finish instantly and the remaining-time estimate is pessimistic until the
    bar reaches work that has to be simulated; the per-point elapsed time is
    printed as each point lands so the estimate can be checked against it.
    """
    use_cache = not args.no_cache
    base = Scenario(L=args.ap_counts[0], M=args.antennas // args.ap_counts[0],
                    K=args.K, Q=args.Q, n_realizations=args.realizations)

    work = [(None, Deployment.COLOCATED)]
    work += [(L, d) for L in args.ap_counts for d in SPLITS]

    results = {d: [] for d in SPLITS}
    colocated = None
    started = time.monotonic()

    bar = tqdm(work, desc="evaluating", unit="pt",
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                          "[elapsed {elapsed}, left {remaining}, {rate_fmt}]")
    for L, deployment in bar:
        scenario = base if L is None else base.with_aps(L)
        label = "co-located" if L is None else f"{deployment.value} L={L}"
        bar.set_postfix_str(label, refresh=True)

        t0 = time.monotonic()
        result = evaluate(scenario, deployment, args.budget, use_cache=use_cache,
                          verbose=False, progress=False)
        dt = time.monotonic() - t0

        if L is None:
            colocated = result
        else:
            results[deployment].append(result)
        tqdm.write(f"  {label:<28} EE = {result.energy_efficiency*1e-6:7.1f} Mbit/J  "
                   f"R = {result.R_total*1e-9:6.2f} Gbit/s  "
                   f"P = {result.power:7.1f} W   ({dt:5.1f} s)")
    bar.close()

    print(f"\ncampaign took {time.monotonic() - started:.0f} s")
    return results, colocated, base


def print_table(results, colocated, Pi_FH) -> None:
    """Efficiencies with the ceiling alongside, so the comparison is explicit."""
    header = (f"{'deployment':<26}{'L':>5}{'M':>5}{'R_tot':>9}{'P_net':>9}"
              f"{'EE':>9}{'ceiling':>10}{'EE/ceil':>9}")
    print(header)
    print(f"{'':<26}{'':>5}{'':>5}{'[Gb/s]':>9}{'[W]':>9}{'[Mb/J]':>9}"
          f"{'[Mb/J]':>10}{'':>9}")
    print("-" * len(header))

    def row(r, name, L, M):
        ceil = 1.0 / (Pi_FH * L) * 1e-6
        print(f"{name:<26}{L:>5}{M:>5}{r.R_total*1e-9:>9.2f}{r.power:>9.1f}"
              f"{r.energy_efficiency*1e-6:>9.1f}{ceil:>10.1f}"
              f"{r.energy_efficiency*1e-6/ceil:>9.2f}")

    row(colocated, colocated.deployment.label, 1, colocated.M)
    print()
    for deployment, series in results.items():
        for r in series:
            row(r, deployment.label, r.L, r.M)
        best = max(series, key=lambda r: r.energy_efficiency)
        print(f"{'':<26}peak {best.energy_efficiency*1e-6:.1f} Mbit/J "
              f"at L={best.L}\n")


def main() -> None:
    args = parse_args()
    for L in args.ap_counts:
        if args.antennas % L:
            raise SystemExit(f"L={L} does not divide the antenna total {args.antennas}")

    print(f"Evaluation 4: energy efficiency over an AP-count sweep at "
          f"P_TX = {args.budget:g} W, {args.antennas} antennas, K={args.K}, "
          f"Q={args.Q}, {args.realizations} drops/point")
    print(f"L: {', '.join(str(L) for L in args.ap_counts)}  (plus the co-located "
          f"baseline at L=1)")
    print()

    results, colocated, base = run(args)
    params = base.power_params(Deployment.CENTRALIZED_S1, args.budget)
    print()
    print(unsourced_parameters(params))
    print()
    print_table(results, colocated, params.Pi_FH)

    plotting.setup_style(use_tex=False, font_size=12)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    title = (f"$P_\\mathrm{{TX}}={args.budget:g}$ W and {args.antennas} "
             f"antennas fixed, $K={args.K}$ users")
    fig, _ = plotting.plot_ee_vs_aps(results, title=title,
                                     colocated_result=colocated,
                                     Pi_FH=params.Pi_FH)
    png = FIG_DIR / "eval4_ee_vs_aps.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")

    tex = pgf.ee_vs_aps(results, FIG_DIR / "eval4_ee_vs_aps.tex",
                        colocated_result=colocated, Pi_FH=params.Pi_FH)

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = [png, tex]
    for p in written:
        shutil.copy(p, PAPER_FIG_DIR / p.name)

    doc = manifest.build(
        name="eval4_ee_vs_aps",
        description=("Evaluation 4: network energy efficiency over an AP-count "
                     "sweep at fixed transmit budget and fixed total antenna "
                     "count, against the fronthaul ceiling of rem:fh_ceiling"),
        scenario=base,
        sweep={"aps": {"variable": "number of APs", "unit": "count",
                       "values": args.ap_counts,
                       "fixed_budget_W": args.budget,
                       "fixed_total_antennas": args.antennas,
                       "metric": "EE = (R_DL + R_UL) / P_net (eq. ee_net)",
                       "ceiling": "1/(Pi_FH L), bounds the data-sharing splits "
                                  "S2 and S3 only",
                       "colocated_baseline": "evaluated separately at L=1, "
                                             "without fronthaul, CPU or sync"}},
        params_by_deployment={d: base.power_params(d, args.budget)
                              for d in Deployment},
        results=[colocated] + [r for s in results.values() for r in s],
        figures=[p.relative_to(OUT_DIR) for p in written],
    )
    path = manifest.write(doc, RUN_DIR)

    print("wrote:")
    for p in written:
        print(f"  {p.relative_to(OUT_DIR)}")
    print(f"  {PAPER_FIG_DIR}  (pgfplots source copied for the manuscript)")
    print(f"  {path.relative_to(OUT_DIR)}   (run parameters)")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
