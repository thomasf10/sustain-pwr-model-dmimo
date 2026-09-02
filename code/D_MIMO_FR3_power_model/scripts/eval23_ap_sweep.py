"""Evaluations 2 and 3 of the manuscript: the cost of chopping one array into L.

Both read the same sweep, so they are computed together. The total antenna count
``L*M = 128`` and the transmit budget ``P_TX`` are held fixed while ``L`` runs
over ``1, 2, 4, ..., 128``, so the only thing that changes along the axis is how
finely the same radiating hardware is spread over the same area.

* **Evaluation 2** reports the network power ``P_net`` against ``L``, one curve
  per functional split.
* **Evaluation 3** resolves the same points into the blocks of eq. pnet and
  eq. pap: amplifier, digital, analog, fronthaul, central unit, synchronization.
  This is what separates the mechanism from the symptom, and it is where
  Remark rem:pa_sizing and Remark rem:fpga are tested. The amplifier block
  should stay flat, since the rating and the total radiated power are both
  fixed, while the digital block should step up as the FPGAs stop being shared.

``L = 1`` is the co-located baseline of the manuscript rather than a one-AP
distributed network: it pays no fronthaul, no central unit and no
synchronization. It is therefore computed once, drawn as a separate marker in
evaluation 2 and as a bar labelled ``1*`` in evaluation 3, and the split curves
start at ``L = 2``. Reading it as the left end of the split curves would credit
the distributed model with a deployment it does not describe.

    python scripts/eval23_ap_sweep.py
    python scripts/eval23_ap_sweep.py --realizations 100 --Q 128
    python scripts/eval23_ap_sweep.py --budget 25.6

Writes ``figures/eval2_power_vs_aps.{png,tex}`` and
``figures/eval3_breakdown_vs_aps.{png,tex}``, copies the pgfplots sources into
the manuscript's ``figs/``, and records the run in a manifest. Rates are cached
in ``data/rates_dmimo.json`` keyed on the rate configuration, so points already
computed by ``eval1_rate_vs_power.py`` at the same budget are reused.
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

#: AP counts of the sweep, at the fixed total of 128 antennas.
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

    A single progress bar covers the whole campaign. Its unit is one evaluated
    point, and because the rate model is cached, points already computed finish
    instantly: the estimate is therefore pessimistic while the cached points go
    by and settles once the bar reaches work that has to be simulated. The
    per-point elapsed time is printed as each point lands, so the remaining
    estimate can be sanity-checked against it.
    """
    use_cache = not args.no_cache
    base = Scenario(L=args.ap_counts[0], M=args.antennas // args.ap_counts[0],
                    K=args.K, Q=args.Q, n_realizations=args.realizations)

    # The co-located point first: it is the reference every panel starts from,
    # and its rate is shared with the L-swept points through the cache.
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
        tqdm.write(f"  {label:<28} P_net = {result.power:8.1f} W   "
                   f"R = {result.R_total*1e-9:6.2f} Gbit/s   ({dt:5.1f} s)")
    bar.close()

    print(f"\ncampaign took {time.monotonic() - started:.0f} s")
    return results, colocated, base


def print_table(results, colocated, args) -> None:
    header = (f"{'deployment':<26}{'L':>5}{'M':>5}{'R_tot':>9}{'P_net':>9}{'EE':>9}"
              f"{'PA':>8}{'dig':>8}{'ana':>8}{'FH':>9}{'CPU':>8}{'sync':>7}")
    print(header)
    print(f"{'':<26}{'':>5}{'':>5}{'[Gb/s]':>9}{'[W]':>9}{'[Mb/J]':>9}"
          f"{'[W]':>8}{'[W]':>8}{'[W]':>8}{'[W]':>9}{'[W]':>8}{'[W]':>7}")
    print("-" * len(header))

    def row(r, name, L, M):
        b = plotting._blocks_of(r)
        print(f"{name:<26}{L:>5}{M:>5}{r.R_total*1e-9:>9.2f}{r.power:>9.1f}"
              f"{r.energy_efficiency*1e-6:>9.1f}"
              f"{b['AP PA']:>8.1f}{b['AP digital']:>8.1f}{b['AP analog']:>8.1f}"
              f"{b['fronthaul']:>9.1f}{b['CPU']:>8.1f}{b['AP sync']:>7.1f}")

    row(colocated, colocated.deployment.label, 1, args.antennas)
    print()
    for deployment, series in results.items():
        for r in series:
            row(r, deployment.label, r.L, r.M)
        print()


def main() -> None:
    args = parse_args()
    for L in args.ap_counts:
        if args.antennas % L:
            raise SystemExit(f"L={L} does not divide the antenna total {args.antennas}")

    print(f"Evaluations 2 and 3: AP-count sweep at P_TX = {args.budget:g} W, "
          f"{args.antennas} antennas, K={args.K}, Q={args.Q}, "
          f"{args.realizations} drops/point")
    print(f"L: {', '.join(str(L) for L in args.ap_counts)}  (plus the co-located "
          f"baseline at L=1)")
    print()

    results, colocated, base = run(args)
    print()
    print(unsourced_parameters(base.power_params(Deployment.CENTRALIZED_S1,
                                                 args.budget)))
    print()
    print_table(results, colocated, args)

    plotting.setup_style(use_tex=False, font_size=12)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    sweep_title = (f"$P_\\mathrm{{TX}}={args.budget:g}$ W and {args.antennas} "
                   f"antennas fixed, $K={args.K}$ users")
    fig2, _ = plotting.plot_power_vs_aps(results, title=sweep_title,
                                         colocated_result=colocated)
    png2 = FIG_DIR / "eval2_power_vs_aps.png"
    fig2.savefig(png2, dpi=200, bbox_inches="tight")

    fig3, _ = plotting.plot_breakdown_vs_aps(results, title=sweep_title,
                                             colocated_result=colocated)
    png3 = FIG_DIR / "eval3_breakdown_vs_aps.png"
    fig3.savefig(png3, dpi=200, bbox_inches="tight")

    tex2 = pgf.power_vs_aps(results, FIG_DIR / "eval2_power_vs_aps.tex",
                            colocated_result=colocated)
    tex3 = pgf.breakdown_vs_aps(results, FIG_DIR / "eval3_breakdown_vs_aps.tex",
                                colocated_result=colocated)

    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    written = [png2, tex2, png3, tex3]
    for p in written:
        shutil.copy(p, PAPER_FIG_DIR / p.name)

    doc = manifest.build(
        name="eval23_ap_sweep",
        description=("Evaluations 2 and 3: network power and its per-block "
                     "breakdown over an AP-count sweep at fixed transmit budget "
                     "and fixed total antenna count"),
        scenario=base,
        sweep={"aps": {"variable": "number of APs", "unit": "count",
                       "values": args.ap_counts,
                       "fixed_budget_W": args.budget,
                       "fixed_total_antennas": args.antennas,
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
    print(f"  {PAPER_FIG_DIR}  (pgfplots sources copied for the manuscript)")
    print(f"  {path.relative_to(OUT_DIR)}   (run parameters)")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
