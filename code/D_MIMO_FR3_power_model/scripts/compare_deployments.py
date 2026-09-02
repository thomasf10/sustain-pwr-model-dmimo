"""Rate versus power consumption of four MIMO deployments.

Evaluates, on both the rate model (``../D_MIMO_rate``) and the power model
(``dmimo_power``):

    co-located MIMO            one site of L*M antennas
    D-MIMO, centralized (S1)   L APs, CPU computes and applies the precoder
    D-MIMO, centralized (S2)   L APs, CPU computes, AP applies
    D-MIMO, distributed (S3)   L APs, AP computes and applies

S1 and S2 realize the same centralized precoder and so deliver the same rate;
they differ only in where the work runs and what crosses the fronthaul, and
their Monte Carlo is computed once and shared.

Two sweeps, on the same axes so they can be read against each other:

* ``--sweep budget`` varies the total transmit power at a fixed topology;
* ``--sweep aps`` fixes the transmit power and redistributes the *same* total
  antenna count over a varying number of APs, from a few large arrays to many
  small ones. The co-located site is the L=1 reference point.
* ``--sweep both`` runs both.

Every run also writes an overview of each distinct network geometry and a JSON
manifest recording the parameters that produced it.

    python scripts/compare_deployments.py
    python scripts/compare_deployments.py --sweep aps --fixed-budget 8
    python scripts/compare_deployments.py --sweep both --realizations 40
    python scripts/compare_deployments.py --no-cache      # recompute the rates

Rates are cached in ``data/rates_dmimo.json``, keyed by every parameter that
affects them, so re-running only recomputes what actually changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmimo_power import manifest, plotting, unsourced_parameters  # noqa: E402
from dmimo_power.scenarios import Deployment, Scenario, evaluate  # noqa: E402

# Total TX power [W]. The nominal budget is P_TX = L*M*P_max = 12.8 W at the
# 0.1 W FR3 amplifier rating, and the sweep re-sizes the amplifiers at every
# point (PASizing.PER_AP_BUDGET), so each point is a deployment dimensioned for
# the budget it is given rather than one deployment being dimmed.
DEFAULT_BUDGETS = (0.8, 1.6, 3.2, 6.4, 12.8, 25.6)
DEFAULT_AP_COUNTS = (2, 4, 8, 16, 32, 64, 128)            # APs, at fixed L*M
OUT_DIR = Path(__file__).resolve().parent.parent
FIG_DIR = OUT_DIR / "figures"
RUN_DIR = OUT_DIR / "data" / "runs"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", choices=("budget", "aps", "both"), default="both",
                    help="which sweep(s) to run")
    ap.add_argument("--budgets", type=float, nargs="+", default=list(DEFAULT_BUDGETS),
                    help="total transmit power budgets for the budget sweep [W]")
    ap.add_argument("--ap-counts", type=int, nargs="+", default=list(DEFAULT_AP_COUNTS),
                    help="AP counts for the fixed-power sweep (must divide L*M)")
    ap.add_argument("--fixed-budget", type=float, default=12.8,
                    help="total transmit power held fixed in the AP-count sweep [W]")
    ap.add_argument("--L", type=int, default=32, help="APs in the budget sweep")
    ap.add_argument("--M", type=int, default=4, help="antennas per AP in the budget sweep")
    ap.add_argument("--K", type=int, default=20, help="users")
    ap.add_argument("--realizations", type=int, default=20,
                    help="Monte Carlo drops per point")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the cached rates and recompute")
    ap.add_argument("--no-show", action="store_true",
                    help="save the figures without opening a window")
    return ap.parse_args()


def print_table(results) -> None:
    """Result table, grouped by deployment."""
    header = (f"{'deployment':<26}{'L':>4}{'M':>4}{'P_TX':>7}{'R_DL':>9}"
              f"{'R_UL':>9}{'R_tot':>9}{'P_net':>9}{'EE':>10}")
    print(header)
    print(f"{'':<26}{'':>4}{'':>4}{'[W]':>7}{'[Gb/s]':>9}{'[Gb/s]':>9}"
          f"{'[Gb/s]':>9}{'[W]':>9}{'[Mb/J]':>10}")
    print("-" * len(header))
    for deployment in Deployment:
        rows = [r for r in results if r.deployment is deployment]
        for r in rows:
            print(f"{deployment.label:<26}{r.L:>4}{r.M:>4}{r.P_budget:>7.1f}"
                  f"{r.rates.R_DL * 1e-9:>9.2f}{r.rates.R_UL * 1e-9:>9.2f}"
                  f"{r.R_total * 1e-9:>9.2f}{r.power:>9.1f}"
                  f"{r.energy_efficiency * 1e-6:>10.1f}")
        if rows:
            print()


def run_budget_sweep(args, scenario, use_cache):
    """Vary the total transmit power at a fixed topology."""
    print("=" * 78)
    print(f"Sweep 1: total transmit power, at L={scenario.L} x M={scenario.M} "
          f"(= {scenario.M_tot} antennas)")
    print("=" * 78)
    results = {d: [] for d in Deployment}
    for deployment in Deployment:
        print(f"{deployment.label}:")
        for budget in args.budgets:
            results[deployment].append(
                evaluate(scenario, deployment, budget, use_cache=use_cache))
    return results


def run_ap_sweep(args, scenario, use_cache):
    """Fix the transmit power and redistribute the same antennas over L APs."""
    budget = args.fixed_budget
    print("=" * 78)
    print(f"Sweep 2: number of APs at fixed P_TX = {budget:g} W and fixed "
          f"{scenario.M_tot} total antennas")
    print("=" * 78)
    results = {d: [] for d in Deployment}

    # The co-located site is the L=1 reference: one array of every antenna. It
    # is a single point, not a curve, since it has no AP count to vary.
    print(f"{Deployment.COLOCATED.label}:")
    results[Deployment.COLOCATED].append(
        evaluate(scenario, Deployment.COLOCATED, budget, use_cache=use_cache))

    for deployment in Deployment:
        if not deployment.is_distributed:
            continue
        print(f"{deployment.label}:")
        for L in args.ap_counts:
            sub = scenario.with_aps(L)
            results[deployment].append(
                evaluate(sub, deployment, budget, use_cache=use_cache))
    return results


def main() -> None:
    args = parse_args()
    scenario = Scenario(L=args.L, M=args.M, K=args.K,
                        n_realizations=args.realizations)
    use_cache = not args.no_cache
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Deployments compared at equal total transmit power and equal antenna "
          f"count ({scenario.M_tot} antennas, K={scenario.K} users, "
          f"{scenario.area_size:.0f} m area, {scenario.n_realizations} drops/point)")
    print(f"frame: tau_DL = {scenario.tau_DL:.2f}, signalling "
          f"{scenario.tau_DLsig:.3f} (DL) / {scenario.tau_ULsig:.3f} (UL), "
          f"load {scenario.xbar_DL:.2f} / {scenario.xbar_UL:.2f}, "
          f"coherence block {scenario.tau_c} samples")
    print()
    print(unsourced_parameters(
        scenario.power_params(Deployment.CENTRALIZED_S1, 1.0)))
    print()

    plotting.setup_style(use_tex=False, font_size=12)
    figures, all_results = [], []

    # --- Sweep 1: transmit power -----------------------------------------
    if args.sweep in ("budget", "both"):
        results = run_budget_sweep(args, scenario, use_cache)
        flat = [r for rs in results.values() for r in rs]
        all_results += flat
        print()
        print_table(flat)

        title = (f"$L={scenario.L}$ APs $\\times$ $M={scenario.M}$ antennas "
                 f"(= {scenario.M_tot} total), $K={scenario.K}$ users")
        fig, _ = plotting.plot_rate_vs_power(
            results, title=title, point_label=plotting.budget_label)
        path = FIG_DIR / "rate_vs_power_budget.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        figures.append(path)

        mid = len(args.budgets) // 2
        fig, _ = plotting.plot_power_breakdown(
            [results[d][mid] for d in Deployment],
            title=f"Power breakdown at $P_\\mathrm{{TX}}={args.budgets[mid]:g}$ W")
        path = FIG_DIR / "power_breakdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        figures.append(path)

    # --- Sweep 2: number of APs ------------------------------------------
    if args.sweep in ("aps", "both"):
        results = run_ap_sweep(args, scenario, use_cache)
        flat = [r for rs in results.values() for r in rs]
        all_results += flat
        print()
        print_table(flat)

        title = (f"$P_\\mathrm{{TX}}={args.fixed_budget:g}$ W fixed, "
                 f"{scenario.M_tot} antennas redistributed over $L$ APs, "
                 f"$K={scenario.K}$ users")
        fig, _ = plotting.plot_rate_vs_power(
            results, title=title, point_label=plotting.ap_count_label)
        path = FIG_DIR / "rate_vs_power_aps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        figures.append(path)

    # --- Network overviews ------------------------------------------------
    # Every topology that appeared in either sweep, each drawn once.
    ap_counts = list(args.ap_counts) if args.sweep in ("aps", "both") else []
    layouts = [scenario.with_aps(L)
               for L in dict.fromkeys([scenario.L] + ap_counts)]
    figures += plotting.save_network_figures(layouts, list(Deployment), FIG_DIR)

    # --- Manifest ---------------------------------------------------------
    sweep_desc = {
        "budget": {"variable": "total transmit power", "unit": "W",
                   "values": args.budgets, "fixed_topology": f"L={args.L}, M={args.M}"},
        "aps": {"variable": "number of APs", "unit": "count",
                "values": args.ap_counts, "fixed_budget_W": args.fixed_budget,
                "fixed_total_antennas": scenario.M_tot},
    }
    params = {d: scenario.power_params(d, args.fixed_budget) for d in Deployment}
    doc = manifest.build(
        name=f"compare_{args.sweep}",
        description=(f"Rate vs power of {len(Deployment)} MIMO deployments; "
                     f"sweep={args.sweep}"),
        scenario=scenario,
        sweep={k: v for k, v in sweep_desc.items()
               if args.sweep in (k, "both")},
        params_by_deployment=params,
        results=all_results,
        figures=[p.relative_to(OUT_DIR) for p in figures],
    )
    path = manifest.write(doc, RUN_DIR)

    print("wrote:")
    for p in figures:
        print(f"  {p.relative_to(OUT_DIR)}")
    print(f"  {path.relative_to(OUT_DIR)}   (run parameters)")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
