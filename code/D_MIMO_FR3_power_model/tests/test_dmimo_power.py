"""Checks on the distributed power model.

The load-bearing one is :func:`test_colocated_is_the_L1_special_case`: the
manuscript states that setting ``L = 1``, dropping the fronthaul and the CPU,
and putting every digital operation at one node must return the co-located
model of eq. pcons unchanged. That is the property which guarantees the
distributed model is an extension of ``../FR3_power_model`` rather than a
second, independently drifting implementation, and it is checked to
floating-point precision.

Run with ``python tests/test_dmimo_power.py`` or under pytest.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dmimo_power import (  # noqa: E402
    DMIMOPowerParams,
    PASizing,
    Split,
    compute_colocated,
    compute_network,
    cpu,
    fronthaul_rate,
    xi_ap,
    xi_cpu,
    xi_precoder_centralized,
    xi_precoder_local,
)
from dmimo_power.network import DL, UL, ap_digital, ap_operating_point  # noqa: E402
from fr3_power import OperatingPoint, PowerParams  # noqa: E402
from fr3_power import power_model as fr3_pm  # noqa: E402

BUDGET = 8.0        # total transmit power [W]
R_DL, R_UL = 12e9, 4e9


def base_params(**kw) -> DMIMOPowerParams:
    """Distributed parameters with the new blocks zeroed, for equivalence tests."""
    defaults = dict(L=16, M=4, K=10, split=Split.S1,
                    P_sync=0.0, P_CPU_0=0.0, P_FH_0=0.0, Pi_FH=0.0,
                    eta_FH_sc=1.0, eta_CPU_sc=PowerParams.eta_dig_sc)
    defaults.update(kw)
    return DMIMOPowerParams(**defaults)


# ======================================================================
# The extension must contain the co-located model
# ======================================================================


def test_colocated_is_the_L1_special_case():
    """L=1, S1, no fronthaul and no CPU overhead reproduces eq. pcons exactly.

    With one AP holding every antenna and split S1, all the MIMO processing sits
    at the CPU and the AP keeps only OFDM, predistortion and filtering. Summing
    the two nodes must give back the single-site digital block, and the analog
    and PA blocks must be untouched.
    """
    M_tot = 64
    p = base_params(L=1, M=M_tot)
    ap_power = np.array([BUDGET])

    net = compute_network(p, ap_power, rho_max=BUDGET, R_DL=R_DL, R_UL=R_UL)
    col = compute_colocated(p, BUDGET, R_DL, R_UL)

    assert np.isclose(net.fronthaul.total, 0.0)
    # Digital: the AP block plus the CPU block is the co-located digital block.
    assert np.isclose(net.ap_digital.total + net.cpu.total, col.digital.total), \
        f"{net.ap_digital.total + net.cpu.total} != {col.digital.total}"
    assert np.isclose(net.ap_analog.total, col.analog.total)
    assert np.isclose(net.ap_pa.total, col.pa.total)
    assert np.isclose(net.total, col.total)


def test_synchronization_is_separable_from_the_analog_block():
    """P_sync is its own term of eq. pap, not an addend inside the analog block.

    Because it was pulled out, the distributed analog block reduces to the
    co-located one *unconditionally* rather than only when P_sync = 0, and the
    whole synchronization cost is a per-AP constant that the network total
    carries linearly in L. It must not depend on the load, the split, or the
    frame, and the co-located baseline must not pay it at all.
    """
    M_tot = 64
    for L, P_sync in ((1, 0.5), (16, 0.5), (32, 2.0)):
        p = base_params(L=L, M=M_tot // L, P_sync=P_sync)
        ap_power = np.full(L, BUDGET / L)

        net = compute_network(p, ap_power, rho_max=BUDGET / L, R_DL=R_DL, R_UL=R_UL)
        off = compute_network(replace(p, P_sync=0.0), ap_power,
                              rho_max=BUDGET / L, R_DL=R_DL, R_UL=R_UL)

        expected = L * P_sync / p.eta_sync_sc
        assert np.isclose(net.ap_sync.total, expected)
        assert np.isclose(net.ap_sync.load_dep, 0.0)      # wholly load-independent
        assert np.isclose(net.ap_analog.total, off.ap_analog.total)
        assert np.isclose(net.total, off.total + expected)

    # The co-located site does not synchronize with anything.
    col = compute_colocated(base_params(L=1, M=M_tot, P_sync=0.5), BUDGET, R_DL, R_UL)
    assert not hasattr(col, "ap_sync")


def test_centralized_precoder_count_matches_colocated():
    """eq. gops_cen at L=1, M=M_RF is the co-located precoder count of eq. gops_pre."""
    M_tot = 64
    p = base_params(L=1, M=M_tot)
    # Co-located: gops = M_RF*(f_sI*2K + f_sI/upsilon*(K^3/(3 M_RF) + 3K^2 + K)).
    gops_colocated = M_tot * (p.f_sI * 2 * p.K + p.f_sI / p.upsilon_coh
                              * (p.K ** 3 / (3 * M_tot) + 3 * p.K ** 2 + p.K))
    gops_dmimo = p.f_sI * xi_cpu(p, DL)
    assert np.isclose(gops_colocated, gops_dmimo)


# ======================================================================
# Functional splits
# ======================================================================


def test_split_places_every_operation_exactly_once():
    """Precoder application appears once across the AP/CPU rows, in every split."""
    for split in (Split.S1, Split.S2, Split.S3):
        p = base_params(split=split)
        # Application, 2*K*N per sample, summed over the nodes that do it. Under
        # S3 the AP row also carries the local factorization, which is a
        # different operation and is subtracted out here.
        computation = p.L * xi_precoder_local(p) if split is Split.S3 else 0.0
        applied = p.L * xi_ap(p, UL) + xi_cpu(p, UL) - computation
        assert np.isclose(applied, 2 * p.K * p.M_tot), \
            f"{split}: application count {applied} != {2 * p.K * p.M_tot}"

    # S1 puts everything at the CPU, S3 everything at the APs.
    p1, p3 = base_params(split=Split.S1), base_params(split=Split.S3)
    assert xi_ap(p1, DL) == 0.0 and xi_ap(p1, UL) == 0.0
    assert xi_cpu(p3, DL) == 0.0 and xi_cpu(p3, UL) == 0.0


def test_s2_splits_computation_from_application():
    """S2: the CPU computes the precoder, the AP applies it, neither does both."""
    p2 = base_params(split=Split.S2)
    # CPU: computation only, and only in the downlink.
    assert np.isclose(xi_cpu(p2, DL), xi_precoder_centralized(p2))
    assert xi_cpu(p2, UL) == 0.0
    # AP: application only, in both directions, and no local computation.
    assert np.isclose(xi_ap(p2, DL), 2 * p2.K * p2.M)
    assert np.isclose(xi_ap(p2, UL), 2 * p2.K * p2.M)

    # S2 computes the same centralized matrix as S1 and applies the same amount
    # at the AP as S3 -- it is the crossover of the two, not a third scheme. The
    # AP rows differ only by the local factorization that S3 alone performs.
    p1, p3 = base_params(split=Split.S1), base_params(split=Split.S3)
    assert np.isclose(xi_cpu(p2, DL), xi_cpu(p1, DL) - 2 * p1.K * p1.M_tot)
    assert np.isclose(xi_ap(p3, UL) - xi_ap(p2, UL), xi_precoder_local(p3))


def test_centralized_computation_charged_to_downlink_only():
    """TDD reciprocity: the ZF matrix is computed once and reused by the combiner."""
    p1, p2 = base_params(split=Split.S1), base_params(split=Split.S2)
    assert np.isclose(xi_cpu(p1, DL) - xi_cpu(p1, UL), xi_precoder_centralized(p1))
    assert np.isclose(xi_cpu(p2, DL) - xi_cpu(p2, UL), xi_precoder_centralized(p2))


def test_local_computation_charged_in_both_directions():
    """S3 shares no factorization between the directions (eq. gops_split, AP row).

    The local downlink precoder inverts ``E_l E_l^H + lambda I`` while the local
    uplink combiner inverts the power-weighted ``E_l P E_l^H + lambda I``. They
    are different matrices, so reciprocity buys nothing and the AP pays the
    Cholesky count twice, unlike the centralized pair.
    """
    p3 = base_params(split=Split.S3)
    assert np.isclose(xi_ap(p3, DL), xi_ap(p3, UL))
    for direction in (DL, UL):
        assert np.isclose(xi_ap(p3, direction),
                          2 * p3.K * p3.M + xi_precoder_local(p3))


def test_centralized_precoding_costs_more_per_antenna():
    """Roughly a factor K/M more, as the manuscript argues for small-M APs."""
    p = base_params(L=16, M=4, K=10)
    per_antenna_cen = xi_precoder_centralized(p) / p.M_tot
    per_antenna_loc = xi_precoder_local(p) / p.M          # per AP, over its M antennas
    ratio = per_antenna_cen / per_antenna_loc
    assert 0.3 * p.K / p.M < ratio < 3 * p.K / p.M, ratio


def test_ap_pays_no_mimo_fpga_when_it_does_no_mimo_work():
    """Under S1 the AP has no precoder/combiner block, only its filter FPGA."""
    p1 = base_params(split=Split.S1)
    p3 = base_params(split=Split.S3)
    op = ap_operating_point(p1, BUDGET / p1.L)
    assert ap_digital(p1, op).total < ap_digital(p3, op).total


# ======================================================================
# Fronthaul
# ======================================================================


def test_s1_fronthaul_is_traffic_independent():
    """eq. fh_s1 is a hardware constant: the delivered rate cannot change it."""
    p = base_params(split=Split.S1, P_FH_0=0.825, Pi_FH=0.25e-9)
    lo = fronthaul_rate(p, R_DL_ap=1e9).total
    hi = fronthaul_rate(p, R_DL_ap=100e9).total
    assert np.isclose(lo, hi)


def test_s3_fronthaul_tracks_the_delivered_rate():
    """Under a data-sharing split the payload is what crosses the link."""
    p = base_params(split=Split.S3, P_FH_0=0.825, Pi_FH=0.25e-9)
    lo = fronthaul_rate(p, R_DL_ap=1e9).total
    hi = fronthaul_rate(p, R_DL_ap=100e9).total
    assert np.isclose(hi - lo, 99e9)


def test_s3_downlink_prelog_is_not_applied_twice():
    """Remark rem:no_double: the delivered rate enters the fronthaul unscaled."""
    p = base_params(split=Split.S3)
    # Difference between two delivered rates must pass through with gain one,
    # not be multiplied by the frame prelog again.
    delta = fronthaul_rate(p, 2e9).total - fronthaul_rate(p, 1e9).total
    assert np.isclose(delta, 1e9)
    assert p.tau_DL * (1 - p.tau_DLsig) < 1.0   # a second prelog would be visible


def test_s1_moves_more_over_the_fronthaul_than_s3_only_when_M_exceeds_K():
    """The sample-forwarding split is not unconditionally the heavier one.

    S1 carries ``M`` sample streams per link in both directions, while S2 and S3
    carry ``K`` scalar partial sums in the uplink (eq. fh_ul). Forwarding partial
    sums is therefore a saving only when an AP has more antennas than it serves
    users. In the regime this paper evaluates, ``K = 20`` users over APs of
    ``M <= 4`` antennas, it is the opposite: the uplink of the data-sharing
    splits is the *heavier* one, by a factor of roughly ``K / M`` on that phase,
    and S1's advantage of never carrying the payload does not make up for it.
    """
    def compare(K, M):
        r1 = fronthaul_rate(base_params(split=Split.S1, K=K, M=M), R_DL_ap=R_DL)
        r3 = fronthaul_rate(base_params(split=Split.S3, K=K, M=M), R_DL_ap=R_DL)
        return r1.total, r3.total

    # Wide APs, few users: samples dominate and S1 is the expensive split.
    r1, r3 = compare(K=4, M=32)
    assert r1 > r3, (r1, r3)

    # The evaluated regime: few antennas per AP, many users. S3 costs more.
    r1, r3 = compare(K=20, M=4)
    assert r3 > r1, (r1, r3)


def test_s2_fronthaul_tracks_traffic_and_adds_coefficients():
    """S2 is a data-sharing split, so its load follows the delivered rate."""
    p = base_params(split=Split.S2)
    lo = fronthaul_rate(p, R_DL_ap=1e9).total
    hi = fronthaul_rate(p, R_DL_ap=100e9).total
    assert np.isclose(hi - lo, 99e9)      # payload passes through with gain one
    # The coefficient term is what separates S2 from S3, and it is amortised
    # over a coherence block, so it is small next to the sample rate of S1.
    r3 = fronthaul_rate(base_params(split=Split.S3), R_DL_ap=1e9).total
    coefficients = p.f_sI / p.upsilon_coh * 2 * p.b_FH * p.M * p.K
    assert lo > r3
    assert lo - r3 < coefficients + 2 * p.b_FH * p.M * p.f_sI


def test_centralized_splits_pay_a_sample_level_fronthaul_for_pilots():
    """The CPU must see the raw pilots under S1 and S2, but not under S3.

    Centralized channel estimation needs the collective received pilots, so the
    uplink signalling phase carries samples whatever the data-phase split; local
    operation consumes its pilots at the AP and sends nothing. The pilot traffic
    is isolated by emptying the uplink data phase (``xbar_UL = 0``) and
    differencing an all-pilot uplink against a no-pilot one, which leaves
    exactly ``tau_UL * R_sig``.
    """
    def pilot_traffic(split):
        p = base_params(split=split)
        with_pilots = fronthaul_rate(replace(p, tau_ULsig=1.0), R_DL, xbar_UL=0.0)
        without = fronthaul_rate(replace(p, tau_ULsig=0.0), R_DL, xbar_UL=0.0)
        return with_pilots.total - without.total, p

    for split in (Split.S1, Split.S2):
        traffic, p = pilot_traffic(split)
        expected = p.tau_UL * 2 * p.b_FH * p.M * p.f_sI     # M streams of samples
        assert np.isclose(traffic, expected), f"{split}: {traffic} != {expected}"

    traffic, _ = pilot_traffic(Split.S3)
    assert np.isclose(traffic, 0.0), f"S3 should send no pilots, got {traffic}"


# ======================================================================
# Network assembly
# ======================================================================


def test_network_rejects_an_ap_over_budget():
    p = base_params()
    over = np.full(p.L, BUDGET / p.L * 1.5)
    try:
        compute_network(p, over, rho_max=BUDGET / p.L, R_DL=R_DL, R_UL=R_UL)
    except ValueError:
        return
    raise AssertionError("an AP radiating above rho_max should be rejected")


def test_local_operation_fills_its_budget_centralized_does_not():
    """Utilisation is the quantity the two operation modes differ on (eq. util)."""
    p = base_params()
    rho = BUDGET / p.L
    full = compute_network(p, np.full(p.L, rho), rho, R_DL, R_UL)
    partial = compute_network(p, np.full(p.L, 0.4 * rho), rho, R_DL, R_UL)
    assert np.isclose(full.utilisation.mean(), 1.0)
    assert np.isclose(partial.utilisation.mean(), 0.4)
    # Radiating less costs less, but the static floor is paid either way.
    assert partial.ap_pa.total < full.ap_pa.total


def test_pa_sizing_conventions_differ_as_documented():
    """Remark rem:pa_sizing: a fully loaded AP consumes rho_max / eta_PAmax."""
    p = base_params(pa_sizing=PASizing.PER_AP_BUDGET, xi=0.1)
    rho = BUDGET / p.L
    net = compute_network(p, np.full(p.L, rho), rho, R_DL, R_UL)
    # Strip the frame averaging and the supply efficiency: at full load the
    # data-mode PA consumption of one AP should be rho_max / eta_PAmax.
    p_full = replace(p, tau_DL=1.0, tau_DLsig=0.0)
    net_full = compute_network(p_full, np.full(p.L, rho), rho, R_DL, R_UL)
    per_ap = net_full.ap_pa.total * p.eta_PA_sc / p.L
    assert np.isclose(per_ap, rho / p.eta_PAmax), (per_ap, rho / p.eta_PAmax)
    assert net.ap_pa.total > 0


def test_fixed_rating_floor_grows_with_antennas():
    """The two sizing conventions must not be mixed: they scale differently."""
    rho = BUDGET / 16
    a = compute_network(base_params(M=4, pa_sizing=PASizing.PER_AP_BUDGET),
                        np.full(16, rho), rho, R_DL, R_UL)
    b = compute_network(base_params(M=8, pa_sizing=PASizing.PER_AP_BUDGET),
                        np.full(16, rho), rho, R_DL, R_UL)
    # Per-AP-budget sizing: the PA block is independent of M at fixed rho_max.
    assert np.isclose(a.ap_pa.total, b.ap_pa.total)


# ======================================================================
# Frame consistency with the rate model
# ======================================================================


def test_frame_is_shared_with_the_rate_config():
    """Both packages hold the same frame, so the join is a copy, not a mapping.

    The prelog the rate model applies to the SE must be exactly the fraction of
    the frame that the power model's averaging charges at the data power level;
    otherwise the delivered rate and the consumption describe two different
    frames.
    """
    from config_dmimo import DMIMOConfig

    cfg = DMIMOConfig(L=16, M=4, K=10, Q=16, tau_c=200,
                      tau_DL=0.75, tau_DLsig=1 / 14, tau_ULsig=1 / 14)
    p = DMIMOPowerParams.from_rate_config(cfg)

    assert np.isclose(p.tau_DL + p.tau_UL, 1.0)
    assert np.isclose(p.tau_DL * (1 - p.tau_DLsig) * cfg.xbar_DL, cfg.dl_prelog)
    assert np.isclose(p.tau_UL * (1 - p.tau_ULsig) * cfg.xbar_UL, cfg.ul_prelog)
    # One coherence block, and one effective bandwidth, govern both packages.
    assert np.isclose(p.upsilon_coh, cfg.tau_c)
    assert np.isclose(p.B_tilde, cfg.B_tilde)
    assert (p.K, p.L, p.M, p.B, p.f_c) == (cfg.K, cfg.L, cfg.M, cfg.B, cfg.f_c)


def test_downlink_only_frame_delivers_no_uplink_rate():
    """tau_ULsig = 1 leaves the uplink phase carrying nothing but pilots."""
    from config_dmimo import DMIMOConfig

    cfg = DMIMOConfig(L=16, M=4, K=10, Q=16, tau_c=200, tau_ULsig=1.0)
    p = DMIMOPowerParams.from_rate_config(cfg)
    assert np.isclose(cfg.ul_prelog, 0.0)
    assert np.isclose(p.tau_ULsig, 1.0)
    # The uplink phase still exists in the frame, so its hardware is still on.
    assert np.isclose(p.tau_UL, 0.25)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  [FAIL] {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
