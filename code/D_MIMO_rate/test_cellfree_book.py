"""Standalone benchmark of the rate model against the cell-free monograph.

Recreates the downlink numerical evaluation of Section 6.6 of Bjornson &
Sanguinetti, *Foundations of User-Centric Cell-Free Massive MIMO*
(arXiv:2108.02541), using the running example of Section 5.5, and compares the
result with the published CDF curves. It is a sanity check on the whole pipeline
(:mod:`config_dmimo`, :mod:`mimo_helpers`, and the channel model), not a unit
test of a single function.

The script has three parts:

1. **Mechanics.** Hard pass/fail checks: the configuration reproduces Table 5.1,
   the analytical channel model of :mod:`cellfree_book_channel` satisfies its
   defining identities (wrap-around metric, shadow covariance, ``Tr(R)/N = beta``,
   ``E{h h^H} = R``), and the precoders respect the 200 mW per-AP budget.
2. **Simulation.** Monte Carlo runs of the two running-example scenarios,
   ``L=400, N=1`` and ``L=100, N=4``, in centralized and distributed operation,
   producing the per-UE SE distributions that Figures 6.3 and 6.5 plot.
3. **Comparison.** A table of the 5th-percentile, median, and mean SE against the
   values read off those figures, plus the CDF plots themselves.

Because this repository uses perfect CSI and serves every UE from every AP, the
reference curves are the book's "(All)" ones and the results here are expected to
sit *above* them; :data:`config_cellfree_book.DEVIATIONS` lists every difference,
and ``--deviations`` prints it.

Usage (from this directory, with the project venv):

    python test_cellfree_book.py                 # full run, saves the CDF plots
    python test_cellfree_book.py --realizations 20 --no-plots
    python test_cellfree_book.py --deviations    # print the difference list only
    python test_cellfree_book.py --sionna        # add the TR 38.901 backend

Exits non-zero if any mechanics check fails. The comparison against the book is
reported but never fails the run, since the documented deviations make an exact
match impossible.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

import cellfree_book_channel as bc
import mimo_helpers as mh
from config_cellfree_book import (
    BOOK_GAIN_AT_1KM_DB,
    BOOK_NOISE_POWER_DBM,
    BOOK_REFERENCE,
    BookExtras,
    FigureReference,
    book_config,
    deviation_report,
    parameter_table,
)
from config_dmimo import DMIMOConfig

TOL = 1 + 1e-9   # slack on the per-AP power budget comparison


# ======================================================================
# Pass/fail harness
# ======================================================================


class Checks:
    """Minimal pass/fail harness with a separate, non-fatal soft channel."""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.notes = 0
        self.failures: List[str] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> None:
        ok = bool(ok)
        line = f"  [{'PASS' if ok else 'FAIL'}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(name)

    def soft(self, name: str, ok: bool, detail: str = "") -> None:
        """Report an expectation whose failure is informational, not fatal."""
        line = f"  [{'ok  ' if ok else 'NOTE'}] {name}"
        if detail:
            line += f"  ({detail})"
        print(line)
        if not ok:
            self.notes += 1

    def section(self, title: str) -> None:
        print(f"\n=== {title} ===")

    def summary(self) -> int:
        print(f"\n{self.passed} passed, {self.failed} failed, {self.notes} notes")
        if self.failures:
            print("FAILED: " + ", ".join(self.failures))
        return 1 if self.failed else 0


# ======================================================================
# 1. Mechanics: configuration against Table 5.1
# ======================================================================


def check_parameters(chk: Checks) -> None:
    """Verify that :func:`book_config` reproduces the running example."""
    chk.section("Running-example parameters (Table 5.1)")

    cfg_a, extras = book_config("A")
    cfg_b, _ = book_config("B")

    chk("scenario A: L = 400 APs, N = 1 antenna", (cfg_a.L, cfg_a.M) == (400, 1))
    chk("scenario B: L = 100 APs, N = 4 antennas", (cfg_b.L, cfg_b.M) == (100, 4))
    chk("both scenarios have M = L*N = 400 antennas",
        cfg_a.M_tot == cfg_b.M_tot == 400)
    chk("K = 40 UEs", cfg_a.K == 40)
    chk("coverage area 1 km x 1 km", cfg_a.area_size == 1000.0)
    chk("bandwidth B = 20 MHz", cfg_a.B == 20e6)

    # sigma^2 = -174 dBm/Hz + 10 log10(20 MHz) + 7 dB = -93.98 dBm, quoted as -94.
    chk("receiver noise power = -94 dBm",
        abs(cfg_a.noise_power_dBm - BOOK_NOISE_POWER_DBM) < 0.05,
        f"{cfg_a.noise_power_dBm:.2f} dBm")
    chk("max DL power per AP = 200 mW", cfg_a.rho_max == 0.2)
    chk("max UL power per UE = 100 mW", extras.p_ul_max == 0.1)
    chk("tau_c = 200, tau_p = 10",
        (cfg_a.tau_c, round(cfg_a.tau_p)) == (200, 10))
    chk("DL prelog (tau_c - tau_p)/tau_c = 190/200",
        np.isclose(cfg_a.dl_prelog, 0.95), f"{cfg_a.dl_prelog:.3f}")

    # beta[dB] = -30.5 - 36.7 log10(d/1m): -140.6 dB at d = 1 km.
    gain_1km = -float(cfg_a.path_loss_dB(1000.0))
    chk("median channel gain at 1 km = -140.6 dB",
        abs(gain_1km - BOOK_GAIN_AT_1KM_DB) < 0.05, f"{gain_1km:.2f} dB")
    chk("pathloss exponent alpha = 3.67", cfg_a.pathloss_exponent == 3.67)
    chk("AP-UE height difference = 10 m",
        np.isclose(cfg_a.ap_height - cfg_a.ue_height, 10.0))
    chk("shadow fading sigma_sf = 4 dB", cfg_a.shadow_std_dB == 4.0)
    chk("ASD sigma_phi = sigma_theta = 15 deg", extras.asd_deg == 15.0)
    chk("frequency-flat block fading (Q = 1)", cfg_a.Q == 1)

    # The (R)ZF precoder is the dual-uplink MMSE combiner, whose loading is
    # sigma^2 / p_max rather than sigma^2 for physically scaled channels.
    chk("RZF loading = sigma^2 / p_max",
        np.isclose(cfg_a.rzf_regularization, cfg_a.noise_power / extras.p_ul_max),
        f"{cfg_a.rzf_regularization:.3e}")

    # eq. (6.35) weights rho_k by beta_k^{-1/2}; eq. (6.36) weights rho_kl by
    # beta_kl^{+1/2}.
    cfg_dist, _ = book_config("A", precoding="L-RZF", operation="distributed")
    chk("power-control exponent v = -1/2 centralized, +1/2 distributed",
        (cfg_a.v, cfg_dist.v) == (-0.5, 0.5))


# ======================================================================
# 2. Mechanics: the analytical channel model of the monograph
# ======================================================================


def check_channel_model(chk: Checks) -> None:
    """Verify the defining identities of :mod:`cellfree_book_channel`."""
    chk.section("Book channel model (Sec. 2.5-2.6)")

    # Small configuration so the second-order statistics can be checked by
    # brute-force Monte Carlo in a second.
    cfg, extras = book_config("B", n_realizations=1, K=6)
    cfg.L, cfg.Q = 8, 1
    rng = np.random.default_rng(0)
    ap_pos, ue_pos = mh.draw_positions(cfg, rng)

    # --- Wrap-around metric ------------------------------------------------
    a = np.array([[10.0, 10.0]])
    b = np.array([[990.0, 990.0]])
    d_wrap = np.linalg.norm(bc.wrapped_displacement(a, b, cfg.area_size))
    d_plain = np.linalg.norm(b - a)
    chk("wrap-around shortens the corner-to-corner distance",
        np.isclose(d_wrap, np.sqrt(2) * 20.0) and d_plain > d_wrap,
        f"{d_wrap:.1f} m vs {d_plain:.1f} m")
    chk("wrap-around distance never exceeds half the area diagonal",
        np.max(np.abs(bc.wrapped_displacement(
            ap_pos[:, None, :], ue_pos[None, :, :], cfg.area_size)))
        <= cfg.area_size / 2 + 1e-9)

    # --- Geometry ----------------------------------------------------------
    d_3d, azimuth, elevation = bc.pair_geometry(cfg, ap_pos, ue_pos, True)
    chk("3-D distance floored by the 10 m height difference",
        d_3d.min() >= cfg.ap_height - cfg.ue_height - 1e-9, f"min {d_3d.min():.1f} m")
    chk("elevation is negative (APs above the UE plane)", np.all(elevation < 0))
    chk("azimuth within [-pi, pi]", np.all(np.abs(azimuth) <= np.pi))

    # --- Shadow fading -----------------------------------------------------
    C = bc.shadow_covariance(cfg, ue_pos, extras)
    chk("shadow covariance symmetric", np.allclose(C, C.T))
    chk("shadow variance on the diagonal = sigma_sf^2",
        np.allclose(np.diag(C), cfg.shadow_std_dB ** 2))
    # E{F_kl F_il} = sigma_sf^2 2^{-delta_ki/9 m}.
    d_ue = np.linalg.norm(
        bc.wrapped_displacement(ue_pos[:, None, :], ue_pos[None, :, :], cfg.area_size),
        axis=2)
    chk("shadow covariance = sigma_sf^2 * 2^(-delta/9 m)",
        np.allclose(C, cfg.shadow_std_dB ** 2 * 2.0 ** (-d_ue / extras.shadow_decorr_m)))

    # Empirical shadowing statistics over many draws of the same drop.
    n_mc = 4000
    rng_s = np.random.default_rng(1)
    beta_mc = np.stack([bc.large_scale_fading(cfg, ap_pos, ue_pos, rng_s, extras)
                        for _ in range(n_mc)])                       # (n, L, K)
    F_mc = 10 * np.log10(beta_mc) + cfg.path_loss_dB(d_3d)[None]     # (n, L, K)
    chk("empirical shadowing is zero mean", abs(F_mc.mean()) < 0.15,
        f"{F_mc.mean():+.3f} dB")
    chk("empirical shadowing std = 4 dB", abs(F_mc.std() - 4.0) < 0.15,
        f"{F_mc.std():.2f} dB")
    emp_C = np.einsum("nlk,nli->ki", F_mc, F_mc) / (n_mc * cfg.L)
    chk("empirical UE-UE shadow covariance matches eq. (2.21)",
        np.max(np.abs(emp_C - C)) < 1.2,
        f"max dev {np.max(np.abs(emp_C - C)):.2f} dB^2")

    # --- Spatial correlation ----------------------------------------------
    beta = bc.large_scale_fading(cfg, ap_pos, ue_pos, np.random.default_rng(2), extras)
    R = bc.local_scattering_correlation(cfg, azimuth, elevation, beta, extras.asd_deg)
    chk("R shape (L, K, N, N)", R.shape == (cfg.L, cfg.K, cfg.M, cfg.M), f"{R.shape}")
    chk("R Hermitian", np.allclose(R, np.conj(np.swapaxes(R, -1, -2))))
    eigs = np.linalg.eigvalsh(R)
    chk("R positive semidefinite", eigs.min() >= -1e-9 * eigs.max(),
        f"min/max eig {eigs.min() / eigs.max():.2e}")
    chk("normalization Tr(R_kl)/N = beta_kl (eq. 2.19)",
        np.allclose(np.trace(R, axis1=2, axis2=3).real / cfg.M, beta))
    chk("R is Toeplitz",
        np.allclose(R[:, :, 0, 0], R[:, :, 1, 1]) and np.allclose(R[:, :, 1, 0], R[:, :, 2, 1]))
    # A wider angular spread decorrelates the array: the adjacent-antenna
    # correlation |[R]_{01}|/beta must fall monotonically with the ASD. It does
    # not vanish, since a half-wavelength ULA only decorrelates fully under
    # genuinely isotropic scattering, which a wide Gaussian is not.
    off = [float(np.abs(bc.local_scattering_correlation(
        cfg, azimuth, elevation, beta, asd)[:, :, 0, 1]).mean() / beta.mean())
        for asd in (5.0, 15.0, 30.0, 60.0)]
    chk("adjacent-antenna correlation decreases with the ASD",
        all(a > b for a, b in zip(off, off[1:])),
        "|R_01|/beta at 5/15/30/60 deg: " + "/".join(f"{o:.3f}" for o in off))
    chk("asd_deg=None gives exactly beta * I_N",
        np.allclose(bc.local_scattering_correlation(cfg, azimuth, elevation, beta, None),
                    beta[:, :, None, None] * np.eye(cfg.M)))

    # --- Channel realizations ---------------------------------------------
    cfg.Q = 4000
    H = bc.correlated_rayleigh(cfg, R, np.random.default_rng(3))
    chk("H shape (Q, K, M_tot) and complex128",
        H.shape == (cfg.Q, cfg.K, cfg.M_tot) and H.dtype == np.complex128, f"{H.shape}")
    chk("H is zero mean", np.abs(H.mean()) / np.abs(H).mean() < 0.05)
    l, k, N = 3, 2, cfg.M
    blk = H[:, k, l * N:(l + 1) * N]                                  # (Q, N)
    emp_R = (blk.T @ np.conj(blk)) / cfg.Q     # [m,n] = mean_q h_m h_n^*
    rel = np.max(np.abs(emp_R - R[l, k])) / beta[l, k]
    chk("empirical E{h_kl h_kl^H} matches R_kl", rel < 0.15, f"max rel dev {rel:.3f}")
    chk("empirical beta = mean |h|^2 over the N antennas",
        abs((np.abs(blk) ** 2).mean() / beta[l, k] - 1) < 0.05)

    # --- Reproducibility ---------------------------------------------------
    cfg.Q = 1
    ch = bc.BookChannel(cfg, extras)
    H1, b1 = ch.generate(ap_pos, ue_pos, np.random.default_rng(7))
    H2, b2 = ch.generate(ap_pos, ue_pos, np.random.default_rng(7))
    chk("same seed reproduces the drop", np.array_equal(H1, H2) and np.array_equal(b1, b2))


# ======================================================================
# 3. Mechanics: power budget and the SE pipeline
# ======================================================================


@dataclass
class RunResult:
    """Per-UE SE samples of one Monte Carlo run."""

    key: str                    # "<scenario>-<label>" index into BOOK_REFERENCE
    label: str                  # legend entry, e.g. "RZF (All), perfect CSI"
    cfg: DMIMOConfig
    se: np.ndarray              # (n_realizations, K) per-UE SE [bit/s/Hz]
    ap_power: np.ndarray        # (L,) mean per-AP transmit power [W]

    @property
    def se_5pct(self) -> float:
        return float(np.percentile(self.se, 5))

    @property
    def se_median(self) -> float:
        return float(np.median(self.se))

    @property
    def se_mean(self) -> float:
        return float(self.se.mean())

    @property
    def se_max(self) -> float:
        return float(self.se.max())

    @property
    def sum_se(self) -> float:
        return float(self.se.sum(axis=1).mean())


def draw_position_sequence(cfg: DMIMOConfig, seed: int, n_drops: int) -> List[Tuple]:
    """Pre-draw ``n_drops`` AP/UE layouts, for reuse across channel backends.

    Feeding the same sequence to two runs makes the comparison paired on
    geometry, so any difference between them is attributable to the propagation
    model alone rather than to different random deployments.
    """
    rng = np.random.default_rng(seed)
    return [mh.draw_positions(cfg, rng) for _ in range(n_drops)]


def run_montecarlo(cfg: DMIMOConfig, extras: BookExtras, label: str, key: str,
                   make_channel: Optional[Callable] = None,
                   positions: Optional[List[Tuple]] = None,
                   progress: bool = True) -> RunResult:
    """Monte Carlo the downlink SE for one configuration.

    Mirrors :func:`dl_rate.simulate_downlink` but takes the channel backend as an
    argument, since the analytical model of the monograph lives outside
    :func:`mimo_helpers.build_channel` (whose ``RAYLEIGH`` branch is an
    unimplemented stub).

    Args:
        cfg: System configuration.
        extras: Book parameters not carried by ``cfg``.
        label: Legend entry for the plots.
        key: Index into :data:`config_cellfree_book.BOOK_REFERENCE`.
        make_channel: Factory returning an object with a
            ``generate(ap_pos, ue_pos, rng)`` method; defaults to
            :class:`cellfree_book_channel.BookChannel`.
        positions: Optional pre-drawn AP/UE layouts from
            :func:`draw_position_sequence`; drawn on the fly when omitted.
        progress: Print a one-line progress indicator.

    Returns:
        A :class:`RunResult`.
    """
    rng = np.random.default_rng(cfg.seed)
    channel = (bc.BookChannel(cfg, extras) if make_channel is None else make_channel(cfg))
    progress = progress and sys.stdout.isatty()   # the \r counter needs a terminal

    se = np.empty((cfg.n_realizations, cfg.K))
    power = np.zeros(cfg.L)
    for i in range(cfg.n_realizations):
        if progress and (i % 10 == 0 or i == cfg.n_realizations - 1):
            print(f"\r    {label}: drop {i+1}/{cfg.n_realizations}   ", end="", flush=True)
        ap_pos, ue_pos = (mh.draw_positions(cfg, rng) if positions is None
                          else positions[i])
        H, beta = channel.generate(ap_pos, ue_pos, rng)
        H_hat = mh.estimate_channels(cfg, rng, H)          # perfect CSI
        Wbar = mh.precoding_directions(cfg, H_hat)
        rho = mh.power_control(cfg, beta, Wbar)
        W = mh.normalize_precoder(cfg, Wbar, rho)
        # Per-subcarrier noise, matching dl_rate. The running example is
        # frequency-flat (Q = 1), so this equals cfg.noise_power here.
        sinr = mh.downlink_sinr(mh.effective_channel(H, W), cfg.noise_power_sc)
        se[i] = mh.spectral_efficiency(sinr, cfg.dl_prelog)
        power += mh.ap_powers(cfg, W)
    if progress:
        print()
    return RunResult(key=key, label=label, cfg=cfg, se=se,
                     ap_power=power / cfg.n_realizations)


def check_pipeline(chk: Checks) -> None:
    """End-to-end checks on a small drop: power budget, SINR, SE range."""
    chk.section("Downlink pipeline on the book setup")

    for operation, scheme in (("centralized", "RZF"), ("distributed", "L-RZF"),
                              ("distributed", "MR")):
        cfg, extras = book_config("B", precoding=scheme, operation=operation,
                                  n_realizations=3)
        res = run_montecarlo(cfg, extras, f"{scheme} ({operation})",
                             key="", progress=False)
        tag = f"{scheme}/{operation}"
        chk(f"{tag}: SE finite and positive",
            np.all(np.isfinite(res.se)) and np.all(res.se > 0),
            f"median {res.se_median:.2f} bit/s/Hz")
        chk(f"{tag}: per-AP power within the 200 mW budget",
            np.all(res.ap_power <= cfg.rho_max * TOL),
            f"max {res.ap_power.max()*1e3:.1f} mW")

    # The distributed rule of eq. (6.36) spends each AP's full budget.
    cfg, extras = book_config("B", precoding="L-RZF", operation="distributed",
                              n_realizations=2)
    res = run_montecarlo(cfg, extras, "L-RZF", key="", progress=False)
    chk("distributed rule uses the full per-AP budget with equality",
        np.allclose(res.ap_power, cfg.rho_max),
        f"{res.ap_power.min()*1e3:.1f}..{res.ap_power.max()*1e3:.1f} mW")

    # eq. (6.36) exactly: rho_kl = rho_max sqrt(beta_kl) / sum_i sqrt(beta_il).
    rng = np.random.default_rng(11)
    ap_pos, ue_pos = mh.draw_positions(cfg, rng)
    beta = bc.large_scale_fading(cfg, ap_pos, ue_pos, rng, extras)
    rho = mh.power_control(cfg, beta, None)
    ref = cfg.rho_max * np.sqrt(beta) / np.sqrt(beta).sum(axis=1, keepdims=True)
    chk("distributed power control reproduces eq. (6.36)", np.allclose(rho, ref))

    # With N = 1 a local precoder is a scalar per (AP, UE) pair, and the
    # per-realization normalization of mimo_helpers.normalize_precoder divides it
    # out, so L-RZF and MR become the same transmitted signal. The book keeps
    # them apart because it normalizes by the *expected* precoder norm.
    runs_a = [run_montecarlo(*book_config("A", precoding=s, operation="distributed",
                                          n_realizations=2), label=s, key="",
                             progress=False)
              for s in ("L-RZF", "MR")]
    chk("scenario A (N=1): L-RZF and MR coincide after per-drop normalization",
        np.allclose(runs_a[0].se, runs_a[1].se),
        f"median {runs_a[0].se_median:.2f} vs {runs_a[1].se_median:.2f} bit/s/Hz")

    # eq. (6.35) weights rho_k by (sum_l beta_kl)^{-1/2}.
    cfg_c, _ = book_config("B", precoding="RZF", operation="centralized")
    rho_c = mh.power_control(cfg_c, beta, None)
    beta_k = beta.sum(axis=0)
    ref_c = (cfg_c.L * cfg_c.rho_max) * beta_k ** -0.5 / (beta_k ** -0.5).sum()
    chk("centralized power control follows the beta^(-1/2) weighting of eq. (6.35)",
        np.allclose(rho_c, ref_c))


# ======================================================================
# 4. Simulation and comparison against the published figures
# ======================================================================


SCHEME_SPECS = [
    ("centralized", "RZF", "centralized", "MMSE-equivalent RZF (All)"),
    ("distributed", "L-RZF", "distributed", "L-MMSE-equivalent L-RZF (All)"),
    ("MR", "MR", "distributed", "MR (All)"),
]


def scenario_runs(scenario: str, n_realizations: int, seed: int,
                  channel_model: str = "rayleigh",
                  positions: Optional[List[Tuple]] = None) -> List[RunResult]:
    """Run the three precoding schemes the book plots for one scenario.

    ``channel_model='rayleigh'`` uses the book's own model from
    :mod:`cellfree_book_channel`; ``'sionna-umi'`` swaps in the TR 38.901 backend
    of this repository and leaves every other parameter untouched.
    """
    make_channel = None if channel_model == "rayleigh" else mh.build_channel
    runs = []
    for suffix, scheme, operation, label in SCHEME_SPECS:
        cfg, extras = book_config(scenario, precoding=scheme, operation=operation,
                                  channel_model=channel_model,
                                  n_realizations=n_realizations, seed=seed)
        runs.append(run_montecarlo(cfg, extras, label, key=f"{scenario}-{suffix}",
                                   make_channel=make_channel, positions=positions))
    return runs


def comparison_table(runs: List[RunResult]) -> str:
    """Table of simulated versus published SE statistics [bit/s/Hz]."""
    head = (f"{'scenario':<9} {'scheme':<30} "
            f"{'5% here':>8} {'5% book':>8} {'med here':>9} {'med book':>9} "
            f"{'mean':>6} {'sum SE':>8}  reference")
    lines = [head, "-" * len(head)]
    for r in runs:
        ref: Optional[FigureReference] = BOOK_REFERENCE.get(r.key)
        scen = r.key.split("-")[0]
        if ref is None:
            lines.append(f"{scen:<9} {r.label:<30} {r.se_5pct:8.2f} {'-':>8} "
                         f"{r.se_median:9.2f} {'-':>9} {r.se_mean:6.2f} "
                         f"{r.sum_se:8.1f}  (no published curve)")
        else:
            lines.append(f"{scen:<9} {r.label:<30} {r.se_5pct:8.2f} {ref.se_5pct:8.2f} "
                         f"{r.se_median:9.2f} {ref.se_median:9.2f} {r.se_mean:6.2f} "
                         f"{r.sum_se:8.1f}  Fig. {ref.figure}, {ref.curve}")
    return "\n".join(lines)


def compare_with_book(chk: Checks, runs: List[RunResult]) -> None:
    """Soft checks of the simulated SE against the published curves."""
    chk.section("Comparison with the published CDFs (soft checks)")
    print(comparison_table(runs))
    print()
    for r in runs:
        ref = BOOK_REFERENCE.get(r.key)
        if ref is None:
            continue
        tag = f"Fig. {ref.figure} {ref.curve} vs {r.label}"
        # Perfect CSI and the genie-aided SE both push the result upwards, so the
        # simulated median should sit at or above the published one.
        chk.soft(f"{tag}: median at or above the book",
                 r.se_median >= ref.se_median - 0.3,
                 f"{r.se_median:.2f} vs {ref.se_median:.2f} bit/s/Hz")
        # ... but not by more than a factor of two, which would point at a
        # modelling error rather than the documented deviations.
        chk.soft(f"{tag}: median within a factor of two",
                 r.se_median <= 2.0 * ref.se_median,
                 f"ratio {r.se_median / ref.se_median:.2f}")
        chk.soft(f"{tag}: 95%-likely SE at or above the book",
                 r.se_5pct >= ref.se_5pct - 0.3,
                 f"{r.se_5pct:.2f} vs {ref.se_5pct:.2f} bit/s/Hz")


def plot_cdfs(runs: List[RunResult], scenario: str, save_path: Optional[str],
              show: bool) -> None:
    """Overlay the per-UE SE CDFs of one scenario, with the book's medians marked."""
    import matplotlib.pyplot as plt

    cfg = runs[0].cfg
    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Dashes keep the curves distinguishable in scenario A, where N = 1 makes
    # L-RZF and MR coincide exactly (see DEVIATIONS, "Precoder normalization").
    styles = ["-", "-", "--"]
    for i, r in enumerate(runs):
        se = np.sort(r.se.ravel())
        cdf = np.arange(1, se.size + 1) / se.size
        line, = ax.plot(se, cdf, lw=2, ls=styles[i % len(styles)], label=r.label)
        ref = BOOK_REFERENCE.get(r.key)
        if ref is not None:
            # Stagger the labels vertically so neighbouring medians do not collide.
            y = 0.30 + 0.16 * i
            ax.axvline(ref.se_median, color=line.get_color(), ls=":", lw=1.2)
            ax.annotate(f"book {ref.curve}: median {ref.se_median:.1f}",
                        (ref.se_median, y), textcoords="offset points",
                        xytext=(5, 0), fontsize=7, color=line.get_color())

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Spectral efficiency [bit/s/Hz]")
    ax.set_ylabel("CDF")
    ax.set_title(f"Downlink SE per UE, running example: L = {cfg.L}, N = {cfg.M}, "
                 f"K = {cfg.K}\n(perfect CSI, all APs serve all UEs; "
                 "dotted = median of the book's '(All)' curve)", fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  wrote {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def large_scale_gain_table(scenario: str, seed: int) -> str:
    """Compare the large-scale gains of both channel models on one shared drop.

    The two backends see the identical AP/UE layout, so the difference in the
    ``beta_kl`` distributions is exactly what the propagation model contributes,
    before any precoding or power control enters. This is the quantity that
    explains most of the SE gap between them.
    """
    cfg_book, extras = book_config(scenario, channel_model="rayleigh", seed=seed)
    cfg_umi, _ = book_config(scenario, channel_model="sionna-umi", seed=seed)
    ap_pos, ue_pos = draw_position_sequence(cfg_book, seed, 1)[0]

    _, beta_book = bc.BookChannel(cfg_book, extras).generate(
        ap_pos, ue_pos, np.random.default_rng(seed))
    _, beta_umi = mh.build_channel(cfg_umi).generate(
        ap_pos, ue_pos, np.random.default_rng(seed))

    rows = []
    for name, beta in (("book (-30.5 - 36.7 log10 d)", beta_book),
                       ("38.901 UMi (forced NLoS)", beta_umi)):
        b = 10 * np.log10(beta)
        # The gain a UE actually rides on is the one to its serving APs, so the
        # best-AP gain matters far more than the all-link average.
        best = b.max(axis=0)
        rows.append((name, np.median(b), b.std(), np.percentile(b, 90),
                     np.median(best)))

    lines = [f"{'large-scale gain beta_kl [dB]':<30} {'median':>8} {'std':>7} "
             f"{'90th pct':>9} {'median best-AP':>15}",
             "-" * 72]
    for name, med, std, p90, best in rows:
        lines.append(f"{name:<30} {med:8.1f} {std:7.1f} {p90:9.1f} {best:15.1f}")
    lines.append(f"{'difference (UMi - book)':<30} "
                 f"{rows[1][1]-rows[0][1]:+8.1f} {rows[1][2]-rows[0][2]:+7.1f} "
                 f"{rows[1][3]-rows[0][3]:+9.1f} {rows[1][4]-rows[0][4]:+15.1f}")
    return "\n".join(lines)


def channel_model_table(book_runs: List[RunResult],
                        umi_runs: List[RunResult]) -> str:
    """Side-by-side SE statistics of the two channel backends [bit/s/Hz]."""
    head = (f"{'scheme':<30} {'5% book':>8} {'5% UMi':>8} {'delta':>7}  "
            f"{'med book':>9} {'med UMi':>8} {'delta':>7} {'ratio':>6}")
    lines = [head, "-" * len(head)]
    for b, u in zip(book_runs, umi_runs):
        lines.append(
            f"{b.label:<30} {b.se_5pct:8.2f} {u.se_5pct:8.2f} "
            f"{u.se_5pct - b.se_5pct:+7.2f}  {b.se_median:9.2f} {u.se_median:8.2f} "
            f"{u.se_median - b.se_median:+7.2f} {u.se_median / b.se_median:6.2f}")
    return "\n".join(lines)


def channel_model_comparison(chk: Checks, scenario: str, n_realizations: int,
                             seed: int, save_path: Optional[str] = None,
                             show: bool = False):
    """Run one scenario on both channel backends and quantify the difference.

    Everything except the propagation model is held fixed, and both runs are fed
    the same pre-drawn AP/UE layouts, so the difference is attributable to
    3GPP TR 38.901 UMi replacing the monograph's ``-30.5 - 36.7 log10(d)`` model.
    The two are not interchangeable: 38.901 UMi is a different (and less lossy)
    propagation model, it applies its own 7.82 dB shadowing rather than the
    monograph's 4 dB, and it builds its geometry from 3-D coordinates so the
    wrap-around topology is lost.

    Returns:
        Tuple ``(book_runs, umi_runs)``, or ``None`` if Sionna is unavailable.
    """
    chk.section(f"Channel model: book vs 3GPP TR 38.901 UMi (scenario {scenario}, "
                f"{n_realizations} drops)")
    try:
        import sionna  # noqa: F401
    except ModuleNotFoundError as exc:
        chk.soft("Sionna available", False, f"skipped: {exc.name} not installed")
        return None

    cfg, _ = book_config(scenario)
    positions = draw_position_sequence(cfg, seed, n_realizations)
    book_runs = scenario_runs(scenario, n_realizations, seed,
                              channel_model="rayleigh", positions=positions)
    umi_runs = scenario_runs(scenario, n_realizations, seed,
                             channel_model="sionna-umi", positions=positions)

    print(large_scale_gain_table(scenario, seed))
    print()
    print(channel_model_table(book_runs, umi_runs))
    print()

    # 38.901 UMi is the less lossy model, so it should raise the SE across the
    # board; anything else would point at a configuration error rather than a
    # genuine modelling difference.
    for b, u in zip(book_runs, umi_runs):
        chk.soft(f"{b.label}: UMi at or above the book model",
                 u.se_median >= b.se_median,
                 f"{u.se_median:.2f} vs {b.se_median:.2f} bit/s/Hz")

    if save_path or show:
        plot_channel_model_cdfs(book_runs, umi_runs, scenario, save_path, show)
    return book_runs, umi_runs


def plot_channel_model_cdfs(book_runs: List[RunResult], umi_runs: List[RunResult],
                            scenario: str, save_path: Optional[str],
                            show: bool) -> None:
    """Overlay the SE CDFs of both channel backends, one colour per scheme."""
    import matplotlib.pyplot as plt

    cfg = book_runs[0].cfg
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (b, u) in enumerate(zip(book_runs, umi_runs)):
        color = f"C{i}"
        for run, style, tag in ((b, "-", "book model"), (u, "--", "38.901 UMi")):
            se = np.sort(run.se.ravel())
            ax.plot(se, np.arange(1, se.size + 1) / se.size, lw=2, ls=style,
                    color=color, label=f"{run.label} - {tag}")

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Spectral efficiency [bit/s/Hz]")
    ax.set_ylabel("CDF")
    ax.set_title(f"Channel model comparison, running example: L = {cfg.L}, "
                 f"N = {cfg.M}, K = {cfg.K}\n"
                 "solid = monograph's correlated Rayleigh, dashed = 3GPP TR 38.901 UMi",
                 fontsize=9)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  wrote {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ======================================================================
# Entry point
# ======================================================================


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--realizations", type=int, default=100,
                        help="Monte Carlo drops per curve (default 100)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument("--scenarios", default="A,B",
                        help="comma-separated scenarios to run (default A,B)")
    parser.add_argument("--no-plots", action="store_true", help="skip the CDF figures")
    parser.add_argument("--show", action="store_true", help="display the figures")
    parser.add_argument("--sionna", action="store_true",
                        help="add the paired book-model vs TR 38.901 UMi comparison")
    parser.add_argument("--sionna-scenario", default="B",
                        help="scenario for the channel-model comparison (default B)")
    parser.add_argument("--sionna-realizations", type=int, default=15,
                        help="drops for the channel-model comparison "
                             "(default 15; the Sionna backend is ~6x slower per drop)")
    parser.add_argument("--deviations", action="store_true",
                        help="print the book-vs-code difference list and exit")
    parser.add_argument("--parameters", action="store_true",
                        help="print the parameter table and exit")
    args = parser.parse_args(argv)

    if args.deviations:
        print(deviation_report())
        return 0
    if args.parameters:
        for scen in ("A", "B"):
            cfg, extras = book_config(scen)
            print(f"\n### Scenario {scen} ###\n")
            print(parameter_table(cfg, extras))
        return 0

    chk = Checks()
    check_parameters(chk)
    check_channel_model(chk)
    check_pipeline(chk)

    all_runs: List[RunResult] = []
    for scenario in [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]:
        chk.section(f"Scenario {scenario}: Monte Carlo over "
                    f"{args.realizations} drops")
        runs = scenario_runs(scenario, args.realizations, args.seed)
        all_runs += runs
        if not args.no_plots:
            plot_cdfs(runs, scenario,
                      save_path=f"cellfree_book_cdf_scenario{scenario}.png",
                      show=args.show)

    compare_with_book(chk, all_runs)

    if args.sionna:
        scen = args.sionna_scenario.strip().upper()
        channel_model_comparison(
            chk, scen, args.sionna_realizations, args.seed,
            save_path=(None if args.no_plots
                       else f"cellfree_book_channel_scenario{scen}.png"),
            show=args.show)

    print("\n" + deviation_report())
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
