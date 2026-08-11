"""Configuration of the cell-free monograph's *running example* as a DMIMOConfig.

Reproduces the network setup that Bjornson & Sanguinetti, *Foundations of
User-Centric Cell-Free Massive MIMO* (Found. Trends Signal Process., vol. 14,
no. 3-4, 2021; arXiv:2108.02541) define in Section 5.5 (Table 5.1, "Key
parameters of running example") and use for the downlink results of Section 6.6,
so that the rate model in this directory can be benchmarked against published
curves.

Two scenarios are defined, matching the two panels of every downlink figure:

    "A"   L = 400 APs, N = 1  antenna  per AP   (Figures 6.3(a), 6.5(a))
    "B"   L = 100 APs, N = 4  antennas per AP   (Figures 6.3(b), 6.5(b))

both with the same total array size ``M = L * N = 400`` and ``K = 40`` UEs.

The parameters that :class:`DMIMOConfig` already carries are set to the book's
values; the ones the book needs but the dataclass does not have (ASD of the local
scattering model, uplink transmit power, shadow-fading decorrelation distance,
pilot reuse) live in :class:`BookExtras` and are consumed by
:mod:`cellfree_book_channel`. :data:`DEVIATIONS` records, item by item, where the
implementation in this repository cannot follow the book; :func:`deviation_report`
prints it. :data:`BOOK_REFERENCE` holds the per-figure statistics that the
benchmark compares against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from config_dmimo import DMIMOConfig

# ======================================================================
# Book parameters (Table 5.1 and the text of Sections 5.5 / 6.6)
# ======================================================================

BOOK_AREA_SIZE = 1000.0        # 1 km x 1 km coverage area, wrap-around  [m]
BOOK_M_TOT = 400               # total number of antennas M = L * N
BOOK_K = 40                    # UEs (Sec. 5.6 / 6.6)
BOOK_BANDWIDTH = 20e6          # B = 20 MHz
BOOK_NOISE_FIGURE_DB = 7.0     # receiver noise figure -> sigma^2 = -94 dBm
BOOK_NOISE_POWER_DBM = -94.0   # quoted total receiver noise power at APs and UEs
BOOK_P_UL_MAX = 0.1            # max uplink power per UE,  100 mW  [W]
BOOK_RHO_MAX = 0.2             # max downlink power per AP, 200 mW [W]
BOOK_TAU_C = 200               # samples per coherence block
BOOK_TAU_P = 10                # pilot sequence length (Sec. 5.6 / 6.6)
BOOK_PATHLOSS_EXPONENT = 3.67  # alpha
BOOK_PATHLOSS_REF_DB = 30.5    # beta_kl [dB] = -30.5 - 36.7 log10(d/1m) + F_kl
BOOK_SHADOW_STD_DB = 4.0       # sigma_sf
BOOK_SHADOW_DECORR = 9.0       # E{F_kl F_il} = sigma_sf^2 2^{-delta_ki/9 m}
BOOK_HEIGHT_DIFF = 10.0        # APs are deployed 10 m above the UE plane [m]
BOOK_GAIN_AT_1KM_DB = -140.6   # median channel gain at d = 1 km
BOOK_ASD_DEG = 15.0            # sigma_phi = sigma_theta of the local scattering model
BOOK_CARRIER = 2e9             # the large-scale fading model is the 2 GHz 3GPP UMi one

# UE height above ground [m]. The book only fixes the 10 m AP-UE height
# *difference*; splitting it as 11.5 m / 1.5 m keeps the difference exact while
# giving the UEs the conventional 1.5 m height that the Sionna backend needs.
BOOK_UE_HEIGHT = 1.5
BOOK_AP_HEIGHT = BOOK_UE_HEIGHT + BOOK_HEIGHT_DIFF

SCENARIOS: Dict[str, Tuple[int, int]] = {
    "A": (400, 1),   # L = 400 APs, N = 1 antenna  per AP
    "B": (100, 4),   # L = 100 APs, N = 4 antennas per AP
}


@dataclass(frozen=True)
class BookExtras:
    """Book parameters that :class:`DMIMOConfig` has no field for.

    These drive :mod:`cellfree_book_channel` (the analytical channel model of the
    monograph) and the precoder loading; they are carried alongside the
    :class:`DMIMOConfig` rather than inside it so the shared config dataclass is
    left untouched.

    Attributes:
        asd_deg: ASD sigma_phi = sigma_theta of the Gaussian local scattering
            model [deg]. ``None`` selects uncorrelated fading R_kl = beta_kl I_N,
            which the book uses as its reference case (and which is exact for
            scenario "A", where N = 1 makes R_kl the scalar beta_kl).
        p_ul_max: Maximum uplink transmit power per UE [W]. Sets the diagonal
            loading sigma^2 / p of the (R)ZF precoders, which are the dual-uplink
            MMSE combiners.
        shadow_decorr_m: Shadow-fading decorrelation distance [m] in
            ``E{F_kl F_il} = sigma_sf^2 2^{-delta_ki / decorr}``.
        wrap_around: Use the toroidal distance metric of the book's simulation
            methodology (step 3 of Sec. 5.6) instead of plain Euclidean distance.
    """

    asd_deg: float | None = BOOK_ASD_DEG
    p_ul_max: float = BOOK_P_UL_MAX
    shadow_decorr_m: float = BOOK_SHADOW_DECORR
    wrap_around: bool = True


def book_config(scenario: str = "B",
                precoding: str = "RZF",
                operation: str = "centralized",
                *,
                channel_model: str = "rayleigh",
                n_realizations: int = 200,
                seed: int = 0,
                K: int = BOOK_K,
                tau_p: int = BOOK_TAU_P,
                asd_deg: float | None = BOOK_ASD_DEG) -> Tuple[DMIMOConfig, BookExtras]:
    """Build the ``(DMIMOConfig, BookExtras)`` pair for one running-example scenario.

    Args:
        scenario: ``"A"`` (L=400, N=1) or ``"B"`` (L=100, N=4).
        precoding: Any :class:`config_dmimo.PrecodingScheme` value. ``"RZF"`` /
            ``"L-RZF"`` with the loading set below are the perfect-CSI form of
            the book's MMSE / L-MMSE precoders (see :data:`DEVIATIONS`).
        operation: ``"centralized"`` or ``"distributed"``; must match the scheme.
        channel_model: ``"rayleigh"`` selects the book's own correlated-Rayleigh
            model (implemented in :mod:`cellfree_book_channel`); ``"sionna-umi"``
            selects the 3GPP TR 38.901 backend of this repository.
        n_realizations: Monte Carlo drops.
        seed: RNG seed.
        K: Number of UEs (the book sweeps this in Figure 6.7).
        tau_p: Pilot length; only enters through the prelog factor here, since
            the pipeline uses perfect CSI.
        asd_deg: ASD of the local scattering model, or ``None`` for uncorrelated
            fading.

    Returns:
        Tuple ``(cfg, extras)``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(SCENARIOS)}, got {scenario!r}")
    L, N = SCENARIOS[scenario]

    # The centralized heuristic of eq. (6.35) makes rho_k proportional to
    # (sum_l beta_kl)^{-1/2}, i.e. v = -1/2 in the fractional rule of
    # mimo_helpers.power_control; the distributed one of eq. (6.36) makes
    # rho_kl proportional to sqrt(beta_kl), i.e. v = +1/2.
    v = -0.5 if operation == "centralized" else 0.5

    cfg = DMIMOConfig(
        # --- Topology (Table 5.1) -----------------------------------------
        L=L, M=N, K=K,
        Q=1,                       # frequency-flat block fading: one subcarrier
        # --- RF band ------------------------------------------------------
        f_c=BOOK_CARRIER,
        B=BOOK_BANDWIDTH,
        Delta_f=15e3,              # unused with Q = 1; LTE numerology for the record
        # --- Noise --------------------------------------------------------
        noise_figure_dB=BOOK_NOISE_FIGURE_DB,
        # --- Power and precoding ------------------------------------------
        rho_max=BOOK_RHO_MAX,
        precoding=precoding,
        operation=operation,
        power_alloc="fractional",
        v=v,
        # Dual-uplink MMSE loading sigma^2 / p_max: the (R)ZF precoder of
        # mimo_helpers assumes unit-power dual uplink signals, while the book's
        # MMSE precoder is built from the uplink combiner at p_k = p_max.
        rzf_reg=None,              # filled in below, once noise_power is known
        # --- Channel backend ----------------------------------------------
        channel_model=channel_model,
        force_nlos=True,           # the book's model is NLoS correlated Rayleigh
        # --- Propagation / deployment --------------------------------------
        area_size=BOOK_AREA_SIZE,
        ap_height=BOOK_AP_HEIGHT,
        ue_height=BOOK_UE_HEIGHT,
        pathloss_exponent=BOOK_PATHLOSS_EXPONENT,
        pathloss_ref_loss_dB=BOOK_PATHLOSS_REF_DB,
        shadow_std_dB=BOOK_SHADOW_STD_DB,
        ref_distance=1.0,
        min_ap_ue_distance=BOOK_HEIGHT_DIFF,   # the height difference is the floor
        # --- Coherence bookkeeping -----------------------------------------
        tau_c=BOOK_TAU_C,
        tau_p=tau_p,
        # --- Monte Carlo ----------------------------------------------------
        n_realizations=n_realizations,
        seed=seed,
    )
    cfg.rzf_reg = cfg.noise_power / BOOK_P_UL_MAX

    extras = BookExtras(asd_deg=asd_deg)
    return cfg, extras


# ======================================================================
# Reference values read off the published figures
# ======================================================================
#
# The monograph reports its downlink results only as CDF plots, so the numbers
# below were read off the vector figures shipped with the source
# (context_papers/cell_free_book/images/section6/*.pdf). They are graphical
# readings accurate to roughly +/- 0.2 bit/s/Hz, not values quoted in the text,
# and are used as soft targets by the benchmark -- not as exact ground truth.
#
# Only the "(All)" curves are listed: the DCC curves rely on the user-centric
# clustering and pilot assignment of Algorithm 4.1, which this repository does
# not implement (see DEVIATIONS).

@dataclass(frozen=True)
class FigureReference:
    """Statistics of one published CDF curve [bit/s/Hz]."""

    figure: str          # figure number in the monograph
    scenario: str        # "A" or "B"
    curve: str           # legend entry
    se_5pct: float       # 95%-likely SE (5th percentile of the CDF)
    se_median: float     # median SE
    se_max: float        # upper end of the empirical support


BOOK_REFERENCE: Dict[str, FigureReference] = {
    "A-centralized": FigureReference("6.3(a)", "A", "MMSE (All)", 5.3, 6.9, 9.4),
    "B-centralized": FigureReference("6.3(b)", "B", "MMSE (All)", 4.0, 5.7, 9.6),
    "A-distributed": FigureReference("6.5(a)", "A", "L-MMSE (All)", 1.3, 2.6, 6.0),
    "B-distributed": FigureReference("6.5(b)", "B", "L-MMSE (All)", 1.4, 4.3, 11.0),
    "A-MR": FigureReference("6.5(a)", "A", "MR (DCC)", 1.2, 2.2, 3.5),
    "B-MR": FigureReference("6.5(b)", "B", "MR (DCC)", 1.0, 2.0, 3.4),
}


# ======================================================================
# Where this implementation departs from the book
# ======================================================================

@dataclass(frozen=True)
class Deviation:
    """One documented difference between the book's setup and this code."""

    topic: str
    book: str
    here: str
    impact: str


DEVIATIONS: Tuple[Deviation, ...] = (
    Deviation(
        topic="Channel state information",
        book="MMSE channel estimates from tau_p = 10 orthogonal pilots shared by "
             "K = 40 UEs, so estimates carry estimation error and pilot contamination.",
        here="Perfect CSI. mimo_helpers.estimate_channels returns the true channels; "
             "no pilot-based estimator is implemented.",
        impact="Optimistic. This is the largest single deviation: with 4 UEs per pilot "
               "the book's interference includes coherent pilot contamination that is "
               "absent here, so the SE reported here should exceed the book's.",
    ),
    Deviation(
        topic="SE expression",
        book="Hardening bound of Theorem 6.1 / Corollary 6.2: the UE knows only the "
             "mean effective channel E{h_k^H D_k w_k}.",
        here="Instantaneous SINR with the realized effective channel h_k^H w_k, i.e. "
             "the genie-aided SE of Corollary 6.3 (eq. 6.37-6.38).",
        impact="Optimistic, but the book shows the two are nearly indistinguishable for "
               "centralized and LP-MMSE precoding (Figures 6.4 and 6.6); the gap is "
               "large only for MR.",
    ),
    Deviation(
        topic="User-centric clustering (DCC)",
        book="Algorithm 4.1 assigns pilots and forms the dynamic cooperation clusters, "
             "so each AP serves only a subset D_l of the UEs.",
        here="Every AP serves every UE (D_kl = I). This matches the book's '(All)' "
             "curves, which are therefore the ones used as reference.",
        impact="Neutral for the comparison, since the '(All)' curves are the target. "
               "It rules out reproducing the scalable P-MMSE / P-RZF / LP-MMSE curves.",
    ),
    Deviation(
        topic="Precoding schemes",
        book="MMSE, P-MMSE, P-RZF (centralized); L-MMSE, LP-MMSE, MR (distributed).",
        here="RZF / MR (centralized, with the RZF loading set to sigma^2 / p_max) and "
             "L-RZF / MR (distributed). P-MMSE, P-RZF, and LP-MMSE raise "
             "NotImplementedError in mimo_helpers.precoding_directions.",
        impact="Under perfect CSI the estimation-error covariance C_il vanishes, so the "
               "book's MMSE and L-MMSE precoders reduce exactly to the RZF and L-RZF "
               "forms used here. The scalable partial variants cannot be reproduced.",
    ),
    Deviation(
        topic="Downlink power allocation",
        book="eq. (6.35): rho_k proportional to (sum_l beta_kl)^{-1/2} omega_k^{-1/2}, "
             "normalized over the serving clusters; eq. (6.36): "
             "rho_kl = rho_max sqrt(beta_kl) / sum_i sqrt(beta_il).",
        here="mimo_helpers.power_control with v = -1/2 (centralized) and v = +1/2 "
             "(distributed). The distributed rule is identical to eq. (6.36). The "
             "centralized rule keeps the beta^{-1/2} weighting but omits the precoder "
             "norm omega_k, and meets the per-AP budget by a single global scaling "
             "that puts the busiest AP at rho_max instead of the book's cluster-wise "
             "normalization.",
        impact="Distributed: exact. Centralized: same fairness weighting, slightly "
               "different normalization, so per-user powers differ by a common factor "
               "and the omega_k correction is missing.",
    ),
    Deviation(
        topic="Precoder power normalization",
        book="Precoders are normalized by their expected norm E{||wbar_k||^2}, taken "
             "over the fading.",
        here="Normalized by the realized norm within the drop "
             "(mimo_helpers.normalize_precoder).",
        impact="It removes the per-realization power fluctuation. Visible consequence in "
               "scenario A: with N = 1 a local precoder is a scalar per (AP, UE) pair, "
               "which the per-drop normalization divides out, so L-RZF and MR produce "
               "identical signals here while the book's Figure 6.5(a) separates them.",
    ),
    Deviation(
        topic="Carrier frequency / path loss",
        book="beta_kl [dB] = -30.5 - 36.7 log10(d/1m) + F_kl, the 3GPP UMi model of "
             "[LTE2017a, Table B.1.2.1-1], calibrated at 2 GHz.",
        here="Identical when channel_model='rayleigh' (DMIMOConfig.path_loss_dB already "
             "implements this model). With channel_model='sionna-umi' the path loss, "
             "shadowing, and LOS probability come from 3GPP TR 38.901 UMi instead.",
        impact="Rayleigh backend: none. Sionna backend: TR 38.901 UMi NLoS is roughly "
               "6 dB less lossy at 1 km at 2 GHz and uses a 7.82 dB shadowing standard "
               "deviation, so its SE is not directly comparable to the book's curves.",
    ),
    Deviation(
        topic="Spatial correlation",
        book="Gaussian local scattering model, eq. (2.23)-(2.24), with ASD "
             "sigma_phi = sigma_theta = 15 deg on a half-wavelength ULA.",
        here="Implemented in cellfree_book_channel for the Rayleigh backend "
             "(Gauss-Hermite quadrature of eq. 2.23). Irrelevant for scenario A, where "
             "N = 1 makes R_kl the scalar beta_kl. The Sionna backend uses the 38.901 "
             "cluster geometry instead.",
        impact="Rayleigh backend: faithful. Sionna backend: a different (and not "
               "parameterizable by ASD) correlation structure.",
    ),
    Deviation(
        topic="Shadow-fading correlation",
        book="F_kl correlated across UEs at a common AP: "
             "E{F_kl F_il} = 4^2 2^{-delta_ki / 9 m}, independent across APs.",
        here="Implemented in cellfree_book_channel via a Cholesky factor of the K x K "
             "covariance. mimo_helpers.large_scale_fading, used by the rest of the "
             "repository, draws independent shadowing instead.",
        impact="Rayleigh backend: faithful. Independent shadowing would slightly narrow "
               "the SE distribution.",
    ),
    Deviation(
        topic="Wrap-around topology",
        book="Toroidal coverage area: distances are the shortest option across the "
             "edges, so no UE sits at a network edge.",
        here="Implemented in cellfree_book_channel for the Rayleigh backend. The Sionna "
             "backend builds its geometry from 3-D coordinates and cannot wrap.",
        impact="Rayleigh backend: faithful. Without wrap-around the lower tail of the "
               "SE CDF degrades, since edge UEs see fewer nearby APs.",
    ),
    Deviation(
        topic="Frequency selectivity",
        book="Single frequency-flat coherence block; no OFDM dimension.",
        here="Q = 1 subcarrier, which reduces the pipeline to the book's flat block "
             "fading exactly.",
        impact="None.",
    ),
)


def deviation_report() -> str:
    """Formatted list of :data:`DEVIATIONS`."""
    lines = ["Differences between the monograph's running example and this implementation",
             "=" * 78]
    for i, d in enumerate(DEVIATIONS, 1):
        lines.append(f"\n{i}. {d.topic}")
        lines.append(f"   book   : {d.book}")
        lines.append(f"   here   : {d.here}")
        lines.append(f"   impact : {d.impact}")
    return "\n".join(lines)


def parameter_table(cfg: DMIMOConfig, extras: BookExtras) -> str:
    """Side-by-side of the book's Table 5.1 and the configuration built here."""
    rows = [
        ("Network area", "1 km x 1 km", f"{cfg.area_size:.0f} m x {cfg.area_size:.0f} m"),
        ("Network layout", "random, wrap-around",
         f"random, wrap-around={extras.wrap_around}"),
        ("Number of APs L", "400 or 100", f"{cfg.L}"),
        ("Antennas per AP N", "1 or 4", f"{cfg.M}"),
        ("Total antennas M", "400", f"{cfg.M_tot}"),
        ("Number of UEs K", "40", f"{cfg.K}"),
        ("Bandwidth B", "20 MHz", f"{cfg.B/1e6:.0f} MHz"),
        ("Noise figure", "7 dB", f"{cfg.noise_figure_dB:.0f} dB"),
        ("Receiver noise power", "-94 dBm", f"{cfg.noise_power_dBm:.2f} dBm"),
        ("Max UL power per UE", "100 mW", f"{extras.p_ul_max*1e3:.0f} mW"),
        ("Max DL power per AP", "200 mW", f"{cfg.rho_max*1e3:.0f} mW"),
        ("Coherence block tau_c", "200", f"{cfg.tau_c}"),
        ("Pilot length tau_p", "10", f"{cfg.tau_p}"),
        ("DL prelog tau_d/tau_c", "190/200 = 0.95", f"{cfg.dl_prelog:.3f}"),
        ("Channel gain at 1 km", "-140.6 dB",
         f"{-float(cfg.path_loss_dB(1000.0)):.1f} dB"),
        ("Pathloss exponent alpha", "3.67", f"{cfg.pathloss_exponent:.2f}"),
        ("AP-UE height difference", "10 m", f"{cfg.ap_height - cfg.ue_height:.1f} m"),
        ("Shadow fading sigma_sf", "4 dB", f"{cfg.shadow_std_dB:.0f} dB"),
        ("Shadow decorrelation", "9 m", f"{extras.shadow_decorr_m:.0f} m"),
        ("ASD sigma_phi = sigma_theta", "15 deg",
         "uncorrelated" if extras.asd_deg is None else f"{extras.asd_deg:.0f} deg"),
        ("Channel model", "correlated Rayleigh", cfg.channel_model.value),
        ("Precoding / operation", "MMSE / L-MMSE",
         f"{cfg.precoding.value} / {cfg.operation.value}"),
        ("Power control exponent v", "-1/2 (DL cent.), +1/2 (DL dist.)", f"{cfg.v:+.1f}"),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [f"{'Parameter'.ljust(width)} | {'Book (Table 5.1)':<32} | This configuration",
             "-" * (width + 3 + 32 + 3 + 24)]
    lines += [f"{name.ljust(width)} | {book:<32} | {here}" for name, book, here in rows]
    return "\n".join(lines)


if __name__ == "__main__":
    for scen in ("A", "B"):
        c, e = book_config(scen)
        print(f"\n### Scenario {scen}: L = {c.L}, N = {c.M} ###\n")
        print(parameter_table(c, e))
    print()
    print(deviation_report())
