"""Configuration for the distributed massive MIMO (cell-free) rate model.

A single dataclass, :class:`DMIMOConfig`, gathers every parameter of the
downlink system model of ``sections/dmimo_sysmodel.tex`` and of the uplink
system model of ``sections/dmimo_ul_sysmodel.tex`` so that the rate scripts
(``dl_rate.py``, ``ul_rate.py``) share one source of truth. The symbols mirror
the manuscript:

    L        number of access points (APs)
    M        antennas per AP            (total array size M_tot = L * M)
    K        single-antenna users
    Q        OFDM data subcarriers
    rho_max  maximum DL transmit power per AP        [W]
    p_max    maximum UL transmit power per user      [W]
    sigma^2  receiver noise power (derived from B and the noise figure)
    v        DL fractional power-control exponent  (eq. fractional-power)
    v_ul     UL fractional power-control exponent  (eq. ul-fractional-power)
    lambda   RZF loading term                   (eq. zf-precoder / RZF)

Parameters are grouped into topology, RF band, noise, downlink power/precoding,
uplink power/combining, propagation, frame, and Monte Carlo blocks. Quantities
that are fully determined by these inputs (M_tot, wavelength, noise power, the
prelog factors, ...) are exposed as read-only properties rather than stored, so
they cannot drift out of sync.

The frame is described in the same fractions as the power model
(``tau_DL``, ``tau_DLsig``, ``tau_ULsig``, ``xbar_DL``, ``xbar_UL``) rather than
in a coherence-block sample split, so no translation stands between the two
packages: the spectral-efficiency prelog is the data fraction
``tau_i (1 - tau_i,sig) xbar_i`` of the frame, which is exactly the fraction the
power model's frame averaging charges at the data power level.

Uplink and downlink share the propagation channel by TDD reciprocity, so the
topology, band, noise, and channel blocks serve both directions; only the
transmit power, the spatial processing, and the share of the coherence block
differ between them.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

SPEED_OF_LIGHT = 3e8                     # [m/s]
THERMAL_NOISE_DENSITY_DBM_HZ = -174.0    # k_B * T at T = 290 K [dBm/Hz]


def db_to_lin(x_db: float) -> float:
    """Convert a dB quantity to linear scale."""
    return 10 ** (x_db / 10)


def lin_to_db(x: float) -> float:
    """Convert a linear quantity to dB."""
    return 10 * math.log10(x)


def dbm_to_watt(p_dbm: float) -> float:
    """Convert power in dBm to watts."""
    return 10 ** ((p_dbm - 30) / 10)


def watt_to_dbm(p_w: float) -> float:
    """Convert power in watts to dBm."""
    return 10 * math.log10(p_w) + 30


class PrecodingScheme(str, Enum):
    """Downlink transmit precoding schemes (Section: Transmit Precoding Schemes).

    Centralized (jointly designed at the CPU): ``MMSE``, ``P_MMSE``, ``P_RZF``,
    ``RZF``, ``ZF``. Distributed (formed locally per AP): ``L_MMSE``, ``L_RZF``,
    ``LP_MMSE``, ``MR``.
    """

    MR = "MR"            # maximum ratio (distributed, closed-form SINR)
    ZF = "ZF"            # zero forcing (centralized; used by the power model)
    RZF = "RZF"          # regularized zero forcing (centralized)
    MMSE = "MMSE"        # nearly optimal, unscalable (centralized)
    P_MMSE = "P-MMSE"    # scalable partial MMSE (centralized)
    P_RZF = "P-RZF"      # scalable partial RZF (centralized)
    L_MMSE = "L-MMSE"    # locally optimal, unscalable (distributed)
    L_RZF = "L-RZF"      # local regularized ZF (distributed)
    LP_MMSE = "LP-MMSE"  # scalable local partial MMSE (distributed)


class CombiningScheme(str, Enum):
    """Uplink receive combining schemes (Section: Combiner and Power Control Designs).

    Centralized (the CPU holds the collective ``y[q]`` and combines over all
    ``LM`` antennas jointly): ``MMSE``, ``RZF``, ``ZF``. Distributed (each AP
    combines its own observation and forwards one scalar per user, which the CPU
    then fuses, see :class:`FusionRule`): ``MR``, ``L_MMSE``, ``L_RZF``.

    The split mirrors :class:`PrecodingScheme` because uplink combining and
    downlink precoding are duals under TDD reciprocity, with one difference: the
    uplink loading is physically determined by the transmit powers
    (``sigma^2 P^{-1}``, eq. ul-centralized-rzf) rather than heuristic. The
    scalable partial variants (P-MMSE, P-RZF, LP-MMSE) need the user-centric
    clustering that this package does not implement and are therefore absent.
    """

    MR = "MR"            # maximum ratio combining (distributed), v_kl = h_kl
    ZF = "ZF"            # zero forcing (centralized); used by the power model
    RZF = "RZF"          # regularized ZF, heuristic loading (centralized)
    MMSE = "MMSE"        # MMSE combining, loading sigma^2 P^{-1} (centralized)
    L_MMSE = "L-MMSE"    # locally optimal per-AP combiner (distributed)
    L_RZF = "L-RZF"      # local regularized ZF, heuristic loading (distributed)


#: Uplink combiner that an unset ``DMIMOConfig.combining`` inherits from the
#: downlink precoder. Under TDD reciprocity each precoder has a combiner of the
#: same name, which is also at the same cooperation level, so this default keeps
#: a downlink-only configuration valid without it having to mention the uplink.
#: The scalable partial precoders have no implemented combiner counterpart (they
#: need the user-centric clustering) and fall back to their unclustered parent.
DUAL_COMBINER = {
    PrecodingScheme.MR: CombiningScheme.MR,
    PrecodingScheme.ZF: CombiningScheme.ZF,
    PrecodingScheme.RZF: CombiningScheme.RZF,
    PrecodingScheme.MMSE: CombiningScheme.MMSE,
    PrecodingScheme.L_MMSE: CombiningScheme.L_MMSE,
    PrecodingScheme.L_RZF: CombiningScheme.L_RZF,
    PrecodingScheme.P_MMSE: CombiningScheme.MMSE,
    PrecodingScheme.P_RZF: CombiningScheme.RZF,
    PrecodingScheme.LP_MMSE: CombiningScheme.L_MMSE,
}


class OperationMode(str, Enum):
    """Where the precoding / combining is computed."""

    CENTRALIZED = "centralized"  # CPU designs all directions from global CSI
    DISTRIBUTED = "distributed"  # each AP designs its directions from local CSI


class FusionRule(str, Enum):
    """How the CPU fuses the ``L`` per-AP soft estimates in distributed uplink
    operation (eq. ul-lsfd).

    ``EQUAL`` simply adds them, ``a_kl = 1``, and needs nothing at the CPU beyond
    the streams. ``LSFD`` weights them with the large-scale-fading decoding
    coefficients of eq. ul-lsfd-weights, which are statistical and are therefore
    refreshed once per large-scale fading realization rather than per coherence
    block. Both are the same collective combiner ``v_k`` with ``l``-th block
    ``a_kl v_kl``, so the SINR expression is unchanged. The rule is meaningless
    under centralized operation, where the CPU combines the raw samples.
    """

    EQUAL = "equal"  # a_kl = 1: unweighted sum of the local estimates
    LSFD = "lsfd"    # large-scale fading decoding weights (statistical)


class SEBound(str, Enum):
    """Which achievable-rate expression the uplink spectral efficiency uses.

    ``INSTANTANEOUS`` evaluates eq. ul-sinr with the realized effective channel
    ``v_k^H h_k``, which assumes it is known wherever the decoding happens. This
    is the genie-aided expression and is what the downlink pipeline uses, so it
    is the consistent choice when the two directions are compared. ``UATF`` is
    the use-and-then-forget bound, in which only the mean effective channel is
    treated as useful gain and its fluctuation becomes extra interference. UatF
    is the rigorous achievable rate when the fusion weights are statistical
    (local operation with :class:`FusionRule`), where the instantaneous
    expression is optimistic by the amount of residual channel hardening.
    """

    INSTANTANEOUS = "instantaneous"  # genie-aided, eq. ul-sinr as written
    UATF = "uatf"                    # use-and-then-forget lower bound


class PowerControlScheme(str, Enum):
    """Power-allocation heuristic (Section: Power Control).

    Downlink: ``EQUAL`` and ``FRACTIONAL`` produce a per-user coefficient and are
    only meaningful for centralized operation; ``FRACTIONAL`` is also the (only)
    local rule, where it produces a per-AP-per-user coefficient. ``EQUAL`` is the
    ``v = 0`` special case of ``FRACTIONAL``.

    Uplink: each user has its own budget rather than a share of a common one, so
    ``EQUAL`` means full power ``p_k = p_max`` for every user and ``FRACTIONAL``
    is eq. ul-fractional-power, normalized by the maximum instead of the sum.
    ``EQUAL`` is again the ``v_ul = 0`` special case. Both are valid in either
    operation mode, since no budget has to be shared.
    """

    EQUAL = "equal"            # DL: rho_k = P_tot / K (centralized); UL: p_k = p_max
    FRACTIONAL = "fractional"  # large-scale-fading fractional allocation (beta^v)


class APPlacement(str, Enum):
    """How the APs are positioned in the coverage area.

    ``RANDOM`` drops them uniformly, which is the cell-free convention and the
    default. ``CENTER`` puts every AP at the area centre, which is only
    meaningful for ``L = 1``: it is the co-located baseline, a single site
    covering the area from the middle rather than from a random corner. Using
    ``RANDOM`` for a one-site deployment would penalize it for an edge position
    that a real single-cell deployment would never choose.
    """

    RANDOM = "random"  # uniform over the coverage area (cell-free default)
    CENTER = "center"  # all APs at the area centre (co-located baseline, L = 1)


class ChannelModel(str, Enum):
    """Backend used to draw channel realizations.

    ``SIONNA_UMI`` generates realistic 3GPP TR 38.901 Urban Microcell channels
    through Sionna's geometry-based stochastic model (path loss, shadowing, and
    spatial correlation all follow 38.901). ``RAYLEIGH`` uses the tractable
    analytical correlated-Rayleigh model built from :meth:`DMIMOConfig.path_loss_dB`
    and the local scattering correlation matrices.
    """

    SIONNA_UMI = "sionna-umi"  # 3GPP TR 38.901 UMi via Sionna (requires sionna)
    RAYLEIGH = "rayleigh"      # analytical correlated-Rayleigh (numpy only)


@dataclass
class DMIMOConfig:
    """Parameters of the distributed massive MIMO system model, both directions.

    Defaults describe an upper mid-band (FR3) cell-free deployment consistent
    with the accompanying power model: a large number of few-antenna APs jointly
    serving several users over a wide bandwidth, with far more distributed
    antennas than users so that zero-forcing is feasible. They reproduce the
    headline topology of the manuscript's parameter table, ``L * M = 128``
    antennas at ``L = 32`` APs serving ``K = 20`` users, with the total transmit
    budget ``P_TX = L * M * P_max = 12.8 W`` shared as ``rho_max = P_TX / L``.

    Both directions carry data: the frame of the system model gives the uplink a
    data phase of ``tau_UL (1 - tau_ULsig)``, so ``ul_prelog > 0``. Setting
    ``tau_ULsig = 1`` is the degenerate downlink-only frame in which the uplink
    phase carries nothing but pilots and the uplink rate vanishes.
    """

    # --- Topology ---------------------------------------------------------
    L: int = 32                 # Number of APs (~35 m spacing over the 200 m area)
    M: int = 4                  # Antennas per AP
    K: int = 20                 # Single-antenna users
    # Subcarriers *evaluated* by the Monte Carlo. The band carries
    # B / Delta_f = 3333 of them (the manuscript's table quotes Q = 3000); the
    # SINR expressions are per subcarrier and every quantity that meets a noise
    # power is divided by this same Q (see noise_power_sc), so a subset samples
    # the band without biasing the SE. It is a runtime cost knob, not a physical
    # parameter: raise it towards the full count for the final figures.
    Q: int = 64                 # OFDM data subcarriers evaluated

    # --- RF band ----------------------------------------------------------
    f_c: float = 10e9           # Carrier frequency [Hz]
    B: float = 400e6            # Bandwidth [Hz]
    Delta_f: float = 120e3      # Subcarrier spacing [Hz]
    B_tilde_factor: float = 0.9  # Effective-bandwidth factor, B_tilde = factor * B

    # --- Noise ------------------------------------------------------------
    noise_figure_dB: float = 8.0  # Receiver noise figure [dB]

    # --- Downlink power and precoding --------------------------------------
    # P_TX / L with P_TX = L * M * P_max at the FR3 amplifier rating
    # P_max = 0.1 W, i.e. 12.8 W / 32 APs (see Remark "Amplifier sizing").
    rho_max: float = 0.4        # Max DL transmit power per AP (26 dBm) [W]
    precoding: PrecodingScheme = PrecodingScheme.ZF
    operation: OperationMode = OperationMode.CENTRALIZED
    power_alloc: PowerControlScheme = PowerControlScheme.FRACTIONAL
    v: float = 0.5              # DL fractional power-control exponent
    rzf_reg: Optional[float] = None  # RZF loading lambda; None -> sigma^2 (see property)

    # --- Uplink power and combining ----------------------------------------
    # The cooperation level ``operation`` is shared with the downlink: a TDD
    # network with a given functional split uses it in both directions.
    p_max: float = 0.1          # Max UL transmit power per user (20 dBm handset) [W]
    combining: Optional[CombiningScheme] = None  # None -> dual of `precoding` (see property)
    fusion: FusionRule = FusionRule.EQUAL
    ul_power_alloc: PowerControlScheme = PowerControlScheme.FRACTIONAL
    v_ul: float = -0.5          # UL fractional power-control exponent (v<0 favours weak users)
    ul_rzf_reg: Optional[float] = None  # UL RZF loading; None -> sigma^2 / p_max (see property)
    ul_se_bound: SEBound = SEBound.INSTANTANEOUS

    # --- Channel model backend --------------------------------------------
    channel_model: ChannelModel = ChannelModel.SIONNA_UMI
    force_nlos: bool = False     # True: force all links NLOS; False: 38.901 distance-dependent LOS probability (LOS paths possible)
    o2i_model: str = "low"      # 38.901 outdoor-to-indoor model; UEs are outdoor here
    antenna_pattern: str = "omni"  # AP element pattern ("omni" or "38.901")

    # --- Propagation / deployment -----------------------------------------
    ap_placement: APPlacement = APPlacement.RANDOM  # AP drop rule; UEs are always uniform
    area_size: float = 200.0    # Square coverage-area side [m] (wrap-around); dense FR3 hotspot
    ap_height: float = 10.0     # AP height [m]
    ue_height: float = 1.5      # UE height [m]
    # The following four parameters drive the analytical RAYLEIGH channel only;
    # the SIONNA_UMI backend takes its path loss and shadowing from 38.901.
    pathloss_exponent: float = 3.67  # Log-distance path-loss exponent (= 36.7/10)
    pathloss_ref_loss_dB: float = 30.5  # Path loss at d0 (3GPP UMi, 2 GHz) [dB]
    shadow_std_dB: float = 4.0  # Log-normal shadowing std [dB]
    ref_distance: float = 1.0   # Path-loss reference distance d0 [m]
    min_ap_ue_distance: float = 1.0  # Floor on AP-UE distance [m]

    # --- Frame structure (eq. load, eq. frame_data) ------------------------
    # The frame is stated directly in fractions, exactly as in the system model
    # and in ``fr3_power.PowerParams``, so one frame describes the rate model and
    # the power model: a TDD frame of length T_f split into a downlink phase
    # tau_DL and an uplink phase tau_UL = 1 - tau_DL, each spending a fraction
    # tau_*sig on reference signalling and carrying data on the fraction xbar_*
    # of what remains, the rest being micro-sleep (DTX).
    tau_DL: float = 0.75        # DL share of the frame
    tau_DLsig: float = 1 / 14   # Signalling share of the DL phase
    tau_ULsig: float = 1 / 14   # Signalling share of the UL phase
    xbar_DL: float = 1.0        # DL average physical resource load, in [0, 1]
    xbar_UL: float = 1.0        # UL average physical resource load, in [0, 1]
    # The frame is taken to span one coherence block, so this single number is
    # both the fading timescale of the rate model and the precoder amortisation
    # ``upsilon_coh`` of the power model.
    tau_c: int = 200            # Coherence block length [samples]

    # --- Monte Carlo ------------------------------------------------------
    n_realizations: int = 100   # Channel realizations averaged per SE point
    seed: int = 0               # RNG seed for reproducibility

    def __post_init__(self) -> None:
        # Accept plain strings for the enum fields (e.g. from a config file).
        self.precoding = PrecodingScheme(self.precoding)
        # An unset combiner takes the TDD dual of the precoder, which keeps the
        # two directions at the same cooperation level by construction and lets
        # a downlink-only configuration stay silent about the uplink.
        self.combining = (DUAL_COMBINER[self.precoding] if self.combining is None
                          else CombiningScheme(self.combining))
        self.operation = OperationMode(self.operation)
        self.power_alloc = PowerControlScheme(self.power_alloc)
        self.ul_power_alloc = PowerControlScheme(self.ul_power_alloc)
        self.fusion = FusionRule(self.fusion)
        self.ul_se_bound = SEBound(self.ul_se_bound)
        self.channel_model = ChannelModel(self.channel_model)
        self.ap_placement = APPlacement(self.ap_placement)

        if self.ap_placement is APPlacement.CENTER and self.L > 1:
            warnings.warn(
                f"ap_placement=CENTER with L={self.L}: all {self.L} APs are stacked at the "
                "area centre, which is a co-located array described as L separate APs, not a "
                "distributed deployment. This is only intended for the L=1 baseline.",
                stacklevel=2,
            )

        for name in ("L", "M", "K", "Q", "tau_c", "n_realizations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {getattr(self, name)}")
        for name in ("f_c", "B", "Delta_f", "rho_max", "p_max", "area_size",
                     "B_tilde_factor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("tau_DL", "tau_DLsig", "tau_ULsig", "xbar_DL", "xbar_UL"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1], got {getattr(self, name)}")

        if self.tau_p < self.K:
            warnings.warn(
                f"the frame leaves {self.tau_p:.1f} signalling samples per coherence block "
                f"but K={self.K} orthogonal pilots are needed, so a pilot-based system would "
                "suffer contamination here. Perfect CSI is assumed, so this affects nothing "
                "computed by this package; it bounds how honest that assumption is.",
                stacklevel=2,
            )
        if self.K > self.M_tot:
            warnings.warn(
                f"K={self.K} exceeds the total number of antennas M_tot={self.M_tot}; "
                "zero-forcing/MMSE precoding is rank-deficient in this regime.",
                stacklevel=2,
            )
        # MR/L-MMSE/L-RZF/LP-MMSE are local schemes; ZF/RZF/MMSE/P-* are centralized.
        local = {PrecodingScheme.MR, PrecodingScheme.L_MMSE, PrecodingScheme.L_RZF,
                 PrecodingScheme.LP_MMSE}
        is_local = self.precoding in local
        if is_local and self.operation is OperationMode.CENTRALIZED:
            raise ValueError(
                f"{self.precoding.value} is a distributed (local) scheme but "
                "operation is CENTRALIZED; use operation=DISTRIBUTED or a centralized scheme."
            )
        if not is_local and self.operation is OperationMode.DISTRIBUTED:
            raise ValueError(
                f"{self.precoding.value} is a centralized scheme but "
                "operation is DISTRIBUTED; use operation=CENTRALIZED or a local scheme."
            )
        # Equal power is a centralized-only rule; distributed operation must use
        # the per-AP fractional allocation (see mimo_helpers.power_control).
        if (self.power_alloc is PowerControlScheme.EQUAL
                and self.operation is OperationMode.DISTRIBUTED):
            raise ValueError(
                "power_alloc=EQUAL is a centralized-only heuristic but operation is "
                "DISTRIBUTED; use power_alloc=FRACTIONAL (the only local rule)."
            )

        # The same cooperation-level coupling on the uplink side. MR/L-MMSE/L-RZF
        # are formed at the APs from local CSI; ZF/RZF/MMSE need the collective
        # y[q] at the CPU. No uplink counterpart of the EQUAL restriction exists,
        # because each user owns its budget instead of sharing a network one.
        local_combiners = {CombiningScheme.MR, CombiningScheme.L_MMSE,
                           CombiningScheme.L_RZF}
        is_local_combiner = self.combining in local_combiners
        if is_local_combiner and self.operation is OperationMode.CENTRALIZED:
            raise ValueError(
                f"{self.combining.value} is a distributed (local) combiner but "
                "operation is CENTRALIZED; use operation=DISTRIBUTED or a centralized combiner."
            )
        if not is_local_combiner and self.operation is OperationMode.DISTRIBUTED:
            raise ValueError(
                f"{self.combining.value} is a centralized combiner but "
                "operation is DISTRIBUTED; use operation=CENTRALIZED or a local combiner."
            )
        # Fusion only exists when the APs produce separate soft estimates.
        if (self.fusion is FusionRule.LSFD
                and self.operation is OperationMode.CENTRALIZED):
            raise ValueError(
                "fusion=LSFD weights the per-AP soft estimates of distributed operation, "
                "but operation is CENTRALIZED, where the CPU combines the raw samples; "
                "use fusion=EQUAL or operation=DISTRIBUTED."
            )
        # Statistical fusion weights leave the CPU without the instantaneous
        # effective channel, so the genie-aided SINR is then optimistic.
        if (self.fusion is FusionRule.LSFD
                and self.ul_se_bound is SEBound.INSTANTANEOUS):
            warnings.warn(
                "fusion=LSFD uses statistical weights, so the CPU does not know the "
                "instantaneous effective channel v_k^H h_k; the INSTANTANEOUS uplink SE "
                "is then optimistic. Set ul_se_bound=SEBound.UATF for the achievable bound.",
                stacklevel=2,
            )

    # --- Derived: array and geometry --------------------------------------
    @property
    def M_tot(self) -> int:
        """Total number of distributed antennas, M_tot = L * M."""
        return self.L * self.M

    @property
    def wavelength(self) -> float:
        """Carrier wavelength lambda_c = c / f_c [m]."""
        return SPEED_OF_LIGHT / self.f_c

    @property
    def antenna_spacing(self) -> float:
        """Half-wavelength antenna spacing [m]."""
        return self.wavelength / 2

    # --- Derived: noise ---------------------------------------------------
    @property
    def noise_power_dBm(self) -> float:
        """Receiver noise power over the full bandwidth B [dBm].

        sigma^2 = -174 dBm/Hz + 10 log10(B) + noise figure.
        """
        return THERMAL_NOISE_DENSITY_DBM_HZ + lin_to_db(self.B) + self.noise_figure_dB

    @property
    def noise_power(self) -> float:
        """Receiver noise power sigma^2 over the full bandwidth B, in watts."""
        return dbm_to_watt(self.noise_power_dBm)

    @property
    def noise_power_sc(self) -> float:
        """Receiver noise power in one of the Q evaluated subcarriers [W].

        Both system models are written per subcarrier, so a per-subcarrier signal
        power has to meet a per-subcarrier noise power. The transmit budgets
        (``rho_max``, ``p_max``) are totals over the OFDM block, and the pipeline
        spreads them over the ``Q`` evaluated subcarriers, so the matching noise
        is ``sigma^2 / Q``.

        Only the *ratio* is physical, and it is the same whichever count is used
        on both sides: an AP really spreads ``rho_max`` over all ``B / Delta_f``
        subcarriers of the band against a noise ``sigma^2 / (B / Delta_f)``, and
        the ``Q`` modelled subcarriers sample that. Dividing signal and noise by
        the same ``Q`` therefore reproduces the true per-subcarrier SNR. What is
        *not* physical is mixing the two, which is why this property exists
        rather than the full-band :attr:`noise_power` being used directly in the
        SINR expressions.

        The uplink never needs this: eq. ul-sinr is invariant to a common factor
        on ``(p, sigma^2)``, so the uplink routines work with block totals and
        the factor cancels itself (see :mod:`mimo_helpers`). The downlink has no
        such invariance, because the precoder carries the power and the receiver
        noise does not scale with it.
        """
        return self.noise_power / self.Q

    # --- Derived: precoding / power control -------------------------------
    @property
    def rzf_regularization(self) -> float:
        """RZF loading term lambda; defaults to the noise power sigma^2 if unset."""
        return self.noise_power if self.rzf_reg is None else self.rzf_reg

    @property
    def ul_rzf_regularization(self) -> float:
        """Uplink RZF/L-RZF loading; defaults to sigma^2 / p_max if unset.

        Unlike the downlink, where the loading is a free heuristic, the uplink
        MMSE loading is physically determined: eq. ul-centralized-rzf loads the
        Gram matrix with ``sigma^2 P^{-1}``, which for equal powers ``p_k = p``
        is the scalar ``sigma^2 / p``. The default takes the reference power to
        be the per-user budget ``p_max``, so RZF coincides with MMSE combining
        when every user transmits at full power.
        """
        return (self.noise_power / self.p_max if self.ul_rzf_reg is None
                else self.ul_rzf_reg)

    # --- Derived: frame bookkeeping ---------------------------------------
    @property
    def tau_UL(self) -> float:
        """Uplink share of the frame, the TDD complement ``1 - tau_DL``."""
        return 1.0 - self.tau_DL

    @property
    def dl_prelog(self) -> float:
        """Downlink prelog ``tau_DL (1 - tau_DL,sig) xbar_DL`` of eq. se.

        The data fraction of the frame: reference signalling and the deactivated
        (micro-sleep) symbols carry no payload, so only this share of the frame
        is charged with the delivered rate.
        """
        return self.tau_DL * (1 - self.tau_DLsig) * self.xbar_DL

    @property
    def ul_prelog(self) -> float:
        """Uplink prelog ``tau_UL (1 - tau_UL,sig) xbar_UL`` of eq. se.

        ``tau_ULsig = 1`` is the degenerate downlink-only frame in which the
        uplink phase carries nothing but pilots and the uplink rate vanishes by
        construction.
        """
        return self.tau_UL * (1 - self.tau_ULsig) * self.xbar_UL

    @property
    def tau_p(self) -> float:
        """Reference-signalling samples per coherence block.

        The frame spans one coherence block of ``tau_c`` samples, so the
        signalling fractions of both directions convert to a sample count. It is
        not used in any SE computation (perfect CSI is assumed and the overhead
        is charged through the prelogs); it exists so the frame can be checked
        against the ``tau_p >= K`` that orthogonal pilots would require.
        """
        return (self.tau_DL * self.tau_DLsig
                + self.tau_UL * self.tau_ULsig) * self.tau_c

    @property
    def B_tilde(self) -> float:
        """Effective bandwidth ``0.9 B`` [Hz] that carries the delivered rate.

        The sum rate of the system model is ``R_i = B_tilde * sum_k SE_k^i``,
        not ``B * sum_k SE_k^i``. The same effective bandwidth drives the
        encoder and decoder models of the power package, which are given
        ``R / B_tilde``, so the two packages must use one convention.
        """
        return self.B_tilde_factor * self.B

    # --- Path-loss model --------------------------------------------------
    def path_loss_dB(self, distance):
        """Mean (shadowing-free) path loss at a 3-D distance [m], returned in dB.

        3GPP Urban Microcell log-distance model used for the simulations in
        Bjornson & Sanguinetti, *Foundations of User-Centric Cell-Free Massive
        MIMO* (Sec. 2.5.2, eq. for beta_kl): the large-scale fading channel gain
        is ``beta_kl [dB] = -30.5 - 36.7 log10(d/1m) + F_kl``, i.e. a path loss
        ``PL(d) = pathloss_ref_loss_dB + 10 * n * log10(d / d0)`` with the
        reference loss (30.5 dB at d0 = 1 m) and exponent n = 3.67 = 36.7/10.
        Add an independent ``N(0, shadow_std_dB^2)`` shadowing term in the caller.
        The returned value is a loss (positive dB); the gain is its negation.

        The reference loss is calibrated for the 2 GHz band; it is representative
        of other sub-6 GHz bands but does not model the extra loss at the FR3
        default carrier ``f_c``. Accepts a scalar or a NumPy array of distances.
        """
        d = np.maximum(distance, self.min_ap_ue_distance)
        return self.pathloss_ref_loss_dB + 10 * self.pathloss_exponent * np.log10(d / self.ref_distance)

    # --- Reporting --------------------------------------------------------
    def summary(self) -> str:
        """Human-readable one-block summary of the configuration."""
        return (
            "D-MIMO configuration\n"
            f"  APs L                 : {self.L}\n"
            f"  antennas/AP M         : {self.M}  (M_tot = {self.M_tot})\n"
            f"  users K               : {self.K}\n"
            f"  subcarriers Q         : {self.Q}\n"
            f"  carrier f_c           : {self.f_c/1e9:.2f} GHz "
            f"(lambda = {self.wavelength*1e3:.2f} mm)\n"
            f"  bandwidth B           : {self.B/1e6:.1f} MHz\n"
            f"  noise power sigma^2   : {self.noise_power_dBm:.1f} dBm "
            f"({self.noise_power*1e12:.3f} pW)\n"
            f"  max power/AP rho_max  : {watt_to_dbm(self.rho_max):.1f} dBm "
            f"({self.rho_max:.2f} W)\n"
            f"  max power/UE p_max    : {watt_to_dbm(self.p_max):.1f} dBm "
            f"({self.p_max*1e3:.0f} mW)\n"
            f"  channel model         : {self.channel_model.value}"
            f"{' (forced NLOS)' if self.force_nlos else ' (38.901 LOS prob.)'}\n"
            f"  operation             : {self.operation.value}\n"
            f"  DL precoding / alloc  : {self.precoding.value} / "
            f"{self.power_alloc.value} (v = {self.v})\n"
            f"  UL combining / alloc  : {self.combining.value} / "
            f"{self.ul_power_alloc.value} (v_ul = {self.v_ul})\n"
            f"  UL fusion / SE bound  : {self.fusion.value} / {self.ul_se_bound.value}\n"
            f"  frame DL / UL         : {self.tau_DL:.2f} / {self.tau_UL:.2f} "
            f"(signalling {self.tau_DLsig:.3f} / {self.tau_ULsig:.3f}, "
            f"load {self.xbar_DL:.2f} / {self.xbar_UL:.2f})\n"
            f"  coherence tau_c       : {self.tau_c} samples "
            f"({self.tau_p:.1f} of them signalling)\n"
            f"  prelogs DL / UL       : {self.dl_prelog:.3f} / {self.ul_prelog:.3f}\n"
            f"  channel realizations  : {self.n_realizations}\n"
            f"  RNG seed              : {self.seed}"
        )


if __name__ == "__main__":
    cfg = DMIMOConfig()
    print(cfg.summary())
