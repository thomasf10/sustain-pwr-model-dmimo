"""Configuration for the distributed massive MIMO (cell-free) rate model.

A single dataclass, :class:`DMIMOConfig`, gathers every parameter of the
downlink system model of ``sections/dmimo_sysmodel.tex`` so that the rate
scripts (``dl_rate.py``, ``ul_rate.py``) share one source of truth. The symbols
mirror the manuscript:

    L        number of access points (APs)
    M        antennas per AP            (total array size M_tot = L * M)
    K        single-antenna users
    Q        OFDM data subcarriers
    rho_max  maximum DL transmit power per AP        [W]
    sigma^2  receiver noise power (derived from B and the noise figure)
    kappa    fractional power-control exponent  (eq. fractional-power)
    lambda   RZF loading term                   (eq. zf-precoder / RZF)

Parameters are grouped into topology, RF band, noise, power/precoding,
propagation, channel-estimation bookkeeping, and Monte Carlo blocks. Quantities
that are fully determined by these inputs (M_tot, wavelength, noise power, the
prelog factor, ...) are exposed as read-only properties rather than stored, so
they cannot drift out of sync.
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


class OperationMode(str, Enum):
    """Where the precoding is computed."""

    CENTRALIZED = "centralized"  # CPU designs all directions from global CSI
    DISTRIBUTED = "distributed"  # each AP designs its directions from local CSI


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
    """Parameters of the downlink distributed massive MIMO system model.

    Defaults describe an upper mid-band (FR3) cell-free deployment consistent
    with the accompanying power model: a large number of few-antenna APs jointly
    serving several users over a wide bandwidth, with far more distributed
    antennas than users so that zero-forcing is feasible.
    """

    # --- Topology ---------------------------------------------------------
    L: int = 10                # Number of APs
    M: int = 4                  # Antennas per AP
    K: int = 20                 # Single-antenna users
    Q: int = 64                 # OFDM data subcarriers evaluated

    # --- RF band ----------------------------------------------------------
    f_c: float = 7e9            # Carrier frequency [Hz]
    B: float = 400e6            # Bandwidth [Hz]
    Delta_f: float = 120e3      # Subcarrier spacing [Hz]

    # --- Noise ------------------------------------------------------------
    noise_figure_dB: float = 8.0  # Receiver noise figure [dB]

    # --- Power and precoding ----------------------------------------------
    rho_max: float = 1.0        # Max DL transmit power per AP [W]
    precoding: PrecodingScheme = PrecodingScheme.ZF
    operation: OperationMode = OperationMode.CENTRALIZED
    kappa: float = 0.5          # Fractional power-control exponent
    rzf_reg: Optional[float] = None  # RZF loading lambda; None -> sigma^2 (see property)

    # --- Channel model backend --------------------------------------------
    channel_model: ChannelModel = ChannelModel.SIONNA_UMI
    force_nlos: bool = True      # Force all links NLOS (keeps the channel zero-mean)
    o2i_model: str = "low"      # 38.901 outdoor-to-indoor model; UEs are outdoor here
    antenna_pattern: str = "omni"  # AP element pattern ("omni" or "38.901")

    # --- Propagation / deployment -----------------------------------------
    area_size: float = 1000.0   # Square coverage-area side [m] (wrap-around)
    ap_height: float = 10.0     # AP height [m]
    ue_height: float = 1.5      # UE height [m]
    # The following four parameters drive the analytical RAYLEIGH channel only;
    # the SIONNA_UMI backend takes its path loss and shadowing from 38.901.
    pathloss_exponent: float = 3.67  # Log-distance path-loss exponent (= 36.7/10)
    pathloss_ref_loss_dB: float = 30.5  # Path loss at d0 (3GPP UMi, 2 GHz) [dB]
    shadow_std_dB: float = 4.0  # Log-normal shadowing std [dB]
    ref_distance: float = 1.0   # Path-loss reference distance d0 [m]
    min_ap_ue_distance: float = 1.0  # Floor on AP-UE distance [m]

    # --- Channel estimation / coherence bookkeeping -----------------------
    tau_c: int = 200            # Coherence block length [samples]
    tau_p: int = 20             # Uplink pilot length [samples], tau_p <= tau_c

    # --- Monte Carlo ------------------------------------------------------
    n_realizations: int = 100   # Channel realizations averaged per SE point
    seed: int = 0               # RNG seed for reproducibility

    def __post_init__(self) -> None:
        # Accept plain strings for the enum fields (e.g. from a config file).
        self.precoding = PrecodingScheme(self.precoding)
        self.operation = OperationMode(self.operation)
        self.channel_model = ChannelModel(self.channel_model)

        for name in ("L", "M", "K", "Q", "tau_c", "tau_p", "n_realizations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer, got {getattr(self, name)}")
        for name in ("f_c", "B", "Delta_f", "rho_max", "area_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.tau_p > self.tau_c:
            raise ValueError(
                f"pilot length tau_p={self.tau_p} exceeds coherence block tau_c={self.tau_c}"
            )
        if self.K > self.M_tot:
            warnings.warn(
                f"K={self.K} exceeds the total number of antennas M_tot={self.M_tot}; "
                "zero-forcing/MMSE precoding is rank-deficient in this regime.",
                stacklevel=2,
            )
        if self.tau_p < self.K:
            warnings.warn(
                f"tau_p={self.tau_p} < K={self.K}: pilots are reused across users, so "
                "channel estimates and the SINR are subject to pilot contamination.",
                stacklevel=2,
            )
        # MR/L-MMSE/LP-MMSE are local schemes; ZF/RZF/MMSE/P-* are centralized.
        local = {PrecodingScheme.MR, PrecodingScheme.L_MMSE, PrecodingScheme.L_RZF,
                 PrecodingScheme.LP_MMSE}
        is_local = self.precoding in local
        if is_local and self.operation is OperationMode.CENTRALIZED:
            warnings.warn(
                f"{self.precoding.value} is a distributed (local) scheme but "
                "operation is CENTRALIZED; check the intended configuration.",
                stacklevel=2,
            )
        if not is_local and self.operation is OperationMode.DISTRIBUTED:
            warnings.warn(
                f"{self.precoding.value} is a centralized scheme but "
                "operation is DISTRIBUTED; check the intended configuration.",
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
        """Receiver noise power sigma^2 in watts."""
        return dbm_to_watt(self.noise_power_dBm)

    # --- Derived: precoding / power control -------------------------------
    @property
    def rzf_regularization(self) -> float:
        """RZF loading term lambda; defaults to the noise power sigma^2 if unset."""
        return self.noise_power if self.rzf_reg is None else self.rzf_reg

    # --- Derived: coherence-block bookkeeping -----------------------------
    @property
    def tau_d(self) -> int:
        """Samples per coherence block available for downlink data, tau_c - tau_p."""
        return self.tau_c - self.tau_p

    @property
    def dl_prelog(self) -> float:
        """Downlink prelog factor (tau_c - tau_p) / tau_c of the SE expression."""
        return self.tau_d / self.tau_c

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
            "D-MIMO downlink configuration\n"
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
            f"  channel model         : {self.channel_model.value}"
            f"{' (NLOS)' if self.force_nlos else ''}\n"
            f"  precoding / operation : {self.precoding.value} / {self.operation.value}\n"
            f"  power-control kappa   : {self.kappa}\n"
            f"  coherence tau_c/tau_p : {self.tau_c}/{self.tau_p} "
            f"(DL prelog {self.dl_prelog:.3f})\n"
            f"  Monte Carlo           : {self.n_realizations} realizations, seed {self.seed}"
        )


if __name__ == "__main__":
    cfg = DMIMOConfig()
    print(cfg.summary())
