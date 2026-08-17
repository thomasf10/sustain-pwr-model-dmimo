"""Parameters of the distributed (cell-free) FR3 power model.

:class:`DMIMOPowerParams` extends the co-located :class:`fr3_power.PowerParams`
rather than restating it, so every hardware constant of the single-site model
(PA, converters, mixers, computational efficiencies, reduction factors) is
inherited and a distributed AP is described by the *same* component models
evaluated at ``M_RF = M``. Only the genuinely new quantities are added here: the
fronthaul, the central unit, the per-AP synchronization, and the functional
split.

The class also owns the two consistency hazards that
``../README.md`` warns about, and resolves both in
:meth:`DMIMOPowerParams.from_rate_config`:

* the **frame**, described as ``tau_DL``/``tau_sig`` here and as
  ``tau_c``/``tau_p``/``tau_u`` in ``D_MIMO_rate``;
* the **coherence block**, which the co-located model derives from Doppler
  (``upsilon_coh``) and the rate model states outright (``tau_c``).

Simplifications, as instructed for this evaluation:

* every AP is active, so the deep-sleep set of eq. pnet is empty and
  ``delta_ds``, ``P_AP_0``, ``P_FH_0_sleep`` are not modelled;
* full cell-free service, ``D_l = {1..K}``, so ``K_l = K`` and every AP carries
  the same load, which collapses eq. frameavg_ap onto eq. frameavg;
* only splits S1 and S3 are implemented (see :class:`Split`).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fr3_power.config import PowerParams


class Split(str, Enum):
    """Functional split: where each digital operation runs (Table `tab:split`).

    ``S1`` forwards precoded samples. The CPU encodes, computes the precoder and
    *applies* it, sending each AP the ``M`` streams of frequency-domain samples
    it must radiate; the AP only does OFDM, predistortion, baseband filtering,
    and conversion. The fronthaul carries samples at the constellation rate,
    independently of the delivered traffic.

    ``S2`` forwards coefficients and data. The CPU encodes and computes
    ``W_l[q]``, but AP ``l`` applies it locally, so the fronthaul carries the
    ``K_l`` data streams plus the precoding coefficients, refreshed once per
    coherence block. Because linear combining is distributive over the APs, an
    AP holding its combining coefficients forms its own partial sum, so S2
    attains the *centralized* uplink rate while forwarding ``K_l`` scalar
    streams instead of ``M`` sample streams.

    ``S3`` is fully distributed (local) operation. The CPU only encodes; each AP
    computes and applies its own precoder from local CSI. The fronthaul carries
    payload in the downlink and per-AP partial sums in the uplink.

    S1 and S2 both realize the centralized design and therefore deliver the
    *same rate*; they differ only in where the work is done and what crosses the
    fronthaul. S3 delivers the lower rate of local processing.
    """

    S1 = "S1"  # CPU computes and applies; fronthaul carries samples
    S2 = "S2"  # CPU computes, AP applies; fronthaul carries data + coefficients
    S3 = "S3"  # AP computes and applies; fronthaul carries data


class PASizing(str, Enum):
    """How the maximum PA output power ``P_max`` relates to the AP budget.

    The two conventions give different scaling laws for the network static floor
    and must not be mixed (Remark `rem:pa_sizing`). ``PER_AP_BUDGET`` sizes each
    PA for its share of the AP budget, ``P_max = rho_max / M``, which makes a
    fully loaded AP consume exactly ``rho_max / eta_PAmax`` and the network floor
    ``L xi rho_max / eta_PAmax``: it grows with the number of APs, not with the
    number of antennas. ``FIXED_RATING`` keeps ``P_max`` at the component rating
    already in :class:`fr3_power.PowerParams`, and the floor becomes
    ``L M xi P_max / eta_PAmax``, growing with the total antenna count.

    ``PER_AP_BUDGET`` is the default because it is the only one that keeps the
    comparison fair as the transmit budget is swept: the amplifiers are assumed
    dimensioned for the budget they are actually given.
    """

    PER_AP_BUDGET = "per-ap-budget"  # P_max = rho_max / M
    FIXED_RATING = "fixed-rating"    # P_max as given in PowerParams


@dataclass
class DMIMOPowerParams(PowerParams):
    """Hardware and model constants of the distributed network.

    Inherits every co-located constant from :class:`fr3_power.PowerParams`; the
    fields below are the distributed additions. Values marked UNSOURCED have no
    counterpart in the co-located model and are not yet backed by a reference
    (see the parameter to-do list at the end of ``dmimo_pwr_model.tex``). They
    are placeholders chosen to be of the right order, and any conclusion that
    turns on one of them should be reported as a sensitivity range rather than a
    point value. :func:`unsourced_parameters` lists them at runtime.
    """

    # --- Topology ---------------------------------------------------------
    L: int = 16                 # Number of APs
    M: int = 4                  # Antennas per AP (fully digital, so M_RF = M)
    split: Split = Split.S1

    # --- Supply and cooling of the new nodes (eq. pnet) --------------------
    # The CPU efficiency is an inverse power-usage effectiveness: the central
    # unit sits in a cooled room, which is not an outdoor AP enclosure, so it
    # should not be set to the macro-base-station eta_c_sc values.
    eta_FH_sc: float = 0.9      # Fronthaul transceiver supply and cooling  [UNSOURCED]
    eta_CPU_sc: float = 0.8     # Central unit, ~ 1/PUE at PUE = 1.25       [UNSOURCED]

    # --- Fronthaul (eq. fh_basic), parametrisation of ngo2018total ---------
    P_FH_0: float = 0.825       # Traffic-independent power per link [W]
    Pi_FH: float = 0.25e-9      # Traffic-dependent coefficient [W per bit/s] (0.25 W/Gbit/s)
    b_FH: int = 12              # Fronthaul bits per real sample            [UNSOURCED]
    delta_FH_micro: float = 0.9  # Idle reduction factor of a transceiver   [UNSOURCED]

    # --- Central unit (eq. cpu) -------------------------------------------
    P_CPU_0: float = 50.0       # Always-on consumption of the central unit [W]  [UNSOURCED]
    # Chains per FPGA on the CPU side. eq. pmimo names Mbar_RF only for the AP;
    # the CPU is taken to process the collective L*M-antenna array, so it runs
    # ceil(L*M / cpu_rf_chains_per_fpga) FPGAs.
    cpu_rf_chains_per_fpga: int = 32

    # --- Per-AP synchronization (eq. ana_avg_ap) ---------------------------
    # Coherent joint transmission across separate nodes needs a shared frequency
    # and phase reference and reciprocity-calibrated TDD chains. It costs power
    # at every AP continuously, which is why it sits outside the direction
    # averaging. A co-located array does not pay it.
    P_sync: float = 0.5         # Per-AP reference distribution + calibration [W]  [UNSOURCED]

    # --- Power amplifier sizing (Remark rem:pa_sizing) ---------------------
    pa_sizing: PASizing = PASizing.PER_AP_BUDGET

    # --- Coherence block ---------------------------------------------------
    # When set, overrides the Doppler-derived PowerParams.upsilon_coh so that the
    # precoder amortisation and the rate model's pilot overhead use one and the
    # same block length. from_rate_config sets it to DMIMOConfig.tau_c.
    upsilon_coh_override: Optional[float] = None

    def __post_init__(self) -> None:
        self.split = Split(self.split)
        self.pa_sizing = PASizing(self.pa_sizing)
        for name in ("L", "M", "K", "b_FH", "cpu_rf_chains_per_fpga"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        for name in ("eta_FH_sc", "eta_CPU_sc"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in (0, 1], got {getattr(self, name)}")

    # --- Derived ----------------------------------------------------------
    @property
    def M_tot(self) -> int:
        """Total number of distributed antennas, L * M."""
        return self.L * self.M

    @property
    def upsilon_coh(self) -> float:
        """Samples per coherence block, overriding the Doppler-derived default.

        :class:`fr3_power.PowerParams` derives this from the delay spread and the
        receiver velocity, while ``D_MIMO_rate`` states ``tau_c`` outright. They
        are the same physical quantity, so letting both stand would make the
        combined model assume two different fading rates: the precoder
        amortisation of eq. gops_cen would use one block length and the pilot
        overhead of the rate model another. ``from_rate_config`` sets the
        override from ``tau_c`` so a single number governs both.
        """
        if self.upsilon_coh_override is not None:
            return float(self.upsilon_coh_override)
        return super().upsilon_coh

    @property
    def cpu_fpgas(self) -> float:
        """FPGAs the CPU runs the centralized precoder on, ceil(L*M / chains)."""
        import numpy as np
        return float(np.ceil(self.M_tot / self.cpu_rf_chains_per_fpga))

    def P_max_for(self, rho_max: float) -> float:
        """Maximum PA output power [W] under the configured sizing convention."""
        if self.pa_sizing is PASizing.PER_AP_BUDGET:
            return rho_max / self.M
        return self.P_max

    # --- Construction from a rate-model configuration ----------------------
    @classmethod
    def from_rate_config(cls, cfg, **overrides) -> "DMIMOPowerParams":
        """Build parameters whose frame and topology match a ``DMIMOConfig``.

        This is the join between the two packages, and it exists because the
        frame is described in two vocabularies that nothing otherwise forces to
        agree. ``D_MIMO_rate`` splits a coherence block of ``tau_c`` samples into
        ``tau_p`` pilots, ``tau_u`` uplink data and ``tau_d`` downlink data; the
        power model works in frame fractions. Charging the pilots to the uplink
        signalling phase, as ``dmimo_sysmodel.tex`` does, pins the mapping:

            tau_DL    = tau_d / tau_c
            tau_UL    = (tau_p + tau_u) / tau_c        ( = 1 - tau_DL )
            tau_ULsig = tau_p / (tau_p + tau_u)
            tau_DLsig = 0                              (downlink is all data)

        so that ``tau_UL * tau_ULsig = tau_p / tau_c`` is the pilot share and
        ``tau_UL * (1 - tau_ULsig) = tau_u / tau_c`` is the uplink data share.
        The degenerate ``tau_u = 0`` gives ``tau_ULsig = 1``, the downlink-only
        frame in which the uplink phase carries nothing but pilots.

        The carrier, bandwidth, subcarrier spacing, user count, and topology are
        copied across, and ``upsilon_coh`` is pinned to ``tau_c``.

        Args:
            cfg: A ``config_dmimo.DMIMOConfig``.
            **overrides: Any field of :class:`DMIMOPowerParams`, applied last.

        Returns:
            Parameters consistent with ``cfg``.
        """
        ul_samples = cfg.tau_p + cfg.tau_u
        if ul_samples == 0:
            raise ValueError(
                "tau_p + tau_u = 0 leaves no uplink phase at all; the frame mapping "
                "charges the pilots to the uplink signalling phase and needs tau_p > 0."
            )
        fields = dict(
            K=cfg.K,
            f_c=cfg.f_c,
            B=cfg.B,
            Delta_f=cfg.Delta_f,
            L=cfg.L,
            M=cfg.M,
            tau_DL=cfg.tau_d / cfg.tau_c,
            tau_DLsig=0.0,
            tau_ULsig=cfg.tau_p / ul_samples,
            upsilon_coh_override=float(cfg.tau_c),
        )
        fields.update(overrides)
        p = cls(**fields)

        # The split and the rate model's cooperation level must agree, or the
        # power of one system would be charged against the rate of another.
        from config_dmimo import OperationMode
        centralized = cfg.operation is OperationMode.CENTRALIZED
        expected = {Split.S1, Split.S2} if centralized else {Split.S3}
        if p.split not in expected:
            names = " or ".join(sorted(s.value for s in expected))
            warnings.warn(
                f"split={p.split.value} but the rate configuration is "
                f"{cfg.operation.value} operation, which is {names}. The power "
                "and the rate would then describe different systems.",
                stacklevel=2,
            )
        return p


#: Parameters with no counterpart in the co-located model and no sourced value.
#: Listed so a report can state plainly which numbers are placeholders.
UNSOURCED = (
    ("eta_FH_sc", "fronthaul transceiver supply and cooling"),
    ("eta_CPU_sc", "central-unit supply and cooling (inverse data-centre PUE)"),
    ("b_FH", "fronthaul quantiser resolution; sets the S1 fronthaul rate outright"),
    ("delta_FH_micro", "idle reduction factor of an optical transceiver"),
    ("P_CPU_0", "always-on consumption of the central unit"),
    ("P_sync", "per-AP reference distribution and reciprocity calibration"),
)


def unsourced_parameters(p: DMIMOPowerParams) -> str:
    """Human-readable list of the placeholder parameters and their values."""
    lines = ["Parameters with no sourced value (placeholders; treat as a sensitivity range):"]
    for name, what in UNSOURCED:
        lines.append(f"  {name:<16} = {getattr(p, name):<10} {what}")
    lines.append("  Inherited PA constants (xi, alpha, eta_PAmax) were fitted for macro-cell")
    lines.append("  amplifiers and should be refitted for AP-class hardware.")
    return "\n".join(lines)
